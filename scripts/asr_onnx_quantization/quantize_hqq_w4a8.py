#!/usr/bin/env python3
"""Convert constant-RHS ONNX MatMul nodes to HQQ W4A8 MatMulNBits.

The ONNX Runtime HQQ quantizer emits weight-only ``MatMulNBits`` nodes but does
not currently expose ``accuracy_level`` on ``HQQWeightOnlyQuantConfig``.  This
tool therefore applies HQQ first and then sets ``accuracy_level=4`` explicitly,
which requests block-wise INT8 activation computation in the CPU kernel.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper


BITS = 4
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4


class QuantizationContractError(RuntimeError):
    """Raised when a graph does not satisfy the requested W4A8 contract."""


def _attribute_int(node: onnx.NodeProto, name: str) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(helper.get_attribute_value(attribute))
    return None


def _set_attribute_int(node: onnx.NodeProto, name: str, value: int) -> None:
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


def _initializer_names(model: ModelProto) -> set[str]:
    return {initializer.name for initializer in model.graph.initializer}


def _gemm_attributes(node: onnx.NodeProto) -> dict[str, float | int]:
    attributes = {
        "alpha": 1.0,
        "beta": 1.0,
        "transA": 0,
        "transB": 0,
    }
    for attribute in node.attribute:
        if attribute.name in attributes:
            attributes[attribute.name] = helper.get_attribute_value(attribute)
    return attributes


def lower_eligible_gemm_to_matmul(model: ModelProto) -> tuple[int, list[str]]:
    """Lower standard linear Gemm nodes to MatMul(+Add) for MatMulNBits.

    Weight transposition is performed once in the initializer, not at runtime.
    Gemm variants with transposed activations or non-unit alpha/beta are kept in
    FP32 and returned in the skipped-node list.
    """

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    used_names = set(initializers)
    lowered = 0
    skipped = []
    replacement_nodes = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "Gemm":
            replacement_nodes.append(node)
            continue
        attributes = _gemm_attributes(node)
        supported = (
            len(node.input) >= 2
            and node.input[1] in initializers
            and int(attributes["transA"]) == 0
            and float(attributes["alpha"]) == 1.0
            and (len(node.input) < 3 or float(attributes["beta"]) == 1.0)
        )
        if not supported:
            skipped.append(node.name or f"Gemm[{index}]")
            replacement_nodes.append(node)
            continue

        rhs_name = node.input[1]
        if int(attributes["transB"]) == 1:
            rhs = np.asarray(numpy_helper.to_array(initializers[rhs_name]))
            if rhs.ndim != 2:
                skipped.append(node.name or f"Gemm[{index}]")
                replacement_nodes.append(node)
                continue
            base_name = f"{rhs_name}__matmul_rhs"
            candidate = base_name
            suffix = 1
            while candidate in used_names:
                suffix += 1
                candidate = f"{base_name}_{suffix}"
            rhs_name = candidate
            used_names.add(rhs_name)
            model.graph.initializer.append(
                numpy_helper.from_array(np.ascontiguousarray(rhs.T), name=rhs_name)
            )

        base_node_name = node.name or f"Gemm_{index}"
        if len(node.input) >= 3 and node.input[2]:
            matmul_output = f"{node.output[0]}__pre_bias"
            replacement_nodes.append(
                helper.make_node(
                    "MatMul",
                    [node.input[0], rhs_name],
                    [matmul_output],
                    name=f"{base_node_name}/MatMul",
                )
            )
            replacement_nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_output, node.input[2]],
                    list(node.output),
                    name=f"{base_node_name}/BiasAdd",
                )
            )
        else:
            replacement_nodes.append(
                helper.make_node(
                    "MatMul",
                    [node.input[0], rhs_name],
                    list(node.output),
                    name=f"{base_node_name}/MatMul",
                )
            )
        lowered += 1

    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)
    return lowered, skipped


def _transpose_perm(node: onnx.NodeProto) -> tuple[int, ...] | None:
    if node.op_type != "Transpose":
        return None
    for attribute in node.attribute:
        if attribute.name == "perm":
            return tuple(int(value) for value in helper.get_attribute_value(attribute))
    return None


def _replace_node_inputs(
    nodes: list[onnx.NodeProto], old_name: str, new_name: str
) -> None:
    for node in nodes:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name


def _pointwise_conv_rhs(
    initializer: onnx.TensorProto, *, output_name: str
) -> onnx.TensorProto:
    weight = np.asarray(numpy_helper.to_array(initializer))
    if weight.ndim != 3 or weight.shape[2] != 1:
        raise QuantizationContractError(
            f"{initializer.name}: expected a Conv1d pointwise weight [M,C,1], got "
            f"{tuple(weight.shape)}"
        )
    rhs = np.ascontiguousarray(weight[:, :, 0].T)
    return numpy_helper.from_array(rhs, name=output_name)


def lower_fastconformer_pointwise_conv_pairs_to_matmul(
    model: ModelProto,
) -> dict[str, int]:
    """Lower FastConformer Conv1d pointwise pairs without extra transposes.

    Each NeMo FastConformer convolution block contains four NTC/NCT
    transposes around ``pointwise_conv1``, the depthwise convolution, and
    ``pointwise_conv2``.  MatMul operates in NTC layout.  Moving the first
    transpose after the GLU and removing both transposes around
    ``pointwise_conv2`` preserves the depthwise NCT layout while reducing the
    block to two transposes.  Weights are transposed once in the initializer.
    """

    nodes = list(model.graph.node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    pair_count = 0
    removed_transpose_count = 0

    conv1_nodes = [
        node
        for node in nodes
        if node.op_type == "Conv" and node.name.endswith("/pointwise_conv1/Conv")
    ]
    for conv1 in conv1_nodes:
        conv2_name = conv1.name.replace("/pointwise_conv1/Conv", "/pointwise_conv2/Conv")
        conv2 = next((node for node in nodes if node.name == conv2_name), None)
        if conv2 is None:
            continue
        if len(conv1.input) not in (2, 3) or len(conv2.input) not in (2, 3):
            continue
        if conv1.input[1] not in initializers or conv2.input[1] not in initializers:
            continue
        if any(
            len(node.input) == 3 and node.input[2] not in initializers
            for node in (conv1, conv2)
        ):
            continue

        producers = {output: node for node in nodes for output in node.output}
        consumers = {}
        for node in nodes:
            for input_name in node.input:
                consumers.setdefault(input_name, []).append(node)

        input_transpose = producers.get(conv1.input[0])
        if input_transpose is None or _transpose_perm(input_transpose) != (0, 2, 1):
            continue
        if consumers.get(input_transpose.output[0], []) != [conv1]:
            continue

        split_consumers = consumers.get(conv1.output[0], [])
        if len(split_consumers) != 1 or split_consumers[0].op_type != "Split":
            continue
        split = split_consumers[0]
        if _attribute_int(split, "axis") != 1 or len(split.output) != 2:
            continue

        sigmoid = next(
            (
                node
                for node in consumers.get(split.output[1], [])
                if node.op_type == "Sigmoid" and len(node.output) == 1
            ),
            None,
        )
        if sigmoid is None:
            continue
        glu = next(
            (
                node
                for node in consumers.get(split.output[0], [])
                if node.op_type == "Mul"
                and split.output[0] in node.input
                and sigmoid.output[0] in node.input
            ),
            None,
        )
        if glu is None or not glu.output:
            continue

        activation = producers.get(conv2.input[0])
        if activation is None or activation.op_type != "Mul":
            continue
        activation_sigmoid = next(
            (
                producers.get(input_name)
                for input_name in activation.input
                if producers.get(input_name) is not None
                and producers[input_name].op_type == "Sigmoid"
            ),
            None,
        )
        if activation_sigmoid is None or not activation_sigmoid.input:
            continue
        activation_base = activation_sigmoid.input[0]
        if activation_base not in activation.input:
            continue
        activation_transpose = producers.get(activation_base)
        output_consumers = consumers.get(conv2.output[0], [])
        if len(output_consumers) != 1:
            continue
        output_transpose = output_consumers[0]
        if _transpose_perm(output_transpose) != (0, 2, 1):
            continue

        # Variant without a dedicated transpose in front of the activation:
        # keep the depthwise NCT layout and transpose after the first
        # bias-add instead.
        if activation_transpose is None or activation_transpose.op_type != "Transpose":
            old_weight_names = (conv1.input[1], conv2.input[1])
            if any(len(consumers.get(name, [])) != 1 for name in old_weight_names):
                continue
            rhs1_name = f"{conv1.input[1]}__matmul_rhs"
            rhs2_name = f"{conv2.input[1]}__matmul_rhs"
            if rhs1_name in initializers or rhs2_name in initializers:
                raise QuantizationContractError(f"initializer name collision in {conv1.name}")
            rhs1 = _pointwise_conv_rhs(
                initializers[conv1.input[1]], output_name=rhs1_name
            )
            rhs2 = _pointwise_conv_rhs(
                initializers[conv2.input[1]], output_name=rhs2_name
            )

            matmul1_output = f"{conv1.output[0]}__ntc_pre_bias"
            bias1_output = f"{conv1.output[0]}__ntc"
            matmul1 = helper.make_node(
                "MatMul",
                [input_transpose.input[0], rhs1_name],
                [matmul1_output],
                name=f"{conv1.name}/MatMul",
            )
            bias1 = helper.make_node(
                "Add",
                [matmul1_output, conv1.input[2]],
                [bias1_output],
                name=f"{conv1.name}/BiasAdd",
            )
            transpose1 = helper.make_node(
                "Transpose",
                [bias1_output],
                list(conv1.output),
                name=f"{conv1.name}/TransposeForDepthwise",
                perm=[0, 2, 1],
            )

            transpose2_output = f"{conv2.input[0]}__ntc"
            transpose2 = helper.make_node(
                "Transpose",
                [conv2.input[0]],
                [transpose2_output],
                name=f"{conv2.name}/TransposeForMatMul",
                perm=[0, 2, 1],
            )
            matmul2_output = f"{output_transpose.output[0]}__pre_bias"
            matmul2 = helper.make_node(
                "MatMul",
                [transpose2_output, rhs2_name],
                [matmul2_output],
                name=f"{conv2.name}/MatMul",
            )
            bias2 = helper.make_node(
                "Add",
                [matmul2_output, conv2.input[2]],
                list(output_transpose.output),
                name=f"{conv2.name}/BiasAdd",
            )

            replacement_nodes = []
            for node in nodes:
                if node is input_transpose:
                    replacement_nodes.extend([matmul1, bias1, transpose1])
                elif node is conv1 or node is output_transpose:
                    continue
                elif node is conv2:
                    replacement_nodes.extend([transpose2, matmul2, bias2])
                else:
                    replacement_nodes.append(node)
            nodes = replacement_nodes

            retained_initializers = [
                initializer
                for initializer in model.graph.initializer
                if initializer.name not in old_weight_names
            ]
            retained_initializers.extend([rhs1, rhs2])
            del model.graph.initializer[:]
            model.graph.initializer.extend(retained_initializers)
            initializers = {
                initializer.name: initializer for initializer in model.graph.initializer
            }
            pair_count += 1
            continue

        if activation_transpose is None or _transpose_perm(activation_transpose) != (0, 2, 1):
            continue
        activation_users = set(id(node) for node in consumers.get(activation_base, []))
        if activation_users != {id(activation), id(activation_sigmoid)}:
            continue

        old_weight_names = (conv1.input[1], conv2.input[1])
        if any(len(consumers.get(name, [])) != 1 for name in old_weight_names):
            continue
        rhs1_name = f"{conv1.input[1]}__matmul_rhs"
        rhs2_name = f"{conv2.input[1]}__matmul_rhs"
        if rhs1_name in initializers or rhs2_name in initializers:
            raise QuantizationContractError(f"initializer name collision in {conv1.name}")
        rhs1 = _pointwise_conv_rhs(initializers[conv1.input[1]], output_name=rhs1_name)
        rhs2 = _pointwise_conv_rhs(initializers[conv2.input[1]], output_name=rhs2_name)

        _set_attribute_int(split, "axis", 2)
        _replace_node_inputs(nodes, activation_transpose.output[0], activation_transpose.input[0])

        glu_channels_first = f"{glu.output[0]}__channels_first"
        glu_users = list(consumers.get(glu.output[0], []))
        for user in glu_users:
            for index, input_name in enumerate(user.input):
                if input_name == glu.output[0]:
                    user.input[index] = glu_channels_first
        post_glu_transpose = helper.make_node(
            "Transpose",
            [glu.output[0]],
            [glu_channels_first],
            name=f"{conv1.name}/TransposeForDepthwise",
            perm=[0, 2, 1],
        )
        matmul1_output = (
            conv1.output[0]
            if len(conv1.input) == 2
            else f"{conv1.output[0]}__pre_bias"
        )
        matmul1 = helper.make_node(
            "MatMul",
            [input_transpose.input[0], rhs1_name],
            [matmul1_output],
            name=f"{conv1.name}/MatMul",
        )
        bias1 = (
            helper.make_node(
                "Add",
                [matmul1_output, conv1.input[2]],
                list(conv1.output),
                name=f"{conv1.name}/BiasAdd",
            )
            if len(conv1.input) == 3
            else None
        )
        matmul2_output = (
            output_transpose.output[0]
            if len(conv2.input) == 2
            else f"{output_transpose.output[0]}__pre_bias"
        )
        matmul2 = helper.make_node(
            "MatMul",
            [conv2.input[0], rhs2_name],
            [matmul2_output],
            name=f"{conv2.name}/MatMul",
        )
        bias2 = (
            helper.make_node(
                "Add",
                [matmul2_output, conv2.input[2]],
                list(output_transpose.output),
                name=f"{conv2.name}/BiasAdd",
            )
            if len(conv2.input) == 3
            else None
        )

        replacement_nodes = []
        for node in nodes:
            if node is input_transpose:
                replacement_nodes.append(matmul1)
                if bias1 is not None:
                    replacement_nodes.append(bias1)
            elif node is conv1 or node is activation_transpose or node is output_transpose:
                continue
            elif node is glu:
                replacement_nodes.extend([node, post_glu_transpose])
            elif node is conv2:
                replacement_nodes.append(matmul2)
                if bias2 is not None:
                    replacement_nodes.append(bias2)
            else:
                replacement_nodes.append(node)
        nodes = replacement_nodes

        retained_initializers = [
            initializer
            for initializer in model.graph.initializer
            if initializer.name not in old_weight_names
        ]
        retained_initializers.extend([rhs1, rhs2])
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained_initializers)
        initializers = {
            initializer.name: initializer for initializer in model.graph.initializer
        }
        pair_count += 1
        removed_transpose_count += 2

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return {
        "pair_count": pair_count,
        "pointwise_conv_count": pair_count * 2,
        "removed_transpose_count": removed_transpose_count,
    }


def audit_quantization(model: ModelProto, *, require_w4a8: bool = False) -> dict[str, Any]:
    """Return graph counts and optionally enforce the HQQ W4A8 node contract."""

    initializer_names = _initializer_names(model)
    node_counts = Counter(node.op_type for node in model.graph.node)
    eligible_matmuls = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) >= 2 and node.input[1] in initializer_names
    ]
    dynamic_rhs_matmuls = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and (len(node.input) < 2 or node.input[1] not in initializer_names)
    ]
    nbits_nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"]

    contracts = Counter(
        (
            _attribute_int(node, "bits"),
            _attribute_int(node, "block_size"),
            _attribute_int(node, "accuracy_level"),
        )
        for node in nbits_nodes
    )
    if require_w4a8:
        for node in nbits_nodes:
            bits = _attribute_int(node, "bits")
            block_size = _attribute_int(node, "block_size")
            accuracy_level = _attribute_int(node, "accuracy_level")
            if bits != BITS:
                raise QuantizationContractError(
                    f"{node.name or '<unnamed>'}: expected bits={BITS}, got {bits}"
                )
            if block_size != BLOCK_SIZE:
                raise QuantizationContractError(
                    f"{node.name or '<unnamed>'}: expected block_size={BLOCK_SIZE}, got {block_size}"
                )
            if accuracy_level != ACCURACY_LEVEL:
                raise QuantizationContractError(
                    f"{node.name or '<unnamed>'}: expected accuracy_level=4 (W4A8), got "
                    f"{accuracy_level}"
                )

    initializer_bytes = sum(
        _tensor_storage_bytes(initializer) for initializer in model.graph.initializer
    )
    return {
        "node_count": len(model.graph.node),
        "node_types": dict(sorted(node_counts.items())),
        "initializer_count": len(model.graph.initializer),
        "initializer_storage_bytes": initializer_bytes,
        "eligible_matmul_count": len(eligible_matmuls),
        "gemm_count": node_counts["Gemm"],
        "dynamic_rhs_matmul_count": len(dynamic_rhs_matmuls),
        "matmul_nbits_count": len(nbits_nodes),
        "matmul_nbits_contracts": [
            {
                "bits": bits,
                "block_size": block_size,
                "accuracy_level": accuracy_level,
                "count": count,
            }
            for (bits, block_size, accuracy_level), count in sorted(
                contracts.items(), key=lambda item: tuple(-1 if value is None else value for value in item[0])
            )
        ],
        "qdq_quantize_linear_count": node_counts["QuantizeLinear"],
        "qdq_dequantize_linear_count": node_counts["DequantizeLinear"],
        "dynamic_quantize_linear_count": node_counts["DynamicQuantizeLinear"],
        "matmul_integer_count": node_counts["MatMulInteger"],
        "conv_integer_count": node_counts["ConvInteger"],
    }


def _tensor_storage_bytes(tensor: onnx.TensorProto) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    if tensor.external_data:
        for entry in tensor.external_data:
            if entry.key == "length":
                return int(entry.value)
    element_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.BFLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.INT64: 8,
        TensorProto.UINT64: 8,
        TensorProto.INT32: 4,
        TensorProto.UINT32: 4,
        TensorProto.INT16: 2,
        TensorProto.UINT16: 2,
        TensorProto.INT8: 1,
        TensorProto.UINT8: 1,
        TensorProto.INT4: 0.5,
        TensorProto.UINT4: 0.5,
    }
    size = element_sizes.get(tensor.data_type, 0)
    elements = 1
    for dimension in tensor.dims:
        elements *= int(dimension)
    return int(elements * size)


def quantize_hqq_w4a8(
    input_path: Path,
    output_path: Path,
    *,
    nodes_to_exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Quantize one model and return a serializable before/after report."""

    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        HQQWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input_path and output_path must differ")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=True)
    before = audit_quantization(model)
    pointwise_lowering = lower_fastconformer_pointwise_conv_pairs_to_matmul(model)
    lowered_gemm_count, skipped_gemm_nodes = lower_eligible_gemm_to_matmul(model)
    prepared = audit_quantization(model)
    if prepared["eligible_matmul_count"] == 0:
        raise QuantizationContractError(
            f"{input_path.name}: no constant-RHS MatMul or supported Gemm nodes found"
        )

    quantizer = MatMulNBitsQuantizer(
        model=model,
        nodes_to_exclude=nodes_to_exclude,
        algo_config=HQQWeightOnlyQuantConfig(
            block_size=BLOCK_SIZE,
            bits=BITS,
            axis=1,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        ),
    )
    quantizer.process()

    quantized_model = quantizer.model.model
    for node in quantized_model.graph.node:
        if node.op_type == "MatMulNBits":
            _set_attribute_int(node, "accuracy_level", ACCURACY_LEVEL)

    after = audit_quantization(quantized_model, require_w4a8=True)
    after["remaining_eligible_matmul_count"] = after["eligible_matmul_count"]
    if after["matmul_nbits_count"] == 0:
        raise QuantizationContractError(f"{input_path.name}: HQQ produced no MatMulNBits nodes")

    quantizer.model.save_model_to_file(str(output_path), use_external_data_format=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "algorithm": "HQQ",
        "bits": BITS,
        "block_size": BLOCK_SIZE,
        "accuracy_level": ACCURACY_LEVEL,
        "eligible_matmul_count": before["eligible_matmul_count"],
        "prepared_eligible_matmul_count": prepared["eligible_matmul_count"],
        "fastconformer_pointwise_lowering": pointwise_lowering,
        "lowered_gemm_count": lowered_gemm_count,
        "skipped_gemm_nodes": skipped_gemm_nodes,
        "excluded_node_count": len(nodes_to_exclude or []),
        "matmul_nbits_count": after["matmul_nbits_count"],
        "remaining_eligible_matmul_count": after["eligible_matmul_count"],
        "before": before,
        "after": after,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = quantize_hqq_w4a8(
        args.input,
        args.output,
        nodes_to_exclude=args.exclude_node,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
