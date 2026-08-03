"""Lower Supertonic 3 affine layers to MatMul and quantize them to Q4 block16.

The audited model contract is the ``onnx`` directory from Hugging Face revision
``724fb5a``. Pointwise Conv nodes become MatMul-based MLP nodes. Ordinary 1-D
Conv nodes become explicit TDNN tap extraction followed by MatMul. Depthwise
Conv and each model's final affine layer remain unchanged.

ONNX Runtime names the emitted operator ``com.microsoft::MatMulNBits``; there
is no separate ONNX operator schema named ``MatMulNBitsMLP``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


BITS = 4
BLOCK_SIZE = 16
HF_REPOSITORY = "Supertone/supertonic-3"
HF_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
REPORT_FILENAME = "quantization-report.json"

MODEL_CONTRACTS = {
    "duration_predictor.onnx": {
        "sha256": "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
        "final_layer": "/predictor/layers.1/Gemm",
        "pointwise_convs": 25,
        "tdnn_convs": 0,
        "gemms": 1,
        "depthwise_convs": 6,
        "matmul_nbits": 26,
        "gather_block_quantized": 1,
    },
    "text_encoder.onnx": {
        "sha256": "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
        "final_layer": "/speech_prompted_text_encoder/attention2/out_fc/linear/MatMul",
        "pointwise_convs": 36,
        "tdnn_convs": 0,
        "gemms": 0,
        "depthwise_convs": 6,
        "matmul_nbits": 43,
        "gather_block_quantized": 1,
    },
    "vector_estimator.onnx": {
        "sha256": "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
        "final_layer": "/vector_estimator/vector_field/proj_out/net/Conv",
        "pointwise_convs": 57,
        "tdnn_convs": 0,
        "gemms": 2,
        "depthwise_convs": 28,
        "matmul_nbits": 95,
        "gather_block_quantized": 0,
    },
    "vocoder.onnx": {
        "sha256": "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
        "final_layer": "/decoder/head/layer2/Conv",
        "pointwise_convs": 20,
        "tdnn_convs": 2,
        "gemms": 0,
        "depthwise_convs": 10,
        "matmul_nbits": 22,
        "gather_block_quantized": 0,
    },
}


class ConversionReport:
    def __init__(self):
        self.pointwise_convs: list[str] = []
        self.layout_optimized_pointwise_pairs: list[list[str]] = []
        self.tdnn_convs: list[str] = []
        self.depthwise_convs: list[str] = []
        self.gemms: list[str] = []
        self.excluded_final_layers: list[str] = []

    def as_dict(self) -> dict[str, object]:
        return {
            "pointwise_convs": self.pointwise_convs,
            "layout_optimized_pointwise_pairs": self.layout_optimized_pointwise_pairs,
            "tdnn_convs": self.tdnn_convs,
            "depthwise_convs": self.depthwise_convs,
            "gemms": self.gemms,
            "excluded_final_layers": self.excluded_final_layers,
        }


def _attributes(node: NodeProto) -> dict[str, object]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _initializer_arrays(model: ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _name(node: NodeProto, suffix: str) -> str:
    prefix = node.name or node.output[0]
    return prefix + suffix


def _pointwise_nodes(
    node: NodeProto, weight: np.ndarray, bias: np.ndarray | None
) -> tuple[list[NodeProto], list[TensorProto]]:
    activation = _name(node, "/TransposeIn_output")
    matmul_output = _name(node, "/MatMul_output")
    biased_output = _name(node, "/Add_output")
    matrix_name = node.input[1] + "__pointwise_matmul"
    matrix = np.ascontiguousarray(weight[:, :, 0].T)
    initializers = [numpy_helper.from_array(matrix, matrix_name)]
    nodes = [
        helper.make_node(
            "Transpose",
            [node.input[0]],
            [activation],
            name=_name(node, "/TransposeIn"),
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "MatMul",
            [activation, matrix_name],
            [matmul_output],
            name=_name(node, "/MatMul"),
        ),
    ]
    transpose_input = matmul_output
    if bias is not None:
        bias_name = node.input[2] + "__pointwise_bias"
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(bias), bias_name))
        nodes.append(
            helper.make_node(
                "Add",
                [matmul_output, bias_name],
                [biased_output],
                name=_name(node, "/Add"),
            )
        )
        transpose_input = biased_output
    nodes.append(
        helper.make_node(
            "Transpose",
            [transpose_input],
            list(node.output),
            name=_name(node, "/TransposeOut"),
            perm=[0, 2, 1],
        )
    )
    return nodes, initializers


def _pointwise_ntc_nodes(
    node: NodeProto,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    input_name: str,
    transpose_output: bool,
) -> tuple[list[NodeProto], list[TensorProto]]:
    """Lower a pointwise Conv while its surrounding ConvNeXt region is NTC."""

    matmul_output = _name(node, "/MatMul_output")
    biased_output = _name(node, "/Add_output")
    matrix_name = node.input[1] + "__pointwise_matmul"
    matrix = np.ascontiguousarray(weight[:, :, 0].T)
    initializers = [numpy_helper.from_array(matrix, matrix_name)]
    final_linear_output = (
        biased_output if bias is not None else matmul_output
    ) if transpose_output else node.output[0]
    nodes = [
        helper.make_node(
            "MatMul",
            [input_name, matrix_name],
            [matmul_output if bias is not None or transpose_output else node.output[0]],
            name=_name(node, "/MatMul"),
        )
    ]
    if bias is not None:
        bias_name = node.input[2] + "__pointwise_bias"
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(bias), bias_name))
        nodes.append(
            helper.make_node(
                "Add",
                [matmul_output, bias_name],
                [final_linear_output],
                name=_name(node, "/Add"),
            )
        )
    if transpose_output:
        nodes.append(
            helper.make_node(
                "Transpose",
                [final_linear_output],
                list(node.output),
                name=_name(node, "/TransposeOut"),
                perm=[0, 2, 1],
            )
        )
    return nodes, initializers


def _transpose_perm(node: NodeProto) -> tuple[int, ...] | None:
    if node.op_type != "Transpose":
        return None
    attributes = _attributes(node)
    perm = attributes.get("perm")
    return tuple(int(value) for value in perm) if perm is not None else None


def _scalar_value_names(
    model: ModelProto, arrays: dict[str, np.ndarray]
) -> set[str]:
    names = {name for name, value in arrays.items() if value.size == 1}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        attributes = _attributes(node)
        value = attributes.get("value")
        if isinstance(value, TensorProto) and numpy_helper.to_array(value).size == 1:
            names.add(node.output[0])
    return names


def _has_layout_invariant_path(
    model: ModelProto,
    pwconv1: NodeProto,
    pwconv2: NodeProto,
    scalar_values: set[str],
) -> bool:
    """Return true when the entire pwconv1-to-pwconv2 region is NCT/NTC agnostic."""

    unary_ops = {"Erf", "Identity", "Relu", "Sigmoid", "Tanh"}
    scalar_elementwise_ops = {"Add", "Div", "Mul", "Sub"}
    layout_values = {pwconv1.output[0]}
    for node in model.graph.node:
        if node.name in {pwconv1.name, pwconv2.name}:
            continue
        if not any(input_name in layout_values for input_name in node.input):
            continue
        if node.op_type in unary_ops and len(node.input) == 1:
            pass
        elif node.op_type in scalar_elementwise_ops and all(
            input_name in layout_values or input_name in scalar_values
            for input_name in node.input
        ):
            pass
        else:
            return False
        layout_values.update(node.output)
    graph_outputs = {output.name for output in model.graph.output}
    return pwconv2.input[0] in layout_values and not graph_outputs.intersection(
        layout_values
    )


def _convnext_ntc_pairs(
    model: ModelProto,
    arrays: dict[str, np.ndarray],
    excluded_node_names: set[str],
) -> tuple[dict[str, tuple[str, str]], set[str], list[list[str]]]:
    """Find audited vector-estimator ConvNeXt pointwise pairs safe to keep as NTC."""

    nodes_by_name = {node.name: node for node in model.graph.node if node.name}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    scalar_values = _scalar_value_names(model, arrays)
    optimized: dict[str, tuple[str, str]] = {}
    removed_transposes: set[str] = set()
    pairs: list[list[str]] = []

    for pwconv1 in model.graph.node:
        suffix = "/pwconv1/Conv"
        if (
            pwconv1.op_type != "Conv"
            or not pwconv1.name.startswith("/vector_estimator/")
            or not pwconv1.name.endswith(suffix)
            or pwconv1.name in excluded_node_names
        ):
            continue
        pwconv2_name = pwconv1.name[: -len(suffix)] + "/pwconv2/Conv"
        pwconv2 = nodes_by_name.get(pwconv2_name)
        if (
            pwconv2 is None
            or pwconv2.op_type != "Conv"
            or pwconv2.name in excluded_node_names
        ):
            continue
        weights = [
            arrays.get(node.input[1]) if len(node.input) > 1 else None
            for node in (pwconv1, pwconv2)
        ]
        if any(
            weight is None or weight.ndim != 3 or weight.shape[2] != 1
            for weight in weights
        ):
            continue
        input_transpose = producers.get(pwconv1.input[0])
        if (
            input_transpose is None
            or _transpose_perm(input_transpose) != (0, 2, 1)
            or consumers.get(input_transpose.output[0]) != [pwconv1]
            or not _has_layout_invariant_path(
                model, pwconv1, pwconv2, scalar_values
            )
        ):
            continue
        optimized[pwconv1.name] = ("input", input_transpose.input[0])
        optimized[pwconv2.name] = ("output", pwconv2.input[0])
        removed_transposes.add(input_transpose.name)
        pairs.append([pwconv1.name, pwconv2.name])
    return optimized, removed_transposes, pairs


def _tdnn_nodes(
    node: NodeProto, weight: np.ndarray, bias: np.ndarray | None
) -> tuple[list[NodeProto], list[TensorProto]]:
    attributes = _attributes(node)
    if attributes.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        raise ValueError(f"{node.name}: auto_pad is not supported by TDNN lowering")
    group = int(attributes.get("group", 1))
    if group != 1:
        raise ValueError(f"{node.name}: grouped non-depthwise Conv is not supported")
    if weight.ndim != 3:
        raise ValueError(f"{node.name}: only 1-D Conv can be lowered to TDNN")

    kernel_size = int(weight.shape[2])
    dilation_values = attributes.get("dilations", [1])
    stride_values = attributes.get("strides", [1])
    pad_values = attributes.get("pads", [0, 0])
    if len(dilation_values) != 1 or len(stride_values) != 1 or len(pad_values) != 2:
        raise ValueError(f"{node.name}: expected 1-D Conv attributes")
    dilation = int(dilation_values[0])
    stride = int(stride_values[0])
    pad_begin, pad_end = (int(value) for value in pad_values)
    if dilation <= 0 or stride <= 0 or min(pad_begin, pad_end) < 0:
        raise ValueError(f"{node.name}: invalid Conv dilation, stride, or pads")

    nodes: list[NodeProto] = []
    initializers: list[TensorProto] = []
    activation = node.input[0]
    if pad_begin or pad_end:
        pads_name = _name(node, "/TDNNPads")
        padded = _name(node, "/TDNNPad_output")
        initializers.append(
            numpy_helper.from_array(
                np.array([0, 0, pad_begin, 0, 0, pad_end], dtype=np.int64),
                pads_name,
            )
        )
        nodes.append(
            helper.make_node(
                "Pad", [activation, pads_name], [padded], name=_name(node, "/TDNNPad")
            )
        )
        activation = padded

    effective_kernel = dilation * (kernel_size - 1) + 1
    slices = []
    max_index = np.iinfo(np.int64).max
    for tap in range(kernel_size):
        offset = tap * dilation
        remaining = effective_kernel - 1 - offset
        starts_name = _name(node, f"/TDNNTap{tap}Starts")
        ends_name = _name(node, f"/TDNNTap{tap}Ends")
        axes_name = _name(node, f"/TDNNTap{tap}Axes")
        steps_name = _name(node, f"/TDNNTap{tap}Steps")
        initializers.extend(
            [
                numpy_helper.from_array(np.array([offset], dtype=np.int64), starts_name),
                numpy_helper.from_array(
                    np.array([-remaining if remaining else max_index], dtype=np.int64),
                    ends_name,
                ),
                numpy_helper.from_array(np.array([2], dtype=np.int64), axes_name),
                numpy_helper.from_array(np.array([stride], dtype=np.int64), steps_name),
            ]
        )
        tap_output = _name(node, f"/TDNNTap{tap}_output")
        nodes.append(
            helper.make_node(
                "Slice",
                [activation, starts_name, ends_name, axes_name, steps_name],
                [tap_output],
                name=_name(node, f"/TDNNTap{tap}"),
            )
        )
        slices.append(tap_output)

    features = _name(node, "/TDNNConcat_output")
    transposed = _name(node, "/TDNNTransposeIn_output")
    matmul_output = _name(node, "/TDNNMatMul_output")
    biased_output = _name(node, "/TDNNAdd_output")
    matrix_name = node.input[1] + "__tdnn_matmul"
    matrix = np.ascontiguousarray(weight.transpose(2, 1, 0).reshape(-1, weight.shape[0]))
    initializers.append(numpy_helper.from_array(matrix, matrix_name))
    nodes.extend(
        [
            helper.make_node(
                "Concat", slices, [features], name=_name(node, "/TDNNConcat"), axis=1
            ),
            helper.make_node(
                "Transpose",
                [features],
                [transposed],
                name=_name(node, "/TDNNTransposeIn"),
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "MatMul",
                [transposed, matrix_name],
                [matmul_output],
                name=_name(node, "/TDNNMatMul"),
            ),
        ]
    )
    transpose_input = matmul_output
    if bias is not None:
        bias_name = node.input[2] + "__tdnn_bias"
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(bias), bias_name))
        nodes.append(
            helper.make_node(
                "Add",
                [matmul_output, bias_name],
                [biased_output],
                name=_name(node, "/TDNNAdd"),
            )
        )
        transpose_input = biased_output
    nodes.append(
        helper.make_node(
            "Transpose",
            [transpose_input],
            list(node.output),
            name=_name(node, "/TDNNTransposeOut"),
            perm=[0, 2, 1],
        )
    )
    return nodes, initializers


def _gemm_nodes(
    node: NodeProto, weight: np.ndarray, bias: np.ndarray | None
) -> tuple[list[NodeProto], list[TensorProto]]:
    attributes = _attributes(node)
    if int(attributes.get("transA", 0)) != 0:
        raise ValueError(f"{node.name}: transA Gemm is not supported")
    alpha = float(attributes.get("alpha", 1.0))
    beta = float(attributes.get("beta", 1.0))
    trans_b = int(attributes.get("transB", 0))
    if trans_b not in (0, 1):
        raise ValueError(f"{node.name}: invalid transB")
    matrix = weight.T if trans_b else weight
    matrix = np.ascontiguousarray(matrix * alpha)
    matrix_name = node.input[1] + "__gemm_matmul"
    matmul_output = _name(node, "/MatMul_output")
    initializers = [numpy_helper.from_array(matrix, matrix_name)]
    nodes = [
        helper.make_node(
            "MatMul",
            [node.input[0], matrix_name],
            [matmul_output if bias is not None else node.output[0]],
            name=_name(node, "/MatMul"),
        )
    ]
    if bias is not None:
        bias_name = node.input[2] + "__gemm_bias"
        initializers.append(
            numpy_helper.from_array(np.ascontiguousarray(bias * beta), bias_name)
        )
        nodes.append(
            helper.make_node(
                "Add",
                [matmul_output, bias_name],
                list(node.output),
                name=_name(node, "/Add"),
            )
        )
    return nodes, initializers


def _remove_unused_initializers(model: ModelProto) -> None:
    used = {input_name for node in model.graph.node for input_name in node.input}
    used.update(output.name for output in model.graph.output)
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def lower_affine_nodes(
    model: ModelProto,
    excluded_node_names: set[str],
    *,
    optimize_convnext_layout: bool = False,
) -> ConversionReport:
    """Mutate a graph so eligible affine layers become constant-weight MatMul."""

    arrays = _initializer_arrays(model)
    report = ConversionReport()
    rewritten_nodes: list[NodeProto] = []
    added_initializers: list[TensorProto] = []
    ntc_nodes, removed_transposes, optimized_pairs = (
        _convnext_ntc_pairs(model, arrays, excluded_node_names)
        if optimize_convnext_layout
        else ({}, set(), [])
    )
    report.layout_optimized_pointwise_pairs.extend(optimized_pairs)

    for node in model.graph.node:
        if node.name in removed_transposes:
            continue
        if node.name in excluded_node_names:
            if node.op_type not in {"Conv", "Gemm", "MatMul"}:
                raise ValueError(f"excluded final layer is not affine: {node.name}")
            report.excluded_final_layers.append(node.name)
            rewritten_nodes.append(node)
            continue

        if node.op_type == "Conv":
            weight = arrays.get(node.input[1]) if len(node.input) > 1 else None
            if weight is None:
                raise ValueError(f"{node.name}: Conv weight is not a constant initializer")
            attributes = _attributes(node)
            group = int(attributes.get("group", 1))
            if group > 1 and weight.ndim == 3 and weight.shape[1] == 1:
                report.depthwise_convs.append(node.name)
                rewritten_nodes.append(node)
                continue
            bias = arrays.get(node.input[2]) if len(node.input) > 2 else None
            if weight.ndim == 3 and weight.shape[2] == 1 and group == 1:
                ntc = ntc_nodes.get(node.name)
                if ntc is None:
                    replacement, initializers = _pointwise_nodes(node, weight, bias)
                else:
                    mode, input_name = ntc
                    replacement, initializers = _pointwise_ntc_nodes(
                        node,
                        weight,
                        bias,
                        input_name=input_name,
                        transpose_output=mode == "output",
                    )
                report.pointwise_convs.append(node.name)
            else:
                replacement, initializers = _tdnn_nodes(node, weight, bias)
                report.tdnn_convs.append(node.name)
            rewritten_nodes.extend(replacement)
            added_initializers.extend(initializers)
            continue

        if node.op_type == "Gemm":
            weight = arrays.get(node.input[1]) if len(node.input) > 1 else None
            if weight is None:
                rewritten_nodes.append(node)
                continue
            bias = arrays.get(node.input[2]) if len(node.input) > 2 else None
            replacement, initializers = _gemm_nodes(node, weight, bias)
            report.gemms.append(node.name)
            rewritten_nodes.extend(replacement)
            added_initializers.extend(initializers)
            continue

        rewritten_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(added_initializers)
    _remove_unused_initializers(model)
    return report


def quantize_weights_q4_block16(
    model: ModelProto, excluded_node_names: set[str] | None = None
) -> None:
    """Quantize constant MatMul and Gather weights using asymmetric Q4 block16."""

    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
        logging.WARNING
    )

    def run_pass(op_type: str) -> None:
        quantizer = MatMulNBitsQuantizer(
            model=model,
            nodes_to_exclude=excluded_node_names or set(),
            algo_config=DefaultWeightOnlyQuantConfig(
                block_size=BLOCK_SIZE,
                is_symmetric=False,
                quant_format=QuantFormat.QOperator,
                op_types_to_quantize=(op_type,),
                bits=BITS,
            ),
        )
        quantizer.process()
        model.CopyFrom(quantizer.model.model)

    run_pass("MatMul")
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    has_constant_gather = any(
        node.op_type == "Gather" and node.input and node.input[0] in initializer_names
        for node in model.graph.node
    )
    if has_constant_gather:
        run_pass("Gather")


def quantize_matmuls_q4_block16(model: ModelProto) -> None:
    """Backward-compatible helper used by the MatMul-only synthetic test."""

    quantize_weights_q4_block16(model)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_attributes(node: NodeProto) -> dict[str, int]:
    return {
        attribute.name: int(attribute.i)
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.INT
    }


def _verify_contract(
    filename: str,
    report: ConversionReport,
    output_model: ModelProto,
    *,
    strict: bool,
) -> dict[str, object]:
    contract = MODEL_CONTRACTS[filename]
    actual = {
        "pointwise_convs": len(report.pointwise_convs),
        "tdnn_convs": len(report.tdnn_convs),
        "gemms": len(report.gemms),
        "depthwise_convs": len(report.depthwise_convs),
        "matmul_nbits": sum(
            node.op_type == "MatMulNBits" for node in output_model.graph.node
        ),
        "gather_block_quantized": sum(
            node.op_type == "GatherBlockQuantized" for node in output_model.graph.node
        ),
    }
    expected = {key: int(contract[key]) for key in actual}
    if strict and actual != expected:
        raise ValueError(f"{filename}: graph contract changed: {actual} != {expected}")
    if report.excluded_final_layers != [contract["final_layer"]]:
        raise ValueError(
            f"{filename}: final-layer exclusion mismatch: "
            f"{report.excluded_final_layers!r}"
        )
    invalid_qnodes = []
    for node in output_model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attributes = _node_attributes(node)
        if attributes.get("bits") != BITS or attributes.get("block_size") != BLOCK_SIZE:
            invalid_qnodes.append({"name": node.name, "attributes": attributes})
    if invalid_qnodes:
        raise ValueError(f"invalid MatMulNBits attributes: {invalid_qnodes}")
    invalid_gathers = []
    for node in output_model.graph.node:
        if node.op_type != "GatherBlockQuantized":
            continue
        attributes = _node_attributes(node)
        expected_attributes = {
            "block_size": BLOCK_SIZE,
            "gather_axis": 0,
            "quantize_axis": 1,
        }
        if attributes != expected_attributes:
            invalid_gathers.append({"name": node.name, "attributes": attributes})
    if invalid_gathers:
        raise ValueError(f"invalid GatherBlockQuantized attributes: {invalid_gathers}")
    return {"counts": actual, "expected_counts": expected}


def _reject_unsafe_output(output_dir: Path, input_dir: Path) -> None:
    if output_dir == Path(output_dir.anchor) or output_dir == Path.cwd().resolve():
        raise ValueError(f"unsafe output directory: {output_dir}")
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise ValueError(f"output directory would contain or replace input: {output_dir}")


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    force: bool,
    allow_unverified_source: bool,
) -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    _reject_unsafe_output(output_dir, input_dir)
    missing = [name for name in MODEL_CONTRACTS if not (input_dir / name).is_file()]
    if missing:
        raise ValueError("missing input model(s): " + ", ".join(missing))
    if output_dir.exists():
        if not force:
            raise ValueError("output directory exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    report_data: dict[str, object] = {
        "source": {"repository": HF_REPOSITORY, "revision": HF_REVISION},
        "quantization": {
            "operators": [
                "com.microsoft::MatMulNBits",
                "com.microsoft::GatherBlockQuantized",
            ],
            "bits": BITS,
            "block_size": BLOCK_SIZE,
            "symmetric": False,
        },
        "models": {},
    }
    for filename, contract in MODEL_CONTRACTS.items():
        source_path = input_dir / filename
        source_hash = _sha256(source_path)
        if not allow_unverified_source and source_hash != contract["sha256"]:
            raise ValueError(
                f"{filename}: SHA-256 {source_hash} does not match audited "
                f"revision {HF_REVISION} ({contract['sha256']})"
            )
        model = onnx.load(str(source_path))
        original_counts = Counter(node.op_type for node in model.graph.node)
        final_layers = {str(contract["final_layer"])}
        embedding_nodes = [
            node.name
            for node in model.graph.node
            if node.op_type == "Gather"
            and node.input
            and any(
                initializer.name == node.input[0]
                for initializer in model.graph.initializer
            )
        ]
        conversion = lower_affine_nodes(model, final_layers)
        quantize_weights_q4_block16(model, final_layers)
        onnx.checker.check_model(model)
        verified = _verify_contract(
            filename, conversion, model, strict=not allow_unverified_source
        )
        output_path = output_dir / filename
        onnx.save_model(model, str(output_path))
        reloaded = onnx.load(str(output_path))
        onnx.checker.check_model(reloaded)
        model_report = {
            "source_sha256": source_hash,
            "source_bytes": source_path.stat().st_size,
            "output_sha256": _sha256(output_path),
            "output_bytes": output_path.stat().st_size,
            "original_op_counts": dict(sorted(original_counts.items())),
            "conversion": conversion.as_dict(),
            "embedding_nodes": embedding_nodes,
            **verified,
        }
        report_data["models"][filename] = model_report
        print(
            f"{filename}: {verified['counts']['matmul_nbits']} MatMulNBits + "
            f"{verified['counts']['gather_block_quantized']} GatherBlockQuantized, "
            f"{source_path.stat().st_size / 1_000_000:.1f} MB -> "
            f"{output_path.stat().st_size / 1_000_000:.1f} MB"
        )

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="allow a graph other than the audited Hugging Face revision",
    )
    args = parser.parse_args()
    try:
        convert_directory(
            args.input_dir,
            args.output_dir,
            force=args.force,
            allow_unverified_source=args.allow_unverified_source,
        )
    except (OSError, ValueError, onnx.checker.ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
