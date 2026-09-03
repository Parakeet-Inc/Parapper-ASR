#!/usr/bin/env python3
"""Build the Parakeet CTC asymmetric HQQ Q4 block32 ONNX variant.

MatMul weights use HQQ optimization. HQQ's optimized zero points are rounded
and packed into UINT8 storage so the resulting MatMulNBits nodes follow the
same runtime contract as the adopted k-quant models. Constant rank-2 Gather
weights, when present, use the CAT Translate row-wise Q8 embedding path.
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

from quantize_hqq_w4a8 import lower_fastconformer_pointwise_conv_pairs_to_matmul


BITS = 4
BLOCK_SIZE = 32
ROW_WISE_Q8_SUFFIX = "_rowwise_q8"
ROW_WISE_SCALE_SUFFIX = "_rowwise_scale"
ROW_WISE_ZERO_POINT_SUFFIX = "_rowwise_zero_point"
ROW_WISE_AXES_SUFFIX = "_rowwise_axes"


class IntegerZeroPointHQQQuantizer:
    """Adapter around ORT HQQ that emits packed UINT8 zero points."""

    @staticmethod
    def install_on(quantizer: Any) -> None:
        from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantizer

        # HQQ optimizes zero points as floats; round them here so the packed
        # UINT8 storage below stays lossless.
        class RoundedZeroPointHQQWeightOnlyQuantizer(HQQWeightOnlyQuantizer):
            @staticmethod
            def optimize_weights(
                tensor: Any,
                scale: Any,
                zero: Any,
                min_max: list[int],
                axis: int = 0,
                opt_params: dict[str, Any] | None = None,
                verbose: bool = False,
            ) -> tuple[Any, Any]:
                optimized_scale, optimized_zero = HQQWeightOnlyQuantizer.optimize_weights(
                    tensor,
                    scale,
                    zero,
                    min_max,
                    axis=axis,
                    opt_params=opt_params,
                    verbose=verbose,
                )
                return optimized_scale, optimized_zero.round().clamp(0, 15)

        quantizer.node_quantizer = RoundedZeroPointHQQWeightOnlyQuantizer(
            quantizer.algo_config
        )

    @staticmethod
    def pack_model_zero_points(model: ModelProto) -> int:
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        replacements = {}
        converted = 0
        for node in model.graph.node:
            if node.op_type != "MatMulNBits" or len(node.input) < 4:
                continue
            zero_name = node.input[3]
            zero_initializer = initializers.get(zero_name)
            if zero_initializer is None:
                raise ValueError(f"{node.name}: missing zero-point initializer {zero_name}")
            if zero_initializer.data_type == TensorProto.UINT8:
                continue
            if zero_initializer.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
                raise ValueError(
                    f"{node.name}: unsupported HQQ zero-point type "
                    f"{TensorProto.DataType.Name(zero_initializer.data_type)}"
                )

            attributes = {
                attribute.name: int(helper.get_attribute_value(attribute))
                for attribute in node.attribute
            }
            rows = attributes["K"]
            cols = attributes["N"]
            block_size = attributes["block_size"]
            block_count = (rows + block_size - 1) // block_size
            unpacked = np.rint(numpy_helper.to_array(zero_initializer)).astype(np.uint8)
            unpacked = np.clip(unpacked.reshape(cols, block_count), 0, 15)
            packed = np.zeros((cols, (block_count + 1) // 2), dtype=np.uint8)
            packed[:, :] = unpacked[:, 0::2]
            if block_count > 1:
                packed[:, : block_count // 2] |= unpacked[:, 1::2] << 4
            replacements[zero_name] = numpy_helper.from_array(packed, name=zero_name)
            converted += 1

        if replacements:
            retained = [
                initializer
                for initializer in model.graph.initializer
                if initializer.name not in replacements
            ]
            retained.extend(replacements.values())
            del model.graph.initializer[:]
            model.graph.initializer.extend(retained)
        return converted


def _initializer_map(model: ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _constant_gather_count(model: ModelProto) -> int:
    initializer_names = set(_initializer_map(model))
    return sum(
        node.op_type == "Gather" and bool(node.input) and node.input[0] in initializer_names
        for node in model.graph.node
    )


def _rowwise_q8_embedding_count(model: ModelProto) -> int:
    return sum(node.name.endswith("_RowwiseDequantize") for node in model.graph.node)


def _audit(model: ModelProto) -> dict[str, Any]:
    initializers = _initializer_map(model)
    node_counts = Counter(node.op_type for node in model.graph.node)
    q4_nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"]
    zero_point_types = Counter(
        TensorProto.DataType.Name(initializers[node.input[3]].data_type)
        for node in q4_nodes
        if len(node.input) >= 4 and node.input[3] in initializers
    )
    contracts = Counter(
        (
            next(
                int(helper.get_attribute_value(attr))
                for attr in node.attribute
                if attr.name == "bits"
            ),
            next(
                int(helper.get_attribute_value(attr))
                for attr in node.attribute
                if attr.name == "block_size"
            ),
        )
        for node in q4_nodes
    )
    return {
        "node_types": dict(sorted(node_counts.items())),
        "initializer_count": len(initializers),
        "matmul_nbits_count": len(q4_nodes),
        "gather_block_quantized_count": node_counts["GatherBlockQuantized"],
        "rowwise_q8_embedding_count": _rowwise_q8_embedding_count(model),
        "constant_gather_count": _constant_gather_count(model),
        "zero_point_types": dict(sorted(zero_point_types.items())),
        "matmul_nbits_contracts": [
            {"bits": bits, "block_size": block_size, "count": count}
            for (bits, block_size), count in sorted(contracts.items())
        ],
    }


def _quantize_embedding_rows(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_min = np.minimum(weights.min(axis=1), np.float32(0.0))
    row_max = np.maximum(weights.max(axis=1), np.float32(0.0))
    row_range = row_max - row_min
    zero_rows = row_range < np.finfo(np.float32).tiny
    scales = np.ones(row_range.shape, dtype=np.float32)
    scales[~zero_rows] = row_range[~zero_rows] / np.float32(255.0)
    zero_points_float = np.zeros(row_range.shape, dtype=np.float32)
    zero_points_float[~zero_rows] = np.clip(
        np.rint(-row_min[~zero_rows] / scales[~zero_rows]), 0, 255
    )
    quantized = np.clip(
        np.rint(weights / scales[:, None]) + zero_points_float[:, None], 0, 255
    ).astype(np.uint8)
    quantized[zero_rows] = 0
    return quantized, scales, zero_points_float.astype(np.uint8)


def _quantize_constant_gathers(model: ModelProto) -> ModelProto:
    initializers = _initializer_map(model)
    candidates = []
    for node in model.graph.node:
        if node.op_type != "Gather" or not node.input:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None or initializer.data_type != TensorProto.FLOAT:
            continue
        weights = numpy_helper.to_array(initializer)
        axis = next(
            (
                int(helper.get_attribute_value(attribute))
                for attribute in node.attribute
                if attribute.name == "axis"
            ),
            0,
        )
        if weights.ndim == 2 and axis == 0:
            candidates.append((node, initializer))

    for gather, source_initializer in candidates:
        consumers = [
            node for node in model.graph.node if source_initializer.name in node.input
        ]
        if consumers != [gather]:
            raise ValueError(
                f"{source_initializer.name}: row-wise Q8 embedding must have one Gather consumer"
            )
        weights = numpy_helper.to_array(source_initializer).astype(
            np.float32, copy=False
        )
        if not np.isfinite(weights).all():
            raise ValueError(f"{source_initializer.name}: embedding contains non-finite values")
        quantized, scales, zero_points = _quantize_embedding_rows(weights)
        source_name = source_initializer.name
        quantized_name = f"{source_name}{ROW_WISE_Q8_SUFFIX}"
        scale_name = f"{source_name}{ROW_WISE_SCALE_SUFFIX}"
        zero_point_name = f"{source_name}{ROW_WISE_ZERO_POINT_SUFFIX}"
        axes_name = f"{source_name}{ROW_WISE_AXES_SUFFIX}"
        generated_names = {quantized_name, scale_name, zero_point_name, axes_name}
        existing_names = {
            initializer.name for initializer in model.graph.initializer
        } | {output for node in model.graph.node for output in node.output}
        collisions = sorted(generated_names & existing_names)
        if collisions:
            raise ValueError(f"{source_name}: row-wise Q8 name collision {collisions}")

        retained = [
            initializer
            for initializer in model.graph.initializer
            if initializer.name != source_name
        ]
        retained.extend(
            [
                numpy_helper.from_array(quantized, name=quantized_name),
                numpy_helper.from_array(scales, name=scale_name),
                numpy_helper.from_array(zero_points, name=zero_point_name),
                numpy_helper.from_array(
                    np.asarray([-1], dtype=np.int64), name=axes_name
                ),
            ]
        )
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)

        input_ids = gather.input[1]
        original_output = gather.output[0]
        prefix = f"{original_output}_rowwise"
        gather.input[0] = quantized_name
        gather.output[0] = f"{prefix}_quantized"
        added_nodes = [
            helper.make_node(
                "Gather",
                [scale_name, input_ids],
                [f"{prefix}_scale"],
                name=f"{gather.name}_Scale",
            ),
            helper.make_node(
                "Gather",
                [zero_point_name, input_ids],
                [f"{prefix}_zero_point"],
                name=f"{gather.name}_ZeroPoint",
            ),
            helper.make_node(
                "Cast",
                [f"{prefix}_quantized"],
                [f"{prefix}_quantized_float"],
                name=f"{gather.name}_QuantizedToFloat",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Cast",
                [f"{prefix}_zero_point"],
                [f"{prefix}_zero_point_float"],
                name=f"{gather.name}_ZeroPointToFloat",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Unsqueeze",
                [f"{prefix}_scale", axes_name],
                [f"{prefix}_scale_expanded"],
                name=f"{gather.name}_ScaleUnsqueeze",
            ),
            helper.make_node(
                "Unsqueeze",
                [f"{prefix}_zero_point_float", axes_name],
                [f"{prefix}_zero_point_expanded"],
                name=f"{gather.name}_ZeroPointUnsqueeze",
            ),
            helper.make_node(
                "Sub",
                [f"{prefix}_quantized_float", f"{prefix}_zero_point_expanded"],
                [f"{prefix}_centered"],
                name=f"{gather.name}_Center",
            ),
            helper.make_node(
                "Mul",
                [f"{prefix}_centered", f"{prefix}_scale_expanded"],
                [original_output],
                name=f"{gather.name}_RowwiseDequantize",
            ),
        ]
        gather_index = next(
            index for index, node in enumerate(model.graph.node) if node is gather
        )
        for offset, node in enumerate(added_nodes, start=1):
            model.graph.node.insert(gather_index + offset, node)
    return model


def quantize_parakeet_q4(
    input_path: Path,
    output_path: Path,
    *,
    nodes_to_exclude: list[str] | None = None,
    lower_fastconformer_pointwise: bool = True,
) -> dict[str, Any]:
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        HQQWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
        logging.WARNING
    )
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input_path and output_path must differ")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=True)
    before = _audit(model)
    pointwise_report = (
        lower_fastconformer_pointwise_conv_pairs_to_matmul(model)
        if lower_fastconformer_pointwise
        else {"pair_count": 0, "pointwise_conv_count": 0, "removed_transpose_count": 0}
    )

    quantizer = MatMulNBitsQuantizer(
        model=model,
        nodes_to_exclude=nodes_to_exclude or [],
        algo_config=HQQWeightOnlyQuantConfig(
            block_size=BLOCK_SIZE,
            bits=BITS,
            axis=1,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        ),
    )
    IntegerZeroPointHQQQuantizer.install_on(quantizer)
    quantizer.process()
    quantized_model = quantizer.model.model
    converted_zero_points = IntegerZeroPointHQQQuantizer.pack_model_zero_points(
        quantized_model
    )
    quantized_model = _quantize_constant_gathers(quantized_model)
    after = _audit(quantized_model)

    if after["matmul_nbits_count"] == 0:
        raise ValueError("HQQ produced no MatMulNBits nodes")
    if after["zero_point_types"] != {"UINT8": after["matmul_nbits_count"]}:
        raise ValueError(f"invalid zero-point contract: {after['zero_point_types']}")
    if after["matmul_nbits_contracts"] != [
        {"bits": BITS, "block_size": BLOCK_SIZE, "count": after["matmul_nbits_count"]}
    ]:
        raise ValueError(
            f"invalid Q4 block32 contract: {after['matmul_nbits_contracts']}"
        )

    data_path = output_path.with_name(output_path.name + ".data")
    if output_path.exists():
        output_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )

    embedding_count = after["rowwise_q8_embedding_count"]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "algorithm": "HQQ",
        "bits": BITS,
        "block_size": BLOCK_SIZE,
        "zero_point_storage": "UINT8",
        "embedding_quantization": (
            "CAT row-wise Q8" if embedding_count else "not present in CTC graph"
        ),
        "excluded_nodes": sorted(nodes_to_exclude or []),
        "converted_zero_point_count": converted_zero_points,
        "fastconformer_pointwise_lowering": pointwise_report,
        "before": before,
        "after": after,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-node", action="append", default=[])
    parser.add_argument("--no-lower-fastconformer-pointwise", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = quantize_parakeet_q4(
        args.input,
        args.output,
        nodes_to_exclude=args.exclude_node,
        lower_fastconformer_pointwise=not args.no_lower_fastconformer_pointwise,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
