#!/usr/bin/env python3
"""Rank Parakeet Q4 MatMul weights by FP32 reconstruction error."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, helper, numpy_helper

from quantize_hqq_w4a8 import lower_fastconformer_pointwise_conv_pairs_to_matmul


def _initializer_map(model: ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {
        attribute.name: int(helper.get_attribute_value(attribute))
        for attribute in node.attribute
    }


def dequantize_matmul_nbits(
    node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]
) -> np.ndarray:
    """Return a MatMulNBits constant weight in its original [K, N] layout."""
    attributes = _attributes(node)
    bits = attributes["bits"]
    block_size = attributes["block_size"]
    rows = attributes["K"]
    cols = attributes["N"]
    if bits != 4:
        raise ValueError(f"{node.name}: expected 4 bits, got {bits}")

    packed = numpy_helper.to_array(initializers[node.input[1]]).reshape(cols, -1)
    quantized = np.empty((cols, packed.shape[1] * 2), dtype=np.float32)
    quantized[:, 0::2] = packed & 15
    quantized[:, 1::2] = packed >> 4

    block_count = (rows + block_size - 1) // block_size
    scales = numpy_helper.to_array(initializers[node.input[2]]).reshape(
        cols, block_count
    )
    packed_zero_points = numpy_helper.to_array(initializers[node.input[3]]).reshape(
        cols, -1
    )
    zero_points = np.empty((cols, packed_zero_points.shape[1] * 2), dtype=np.uint8)
    zero_points[:, 0::2] = packed_zero_points & 15
    zero_points[:, 1::2] = packed_zero_points >> 4
    zero_points = zero_points[:, :block_count]

    dequantized = (
        quantized[:, : block_count * block_size]
        - np.repeat(zero_points.astype(np.float32), block_size, axis=1)
    ) * np.repeat(scales.astype(np.float32), block_size, axis=1)
    return dequantized[:, :rows].T


def _component(node_name: str) -> str:
    if "/pre_encode/" in node_name:
        return "pre_encode"
    if "/pointwise_conv" in node_name:
        return "pointwise_conv"
    if "/feed_forward" in node_name:
        return "feed_forward"
    if "/self_attn/" in node_name:
        return "self_attention_projection"
    return "other"


def analyze(fp32_path: Path, q4_path: Path) -> dict[str, Any]:
    fp32_model = onnx.load(str(fp32_path), load_external_data=True)
    lowering = lower_fastconformer_pointwise_conv_pairs_to_matmul(fp32_model)
    fp32_initializers = _initializer_map(fp32_model)

    q4_model = onnx.load(str(q4_path), load_external_data=True)
    q4_initializers = _initializer_map(q4_model)
    rows = []
    for node in q4_model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        q4_weight_name = node.input[1]
        if not q4_weight_name.endswith("_Q4"):
            raise ValueError(f"{node.name}: unexpected weight name {q4_weight_name}")
        fp32_weight_name = q4_weight_name.removesuffix("_Q4")
        fp32_initializer = fp32_initializers.get(fp32_weight_name)
        if fp32_initializer is None:
            raise ValueError(f"{node.name}: missing FP32 weight {fp32_weight_name}")

        expected = numpy_helper.to_array(fp32_initializer).astype(np.float32, copy=False)
        actual = dequantize_matmul_nbits(node, q4_initializers)
        if expected.shape != actual.shape:
            raise ValueError(
                f"{node.name}: shape mismatch {expected.shape} != {actual.shape}"
            )
        difference = actual - expected
        signal_rms = float(np.sqrt(np.mean(np.square(expected, dtype=np.float64))))
        error_rms = float(np.sqrt(np.mean(np.square(difference, dtype=np.float64))))
        relative_rmse = error_rms / signal_rms if signal_rms else 0.0
        rows.append(
            {
                "node": node.name,
                "component": _component(node.name),
                "shape": list(expected.shape),
                "parameter_count": int(expected.size),
                "relative_rmse": relative_rmse,
                "snr_db": (
                    20.0 * math.log10(signal_rms / error_rms)
                    if error_rms and signal_rms
                    else None
                ),
                "max_absolute_error": float(np.max(np.abs(difference))),
            }
        )

    rows.sort(key=lambda row: row["relative_rmse"], reverse=True)
    component_summary = []
    for component in sorted({row["component"] for row in rows}):
        members = [row for row in rows if row["component"] == component]
        parameters = sum(row["parameter_count"] for row in members)
        component_summary.append(
            {
                "component": component,
                "node_count": len(members),
                "parameter_count": parameters,
                "relative_rmse_parameter_weighted": sum(
                    row["relative_rmse"] * row["parameter_count"] for row in members
                )
                / parameters,
                "max_relative_rmse": max(row["relative_rmse"] for row in members),
            }
        )
    return {
        "fp32": str(fp32_path),
        "q4": str(q4_path),
        "fastconformer_pointwise_lowering": lowering,
        "quantized_node_count": len(rows),
        "components": component_summary,
        "nodes": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = analyze(args.fp32.resolve(), args.q4.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "nodes"}
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
