from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_parakeet_q4_error import dequantize_matmul_nbits


class AnalyzeParakeetQ4ErrorTests(unittest.TestCase):
    def test_dequantize_restores_block32_uint8_zero_point_layout(self):
        quantized = np.tile(np.arange(16, dtype=np.uint8), 4).reshape(1, 64)
        packed = quantized[:, 0::2] | quantized[:, 1::2] << 4
        packed = packed.reshape(1, 2, 16)
        scales = np.array([0.5, 2.0], dtype=np.float32)
        zero_points = np.array([[179]], dtype=np.uint8)
        initializers = {
            "weight_Q4": numpy_helper.from_array(packed, name="weight_Q4"),
            "weight_scales": numpy_helper.from_array(scales, name="weight_scales"),
            "weight_zero_points": numpy_helper.from_array(
                zero_points, name="weight_zero_points"
            ),
        }
        node = helper.make_node(
            "MatMulNBits",
            ["input", "weight_Q4", "weight_scales", "weight_zero_points"],
            ["output"],
            domain="com.microsoft",
            K=64,
            N=1,
            bits=4,
            block_size=32,
        )

        actual = dequantize_matmul_nbits(node, initializers)

        expected = np.concatenate(
            [
                (np.arange(16, dtype=np.float32) - 3) * 0.5,
                (np.arange(16, dtype=np.float32) - 3) * 0.5,
                (np.arange(16, dtype=np.float32) - 11) * 2.0,
                (np.arange(16, dtype=np.float32) - 11) * 2.0,
            ]
        ).reshape(64, 1)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
