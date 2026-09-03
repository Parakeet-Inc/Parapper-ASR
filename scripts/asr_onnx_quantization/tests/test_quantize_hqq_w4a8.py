from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from quantize_hqq_w4a8 import (
    QuantizationContractError,
    audit_quantization,
    lower_fastconformer_pointwise_conv_pairs_to_matmul,
    quantize_hqq_w4a8,
)


def _matmul_model(*, constant_rhs: bool = True) -> onnx.ModelProto:
    rng = np.random.default_rng(7)
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 32])]
    initializers = []
    if constant_rhs:
        initializers.append(
            numpy_helper.from_array(
                rng.normal(0.0, 0.25, size=(32, 16)).astype(np.float32),
                name="weight",
            )
        )
    else:
        inputs.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [32, 16]))

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="projection")],
        "w4a8_contract",
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 16])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def _gemm_model() -> onnx.ModelProto:
    rng = np.random.default_rng(17)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gemm",
                ["x", "weight", "bias"],
                ["y"],
                name="output_projection",
                transB=1,
            )
        ],
        "gemm_w4a8_contract",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 16])],
        [
            numpy_helper.from_array(
                rng.normal(0.0, 0.25, size=(16, 32)).astype(np.float32),
                name="weight",
            ),
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(16,)).astype(np.float32),
                name="bias",
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def _fastconformer_pointwise_pair_model() -> onnx.ModelProto:
    rng = np.random.default_rng(23)
    nodes = [
        helper.make_node(
            "Transpose",
            ["x"],
            ["channels_first"],
            name="/layers.0/conv/Transpose",
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Conv",
            ["channels_first", "pointwise1.weight", "pointwise1.bias"],
            ["pointwise1"],
            name="/layers.0/conv/pointwise_conv1/Conv",
            kernel_shape=[1],
        ),
        helper.make_node(
            "Split",
            ["pointwise1"],
            ["split_value", "split_gate"],
            name="/layers.0/conv/Split",
            axis=1,
            num_outputs=2,
        ),
        helper.make_node("Sigmoid", ["split_gate"], ["gate"], name="/layers.0/conv/Sigmoid"),
        helper.make_node("Mul", ["split_value", "gate"], ["glu"], name="/layers.0/conv/Mul"),
        helper.make_node(
            "Conv",
            ["glu", "depthwise.weight"],
            ["depthwise"],
            name="/layers.0/conv/depthwise_conv/Conv",
            group=32,
            kernel_shape=[3],
            pads=[1, 1],
        ),
        helper.make_node(
            "Transpose",
            ["depthwise"],
            ["depthwise_time_major"],
            name="/layers.0/conv/Transpose_1",
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Transpose",
            ["depthwise_time_major"],
            ["activation_channels_first"],
            name="/layers.0/conv/Transpose_2",
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Sigmoid",
            ["activation_channels_first"],
            ["activation_gate"],
            name="/layers.0/conv/activation/Sigmoid",
        ),
        helper.make_node(
            "Mul",
            ["activation_channels_first", "activation_gate"],
            ["activation"],
            name="/layers.0/conv/activation/Mul",
        ),
        helper.make_node(
            "Conv",
            ["activation", "pointwise2.weight", "pointwise2.bias"],
            ["pointwise2"],
            name="/layers.0/conv/pointwise_conv2/Conv",
            kernel_shape=[1],
        ),
        helper.make_node(
            "Transpose",
            ["pointwise2"],
            ["y"],
            name="/layers.0/conv/Transpose_3",
            perm=[0, 2, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "fastconformer_pointwise_pair",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 7, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 7, 32])],
        [
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(64, 32, 1)).astype(np.float32),
                name="pointwise1.weight",
            ),
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(64,)).astype(np.float32),
                name="pointwise1.bias",
            ),
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(32, 1, 3)).astype(np.float32),
                name="depthwise.weight",
            ),
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(32, 32, 1)).astype(np.float32),
                name="pointwise2.weight",
            ),
            numpy_helper.from_array(
                rng.normal(0.0, 0.1, size=(32,)).astype(np.float32),
                name="pointwise2.bias",
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def _compact_fastconformer_pointwise_pair_model() -> onnx.ModelProto:
    model = _fastconformer_pointwise_pair_model()
    nodes = list(model.graph.node)
    activation_transpose = next(node for node in nodes if node.name.endswith("Transpose_1"))
    redundant_transpose = next(node for node in nodes if node.name.endswith("Transpose_2"))
    activation = next(node for node in nodes if node.name.endswith("activation/Mul"))
    sigmoid = next(node for node in nodes if node.name.endswith("activation/Sigmoid"))
    depthwise_output = activation_transpose.input[0]
    activation.input[:] = [
        depthwise_output if value == redundant_transpose.output[0] else value
        for value in activation.input
    ]
    sigmoid.input[:] = [depthwise_output]
    retained = [
        node
        for node in nodes
        if node is not activation_transpose and node is not redundant_transpose
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    return model


class QuantizeHqqW4A8Tests(unittest.TestCase):
    def test_compact_fastconformer_pointwise_pair_moves_two_transposes_and_preserves_values(self):
        import onnxruntime as ort

        model = _compact_fastconformer_pointwise_pair_model()
        x = np.random.default_rng(37).normal(size=(1, 7, 32)).astype(np.float32)
        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]

        report = lower_fastconformer_pointwise_conv_pairs_to_matmul(model)

        onnx.checker.check_model(model)
        actual = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]
        node_types = [node.op_type for node in model.graph.node]
        self.assertEqual(
            report,
            {"pair_count": 1, "pointwise_conv_count": 2, "removed_transpose_count": 0},
        )
        self.assertEqual(node_types.count("Conv"), 1)
        self.assertEqual(node_types.count("MatMul"), 2)
        self.assertEqual(node_types.count("Transpose"), 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_fastconformer_pointwise_pair_lowering_preserves_values_and_removes_two_transposes(self):
        import onnxruntime as ort

        model = _fastconformer_pointwise_pair_model()
        x = np.random.default_rng(29).normal(size=(1, 7, 32)).astype(np.float32)
        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]

        report = lower_fastconformer_pointwise_conv_pairs_to_matmul(model)

        onnx.checker.check_model(model)
        actual = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]
        node_types = [node.op_type for node in model.graph.node]
        self.assertEqual(
            report,
            {"pair_count": 1, "pointwise_conv_count": 2, "removed_transpose_count": 2},
        )
        self.assertEqual(node_types.count("Conv"), 1)
        self.assertEqual(node_types.count("MatMul"), 2)
        self.assertEqual(node_types.count("Transpose"), 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_hqq_conversion_emits_w4a8_block32_contract_and_runs(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fp32_path = root / "fp32.onnx"
            q4_path = root / "q4.onnx"
            onnx.save(_matmul_model(), fp32_path)

            report = quantize_hqq_w4a8(fp32_path, q4_path)

            self.assertEqual(report["eligible_matmul_count"], 1)
            self.assertEqual(report["matmul_nbits_count"], 1)
            self.assertEqual(report["remaining_eligible_matmul_count"], 0)
            node = next(
                node for node in onnx.load(q4_path).graph.node if node.op_type == "MatMulNBits"
            )
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            self.assertEqual(
                {name: attributes[name] for name in ("bits", "block_size", "accuracy_level")},
                {"bits": 4, "block_size": 32, "accuracy_level": 4},
            )

            x = np.random.default_rng(11).normal(size=(2, 32)).astype(np.float32)
            fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            q4 = ort.InferenceSession(str(q4_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            self.assertEqual(q4.shape, fp32.shape)
            self.assertLess(float(np.mean(np.abs(fp32 - q4))), 0.2)

    def test_dynamic_rhs_is_reported_and_not_silently_claimed_as_quantized(self):
        audit = audit_quantization(_matmul_model(constant_rhs=False))

        self.assertEqual(audit["eligible_matmul_count"], 0)
        self.assertEqual(audit["dynamic_rhs_matmul_count"], 1)

    def test_gemm_output_projection_is_lowered_then_quantized_without_changing_shape(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fp32_path = root / "gemm-fp32.onnx"
            q4_path = root / "gemm-q4.onnx"
            onnx.save(_gemm_model(), fp32_path)

            report = quantize_hqq_w4a8(fp32_path, q4_path)

            self.assertEqual(report["lowered_gemm_count"], 1)
            self.assertEqual(report["matmul_nbits_count"], 1)
            self.assertNotIn("Gemm", report["after"]["node_types"])
            x = np.random.default_rng(19).normal(size=(2, 32)).astype(np.float32)
            fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            q4 = ort.InferenceSession(str(q4_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            self.assertEqual(q4.shape, fp32.shape)
            self.assertLess(float(np.mean(np.abs(fp32 - q4))), 0.2)

    def test_contract_rejects_non_w4a8_matmul_nbits(self):
        model = _matmul_model()
        node = model.graph.node[0]
        node.op_type = "MatMulNBits"
        node.domain = "com.microsoft"
        node.attribute.extend(
            [
                helper.make_attribute("K", 32),
                helper.make_attribute("N", 16),
                helper.make_attribute("bits", 4),
                helper.make_attribute("block_size", 32),
                helper.make_attribute("accuracy_level", 1),
            ]
        )

        with self.assertRaisesRegex(QuantizationContractError, "accuracy_level=4"):
            audit_quantization(model, require_w4a8=True)


if __name__ == "__main__":
    unittest.main()
