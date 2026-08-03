import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


SCRIPT = Path(__file__).parents[1] / "quantize_supertonic3_onnx.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quantize = load_module("quantize_supertonic3_onnx", SCRIPT)


def make_convnext_pointwise_pair(*, layout_sensitive_activation: bool) -> onnx.ModelProto:
    prefix = "/vector_estimator/vector_field/main_blocks.0/convnext.0"
    rng = np.random.default_rng(1234)
    initializers = [
        numpy_helper.from_array(
            rng.normal(0, 0.2, (4, 2, 1)).astype(np.float32), "pwconv1.weight"
        ),
        numpy_helper.from_array(
            rng.normal(0, 0.1, (4,)).astype(np.float32), "pwconv1.bias"
        ),
        numpy_helper.from_array(
            rng.normal(0, 0.2, (2, 4, 1)).astype(np.float32), "pwconv2.weight"
        ),
        numpy_helper.from_array(
            rng.normal(0, 0.1, (2,)).astype(np.float32), "pwconv2.bias"
        ),
    ]
    nodes = [
        helper.make_node(
            "Transpose",
            ["x"],
            ["norm_in"],
            name=f"{prefix}/norm/Transpose",
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Identity", ["norm_in"], ["norm_out"], name=f"{prefix}/norm/Identity"
        ),
        helper.make_node(
            "Transpose",
            ["norm_out"],
            ["pwconv1_in"],
            name=f"{prefix}/norm/Transpose_1",
            perm=[0, 2, 1],
        ),
        helper.make_node(
            "Conv",
            ["pwconv1_in", "pwconv1.weight", "pwconv1.bias"],
            ["hidden"],
            name=f"{prefix}/pwconv1/Conv",
            kernel_shape=[1],
        ),
    ]
    if layout_sensitive_activation:
        initializers.append(
            numpy_helper.from_array(
                np.array([[[0.5], [0.75], [1.25], [1.5]]], dtype=np.float32),
                "channel_scale",
            )
        )
        nodes.append(
            helper.make_node(
                "Mul", ["hidden", "channel_scale"], ["activated"], name=f"{prefix}/act/Mul"
            )
        )
    else:
        initializers.extend(
            [
                numpy_helper.from_array(np.array(np.sqrt(2), np.float32), "sqrt2"),
                numpy_helper.from_array(np.array(1.0, np.float32), "one"),
                numpy_helper.from_array(np.array(0.5, np.float32), "half"),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Div", ["hidden", "sqrt2"], ["scaled"], name=f"{prefix}/act/Div"),
                helper.make_node("Erf", ["scaled"], ["erf"], name=f"{prefix}/act/Erf"),
                helper.make_node("Add", ["erf", "one"], ["shifted"], name=f"{prefix}/act/Add"),
                helper.make_node("Mul", ["hidden", "shifted"], ["gated"], name=f"{prefix}/act/Mul"),
                helper.make_node("Mul", ["gated", "half"], ["activated"], name=f"{prefix}/act/Mul_1"),
            ]
        )
    nodes.append(
        helper.make_node(
            "Conv",
            ["activated", "pwconv2.weight", "pwconv2.bias"],
            ["y"],
            name=f"{prefix}/pwconv2/Conv",
            kernel_shape=[1],
        )
    )
    graph = helper.make_graph(
        nodes,
        "convnext_pointwise_pair",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 5])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 5])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class ConvNextPointwiseLayoutTests(unittest.TestCase):
    def test_scalar_activation_pair_keeps_ntc_layout_and_matches_original_output(self):
        original = make_convnext_pointwise_pair(layout_sensitive_activation=False)
        lowered = onnx.ModelProto()
        lowered.CopyFrom(original)
        report = quantize.lower_affine_nodes(
            lowered, set(), optimize_convnext_layout=True
        )
        onnx.checker.check_model(lowered)

        x = np.random.default_rng(99).normal(size=(1, 2, 5)).astype(np.float32)
        expected = ReferenceEvaluator(original).run(None, {"x": x})[0]
        actual = ReferenceEvaluator(lowered).run(None, {"x": x})[0]

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(sum(node.op_type == "Transpose" for node in lowered.graph.node), 2)
        self.assertEqual(sum(node.op_type == "MatMul" for node in lowered.graph.node), 2)
        self.assertEqual(len(report.pointwise_convs), 2)

    def test_channel_broadcast_pair_is_not_moved_across_layout(self):
        original = make_convnext_pointwise_pair(layout_sensitive_activation=True)
        lowered = onnx.ModelProto()
        lowered.CopyFrom(original)
        quantize.lower_affine_nodes(lowered, set(), optimize_convnext_layout=True)
        onnx.checker.check_model(lowered)

        x = np.random.default_rng(101).normal(size=(1, 2, 5)).astype(np.float32)
        expected = ReferenceEvaluator(original).run(None, {"x": x})[0]
        actual = ReferenceEvaluator(lowered).run(None, {"x": x})[0]

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(sum(node.op_type == "Transpose" for node in lowered.graph.node), 6)


if __name__ == "__main__":
    unittest.main()
