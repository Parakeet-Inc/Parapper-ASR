import importlib.util
import sys
import unittest
from pathlib import Path

import onnx
from onnx import helper


SCRIPT = Path(__file__).parents[1] / "verify_supertonic3_q4_distribution.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DistributionGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verify = load_module("verify_supertonic3_q4_distribution", SCRIPT)

    def test_fp32_components_reject_quantized_operators(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMulNBits", [], [], domain="com.microsoft")],
                "unexpected_q4",
                [],
                [],
            )
        )
        with self.assertRaises(self.verify.VerificationError):
            self.verify.validate_model_graph("onnx/duration_predictor.onnx", model)
        with self.assertRaises(self.verify.VerificationError):
            self.verify.validate_model_graph("onnx/text_encoder.onnx", model)

    def test_vector_contract_requires_exact_q4_count_and_fp32_final_projection(self):
        nodes = [
            helper.make_node(
                "MatMulNBits",
                [],
                [],
                name=f"q4-{index}",
                domain="com.microsoft",
                bits=4,
                block_size=16,
            )
            for index in range(95)
        ]
        nodes.append(
            helper.make_node(
                "Conv",
                [],
                [],
                name="/vector_estimator/vector_field/proj_out/net/Conv",
            )
        )
        model = helper.make_model(helper.make_graph(nodes, "vector", [], []))
        self.verify.validate_model_graph("onnx/vector_estimator.onnx", model)
        del model.graph.node[-1]
        with self.assertRaises(self.verify.VerificationError):
            self.verify.validate_model_graph("onnx/vector_estimator.onnx", model)

    def test_vocoder_contract_requires_all_five_fp32_boundaries(self):
        nodes = [
            helper.make_node(
                "MatMulNBits",
                [],
                [],
                name=f"q4-{index}",
                domain="com.microsoft",
                bits=4,
                block_size=16,
            )
            for index in range(18)
        ]
        nodes.extend(
            helper.make_node("Conv", [], [], name=name)
            for name in self.verify.VOCODER_D_FP32_NODES
        )
        model = helper.make_model(helper.make_graph(nodes, "vocoder", [], []))
        self.verify.validate_model_graph("onnx/vocoder.onnx", model)
        del model.graph.node[-1]
        with self.assertRaises(self.verify.VerificationError):
            self.verify.validate_model_graph("onnx/vocoder.onnx", model)


if __name__ == "__main__":
    unittest.main()
