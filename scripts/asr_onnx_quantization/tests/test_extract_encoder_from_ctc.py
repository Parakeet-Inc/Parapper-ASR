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

from extract_encoder_from_ctc import extract_ctc_head


def _ctc_graph() -> onnx.ModelProto:
    rng = np.random.default_rng(13)
    weight = rng.normal(size=(4, 1024, 1)).astype(np.float32)
    bias = rng.normal(size=(4,)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["features"],
                ["/Transpose_2_output_0"],
                name="encoder_boundary",
            ),
            helper.make_node(
                "Identity", ["length"], ["/Cast_output_0"], name="length_boundary"
            ),
            helper.make_node(
                "Conv",
                ["/Transpose_2_output_0", "ctc.weight", "ctc.bias"],
                ["ctc_logits"],
                name="ctc_decoder.output",
            ),
            helper.make_node(
                "LogSoftmax", ["ctc_logits"], ["ctc_logprobs"], name="ctc_logsoftmax", axis=1
            ),
            helper.make_node(
                "Transpose",
                ["ctc_logprobs"],
                ["logprobs"],
                name="ctc_output_transpose",
                perm=[0, 2, 1],
            ),
        ],
        "ctc_contract",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1024, 3]),
            helper.make_tensor_value_info("length", TensorProto.INT64, [1]),
        ],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, [1, 3, 4])],
        [
            numpy_helper.from_array(weight, name="ctc.weight"),
            numpy_helper.from_array(bias, name="ctc.bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


class ExtractCtcHeadTests(unittest.TestCase):
    def test_ctc_head_uses_encoder_boundary_and_matches_monolithic_logits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model.onnx"
            head = root / "ctc-head-model.onnx"
            model = _ctc_graph()
            onnx.save(model, source)

            extract_ctc_head(source, head)
            self.assertTrue((root / "ctc-head-model.onnx_data").exists())
            onnx.checker.check_model(onnx.load(head, load_external_data=True))

            import onnxruntime as ort

            features = np.random.default_rng(7).normal(size=(1, 1024, 3)).astype(np.float32)
            length = np.array([3], dtype=np.int64)
            inputs = {"features": features, "length": length}
            full_result = ort.InferenceSession(
                str(source), providers=["CPUExecutionProvider"]
            ).run(None, inputs)[0]
            head_session = ort.InferenceSession(
                str(head), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(
                [item.name for item in head_session.get_inputs()],
                ["encoder_outputs", "encoded_lengths"],
            )
            head_result = head_session.run(
                None, {"encoder_outputs": features, "encoded_lengths": length}
            )[0]
            np.testing.assert_allclose(head_result, full_result, rtol=1e-5, atol=1e-6)

            head_model = onnx.load(head, load_external_data=True)
            self.assertEqual([item.name for item in head_model.graph.output], ["logprobs"])
            self.assertNotIn("features", {item.name for item in head_model.graph.input})
            self.assertEqual(
                {item.name for item in head_model.graph.input},
                {"encoder_outputs", "encoded_lengths"},
            )
            self.assertFalse(any("encoder_boundary" == node.name for node in head_model.graph.node))


if __name__ == "__main__":
    unittest.main()
