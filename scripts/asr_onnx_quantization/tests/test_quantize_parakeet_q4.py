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

from quantize_parakeet_q4 import quantize_parakeet_q4


def _model_with_matmul_and_embedding() -> onnx.ModelProto:
    rng = np.random.default_rng(31)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Gather",
                ["embedding", "token_ids"],
                ["embedded"],
                name="token_embedding",
                axis=0,
            ),
            helper.make_node(
                "MatMul",
                ["embedded", "projection.weight"],
                ["projected"],
                name="encoder.layers.0.projection",
            ),
            helper.make_node(
                "MatMul",
                ["projected", "output.weight"],
                ["output"],
                name="ctc_decoder.output",
            ),
        ],
        "parakeet_q4_contract",
        [helper.make_tensor_value_info("token_ids", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 8])],
        [
            numpy_helper.from_array(
                rng.normal(size=(32, 16)).astype(np.float32), name="embedding"
            ),
            numpy_helper.from_array(
                rng.normal(size=(16, 16)).astype(np.float32),
                name="projection.weight",
            ),
            numpy_helper.from_array(
                rng.normal(size=(16, 8)).astype(np.float32), name="output.weight"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


class QuantizeParakeetQ4Tests(unittest.TestCase):
    def test_hqq_block32_uses_uint8_zero_points_and_cat_rowwise_q8_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "fp32.onnx"
            output_path = root / "q4.onnx"
            onnx.save(_model_with_matmul_and_embedding(), input_path)

            report = quantize_parakeet_q4(
                input_path,
                output_path,
                nodes_to_exclude=["ctc_decoder.output"],
                lower_fastconformer_pointwise=False,
            )

            model = onnx.load(output_path)
            initializers = {initializer.name: initializer for initializer in model.graph.initializer}
            q4_node = next(node for node in model.graph.node if node.op_type == "MatMulNBits")
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in q4_node.attribute
            }
            self.assertEqual(attributes["bits"], 4)
            self.assertEqual(attributes["block_size"], 32)
            self.assertNotIn("accuracy_level", attributes)
            self.assertEqual(initializers[q4_node.input[3]].data_type, TensorProto.UINT8)
            self.assertEqual(
                sum(node.op_type == "GatherBlockQuantized" for node in model.graph.node),
                0,
            )
            self.assertEqual(
                initializers["embedding_rowwise_q8"].data_type,
                TensorProto.UINT8,
            )
            self.assertEqual(
                list(initializers["embedding_rowwise_scale"].dims),
                [32],
            )
            self.assertEqual(
                initializers["embedding_rowwise_zero_point"].data_type,
                TensorProto.UINT8,
            )
            self.assertEqual(
                [node.name for node in model.graph.node if node.op_type == "MatMul"],
                ["ctc_decoder.output"],
            )
            self.assertEqual(report["algorithm"], "HQQ")
            self.assertEqual(report["zero_point_storage"], "UINT8")
            self.assertEqual(report["embedding_quantization"], "CAT row-wise Q8")

            import onnxruntime as ort

            session = ort.InferenceSession(
                str(output_path), providers=["CPUExecutionProvider"]
            )
            result = session.run(None, {"token_ids": np.array([1, 3], dtype=np.int64)})[0]
            self.assertEqual(result.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
