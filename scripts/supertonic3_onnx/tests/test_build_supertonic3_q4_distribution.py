import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build_supertonic3_q4_distribution.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AdoptedDistributionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = load_module("build_supertonic3_q4_distribution", SCRIPT)

    def test_only_vector_estimator_and_vocoder_are_quantized(self):
        self.assertEqual(
            {
                name: self.build.model_transform_kind(name)
                for name in self.build.ONNX_MODEL_FILES
            },
            {
                "onnx/duration_predictor.onnx": "copy_fp32",
                "onnx/text_encoder.onnx": "copy_fp32",
                "onnx/vector_estimator.onnx": "q4_block16",
                "onnx/vocoder.onnx": "q4_block16_vocoder_d",
            },
        )

    def test_vocoder_d_keeps_the_audited_five_layers_in_fp32(self):
        self.assertEqual(
            self.build.VOCODER_D_FP32_NODES,
            {
                "/decoder/embed/net/Conv",
                "/decoder/convnext.9/pwconv1/Conv",
                "/decoder/convnext.9/pwconv2/Conv",
                "/decoder/head/layer1/net/Conv",
                "/decoder/head/layer2/Conv",
            },
        )

    def test_output_cannot_replace_or_contain_the_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            with self.assertRaises(self.build.BuildError):
                self.build.reject_unsafe_output(source, source)
            with self.assertRaises(self.build.BuildError):
                self.build.reject_unsafe_output(root, source)


if __name__ == "__main__":
    unittest.main()
