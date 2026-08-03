import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "quantize_cat_embedding_gather.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


quantize = load_module("quantize_cat_embedding_gather", SCRIPT)


class QuantizeCatEmbeddingGatherSafetyTests(unittest.TestCase):
    def test_output_that_contains_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            input_dir = root / "input"
            input_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "unsafe out_dir"):
                quantize._reject_unsafe_output(root, (input_dir,))


if __name__ == "__main__":
    unittest.main()
