import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "stage_supertonic3_q4_hf_release.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HuggingFaceStageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = load_module("stage_supertonic3_q4_hf_release", SCRIPT)

    def test_release_tools_do_not_publish_benchmarks_or_performance_results(self):
        names = set(self.stage.RELEASE_TOOL_FILES)
        self.assertFalse(any("bench" in name.casefold() for name in names))
        self.assertFalse(any("rtf" in name.casefold() for name in names))

    def test_output_cannot_contain_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            candidate = root / "candidate"
            candidate.mkdir()
            with self.assertRaises(self.stage.StagingError):
                self.stage.reject_unsafe_output(root, (candidate,))


if __name__ == "__main__":
    unittest.main()
