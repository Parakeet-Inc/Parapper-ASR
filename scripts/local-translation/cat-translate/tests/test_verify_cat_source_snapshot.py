import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "verify_cat_source_snapshot.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


source_verifier = load_module("verify_cat_source_snapshot", SCRIPT)


class VerifyCatSourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = (
            Path(self.temp_dir.name) / source_verifier.SOURCE_REVISION
        )
        self.source_dir.mkdir()
        self.contents = {
            "config.json": b"config\n",
            "model.safetensors": b"weights\n",
        }
        for name, content in self.contents.items():
            (self.source_dir / name).write_bytes(content)
        self.expected = {
            name: (len(content), hashlib.sha256(content).hexdigest())
            for name, content in self.contents.items()
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_snapshot_records_are_accepted(self):
        source_verifier.verify_snapshot(self.source_dir, self.expected)

    def test_changed_weight_bytes_are_rejected(self):
        (self.source_dir / "model.safetensors").write_bytes(b"changed\n")

        with self.assertRaisesRegex(
            source_verifier.SourceVerificationError, "SHA-256 mismatch"
        ):
            source_verifier.verify_snapshot(self.source_dir, self.expected)

    def test_non_consumed_snapshot_files_do_not_invalidate_consumed_records(self):
        (self.source_dir / "unreviewed.json").write_text("{}\n", encoding="utf-8")

        source_verifier.verify_snapshot(self.source_dir, self.expected)


if __name__ == "__main__":
    unittest.main()
