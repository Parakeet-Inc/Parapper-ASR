import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "stage_cat_hf_release.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = load_module("stage_cat_hf_release", SCRIPT)


class StageCatHfReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        self.tools = self.root / "tools"
        (self.tools / "assets").mkdir(parents=True)
        (self.tools / "LICENSE").write_text(
            "Parakeet MIT license\n", encoding="utf-8"
        )
        self.documentation = self.root / "procedure.md"
        self.documentation.write_text(
            r".\scripts\local-translation\cat-translate\verify_cat_onnx_distribution.py"
            + "\n",
            encoding="utf-8",
        )
        self.output = self.root / "hf-upload"
        self._write_candidate()
        self._write_tools()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_candidate(self):
        for name in stage.CORE_PAYLOAD_FILES + stage.CORE_METADATA_FILES:
            (self.candidate / name).write_text(f"{name}\n", encoding="utf-8")
        (self.candidate / "distribution-manifest.json").write_text(
            json.dumps({"files": {name: {} for name in stage.CORE_PAYLOAD_FILES}}),
            encoding="utf-8",
        )

    def _write_tools(self):
        for name in stage.RELEASE_TOOL_FILES:
            (self.tools / name).write_text(f"{name}\n", encoding="utf-8")
        for name in stage.RELEASE_ASSET_FILES:
            (self.tools / "assets" / name).write_text(f"{name}\n", encoding="utf-8")

    def test_verified_candidate_stages_one_complete_hf_upload_folder(self):
        verified = []

        stage.stage_release(
            self.candidate,
            self.output,
            self.tools,
            self.documentation,
            lambda path: verified.append(path),
            force=False,
        )

        self.assertEqual(verified[0], self.candidate.resolve())
        self.assertEqual(len(verified), 2)
        self.assertEqual(verified[1].name, "bundle")
        expected_top_level = set(stage.CORE_PAYLOAD_FILES + stage.CORE_METADATA_FILES) | {
            "README.md",
            "release-tools",
            stage.HF_CHECKSUMS_NAME,
        }
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            expected_top_level,
        )
        self.assertEqual(
            (self.output / "README.md").read_bytes(),
            (self.output / "MODEL_CARD.md").read_bytes(),
        )
        checksum_text = (self.output / stage.HF_CHECKSUMS_NAME).read_text(
            encoding="utf-8"
        )
        self.assertIn("model_q4.onnx.data", checksum_text)
        self.assertIn("release-tools/export_cat_onnx_variants.ps1", checksum_text)
        self.assertIn("release-tools/quantize_cat_embedding_gather.py", checksum_text)
        self.assertNotIn("bench_translation_onnx.py", checksum_text)
        self.assertFalse(
            (self.output / "release-tools" / "bench_translation_onnx.py").exists()
        )
        self.assertTrue((self.output / "release-tools" / "LICENSE").is_file())
        procedure = (
            self.output / "release-tools" / "RELEASE_PROCEDURE.md"
        ).read_text(encoding="utf-8")
        self.assertIn(r".\release-tools\verify_cat_onnx_distribution.py", procedure)
        self.assertNotIn(r".\release-tools\cat-translate", procedure)
        self.assertNotIn(r".\scripts\local-translation", procedure)

    def test_verification_failure_does_not_replace_existing_output(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing\n", encoding="utf-8")

        def reject(_path):
            raise stage.StagingError("candidate rejected")

        with self.assertRaisesRegex(stage.StagingError, "candidate rejected"):
            stage.stage_release(
                self.candidate,
                self.output,
                self.tools,
                self.documentation,
                reject,
                force=True,
            )

        self.assertEqual(marker.read_text(encoding="utf-8"), "existing\n")

    def test_manifest_with_unexpected_payload_contract_is_rejected(self):
        manifest = json.loads(
            (self.candidate / "distribution-manifest.json").read_text(encoding="utf-8")
        )
        manifest["files"]["unreviewed.bin"] = {}
        (self.candidate / "distribution-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        with self.assertRaisesRegex(stage.StagingError, "exact audited CAT payload"):
            stage.stage_release(
                self.candidate,
                self.output,
                self.tools,
                self.documentation,
                lambda _path: None,
                force=False,
            )

        self.assertFalse(self.output.exists())

    def test_force_rejects_an_output_directory_that_contains_its_inputs(self):
        with self.assertRaisesRegex(stage.StagingError, "unsafe output"):
            stage.stage_release(
                self.candidate,
                self.root,
                self.tools,
                self.documentation,
                lambda _path: None,
                force=True,
            )

        self.assertTrue(self.candidate.is_dir())
        self.assertTrue(self.tools.is_dir())


if __name__ == "__main__":
    unittest.main()
