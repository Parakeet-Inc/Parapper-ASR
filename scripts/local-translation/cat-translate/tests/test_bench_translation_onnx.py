import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "bench_translation_onnx.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("bench_translation_onnx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchTranslationOnnxContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_benchmark_module()

    def test_extended_release_set_keeps_all_28_bidirectional_cases(self):
        cases = self.bench.EXTENDED_CASES

        self.assertEqual(len(cases), 28)
        self.assertEqual(len({case_id for case_id, _, _, _ in cases}), 28)
        self.assertEqual(sum(target == "English" for _, _, target, _ in cases), 13)
        self.assertEqual(sum(target == "Japanese" for _, _, target, _ in cases), 15)

    def test_official_prompt_matches_cat_instruction_and_chat_tokens(self):
        prompt = self.bench.build_prompt(
            "official", "Japanese", "English", "こんにちは。"
        )

        self.assertEqual(
            prompt,
            "<|user|>Translate the following Japanese text into English.\n\n"
            "こんにちは。</s><|assistant|>",
        )

    def test_empty_or_symbol_only_english_output_is_a_language_failure(self):
        self.assertFalse(self.bench.target_language_ok("English", ""))
        self.assertFalse(self.bench.target_language_ok("English", "123!?"))
        self.assertTrue(self.bench.target_language_ok("English", "Hello."))

    def test_release_cli_rejects_single_request_before_loading_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    f"candidate={temp_dir}",
                    "--json-out",
                    str(Path(temp_dir) / "report.json"),
                    "--repeats",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release evidence requires --repeats >= 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
