from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from compare_asr_transcripts import compare_reports


class CompareAsrTranscriptsTests(unittest.TestCase):
    def test_comparison_aligns_by_wav_name_and_reports_text_drift_and_rtf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"wav": "reference/a.wav", "repeat": 1, "text": "abc", "rtf": 0.2},
                            {"wav": "reference/b.wav", "repeat": 1, "text": "日本", "rtf": 0.4},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"wav": "candidate/b.wav", "repeat": 1, "text": "日語", "rtf": 0.6},
                            {"wav": "candidate/a.wav", "repeat": 1, "text": "abc", "rtf": 0.2},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = compare_reports(reference_path, {"candidate": candidate_path})

            self.assertEqual(result["reference"]["run_count"], 2)
            candidate = result["candidates"]["candidate"]
            self.assertEqual(candidate["exact_text_match_count"], 1)
            self.assertEqual(candidate["edit_distance"], 1)
            self.assertEqual(candidate["reference_character_count"], 5)
            self.assertAlmostEqual(candidate["character_error_rate_vs_reference"], 0.2)
            self.assertAlmostEqual(candidate["mean_rtf"], 0.4)

    def test_comparison_accepts_benchmark_rows_and_aligns_by_path_and_run_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "runs": 1,
                        "rows": [
                            {
                                "path": "reference/a.wav",
                                "run_index": 0,
                                "text": "alpha",
                                "rtf": 0.3,
                            },
                            {
                                "path": "reference/b.wav",
                                "run_index": 0,
                                "text": "beta",
                                "rtf": 0.5,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        "runs": 1,
                        "rows": [
                            {
                                "path": "candidate/b.wav",
                                "run_index": 0,
                                "text": "beto",
                                "rtf": 0.4,
                            },
                            {
                                "path": "candidate/a.wav",
                                "run_index": 0,
                                "text": "alpha",
                                "rtf": 0.2,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = compare_reports(reference_path, {"candidate": candidate_path})

            candidate = result["candidates"]["candidate"]
            self.assertEqual(candidate["exact_text_match_count"], 1)
            self.assertEqual(candidate["edit_distance"], 1)
            self.assertAlmostEqual(candidate["mean_rtf"], 0.3)
            self.assertEqual(candidate["differences"][0]["wav"], "b.wav")


if __name__ == "__main__":
    unittest.main()
