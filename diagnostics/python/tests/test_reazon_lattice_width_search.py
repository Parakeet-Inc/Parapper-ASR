from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_reazon_lattice_width_search import (  # noqa: E402
    REPOSITORY_ROOT,
    evidence_candidates,
    public_artifact_path,
)


class ReazonLatticeWidthSearchTests(unittest.TestCase):
    def test_public_artifact_path_is_target_relative_for_repository_inputs(self) -> None:
        artifact = (
            REPOSITORY_ROOT
            / "target"
            / "asr-eval"
            / "fixture"
            / "beam4.jsonl"
        )

        self.assertEqual(
            public_artifact_path(artifact),
            "target/asr-eval/fixture/beam4.jsonl",
        )

    def test_public_artifact_path_does_not_publish_external_absolute_paths(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            external_artifact = Path(temporary_directory) / "beam8.jsonl"

            self.assertEqual(
                public_artifact_path(external_artifact),
                "external-artifact/beam8.jsonl",
            )

    def test_evidence_lattice_marks_original_and_one_splice_paths(self) -> None:
        row = {
            "seeds": [
                {
                    "rank": 1,
                    "hypothesis": "ABCD",
                    "token_ids": [1, 2, 3, 4],
                    "raw_score": 0.0,
                },
                {
                    "rank": 2,
                    "hypothesis": "XBCY",
                    "token_ids": [5, 2, 3, 6],
                    "raw_score": -1.0,
                },
            ]
        }

        candidates = evidence_candidates(row)
        by_text = {candidate["hypothesis"]: candidate for candidate in candidates}

        self.assertEqual(set(by_text), {"ABCD", "ABCY", "XBCD", "XBCY"})
        self.assertEqual(by_text["ABCD"]["min_source_switches"], 0)
        self.assertEqual(by_text["XBCY"]["min_source_switches"], 0)
        self.assertEqual(by_text["ABCY"]["min_source_switches"], 1)
        self.assertEqual(by_text["XBCD"]["min_source_switches"], 1)
        self.assertGreater(
            by_text["ABCD"]["support_scores"]["1.0"],
            by_text["XBCY"]["support_scores"]["1.0"],
        )


if __name__ == "__main__":
    unittest.main()
