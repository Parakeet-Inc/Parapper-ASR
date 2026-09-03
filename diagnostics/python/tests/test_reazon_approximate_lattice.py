from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_reazon_approximate_lattice import (  # noqa: E402
    select_oracle,
    structural_lattice_candidates,
    summarize_selection,
    union_candidates,
)


class ApproximateLatticeTests(unittest.TestCase):
    def test_structural_lattice_enumerates_seed_and_recombined_histories(self) -> None:
        row = {
            "utterance_id": "sample",
            "seeds": [
                {"rank": 1, "hypothesis": "ABCD", "token_ids": [1, 2, 3, 4]},
                {"rank": 2, "hypothesis": "XBCY", "token_ids": [5, 2, 3, 6]},
            ],
        }

        candidates = structural_lattice_candidates(row)

        self.assertEqual(
            {candidate["hypothesis"] for candidate in candidates},
            {"ABCD", "ABCY", "XBCD", "XBCY"},
        )
        self.assertEqual(sum(candidate["is_seed"] for candidate in candidates), 2)

    def test_union_oracle_can_select_a_novel_recombined_transcript(self) -> None:
        row = {
            "utterance_id": "sample",
            "reference": "橋を渡る。",
            "seeds": [
                {"rank": 1, "hypothesis": "箸を渡る", "token_ids": [1, 2, 3]},
                {"rank": 2, "hypothesis": "橋が渡る", "token_ids": [4, 5, 3]},
            ],
            "candidates": [
                {
                    "rank": 1,
                    "hypothesis": "橋を渡る",
                    "token_ids": [4, 2, 3],
                    "lattice_score": -1.0,
                    "is_seed": False,
                }
            ],
        }

        seed = summarize_selection(
            [row], lambda item: select_oracle(item["reference"], item["seeds"])
        )
        union = summarize_selection(
            [row],
            lambda item: select_oracle(
                item["reference"], union_candidates(item)
            ),
        )

        self.assertEqual(seed["edits"], 1)
        self.assertEqual(union["edits"], 0)
        self.assertEqual(union["exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
