from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_reazon_fixed_sequence_rescore import (  # noqa: E402
    select_by_score,
    summarize_selection,
)


class FixedSequenceRescoreTests(unittest.TestCase):
    def test_fixed_sequence_scores_can_recover_a_better_second_search_candidate(
        self,
    ) -> None:
        rows = [
            {
                "utterance_id": "sample",
                "reference": "橋を渡る。",
                "candidates": [
                    {
                        "rank": 1,
                        "hypothesis": "箸を渡る",
                        "token_ids": [1, 2, 3],
                        "search_raw_score": -1.0,
                        "viterbi_score": -2.0,
                        "forward_score": -1.5,
                    },
                    {
                        "rank": 2,
                        "hypothesis": "橋を渡る",
                        "token_ids": [4, 2, 3],
                        "search_raw_score": -1.2,
                        "viterbi_score": -1.0,
                        "forward_score": -0.8,
                    },
                ],
            }
        ]

        search = summarize_selection(
            rows,
            lambda row: select_by_score(
                row["candidates"], "search_raw_score", 0.0
            ),
        )
        forward = summarize_selection(
            rows,
            lambda row: select_by_score(row["candidates"], "forward_score", 0.0),
        )

        self.assertEqual(search["edits"], 1)
        self.assertEqual(search["selected_rank_histogram"], {"1": 1})
        self.assertEqual(forward["edits"], 0)
        self.assertEqual(forward["exact_rate"], 1.0)
        self.assertEqual(forward["selected_rank_histogram"], {"2": 1})

    def test_score_selection_applies_the_same_token_count_exponent_contract(
        self,
    ) -> None:
        candidates = [
            {"rank": 1, "token_ids": [1], "forward_score": -1.0},
            {"rank": 2, "token_ids": [1, 2, 3], "forward_score": -1.2},
        ]

        self.assertEqual(
            select_by_score(candidates, "forward_score", 0.0)["rank"], 1
        )
        self.assertEqual(
            select_by_score(candidates, "forward_score", 1.0)["rank"], 2
        )
