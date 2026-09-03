from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from summarize_reazon_nbest_ablation import (  # noqa: E402
    select_by_final_exponent,
    select_oracle_candidate,
    summarize_selection,
)


def candidate(rank: int, text: str, score: float, token_count: int) -> dict:
    return {
        "rank": rank,
        "hypothesis": text,
        "raw_score": score,
        "token_ids": list(range(token_count)),
        "timestamps": [],
    }


def test_final_length_exponent_can_select_a_longer_lower_raw_score_candidate() -> None:
    candidates = [
        candidate(1, "短", -1.0, 0),
        candidate(2, "長い候補", -1.2, 4),
    ]

    assert select_by_final_exponent(candidates, 0.0)["hypothesis"] == "短"
    assert select_by_final_exponent(candidates, 1.0)["hypothesis"] == "長い候補"


def test_oracle_reports_the_best_text_even_when_the_model_ranks_it_second() -> None:
    rows = [
        {
            "reference": "今日は晴れです。",
            "candidates": [
                candidate(1, "今日は雨です", -1.0, 6),
                candidate(2, "今日は晴れです", -1.2, 7),
            ],
        }
    ]

    oracle = summarize_selection(
        rows,
        lambda row: select_oracle_candidate(row["reference"], row["candidates"]),
    )

    assert oracle == {
        "samples": 1,
        "reference_characters": 7,
        "edits": 0,
        "micro_cer": 0.0,
        "macro_cer": 0.0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "exact": 1,
        "exact_rate": 1.0,
        "selected_rank_histogram": {"2": 1},
    }
