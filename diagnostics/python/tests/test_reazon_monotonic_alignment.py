from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_reazon_monotonic_alignment import (  # noqa: E402
    SelectionConfig,
    extension_features,
    select_extension,
)


def alignment(expected: float, lower: float, upper: float, entropy: float = 0.5):
    return {
        "expected_timestamp": expected,
        "posterior_lower_timestamp": lower,
        "posterior_upper_timestamp": upper,
        "entropy": entropy,
    }


def candidate(rank, text, tokens, search_score, forward_score, times, alignments):
    return {
        "rank": rank,
        "hypothesis": text,
        "token_ids": tokens,
        "search_raw_score": search_score,
        "forward_score": forward_score,
        "search_timestamps": times,
        "token_alignments": alignments,
    }


def test_repeated_context_tokens_are_compared_as_distinct_output_positions() -> None:
    top = candidate(
        1,
        "ABAB",
        [1, 2, 1, 2],
        -1.0,
        -0.8,
        [0.4, 0.8, 1.2, 1.6],
        [alignment(0.4, 0.36, 0.44), alignment(0.8, 0.76, 0.84), alignment(1.2, 1.16, 1.24), alignment(1.6, 1.56, 1.64)],
    )
    extended = candidate(
        2,
        "ABABA",
        [1, 2, 1, 2, 1],
        -1.5,
        -1.2,
        [0.4, 0.8, 1.2, 1.6, 2.4],
        [alignment(0.4, 0.36, 0.44), alignment(0.8, 0.76, 0.84), alignment(1.2, 1.16, 1.24), alignment(1.6, 1.56, 1.64), alignment(2.5, 2.4, 2.6)],
    )

    features = extension_features(top, extended)

    assert features is not None
    assert features["direction"] == "tail"
    assert abs(features["expected_time_gap"] - 0.9) < 1.0e-6


def test_selector_rejects_punctuation_only_and_uses_posterior_gap() -> None:
    top = candidate(1, "文", [1], -1.0, -0.8, [0.4], [alignment(0.5, 0.4, 0.6)])
    punctuation = candidate(2, "文。", [1, 9], -1.1, -0.9, [0.4, 1.6], [alignment(0.5, 0.4, 0.6), alignment(1.7, 1.6, 1.8)])
    valid = candidate(3, "文です", [1, 2], -1.2, -1.0, [0.4, 1.2], [alignment(0.5, 0.4, 0.6), alignment(1.4, 1.2, 1.6)])
    config = SelectionConfig(3.0, 2.0, 0.08, 0.8, 2.0, "expected")

    selected, features = select_extension(
        {"candidates": [top, punctuation, valid]}, config
    )

    assert selected["hypothesis"] == "文です"
    assert features is not None
    assert features["direction"] == "tail"
