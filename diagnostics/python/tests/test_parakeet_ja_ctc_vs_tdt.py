from __future__ import annotations

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from summarize_parakeet_ja_ctc_vs_tdt import (  # noqa: E402
    attribute_errors,
    bootstrap_delta,
    classify_reference,
    okurigana_positions,
    pooled_class_rate,
)


def test_okurigana_positions_mark_kana_after_kanji_inside_one_token() -> None:
    # 買わなくて -> 買わ|なく|て: only わ trails a kanji inside its token.
    tokens = [("買わ", 0), ("なく", 2), ("て", 4)]
    assert okurigana_positions(tokens) == {1}

    # 呼び起こす as one token: び, こ, す all follow a kanji in the token.
    assert okurigana_positions([("呼び起こす", 0)]) == {1, 3, 4}

    # A particle after a kanji is its own token, so it is never okurigana.
    assert okurigana_positions([("山", 0), ("が", 1)]) == set()

    # Kana-only and katakana tokens contribute nothing.
    assert okurigana_positions([("タワー", 0), ("です", 3)]) == set()


def test_classify_reference_uses_okurigana_before_script_classes() -> None:
    reference = "買わない水5リットル"
    classes = classify_reference(reference, okurigana_positions([("買わ", 0)]))
    assert classes == [
        "kanji",
        "okurigana",
        "hiragana_other",
        "hiragana_other",
        "kanji",
        "other",
        "katakana",
        "katakana",
        "katakana",
        "katakana",
    ]


def test_attribute_errors_anchors_substitutions_and_deletions_to_the_reference() -> None:
    reference = "買わない"
    classes = classify_reference(reference, {1})

    dropped_okurigana = attribute_errors(reference, "買ない", classes)
    assert dropped_okurigana["per_class"]["okurigana"] == {
        "substitutions": 0,
        "deletions": 1,
        "total": 1,
    }
    assert dropped_okurigana["per_class"]["kanji"]["deletions"] == 0
    assert dropped_okurigana["insertions"] == 0
    assert dropped_okurigana["edits"] == 1

    substituted_kanji = attribute_errors(reference, "飼わない", classes)
    assert substituted_kanji["per_class"]["kanji"] == {
        "substitutions": 1,
        "deletions": 0,
        "total": 1,
    }
    assert substituted_kanji["edits"] == 1

    inserted_kana = attribute_errors(reference, "買わないい", classes)
    assert inserted_kana["insertions"] == 1
    assert inserted_kana["edits"] == 1


def _scored(per_utterance: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"per_utterance": per_utterance}


def _attribution(okurigana_errors: int, total: int) -> dict[str, object]:
    per_class = {
        char_class: {"substitutions": 0, "deletions": 0, "total": 0}
        for char_class in ("kanji", "okurigana", "hiragana_other", "katakana", "other")
    }
    per_class["okurigana"] = {
        "substitutions": okurigana_errors,
        "deletions": 0,
        "total": total,
    }
    return {"per_class": per_class, "insertions": 0, "edits": okurigana_errors}


def test_bootstrap_delta_is_seeded_and_brackets_the_pooled_delta() -> None:
    ids = [f"utt-{index}" for index in range(8)]
    worse = _scored({utt: _attribution(2, 4) for utt in ids})
    better = _scored({utt: _attribution(1, 4) for utt in ids})

    assert pooled_class_rate(worse, ids, "okurigana") == 0.5
    assert pooled_class_rate(better, ids, "okurigana") == 0.25

    first = bootstrap_delta(worse, better, ids, "okurigana", 200, seed=7)
    second = bootstrap_delta(worse, better, ids, "okurigana", 200, seed=7)
    assert first == second
    assert first["delta"] == 0.25
    # Identical per-utterance rates make every resample delta exactly 0.25.
    assert first["bootstrap_lower"] == 0.25
    assert first["bootstrap_upper"] == 0.25
