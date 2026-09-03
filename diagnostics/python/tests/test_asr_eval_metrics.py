from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from asr_eval_metrics import (  # noqa: E402
    align_characters,
    diagnostic_normalize,
    semantic_normalize,
    summarize_alignment,
)


def test_normalizers_keep_semantic_word_boundaries_but_cer_does_not() -> None:
    text = "ＡＢＣ、東京  タワー。"

    assert diagnostic_normalize(text) == "abc東京タワー"
    assert semantic_normalize(text) == "abc 東京 タワー"


def test_alignment_reports_missing_prefix_as_one_leading_deletion_run() -> None:
    operations = align_characters("案の定あの業者", "あの業者")

    assert summarize_alignment(operations) == {
        "substitutions": 0,
        "deletions": 3,
        "insertions": 0,
        "max_deletion_run": 3,
        "leading_deletions": 3,
        "trailing_deletions": 0,
    }


def test_alignment_tie_break_prefers_substitution_over_delete_insert_pair() -> None:
    operations = align_characters("甲", "乙")

    assert operations == ["substitution"]


def test_internal_deletions_do_not_count_as_leading_or_trailing() -> None:
    operations = align_characters("abcdef", "abef")

    assert summarize_alignment(operations) == {
        "substitutions": 0,
        "deletions": 2,
        "insertions": 0,
        "max_deletion_run": 2,
        "leading_deletions": 0,
        "trailing_deletions": 0,
    }
