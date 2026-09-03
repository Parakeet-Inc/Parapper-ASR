from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from summarize_nc_ulunas_asr import (  # noqa: E402
    bootstrap_micro_cer_delta,
    build_examples,
    run,
    score_utterance,
)


def completed(
    utterance_id: str,
    reference: str,
    hypothesis: str,
    elapsed_ms: float = 10.0,
) -> dict[str, object]:
    return {
        "status": "completed",
        "schema_version": 1,
        "utterance_id": utterance_id,
        "reference": reference,
        "hypothesis": hypothesis,
        "duration_samples": 16000,
        "inference_elapsed_ms": elapsed_ms,
    }


def failed(utterance_id: str, stage: str = "decode") -> dict[str, object]:
    return {
        "status": "failed",
        "schema_version": 1,
        "utterance_id": utterance_id,
        "stage": stage,
        "message": "synthetic failure",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def summarize(tmp_path: Path, conditions: dict[str, Path], pairs: list[str], **kwargs):
    output_dir = kwargs.pop("output_dir", tmp_path / "analysis")
    argv = []
    for name, path in conditions.items():
        argv += ["--condition", f"{name}={path}"]
    for pair in pairs:
        argv += ["--pair", pair]
    argv += [
        "--bootstrap-samples",
        str(kwargs.pop("bootstrap_samples", 64)),
        "--seed",
        str(kwargs.pop("seed", 20260817)),
        "--output-dir",
        str(output_dir),
    ]
    assert not kwargs, f"unexpected kwargs {kwargs}"
    return run(argv), output_dir


def test_score_utterance_counts_edits_exact_match_and_empty_hypothesis() -> None:
    perfect = score_utterance("あいうえお", "あいうえお")
    assert perfect["edits"] == 0
    assert perfect["cer"] == 0.0
    assert perfect["exact_match"] == 1
    assert perfect["empty_hypothesis"] == 0

    truncated = score_utterance("かきくけこさ", "かきくけこ")
    assert truncated["deletions"] == 1
    assert truncated["trailing_deletions"] == 1
    assert truncated["leading_deletions"] == 0
    assert truncated["edits"] == 1
    assert truncated["cer"] == pytest.approx(1 / 6)
    assert truncated["exact_match"] == 0

    empty = score_utterance("さし", "")
    assert empty["deletions"] == 2
    assert empty["leading_deletions"] == 2
    assert empty["trailing_deletions"] == 2
    assert empty["empty_hypothesis"] == 1
    assert empty["cer"] == 1.0

    with pytest.raises(ValueError):
        score_utterance("", "あ")


def test_condition_summary_matches_hand_computed_micro_and_macro_cer(
    tmp_path: Path,
) -> None:
    # 0/5, 1/6 and 2/2 edits -> micro 3/13, macro (0 + 1/6 + 1) / 3.
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお", 10.0),
            completed("u2", "かきくけこさ", "かきくけこ", 20.0),
            completed("u3", "さし", "", 30.0),
        ],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこさ", "かきくけこさ"),
            completed("u3", "さし", "さし"),
        ],
    )
    summary, output_dir = summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
    )

    block = summary["conditions"]["clean"]
    assert block["utterances"] == 3
    assert block["failed"] == 0
    assert block["reference_characters"] == 13
    assert block["edits"] == 3
    assert block["micro_cer"] == pytest.approx(3 / 13)
    assert block["macro_cer"] == pytest.approx((0 + 1 / 6 + 1) / 3)
    assert block["exact_matches"] == 1
    assert block["empty_hypotheses"] == 1
    assert block["deletions"] == 3
    assert block["substitutions"] == 0
    assert block["insertions"] == 0
    assert block["deletion_rate"] == pytest.approx(3 / 13)
    assert block["substitution_rate"] == 0.0
    assert block["insertion_rate"] == 0.0
    assert block["deletion_share"] == 1.0
    assert block["substitution_share"] == 0.0
    assert block["insertion_share"] == 0.0
    assert block["leading_deletions"] == 2
    assert block["trailing_deletions"] == 3
    assert block["mean_inference_elapsed_ms"] == pytest.approx(20.0)

    perfect = summary["conditions"]["nc"]
    assert perfect["micro_cer"] == 0.0
    assert perfect["exact_matches"] == 3
    assert perfect["empty_hypotheses"] == 0
    # No edits at all: every share degenerates to 0.0 rather than dividing by zero.
    assert perfect["substitution_share"] == 0.0
    assert perfect["deletion_share"] == 0.0
    assert perfect["insertion_share"] == 0.0
    assert perfect["deletion_rate"] == 0.0

    comparison = summary["comparisons"]["nc_vs_clean"]
    assert comparison["paired_utterances"] == 3
    assert comparison["dropped_from_pair"] == 0
    assert comparison["delta_micro_cer_pp"] == pytest.approx(-100.0 * 3 / 13)
    assert (comparison["wins"], comparison["losses"], comparison["ties"]) == (2, 0, 1)
    assert comparison["delta_leading_deletions"] == -2
    assert comparison["delta_trailing_deletions"] == -3
    assert comparison["delta_empty_hypotheses"] == -1
    assert comparison["delta_exact_matches"] == 2

    assert (output_dir / "summary.json").exists()
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "## ペア比較" in report
    assert "nc_vs_clean" in report


def test_edit_type_rates_sum_to_micro_cer_and_pair_deltas_match(tmp_path: Path) -> None:
    # One deletion, one insertion and one substitution over 15 reference chars:
    # micro CER 3/15, each rate 1/15, each share 1/3.
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [
            completed("u1", "あいうえお", "あいうえ"),
            completed("u2", "かきくけこ", "かきくけこさ"),
            completed("u3", "さしすせそ", "さしすせた"),
        ],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこ", "かきくけこ"),
            completed("u3", "さしすせそ", "さしすせそ"),
        ],
    )
    summary, output_dir = summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
    )

    block = summary["conditions"]["clean"]
    assert block["reference_characters"] == 15
    assert (block["substitutions"], block["deletions"], block["insertions"]) == (1, 1, 1)
    assert block["micro_cer"] == pytest.approx(3 / 15)
    assert block["substitution_rate"] == pytest.approx(1 / 15)
    assert block["deletion_rate"] == pytest.approx(1 / 15)
    assert block["insertion_rate"] == pytest.approx(1 / 15)
    assert (
        block["substitution_rate"] + block["deletion_rate"] + block["insertion_rate"]
        == pytest.approx(block["micro_cer"])
    )
    assert block["substitution_share"] == pytest.approx(1 / 3)
    assert block["deletion_share"] == pytest.approx(1 / 3)
    assert block["insertion_share"] == pytest.approx(1 / 3)
    assert (
        block["substitution_share"] + block["deletion_share"] + block["insertion_share"]
        == pytest.approx(1.0)
    )

    comparison = summary["comparisons"]["nc_vs_clean"]
    assert comparison["delta_substitutions"] == -1
    assert comparison["delta_deletions"] == -1
    assert comparison["delta_insertions"] == -1
    assert comparison["delta_substitution_rate_pp"] == pytest.approx(-100.0 / 15)
    assert comparison["delta_deletion_rate_pp"] == pytest.approx(-100.0 / 15)
    assert comparison["delta_insertion_rate_pp"] == pytest.approx(-100.0 / 15)
    assert (
        comparison["delta_substitution_rate_pp"]
        + comparison["delta_deletion_rate_pp"]
        + comparison["delta_insertion_rate_pp"]
        == pytest.approx(comparison["delta_micro_cer_pp"])
    )

    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "## 編集タイプ内訳" in report
    assert "### 編集タイプ別差分" in report
    assert "置換率" in report


def test_failed_record_is_counted_and_dropped_from_the_pair(tmp_path: Path) -> None:
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこ", "かきくけ"),
            completed("u3", "さしすせそ", "さしすせそ"),
        ],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこ", "かきくけこ"),
            failed("u3"),
        ],
    )
    summary, _ = summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
    )

    assert summary["conditions"]["nc"]["failed"] == 1
    assert summary["conditions"]["nc"]["failed_utterance_ids"] == ["u3"]
    assert summary["conditions"]["nc"]["utterances"] == 2
    assert summary["conditions"]["clean"]["utterances"] == 3
    assert summary["utterances"] == 3

    comparison = summary["comparisons"]["nc_vs_clean"]
    assert comparison["paired_utterances"] == 2
    assert comparison["dropped_from_pair"] == 1
    assert comparison["dropped_utterance_ids"] == ["u3"]
    # Paired micro CER covers only u1 and u2: baseline 1/10, treatment 0/10.
    assert comparison["baseline_micro_cer"] == pytest.approx(0.1)
    assert comparison["treatment_micro_cer"] == 0.0
    assert comparison["delta_micro_cer_pp"] == pytest.approx(-10.0)


def test_pairs_reject_conditions_that_cover_different_utterances(
    tmp_path: Path,
) -> None:
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこ", "かきくけこ"),
        ],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [completed("u1", "あいうえお", "あいうえお")],
    )
    with pytest.raises(ValueError, match="cover different utterances"):
        summarize(tmp_path, {"clean": clean, "nc": noisy}, ["p=clean:nc"])


def test_mismatched_references_are_a_hard_error_but_punctuation_is_not(
    tmp_path: Path,
) -> None:
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [completed("u1", "こんにちは", "こんにちは")],
    )
    punctuated = write_jsonl(
        tmp_path / "punctuated.jsonl",
        [completed("u1", "こんにちは。", "こんにちは")],
    )
    summary, _ = summarize(
        tmp_path,
        {"clean": clean, "punctuated": punctuated},
        ["p=clean:punctuated"],
    )
    assert summary["comparisons"]["p"]["paired_utterances"] == 1

    different = write_jsonl(
        tmp_path / "different.jsonl",
        [completed("u1", "こんばんは", "こんにちは")],
    )
    with pytest.raises(ValueError, match="disagree on the normalized reference"):
        summarize(
            tmp_path,
            {"clean": clean, "different": different},
            ["p=clean:different"],
            output_dir=tmp_path / "analysis-2",
        )


def test_bootstrap_is_seeded_and_the_summary_is_byte_identical(tmp_path: Path) -> None:
    baseline = {
        f"u{index}": {"edits": index % 3, "reference_characters": 10}
        for index in range(12)
    }
    treatment = {
        f"u{index}": {"edits": (index + 1) % 3, "reference_characters": 10}
        for index in range(12)
    }
    ids = sorted(baseline)
    first = bootstrap_micro_cer_delta(baseline, treatment, ids, 200, seed=7)
    second = bootstrap_micro_cer_delta(baseline, treatment, ids, 200, seed=7)
    assert first == second
    assert first["bootstrap_lower_pp"] <= first["delta_micro_cer_pp"]
    assert first["delta_micro_cer_pp"] <= first["bootstrap_upper_pp"]

    identical = {f"u{index}": {"edits": 2, "reference_characters": 10} for index in range(8)}
    flat = bootstrap_micro_cer_delta(
        identical,
        {key: dict(value) for key, value in identical.items()},
        sorted(identical),
        50,
        seed=7,
    )
    assert flat == {
        "delta_micro_cer_pp": 0.0,
        "bootstrap_lower_pp": 0.0,
        "bootstrap_upper_pp": 0.0,
    }

    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [completed(f"u{index}", "あいうえお", "あいうえ" if index % 2 else "あいうえお")
         for index in range(10)],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [completed(f"u{index}", "あいうえお", "あいうお" if index % 3 else "あいうえお")
         for index in range(10)],
    )
    summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
        output_dir=tmp_path / "analysis-a",
        bootstrap_samples=500,
    )
    summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
        output_dir=tmp_path / "analysis-a",
        bootstrap_samples=500,
    )
    first_bytes = (tmp_path / "analysis-a" / "summary.json").read_bytes()
    summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
        output_dir=tmp_path / "analysis-b",
        bootstrap_samples=500,
    )
    second_bytes = (tmp_path / "analysis-b" / "summary.json").read_bytes()
    assert first_bytes == second_bytes


def test_examples_are_ranked_by_edit_delta_with_utterance_id_tie_break() -> None:
    deltas = [(2, "utt-b"), (2, "utt-a"), (5, "utt-c"), (-3, "utt-e"), (-3, "utt-d"), (0, "utt-f")]
    references = {utterance_id: "あいうえお" for _, utterance_id in deltas}
    baseline_hypotheses = {utterance_id: "あい" for _, utterance_id in deltas}
    treatment_hypotheses = {utterance_id: "うえお" for _, utterance_id in deltas}
    scores = {
        utterance_id: {"edits": 4, "reference_characters": 5}
        for _, utterance_id in deltas
    }
    examples = build_examples(
        deltas,
        references,
        baseline_hypotheses,
        treatment_hypotheses,
        scores,
        scores,
        limit=2,
    )
    assert [example["utterance_id"] for example in examples["regressions"]] == [
        "utt-c",
        "utt-a",
    ]
    assert [example["utterance_id"] for example in examples["improvements"]] == [
        "utt-d",
        "utt-e",
    ]
    assert examples["regressions"][0]["edit_delta"] == 5
    assert examples["improvements"][0]["reference"] == "あいうえお"
    assert examples["improvements"][0]["baseline_hypothesis"] == "あい"
    assert examples["improvements"][0]["treatment_hypothesis"] == "うえお"


def test_examples_from_a_full_run_are_ordered_and_normalized(tmp_path: Path) -> None:
    clean = write_jsonl(
        tmp_path / "clean.jsonl",
        [
            completed("u1", "あいうえお", "あいうえお"),
            completed("u2", "かきくけこ", "かきくけこ"),
            completed("u3", "さしすせそ", "さ"),
        ],
    )
    noisy = write_jsonl(
        tmp_path / "nc.jsonl",
        [
            completed("u1", "あいうえお", "あ、い、う"),
            completed("u2", "かきくけこ", "か"),
            completed("u3", "さしすせそ", "さしすせそ"),
        ],
    )
    summary, _ = summarize(
        tmp_path,
        {"clean": clean, "nc": noisy},
        ["nc_vs_clean=clean:nc"],
    )
    examples = summary["examples"]["nc_vs_clean"]
    assert [example["utterance_id"] for example in examples["regressions"]] == [
        "u2",
        "u1",
    ]
    assert [example["utterance_id"] for example in examples["improvements"]] == ["u3"]
    # Punctuation is stripped before the hypotheses are stored.
    assert examples["regressions"][1]["treatment_hypothesis"] == "あいう"
