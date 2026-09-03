#!/usr/bin/env python3
"""Summarize paired ASR runs that differ only by an audio front end (UL-UNAS NC).

Inputs are `run_asr_eval` JSONL outputs (``EvalRecordV1``): one condition per
file, one JSON object per line. Conditions are named on the command line and
compared in explicitly declared pairs, so the same script serves any front-end
ablation (clean vs noise-cancelled, padded vs unpadded, ...).

Scoring contract:

- Every text goes through :func:`asr_eval_metrics.diagnostic_normalize`
  (NFKC -> lower -> drop whitespace and Unicode punctuation) before alignment.
- ``status != "completed"`` records are collected as failures, counted per
  condition, and excluded from scoring.
- The two conditions of a pair must cover the same utterance ids (any status).
  Utterances that failed in either half are dropped from that pair's paired
  statistics and reported as ``dropped_from_pair``.
- References must agree across conditions after normalization; the front end
  never rewrites references, so a mismatch is a hard error.
- CER is broken down by edit type: ``substitution_rate`` + ``deletion_rate`` +
  ``insertion_rate`` == ``micro_cer`` (all divided by reference characters),
  and the ``*_share`` fields give each type's share of all edit operations
  (0.0 for every type when a condition made no edits at all).

Usage:
  python summarize_nc_ulunas_asr.py \
    --condition parakeet_clean=<dir>/parakeet_clean.jsonl \
    --condition parakeet_nc=<dir>/parakeet_nc.jsonl \
    --pair parakeet_nc_vs_clean=parakeet_clean:parakeet_nc \
    --bootstrap-samples 10000 --seed 20260817 \
    --output-dir <dir>/analysis-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from asr_eval_metrics import (  # noqa: E402
    align_characters,
    diagnostic_normalize,
    summarize_alignment,
)

SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 20260817
DEFAULT_EXAMPLES = 5
REPORT_EXAMPLES = 3
REPORT_TEXT_LIMIT = 60
EDIT_TYPES = ("substitutions", "deletions", "insertions")


@dataclass
class Condition:
    """One JSONL run: completed records plus the failures excluded from scoring."""

    name: str
    path: Path
    records: dict[str, dict] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    observed_ids: set[str] = field(default_factory=set)


def load_condition(name: str, path: Path) -> Condition:
    """Read one condition JSONL, splitting completed records from failures."""
    condition = Condition(name=name, path=path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = record["utterance_id"]
            if utterance_id in condition.observed_ids:
                raise ValueError(f"duplicate utterance {utterance_id} in {path}")
            condition.observed_ids.add(utterance_id)
            if record.get("status") != "completed":
                condition.failures.append(
                    {
                        "utterance_id": utterance_id,
                        "stage": str(record.get("stage", "")),
                        "message": str(record.get("message", "")),
                    }
                )
                continue
            condition.records[utterance_id] = record
    condition.failures.sort(key=lambda failure: failure["utterance_id"])
    return condition


def collect_references(conditions: Sequence[Condition]) -> dict[str, str]:
    """Return the normalized reference per utterance, rejecting disagreements."""
    references: dict[str, str] = {}
    for condition in conditions:
        for utterance_id in sorted(condition.records):
            normalized = diagnostic_normalize(condition.records[utterance_id]["reference"])
            existing = references.setdefault(utterance_id, normalized)
            if existing != normalized:
                raise ValueError(
                    f"conditions disagree on the normalized reference of {utterance_id}: "
                    f"{existing!r} != {normalized!r} (from {condition.name})"
                )
    return references


def score_utterance(reference: str, hypothesis: str) -> dict[str, float]:
    """Align one normalized pair and return its edit counts and CER."""
    if not reference:
        raise ValueError("normalized reference is empty; per-utterance CER is undefined")
    counts = summarize_alignment(align_characters(reference, hypothesis))
    edits = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    scored: dict[str, float] = dict(counts)
    scored["edits"] = edits
    scored["reference_characters"] = len(reference)
    scored["cer"] = edits / len(reference)
    scored["exact_match"] = int(hypothesis == reference)
    scored["empty_hypothesis"] = int(not hypothesis)
    return scored


def score_condition(
    condition: Condition, references: dict[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, object]]:
    """Score every completed record of one condition.

    Returns the per-utterance scores, the normalized hypotheses, and the
    condition-level summary block written into ``summary.json``.
    """
    per_utterance: dict[str, dict[str, float]] = {}
    hypotheses: dict[str, str] = {}
    for utterance_id in sorted(condition.records):
        hypothesis = diagnostic_normalize(condition.records[utterance_id]["hypothesis"])
        hypotheses[utterance_id] = hypothesis
        try:
            per_utterance[utterance_id] = score_utterance(
                references[utterance_id], hypothesis
            )
        except ValueError as error:
            raise ValueError(
                f"{condition.name}/{utterance_id}: {error}"
            ) from error

    ordered = sorted(per_utterance)
    elapsed = [
        float(condition.records[utterance_id]["inference_elapsed_ms"])
        for utterance_id in ordered
    ]
    reference_characters = _total(per_utterance, ordered, "reference_characters")
    edits = _total(per_utterance, ordered, "edits")
    counts = {
        edit_type: _total(per_utterance, ordered, edit_type) for edit_type in EDIT_TYPES
    }
    if sum(counts.values()) != edits:
        raise RuntimeError(
            f"{condition.name}: edit-type counts {counts} do not sum to {edits} edits"
        )
    micro = edits / reference_characters if reference_characters else 0.0
    rates = {
        edit_type: (counts[edit_type] / reference_characters if reference_characters else 0.0)
        for edit_type in EDIT_TYPES
    }
    if not math.isclose(sum(rates.values()), micro, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(
            f"{condition.name}: per-type rates {rates} do not sum to micro CER {micro}"
        )
    # Shares are undefined without edits; a flawless condition reports 0.0 for all three.
    shares = {
        edit_type: (counts[edit_type] / edits if edits else 0.0)
        for edit_type in EDIT_TYPES
    }
    summary = {
        "utterances": len(ordered),
        "failed": len(condition.failures),
        "failed_utterance_ids": [failure["utterance_id"] for failure in condition.failures],
        "reference_characters": reference_characters,
        "edits": edits,
        "micro_cer": micro,
        "macro_cer": (
            sum(per_utterance[utterance_id]["cer"] for utterance_id in ordered)
            / len(ordered)
            if ordered
            else 0.0
        ),
        "substitution_rate": rates["substitutions"],
        "deletion_rate": rates["deletions"],
        "insertion_rate": rates["insertions"],
        "substitution_share": shares["substitutions"],
        "deletion_share": shares["deletions"],
        "insertion_share": shares["insertions"],
        "exact_matches": _total(per_utterance, ordered, "exact_match"),
        "empty_hypotheses": _total(per_utterance, ordered, "empty_hypothesis"),
        "substitutions": counts["substitutions"],
        "deletions": counts["deletions"],
        "insertions": counts["insertions"],
        "leading_deletions": _total(per_utterance, ordered, "leading_deletions"),
        "trailing_deletions": _total(per_utterance, ordered, "trailing_deletions"),
        "mean_inference_elapsed_ms": sum(elapsed) / len(elapsed) if elapsed else 0.0,
    }
    return per_utterance, hypotheses, summary


def _total(
    per_utterance: dict[str, dict[str, float]], ids: Iterable[str], key: str
) -> int:
    return int(sum(per_utterance[utterance_id][key] for utterance_id in ids))


def micro_cer(per_utterance: dict[str, dict[str, float]], ids: Sequence[str]) -> float:
    """Pooled CER over ``ids`` (a resample may repeat an utterance)."""
    edits = 0
    characters = 0
    for utterance_id in ids:
        edits += per_utterance[utterance_id]["edits"]
        characters += per_utterance[utterance_id]["reference_characters"]
    return edits / characters if characters else 0.0


def edit_type_rate(
    per_utterance: dict[str, dict[str, float]], ids: Sequence[str], edit_type: str
) -> float:
    """Pooled per-type error rate (one edit type / reference characters)."""
    errors = 0
    characters = 0
    for utterance_id in ids:
        errors += per_utterance[utterance_id][edit_type]
        characters += per_utterance[utterance_id]["reference_characters"]
    return errors / characters if characters else 0.0


def bootstrap_micro_cer_delta(
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    ordered_ids: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap of the pooled ``treatment - baseline`` CER, in points."""
    if not ordered_ids:
        raise ValueError("cannot bootstrap an empty utterance set")
    generator = random.Random(seed)
    count = len(ordered_ids)
    deltas: list[float] = []
    for _ in range(samples):
        resample = [ordered_ids[generator.randrange(count)] for _ in range(count)]
        deltas.append(
            micro_cer(treatment, resample) - micro_cer(baseline, resample)
        )
    deltas.sort()
    lower = deltas[int(0.025 * samples)]
    upper = deltas[min(int(0.975 * samples), samples - 1)]
    return {
        "delta_micro_cer_pp": 100.0
        * (micro_cer(treatment, ordered_ids) - micro_cer(baseline, ordered_ids)),
        "bootstrap_lower_pp": 100.0 * lower,
        "bootstrap_upper_pp": 100.0 * upper,
    }


def edit_deltas(
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    ordered_ids: Sequence[str],
) -> list[tuple[int, str]]:
    """Per-utterance ``treatment - baseline`` edit-count deltas."""
    return [
        (
            int(treatment[utterance_id]["edits"] - baseline[utterance_id]["edits"]),
            utterance_id,
        )
        for utterance_id in ordered_ids
    ]


def build_examples(
    deltas: Sequence[tuple[int, str]],
    references: dict[str, str],
    baseline_hypotheses: dict[str, str],
    treatment_hypotheses: dict[str, str],
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    limit: int,
) -> dict[str, list[dict[str, object]]]:
    """Top regressions and improvements, ties broken by utterance id."""
    regressions = sorted(
        (item for item in deltas if item[0] > 0),
        key=lambda item: (-item[0], item[1]),
    )[:limit]
    improvements = sorted(
        (item for item in deltas if item[0] < 0),
        key=lambda item: (item[0], item[1]),
    )[:limit]

    def render(entries: Sequence[tuple[int, str]]) -> list[dict[str, object]]:
        return [
            {
                "utterance_id": utterance_id,
                "edit_delta": delta,
                "baseline_edits": int(baseline[utterance_id]["edits"]),
                "treatment_edits": int(treatment[utterance_id]["edits"]),
                "reference": references[utterance_id],
                "baseline_hypothesis": baseline_hypotheses[utterance_id],
                "treatment_hypothesis": treatment_hypotheses[utterance_id],
            }
            for delta, utterance_id in entries
        ]

    return {"regressions": render(regressions), "improvements": render(improvements)}


def compare_pair(
    label: str,
    baseline_condition: Condition,
    treatment_condition: Condition,
    scores: dict[str, dict[str, dict[str, float]]],
    hypotheses: dict[str, dict[str, str]],
    references: dict[str, str],
    samples: int,
    seed: int,
    example_limit: int,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    """Paired statistics for one baseline/treatment pair."""
    if baseline_condition.observed_ids != treatment_condition.observed_ids:
        only_baseline = sorted(
            baseline_condition.observed_ids - treatment_condition.observed_ids
        )
        only_treatment = sorted(
            treatment_condition.observed_ids - baseline_condition.observed_ids
        )
        raise ValueError(
            f"pair {label}: {baseline_condition.name} and {treatment_condition.name} "
            f"cover different utterances "
            f"(only in {baseline_condition.name}: {len(only_baseline)} "
            f"{only_baseline[:5]}, only in {treatment_condition.name}: "
            f"{len(only_treatment)} {only_treatment[:5]})"
        )

    baseline = scores[baseline_condition.name]
    treatment = scores[treatment_condition.name]
    common = sorted(set(baseline) & set(treatment))
    dropped = sorted(baseline_condition.observed_ids - set(common))
    if not common:
        raise ValueError(f"pair {label}: no utterance completed in both conditions")

    deltas = edit_deltas(baseline, treatment, common)
    wins = sum(1 for delta, _ in deltas if delta < 0)
    losses = sum(1 for delta, _ in deltas if delta > 0)
    ties = sum(1 for delta, _ in deltas if delta == 0)
    bootstrap = bootstrap_micro_cer_delta(baseline, treatment, common, samples, seed)

    comparison: dict[str, object] = {
        "baseline_condition": baseline_condition.name,
        "treatment_condition": treatment_condition.name,
        "paired_utterances": len(common),
        "dropped_from_pair": len(dropped),
        "dropped_utterance_ids": dropped,
        "baseline_micro_cer": micro_cer(baseline, common),
        "treatment_micro_cer": micro_cer(treatment, common),
        "delta_micro_cer_pp": bootstrap["delta_micro_cer_pp"],
        "bootstrap_lower_pp": bootstrap["bootstrap_lower_pp"],
        "bootstrap_upper_pp": bootstrap["bootstrap_upper_pp"],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "delta_substitutions": _total(treatment, common, "substitutions")
        - _total(baseline, common, "substitutions"),
        "delta_deletions": _total(treatment, common, "deletions")
        - _total(baseline, common, "deletions"),
        "delta_insertions": _total(treatment, common, "insertions")
        - _total(baseline, common, "insertions"),
        "delta_substitution_rate_pp": 100.0
        * (
            edit_type_rate(treatment, common, "substitutions")
            - edit_type_rate(baseline, common, "substitutions")
        ),
        "delta_deletion_rate_pp": 100.0
        * (
            edit_type_rate(treatment, common, "deletions")
            - edit_type_rate(baseline, common, "deletions")
        ),
        "delta_insertion_rate_pp": 100.0
        * (
            edit_type_rate(treatment, common, "insertions")
            - edit_type_rate(baseline, common, "insertions")
        ),
        "delta_leading_deletions": _total(treatment, common, "leading_deletions")
        - _total(baseline, common, "leading_deletions"),
        "delta_trailing_deletions": _total(treatment, common, "trailing_deletions")
        - _total(baseline, common, "trailing_deletions"),
        "delta_empty_hypotheses": _total(treatment, common, "empty_hypothesis")
        - _total(baseline, common, "empty_hypothesis"),
        "delta_exact_matches": _total(treatment, common, "exact_match")
        - _total(baseline, common, "exact_match"),
    }
    examples = build_examples(
        deltas,
        references,
        hypotheses[baseline_condition.name],
        hypotheses[treatment_condition.name],
        baseline,
        treatment,
        example_limit,
    )
    return comparison, examples


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truncate(text: str, limit: int = REPORT_TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_report(summary: dict[str, object]) -> str:
    """Render the Japanese REPORT.md for one summary payload."""
    parameters = summary["parameters"]
    lines = [
        "# UL-UNAS ノイズキャンセル ASR 評価サマリ",
        "",
        f"- split: `{summary['split_id']}`",
        f"- 条件数: {len(summary['conditions'])} / 観測発話数(全条件の和集合): "
        f"{summary['utterances']}",
        f"- bootstrap: {parameters['bootstrap_samples']} リサンプル, "
        f"seed {parameters['seed']}",
        "- CER は diagnostic 正規化 (NFKC → 小文字化 → 空白・句読点除去) 後の文字 CER。",
        "- 平均推論時間は同一エンジン・同一ランナー内でのみ比較可能。",
        "",
        "## 結論",
        "",
    ]
    for label, comparison in summary["comparisons"].items():
        lines.append(
            f"- `{label}` ({comparison['baseline_condition']} → "
            f"{comparison['treatment_condition']}, n={comparison['paired_utterances']}, "
            f"除外 {comparison['dropped_from_pair']}): "
            f"micro CER {100.0 * comparison['baseline_micro_cer']:.3f}% → "
            f"{100.0 * comparison['treatment_micro_cer']:.3f}% "
            f"(Δ {comparison['delta_micro_cer_pp']:+.3f} pp, 95% CI "
            f"[{comparison['bootstrap_lower_pp']:+.3f}, "
            f"{comparison['bootstrap_upper_pp']:+.3f}] pp), "
            f"改善/悪化/同点 = {comparison['wins']}/{comparison['losses']}/"
            f"{comparison['ties']}, 内訳 Δ置換 "
            f"{comparison['delta_substitution_rate_pp']:+.3f} pp / Δ削除 "
            f"{comparison['delta_deletion_rate_pp']:+.3f} pp / Δ挿入 "
            f"{comparison['delta_insertion_rate_pp']:+.3f} pp, Δ先頭削除 "
            f"{comparison['delta_leading_deletions']:+d}, Δ空仮説 "
            f"{comparison['delta_empty_hypotheses']:+d}"
        )
    lines += [
        "",
        "## 条件",
        "",
        "| 条件 | JSONL | 記録数 | 完了 | 失敗 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for condition, block in summary["conditions"].items():
        lines.append(
            f"| {condition} | `{parameters['condition_paths'][condition]}` "
            f"| {block['utterances'] + block['failed']} | {block['utterances']} "
            f"| {block['failed']} |"
        )
    lines += [
        "",
        "## 条件別サマリ",
        "",
        "置換率/削除率/挿入率はいずれも正規化参照文字数で割った値で、3 つの合計が "
        "micro CER に一致する。",
        "",
        "| 条件 | n | 失敗 | micro CER | macro CER | 置換率 | 削除率 | 挿入率 | 完全一致 "
        "| 空仮説 | 先頭削除 | 末尾削除 | 平均推論ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: |",
    ]
    for condition, block in summary["conditions"].items():
        lines.append(
            f"| {condition} | {block['utterances']} | {block['failed']} "
            f"| {100.0 * block['micro_cer']:.3f}% | {100.0 * block['macro_cer']:.3f}% "
            f"| {100.0 * block['substitution_rate']:.3f}% "
            f"| {100.0 * block['deletion_rate']:.3f}% "
            f"| {100.0 * block['insertion_rate']:.3f}% "
            f"| {block['exact_matches']} | {block['empty_hypotheses']} "
            f"| {block['leading_deletions']} | {block['trailing_deletions']} "
            f"| {block['mean_inference_elapsed_ms']:.1f} |"
        )
    lines += [
        "",
        "## 編集タイプ内訳",
        "",
        "括弧内は全編集操作に占める割合 (編集 0 件の条件は 0.0% と表示)。",
        "",
        "| 条件 | 編集合計 | 置換 | 削除 | 挿入 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition, block in summary["conditions"].items():
        lines.append(
            f"| {condition} | {block['edits']} "
            f"| {block['substitutions']} ({100.0 * block['substitution_share']:.1f}%) "
            f"| {block['deletions']} ({100.0 * block['deletion_share']:.1f}%) "
            f"| {block['insertions']} ({100.0 * block['insertion_share']:.1f}%) |"
        )
    lines += [
        "",
        "## ペア比較",
        "",
        "ΔCER は treatment − baseline (正 = 悪化)。改善/悪化は発話ごとの編集数比較。",
        "ペア統計は両条件で完了した発話 (n 列) のみで再計算しているため、"
        "条件別サマリの micro CER とは母集団が異なる場合がある。",
        "",
        "| ペア | baseline | treatment | n | 除外 | ΔCER (pp) | 95% CI (pp) | 改善 "
        "| 悪化 | 同点 | Δ先頭削除 | Δ末尾削除 | Δ空仮説 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: "
        "| ---: | ---: |",
    ]
    for label, comparison in summary["comparisons"].items():
        lines.append(
            f"| {label} | {comparison['baseline_condition']} "
            f"| {comparison['treatment_condition']} "
            f"| {comparison['paired_utterances']} | {comparison['dropped_from_pair']} "
            f"| {comparison['delta_micro_cer_pp']:+.3f} "
            f"| [{comparison['bootstrap_lower_pp']:+.3f}, "
            f"{comparison['bootstrap_upper_pp']:+.3f}] "
            f"| {comparison['wins']} | {comparison['losses']} | {comparison['ties']} "
            f"| {comparison['delta_leading_deletions']:+d} "
            f"| {comparison['delta_trailing_deletions']:+d} "
            f"| {comparison['delta_empty_hypotheses']:+d} |"
        )
    lines += [
        "",
        "### 編集タイプ別差分",
        "",
        "件数はペア共通集合での treatment − baseline、率差は正規化参照文字数基準の pp。",
        "",
        "| ペア | Δ置換 | Δ削除 | Δ挿入 | Δ置換率 (pp) | Δ削除率 (pp) | Δ挿入率 (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, comparison in summary["comparisons"].items():
        lines.append(
            f"| {label} | {comparison['delta_substitutions']:+d} "
            f"| {comparison['delta_deletions']:+d} "
            f"| {comparison['delta_insertions']:+d} "
            f"| {comparison['delta_substitution_rate_pp']:+.3f} "
            f"| {comparison['delta_deletion_rate_pp']:+.3f} "
            f"| {comparison['delta_insertion_rate_pp']:+.3f} |"
        )
    lines += ["", "## 代表例", ""]
    for label, examples in summary["examples"].items():
        comparison = summary["comparisons"][label]
        lines += [
            f"### {label} ({comparison['baseline_condition']} → "
            f"{comparison['treatment_condition']})",
            "",
        ]
        for title, key in (("悪化", "regressions"), ("改善", "improvements")):
            entries = examples[key][:REPORT_EXAMPLES]
            lines.append(f"- {title}上位 {len(entries)} 件")
            if not entries:
                lines.append("  - (該当なし)")
            for example in entries:
                lines += [
                    f"  - `{example['utterance_id']}` "
                    f"(編集数 {example['baseline_edits']} → "
                    f"{example['treatment_edits']}, Δ{example['edit_delta']:+d})",
                    f"    - ref: {truncate(example['reference'])}",
                    f"    - {comparison['baseline_condition']}: "
                    f"{truncate(example['baseline_hypothesis'])}",
                    f"    - {comparison['treatment_condition']}: "
                    f"{truncate(example['treatment_hypothesis'])}",
                ]
        lines.append("")
    lines += [
        "## 入力アーティファクト",
        "",
        "| 条件 | パス | sha256 |",
        "| --- | --- | --- |",
    ]
    for condition, digest in parameters["conditions"].items():
        lines.append(
            f"| {condition} | `{parameters['condition_paths'][condition]}` | `{digest}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_condition_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            f"--condition expects NAME=PATH, got {value!r}"
        )
    return name, Path(path)


def parse_pair_argument(value: str) -> tuple[str, str, str]:
    label, separator, endpoints = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError(
            f"--pair expects LABEL=BASELINE:TREATMENT, got {value!r}"
        )
    parts = endpoints.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--pair expects LABEL=BASELINE:TREATMENT, got {value!r}"
        )
    return label, parts[0], parts[1]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        type=parse_condition_argument,
        metavar="NAME=PATH",
        help="condition name and its run_asr_eval JSONL (repeatable)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        type=parse_pair_argument,
        metavar="LABEL=BASELINE:TREATMENT",
        help="paired comparison over two declared conditions (repeatable)",
    )
    parser.add_argument("--split-id", default="")
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def build_summary(arguments: argparse.Namespace) -> dict[str, object]:
    conditions: dict[str, Condition] = {}
    for name, path in arguments.condition:
        if name in conditions:
            raise ValueError(f"duplicate condition name {name}")
        conditions[name] = load_condition(name, path)

    pairs: list[tuple[str, str, str]] = []
    seen_labels: set[str] = set()
    for label, baseline_name, treatment_name in arguments.pair:
        if label in seen_labels:
            raise ValueError(f"duplicate pair label {label}")
        seen_labels.add(label)
        for name in (baseline_name, treatment_name):
            if name not in conditions:
                raise ValueError(f"pair {label} references unknown condition {name}")
        if baseline_name == treatment_name:
            raise ValueError(f"pair {label} compares {baseline_name} with itself")
        pairs.append((label, baseline_name, treatment_name))

    references = collect_references(list(conditions.values()))
    scores: dict[str, dict[str, dict[str, float]]] = {}
    hypotheses: dict[str, dict[str, str]] = {}
    condition_summaries: dict[str, object] = {}
    for name, condition in conditions.items():
        (
            scores[name],
            hypotheses[name],
            condition_summaries[name],
        ) = score_condition(condition, references)

    comparisons: dict[str, object] = {}
    examples: dict[str, object] = {}
    for label, baseline_name, treatment_name in pairs:
        comparisons[label], examples[label] = compare_pair(
            label,
            conditions[baseline_name],
            conditions[treatment_name],
            scores,
            hypotheses,
            references,
            arguments.bootstrap_samples,
            arguments.seed,
            arguments.examples,
        )

    observed: set[str] = set()
    for condition in conditions.values():
        observed |= condition.observed_ids

    return {
        "schema_version": SCHEMA_VERSION,
        "split_id": arguments.split_id,
        "utterances": len(observed),
        "parameters": {
            "bootstrap_samples": arguments.bootstrap_samples,
            "seed": arguments.seed,
            "examples": arguments.examples,
            "normalization": "diagnostic_normalize",
            "conditions": {
                name: sha256_of(condition.path)
                for name, condition in conditions.items()
            },
            "condition_paths": {
                name: str(condition.path) for name, condition in conditions.items()
            },
            "pairs": {
                label: {"baseline": baseline_name, "treatment": treatment_name}
                for label, baseline_name, treatment_name in pairs
            },
        },
        "conditions": condition_summaries,
        "comparisons": comparisons,
        "examples": examples,
    }


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = parse_arguments(argv)
    summary = build_summary(arguments)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report_path = arguments.output_dir / "REPORT.md"
    report_path.write_text(build_report(summary), encoding="utf-8", newline="\n")
    print(f"wrote {summary_path} and {report_path}")
    return summary


if __name__ == "__main__":
    run()
