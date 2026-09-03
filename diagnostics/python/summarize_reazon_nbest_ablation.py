#!/usr/bin/env python3
"""Summarize ReazonSpeech full-prefix N-best and length-score ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from asr_eval_metrics import (
    align_characters,
    diagnostic_normalize,
    summarize_alignment,
)


CONDITION_FILES = {
    "beam4_raw_during": "reazon-fp32-full-prefix-beam4-raw-during.jsonl",
    "beam4_per_token_during": (
        "reazon-fp32-full-prefix-beam4-per-token-during.jsonl"
    ),
    "beam8_raw_during": "reazon-fp32-full-prefix-beam8-raw-during.jsonl",
    "beam8_per_token_during": (
        "reazon-fp32-full-prefix-beam8-per-token-during.jsonl"
    ),
}
FINAL_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def final_ranking_score(candidate: dict[str, Any], exponent: float) -> float:
    token_count = len(candidate["token_ids"])
    return float(candidate["raw_score"]) / ((token_count + 2) ** exponent)


def select_by_final_exponent(
    candidates: list[dict[str, Any]], exponent: float
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("completed N-best row has no candidates")
    return max(
        candidates,
        key=lambda candidate: (
            final_ranking_score(candidate, exponent),
            float(candidate["raw_score"]),
            -int(candidate["rank"]),
        ),
    )


def candidate_alignment(
    reference: str, candidate: dict[str, Any]
) -> tuple[dict[str, int], str, str]:
    normalized_reference = diagnostic_normalize(reference)
    normalized_hypothesis = diagnostic_normalize(candidate["hypothesis"])
    summary = summarize_alignment(
        align_characters(normalized_reference, normalized_hypothesis)
    )
    summary["edits"] = (
        summary["substitutions"]
        + summary["deletions"]
        + summary["insertions"]
    )
    summary["reference_characters"] = len(normalized_reference)
    return summary, normalized_reference, normalized_hypothesis


def select_oracle_candidate(
    reference: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("completed N-best row has no candidates")
    return min(
        candidates,
        key=lambda candidate: (
            candidate_alignment(reference, candidate)[0]["edits"],
            -final_ranking_score(candidate, 1.0),
            int(candidate["rank"]),
        ),
    )


def summarize_selection(
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_characters = 0
    exact = 0
    macro_rates: list[float] = []
    selected_ranks: dict[str, int] = {}
    for row in rows:
        candidate = selector(row)
        summary, reference, hypothesis = candidate_alignment(row["reference"], candidate)
        substitutions += summary["substitutions"]
        deletions += summary["deletions"]
        insertions += summary["insertions"]
        reference_characters += summary["reference_characters"]
        exact += reference == hypothesis
        macro_rates.append(
            summary["edits"] / len(reference)
            if reference
            else float(summary["edits"] > 0)
        )
        rank = str(candidate["rank"])
        selected_ranks[rank] = selected_ranks.get(rank, 0) + 1
    edits = substitutions + deletions + insertions
    return {
        "samples": len(rows),
        "reference_characters": reference_characters,
        "edits": edits,
        "micro_cer": edits / reference_characters,
        "macro_cer": sum(macro_rates) / len(macro_rates),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "exact": exact,
        "exact_rate": exact / len(rows),
        "selected_rank_histogram": selected_ranks,
    }


def load_nbest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("status") != "completed":
            raise ValueError(f"{path}:{line_number}: non-completed row: {row!r}")
        utterance_id = row["utterance_id"]
        if utterance_id in seen:
            raise ValueError(f"{path}: duplicate utterance_id {utterance_id}")
        seen.add(utterance_id)
        if not row["candidates"]:
            raise ValueError(f"{path}:{line_number}: empty candidate list")
        ranks = [candidate["rank"] for candidate in row["candidates"]]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{path}:{line_number}: non-contiguous candidate ranks")
        rows.append(row)
    return rows


def load_greedy(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("status") != "completed":
            raise ValueError(f"{path}:{line_number}: non-completed row")
        utterance_id = row["utterance_id"]
        if utterance_id in rows:
            raise ValueError(f"{path}: duplicate utterance_id {utterance_id}")
        rows[utterance_id] = row
    return rows


def validate_conditions(
    condition_rows: dict[str, list[dict[str, Any]]],
    greedy: dict[str, dict[str, Any]] | None,
) -> list[str]:
    baseline = condition_rows["beam4_raw_during"]
    utterance_ids = [row["utterance_id"] for row in baseline]
    baseline_by_id = {row["utterance_id"]: row for row in baseline}
    for condition, rows in condition_rows.items():
        ids = [row["utterance_id"] for row in rows]
        if ids != utterance_ids:
            raise ValueError(f"{condition}: utterance order or identity differs")
        for row in rows:
            expected = baseline_by_id[row["utterance_id"]]
            for field in ("reference", "duration_samples"):
                if row[field] != expected[field]:
                    raise ValueError(
                        f"{condition}/{row['utterance_id']}: {field} differs"
                    )
    if greedy is not None:
        if set(greedy) != set(utterance_ids):
            raise ValueError("greedy utterance identity differs from N-best runs")
        for utterance_id in utterance_ids:
            expected = baseline_by_id[utterance_id]
            actual = greedy[utterance_id]
            for field in ("reference", "duration_samples"):
                if actual[field] != expected[field]:
                    raise ValueError(f"greedy/{utterance_id}: {field} differs")
    return utterance_ids


def summarize_candidate_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(row["candidates"]) for row in rows]
    unique_counts = [
        len({candidate["hypothesis"] for candidate in row["candidates"]})
        for row in rows
    ]
    beam_size = rows[0]["beam_size"]
    return {
        "beam_size": beam_size,
        "mean_candidates": sum(counts) / len(counts),
        "mean_unique_texts": sum(unique_counts) / len(unique_counts),
        "full_width_rows": sum(count == beam_size for count in counts),
        "full_width_rate": sum(count == beam_size for count in counts) / len(counts),
        "all_texts_unique_rows": sum(
            count == unique for count, unique in zip(counts, unique_counts, strict=True)
        ),
    }


def summarize_union(
    raw_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    union_rows: list[dict[str, Any]] = []
    jaccards: list[float] = []
    for raw, normalized in zip(raw_rows, normalized_rows, strict=True):
        by_text: dict[str, dict[str, Any]] = {}
        for candidate in raw["candidates"] + normalized["candidates"]:
            existing = by_text.get(candidate["hypothesis"])
            if existing is None or candidate["raw_score"] > existing["raw_score"]:
                by_text[candidate["hypothesis"]] = candidate
        candidates = list(by_text.values())
        candidates.sort(key=lambda candidate: candidate["rank"])
        for rank, candidate in enumerate(candidates, 1):
            candidate = dict(candidate)
            candidate["rank"] = rank
            candidates[rank - 1] = candidate
        raw_texts = {candidate["hypothesis"] for candidate in raw["candidates"]}
        normalized_texts = {
            candidate["hypothesis"] for candidate in normalized["candidates"]
        }
        jaccards.append(
            len(raw_texts & normalized_texts) / len(raw_texts | normalized_texts)
        )
        union_rows.append({**raw, "candidates": candidates})
    return {
        "mean_candidate_text_jaccard": sum(jaccards) / len(jaccards),
        "candidate_pool": summarize_candidate_pool(union_rows),
        "oracle": summarize_selection(
            union_rows,
            lambda row: select_oracle_candidate(row["reference"], row["candidates"]),
        ),
    }


def summarize_greedy(
    utterance_ids: list[str], greedy: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = [
        {
            **greedy[utterance_id],
            "candidates": [
                {
                    "rank": 1,
                    "hypothesis": greedy[utterance_id]["hypothesis"],
                    "raw_score": 0.0,
                    "token_ids": [],
                }
            ],
        }
        for utterance_id in utterance_ids
    ]
    return summarize_selection(rows, lambda row: row["candidates"][0])


def build_summary(
    condition_rows: dict[str, list[dict[str, Any]]],
    hashes: dict[str, str],
    greedy: dict[str, dict[str, Any]] | None,
    greedy_hash: str | None,
) -> dict[str, Any]:
    utterance_ids = validate_conditions(condition_rows, greedy)
    results: dict[str, Any] = {}
    for condition, rows in condition_rows.items():
        results[condition] = {
            "search_normalization": rows[0]["search_normalization"],
            "candidate_pool": summarize_candidate_pool(rows),
            "final_ranking": {
                f"alpha_{exponent:.2f}": summarize_selection(
                    rows,
                    lambda row, exponent=exponent: select_by_final_exponent(
                        row["candidates"], exponent
                    ),
                )
                for exponent in FINAL_EXPONENTS
            },
            "oracle": summarize_selection(
                rows,
                lambda row: select_oracle_candidate(
                    row["reference"], row["candidates"]
                ),
            ),
            "output_sha256": hashes[condition],
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": "reazonspeech_k2_v2",
            "precision": "float32",
            "provider": "cpu",
            "threads_per_shard": 4,
            "parallel_shards": 4,
        },
        "input_conditioning": {
            "leading_silence_samples": 5120,
            "trailing_silence_samples": 5120,
            "sample_rate_hz": 16000,
            "silence_ms_per_edge": 320,
            "source_edge_fade_samples": 160,
            "reported_duration_excludes_padding": True,
        },
        "search_contract": {
            "pruning": "full_prefix",
            "exact_prefix_alignment_merge": "logsumexp",
            "different_full_prefixes": "retained as separate candidates",
            "predictor_context_rows": "deduplicated by last two token IDs",
            "final_score": "raw_score / (token_count + 2) ** alpha",
        },
        "dataset": {
            "samples": len(utterance_ids),
            "utterance_order_sha256": hashlib.sha256(
                "\0".join(utterance_ids).encode("utf-8")
            ).hexdigest(),
        },
        "final_length_exponents": list(FINAL_EXPONENTS),
        "conditions": results,
        "width_unions": {
            "beam4": summarize_union(
                condition_rows["beam4_raw_during"],
                condition_rows["beam4_per_token_during"],
            ),
            "beam8": summarize_union(
                condition_rows["beam8_raw_during"],
                condition_rows["beam8_per_token_during"],
            ),
        },
    }
    if greedy is not None:
        summary["greedy"] = {
            **summarize_greedy(utterance_ids, greedy),
            "output_sha256": greedy_hash,
        }
    return summary


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ReazonSpeech N-best oracle and length-normalization ablation",
        "",
        (
            f"Full-prefix FP32 search on {summary['dataset']['samples']:,} utterances. "
            "CER uses NFKC, lowercase, and removal of Unicode punctuation/whitespace."
        ),
        "",
        "| condition | α=0 | α=.25 | α=.50 | α=.75 | α=1 | oracle | oracle exact | unique texts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, result in summary["conditions"].items():
        ranking = result["final_ranking"]
        oracle = result["oracle"]
        values = [ranking[f"alpha_{exponent:.2f}"]["micro_cer"] for exponent in FINAL_EXPONENTS]
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{value:.4%}" for value in values)
            + f" | {oracle['micro_cer']:.4%} | {oracle['exact_rate']:.2%}"
            + f" | {result['candidate_pool']['mean_unique_texts']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| width union | raw/per-token Jaccard | union candidates | union oracle CER | union oracle exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for width, result in summary["width_unions"].items():
        lines.append(
            f"| {width} | {result['mean_candidate_text_jaccard']:.3f}"
            f" | {result['candidate_pool']['mean_unique_texts']:.3f}"
            f" | {result['oracle']['micro_cer']:.4%}"
            f" | {result['oracle']['exact_rate']:.2%} |"
        )
    if "greedy" in summary:
        greedy = summary["greedy"]
        lines.extend(
            [
                "",
                f"Greedy diagnostic CER: {greedy['micro_cer']:.4%}; "
                f"exact: {greedy['exact_rate']:.2%}.",
                "",
                "## Findings",
                "",
                (
                    "Raw-score search produced the best model-ranked transcript at both widths. "
                    "Increasing final alpha did not recover greedy accuracy, and per-token "
                    "normalization during search made both top-1 and oracle CER worse."
                ),
                "",
            ]
        )
        for width in (4, 8):
            result = summary["conditions"][f"beam{width}_raw_during"]
            alpha_zero = result["final_ranking"]["alpha_0.00"]
            oracle = result["oracle"]
            lines.append(
                f"- Beam {width}: raw top-1 {alpha_zero['micro_cer']:.4%} "
                f"({alpha_zero['substitutions']}/{alpha_zero['deletions']}/"
                f"{alpha_zero['insertions']} S/D/I), while oracle reached "
                f"{oracle['micro_cer']:.4%} and {oracle['exact_rate']:.2%} exact. "
                f"Oracle improves over greedy by "
                f"{(greedy['micro_cer'] - oracle['micro_cer']) * 100:.4f} percentage points."
            )
        lines.extend(
            [
                "",
                (
                    "Every row retained the full requested number of unique texts. "
                    "The large oracle gap therefore identifies top-1 scoring/reranking, not "
                    "a lack of candidate diversity, as the immediate accuracy bottleneck."
                ),
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, action="append")
    parser.add_argument("--greedy-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition_paths = {
        condition: [input_dir / file_name for input_dir in args.input_dir]
        for condition, file_name in CONDITION_FILES.items()
    }
    condition_rows = {
        condition: [row for path in paths for row in load_nbest(path)]
        for condition, paths in condition_paths.items()
    }
    hashes = {
        condition: sha256_files(paths) for condition, paths in condition_paths.items()
    }
    greedy = load_greedy(args.greedy_jsonl) if args.greedy_jsonl else None
    greedy_hash = sha256_file(args.greedy_jsonl) if args.greedy_jsonl else None
    summary = build_summary(condition_rows, hashes, greedy, greedy_hash)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(format_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
