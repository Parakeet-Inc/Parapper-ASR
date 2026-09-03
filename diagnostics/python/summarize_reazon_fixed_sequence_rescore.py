#!/usr/bin/env python3
"""Summarize width-8 fixed-sequence Viterbi/forward rescoring."""

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


EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_by_score(
    candidates: list[dict[str, Any]], score_name: str, exponent: float
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("completed rescore row has no candidates")

    def ranking(candidate: dict[str, Any]) -> tuple[float, float, int]:
        score = float(candidate[score_name])
        normalized = score / ((len(candidate["token_ids"]) + 2) ** exponent)
        return normalized, score, -int(candidate["rank"])

    return max(candidates, key=ranking)


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


def summarize_selection(
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0}
    reference_characters = 0
    exact = 0
    macro_rates: list[float] = []
    selected_ranks: dict[str, int] = {}
    utterance_edits: dict[str, int] = {}
    for row in rows:
        candidate = selector(row)
        summary, reference, hypothesis = candidate_alignment(row["reference"], candidate)
        for operation in totals:
            totals[operation] += summary[operation]
        reference_characters += summary["reference_characters"]
        exact += reference == hypothesis
        macro_rates.append(
            summary["edits"] / len(reference)
            if reference
            else float(summary["edits"] > 0)
        )
        rank = str(candidate["rank"])
        selected_ranks[rank] = selected_ranks.get(rank, 0) + 1
        utterance_edits[row["utterance_id"]] = summary["edits"]
    edits = sum(totals.values())
    return {
        "samples": len(rows),
        "reference_characters": reference_characters,
        "edits": edits,
        "micro_cer": edits / reference_characters,
        "macro_cer": sum(macro_rates) / len(macro_rates),
        **totals,
        "exact": exact,
        "exact_rate": exact / len(rows),
        "selected_rank_histogram": selected_ranks,
        "utterance_edits": utterance_edits,
    }


def select_oracle(row: dict[str, Any]) -> dict[str, Any]:
    return min(
        row["candidates"],
        key=lambda candidate: (
            candidate_alignment(row["reference"], candidate)[0]["edits"],
            -float(candidate["forward_score"]),
            int(candidate["rank"]),
        ),
    )


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("status") != "completed":
                raise ValueError(f"{path}:{line_number}: non-completed row")
            utterance_id = row["utterance_id"]
            if utterance_id in seen:
                raise ValueError(f"duplicate utterance_id {utterance_id}")
            seen.add(utterance_id)
            if row["beam_size"] != 8 or row["search_normalization"] != "raw":
                raise ValueError(f"{path}:{line_number}: unexpected search contract")
            candidates = row["candidates"]
            if len(candidates) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 candidates")
            if [candidate["rank"] for candidate in candidates] != list(range(1, 9)):
                raise ValueError(f"{path}:{line_number}: non-contiguous ranks")
            for candidate in candidates:
                if candidate["forward_score"] + 1.0e-5 < candidate["viterbi_score"]:
                    raise ValueError(
                        f"{path}:{line_number}: forward score below Viterbi score"
                    )
            rows.append(row)
    return sorted(rows, key=lambda row: row["utterance_id"])


def strip_internal(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "utterance_edits"}


def paired(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    wins = losses = ties = 0
    for utterance_id, baseline_edits in baseline["utterance_edits"].items():
        candidate_edits = candidate["utterance_edits"][utterance_id]
        if candidate_edits < baseline_edits:
            wins += 1
        elif candidate_edits > baseline_edits:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def load_greedy(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        if record.get("status") != "completed":
            raise ValueError(f"{path}:{line_number}: non-completed greedy row")
        utterance_id = record["utterance_id"]
        if utterance_id in seen:
            raise ValueError(f"{path}: duplicate greedy utterance_id {utterance_id}")
        seen.add(utterance_id)
        rows.append(
            {
                "utterance_id": utterance_id,
                "reference": record["reference"],
                "candidates": [
                    {
                        "rank": 1,
                        "hypothesis": record["hypothesis"],
                        "token_ids": [],
                    }
                ],
            }
        )
    return sorted(rows, key=lambda row: row["utterance_id"])


def summarize(paths: list[Path], greedy_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(paths)
    search = summarize_selection(
        rows,
        lambda row: select_by_score(row["candidates"], "search_raw_score", 0.0),
    )
    oracle = summarize_selection(rows, select_oracle)
    score_results: dict[str, Any] = {}
    for score_name in ("viterbi_score", "forward_score"):
        by_exponent: dict[str, Any] = {}
        for exponent in EXPONENTS:
            selection = summarize_selection(
                rows,
                lambda row, score_name=score_name, exponent=exponent: select_by_score(
                    row["candidates"], score_name, exponent
                ),
            )
            by_exponent[f"alpha_{exponent:.2f}"] = {
                **strip_internal(selection),
                "paired_vs_search_raw": paired(selection, search),
                "delta_edits_vs_search_raw": selection["edits"] - search["edits"],
                "delta_cer_percentage_points_vs_search_raw": (
                    selection["micro_cer"] - search["micro_cer"]
                )
                * 100,
            }
        score_results[score_name] = by_exponent

    candidate_count = sum(len(row["candidates"]) for row in rows)
    forward_minus_search = [
        float(candidate["forward_score"]) - float(candidate["search_raw_score"])
        for row in rows
        for candidate in row["candidates"]
    ]
    forward_rank_changes = sum(
        select_by_score(row["candidates"], "forward_score", 0.0)["rank"] != 1
        for row in rows
    )
    viterbi_rank_changes = sum(
        select_by_score(row["candidates"], "viterbi_score", 0.0)["rank"] != 1
        for row in rows
    )
    elapsed = [float(row["search_and_rescore_elapsed_ms"]) for row in rows]
    oracle_denominator = search["edits"] - oracle["edits"]
    for score_name in score_results:
        for result in score_results[score_name].values():
            result["oracle_edit_recovery_rate"] = (
                (search["edits"] - result["edits"]) / oracle_denominator
                if oracle_denominator
                else 0.0
            )

    result = {
        "schema_version": 1,
        "condition": {
            "model": "reazonspeech_k2_v2",
            "precision": "float32",
            "provider": "cpu",
            "beam_size": 8,
            "search_normalization": "raw",
            "fixed_sequence_graph": "one_symbol_per_frame",
            "input_edge_silence_ms": 320,
            "accuracy_only": True,
            "timing_comparable": False,
        },
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "search_raw": strip_internal(search),
        "fixed_sequence": score_results,
        "oracle": strip_internal(oracle),
        "score_diagnostics": {
            "candidates": candidate_count,
            "forward_minus_search_raw_mean": sum(forward_minus_search)
            / candidate_count,
            "forward_minus_search_raw_min": min(forward_minus_search),
            "forward_minus_search_raw_max": max(forward_minus_search),
            "forward_raw_changed_top1_utterances": forward_rank_changes,
            "viterbi_raw_changed_top1_utterances": viterbi_rank_changes,
            "mean_search_and_rescore_elapsed_ms": sum(elapsed) / len(elapsed),
        },
    }
    if greedy_path is not None:
        greedy_rows = load_greedy(greedy_path)
        if [
            (row["utterance_id"], row["reference"]) for row in greedy_rows
        ] != [(row["utterance_id"], row["reference"]) for row in rows]:
            raise ValueError("greedy and rescore utterance/reference contracts differ")
        greedy = summarize_selection(
            greedy_rows, lambda row: row["candidates"][0]
        )
        result["greedy"] = strip_internal(greedy)
        result["greedy_artifact"] = {
            "path": str(greedy_path),
            "sha256": sha256_file(greedy_path),
        }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ReazonSpeech width-8 fixed-sequence rescoring",
        "",
        "JSUT BASIC5000, FP32 CPU, width-8 full-prefix raw search, with 320 ms silence on each edge.",
        "",
        "| selector | α=0 | α=.25 | α=.50 | α=.75 | α=1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score_name, label in (("viterbi_score", "Viterbi"), ("forward_score", "Forward")):
        values = summary["fixed_sequence"][score_name]
        cells = [f"{values[f'alpha_{alpha:.2f}']['micro_cer']:.4%}" for alpha in EXPONENTS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    search = summary["search_raw"]
    oracle = summary["oracle"]
    if "greedy" in summary:
        greedy = summary["greedy"]
        lines.extend(
            [
                "",
                f"Greedy: {greedy['micro_cer']:.4%} ({greedy['edits']} edits), exact {greedy['exact_rate']:.2%}.",
            ]
        )
    lines.extend(
        [
            "",
            f"Search raw top-1: {search['micro_cer']:.4%} ({search['edits']} edits), exact {search['exact_rate']:.2%}.",
            f"Oracle: {oracle['micro_cer']:.4%} ({oracle['edits']} edits), exact {oracle['exact_rate']:.2%}.",
            "",
            "Each fixed transcript is rescored over all blank/token alignments in the same one-symbol-per-frame graph. Forward log-adds alignments; Viterbi keeps only the maximum path.",
        ]
    )
    best = summary["fixed_sequence"]["viterbi_score"]["alpha_0.25"]
    forward = summary["fixed_sequence"]["forward_score"]["alpha_0.00"]
    diagnostics = summary["score_diagnostics"]
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- Exact forward rescoring changed top-1 for only {diagnostics['forward_raw_changed_top1_utterances']}/5000 utterances and increased errors by {forward['delta_edits_vs_search_raw']} ({forward['paired_vs_search_raw']['wins']} wins / {forward['paired_vs_search_raw']['losses']} losses).",
            f"- Viterbi with α=.25 was the best fixed-sequence condition: {best['micro_cer']:.4%}, {best['edits']} edits, {best['exact_rate']:.2%} exact. It removed {-best['delta_edits_vs_search_raw']} edits ({best['paired_vs_search_raw']['wins']} wins / {best['paired_vs_search_raw']['losses']} losses) but recovered only {best['oracle_edit_recovery_rate']:.2%} of the oracle edit gap.",
            f"- Greedy still has {best['edits'] - summary['greedy']['edits']} fewer edits than the best fixed-sequence result. Exact acoustic alignment rescoring is therefore not enough to realize the width-8 oracle gain.",
            "- α=.25 was selected on this same JSUT set; treat it as an ablation result, not a held-out generalization claim.",
            "- Timing is accuracy-only: predictor entries are cached across utterances and four shards ran concurrently.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--greedy", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--ort-runtime", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.inputs, args.greedy)
    summary["audit_artifacts"] = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in (
            ("manifest", args.manifest),
            ("runner", args.runner),
            ("ort_runtime", args.ort_runtime),
        )
        if path is not None
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
