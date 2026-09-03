#!/usr/bin/env python3
"""Summarize width-8 approximate time-free lattice evaluation."""

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


def select_oracle(
    reference: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("oracle candidate pool is empty")
    return min(
        candidates,
        key=lambda candidate: (
            candidate_alignment(reference, candidate)[0]["edits"],
            int(candidate.get("rank", 0)),
        ),
    )


def select_lattice(
    candidates: list[dict[str, Any]], exponent: float
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("lattice candidate pool is empty")
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["lattice_score"])
            / ((len(candidate["token_ids"]) + 2) ** exponent),
            float(candidate["lattice_score"]),
            -int(candidate["rank"]),
        ),
    )


def union_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in [*row["seeds"], *row["candidates"]]:
        key = tuple(candidate["token_ids"])
        candidates.setdefault(key, candidate)
    return list(candidates.values())


def structural_lattice_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every path in the seed-derived time-free state DAG.

    This intentionally ignores arc scores and frame compatibility.  It is an
    oracle-only upper bound for deciding whether a later language model could
    benefit from the recombined candidate space.
    """
    root = (0, 0, 0)
    token_text: dict[int, str] = {}
    seed_sequences: set[tuple[int, ...]] = set()
    arcs: dict[tuple[int, int, int], set[tuple[int, tuple[int, int, int]]]] = {}
    terminals: set[tuple[int, int, int]] = set()
    states = {root}
    for seed in row["seeds"]:
        token_ids = tuple(int(token_id) for token_id in seed["token_ids"])
        hypothesis = seed["hypothesis"]
        if len(token_ids) != len(hypothesis):
            raise ValueError(
                f"{row['utterance_id']}: token-to-character mapping is not one-to-one"
            )
        for token_id, character in zip(token_ids, hypothesis):
            existing = token_text.setdefault(token_id, character)
            if existing != character:
                raise ValueError(
                    f"{row['utterance_id']}: token {token_id} maps to multiple characters"
                )
        seed_sequences.add(token_ids)
        state = root
        for token_count, token_id in enumerate(token_ids, 1):
            destination = (token_count, state[2], token_id)
            arcs.setdefault(state, set()).add((token_id, destination))
            states.add(destination)
            state = destination
        terminals.add(state)

    paths: dict[tuple[int, int, int], set[tuple[int, ...]]] = {root: {()}}
    for state in sorted(states):
        for prefix in paths.get(state, ()):
            for token_id, destination in sorted(arcs.get(state, ())):
                paths.setdefault(destination, set()).add(prefix + (token_id,))
    sequences: set[tuple[int, ...]] = set()
    for terminal in terminals:
        sequences.update(paths.get(terminal, ()))
    return [
        {
            "rank": rank,
            "hypothesis": "".join(token_text[token_id] for token_id in sequence),
            "token_ids": list(sequence),
            "is_seed": sequence in seed_sequences,
        }
        for rank, sequence in enumerate(sorted(sequences), 1)
    ]


def summarize_selection(
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0}
    reference_characters = 0
    exact = 0
    macro_rates: list[float] = []
    selected_seed = 0
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
        selected_seed += bool(candidate.get("is_seed", True))
        rank = str(candidate.get("rank", 0))
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
        "selected_seed": selected_seed,
        "selected_novel": len(rows) - selected_seed,
        "selected_rank_histogram": selected_ranks,
        "utterance_edits": utterance_edits,
    }


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
            if (
                row["beam_size"] != 8
                or row["search_normalization"] != "raw"
                or row["arc_merge"] != "maximum"
            ):
                raise ValueError(f"{path}:{line_number}: unexpected lattice contract")
            if len(row["seeds"]) != 8 or not row["candidates"]:
                raise ValueError(f"{path}:{line_number}: invalid candidate pool")
            if [candidate["rank"] for candidate in row["seeds"]] != list(range(1, 9)):
                raise ValueError(f"{path}:{line_number}: invalid seed ranks")
            if [candidate["rank"] for candidate in row["candidates"]] != list(
                range(1, len(row["candidates"]) + 1)
            ):
                raise ValueError(f"{path}:{line_number}: invalid lattice ranks")
            seed_ids = {tuple(seed["token_ids"]) for seed in row["seeds"]}
            for candidate in row["candidates"]:
                if candidate["is_seed"] != (
                    tuple(candidate["token_ids"]) in seed_ids
                ):
                    raise ValueError(f"{path}:{line_number}: incorrect is_seed marker")
            rows.append(row)
    return sorted(rows, key=lambda row: row["utterance_id"])


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows = load_rows(paths)
    structural_candidates = {
        row["utterance_id"]: structural_lattice_candidates(row) for row in rows
    }
    seed_top1 = summarize_selection(rows, lambda row: row["seeds"][0])
    seed_oracle = summarize_selection(
        rows, lambda row: select_oracle(row["reference"], row["seeds"])
    )
    lattice_oracle = summarize_selection(
        rows, lambda row: select_oracle(row["reference"], row["candidates"])
    )
    union_oracle = summarize_selection(
        rows,
        lambda row: select_oracle(row["reference"], union_candidates(row)),
    )
    structural_oracle = summarize_selection(
        rows,
        lambda row: select_oracle(
            row["reference"], structural_candidates[row["utterance_id"]]
        ),
    )
    rankings: dict[str, Any] = {}
    for exponent in EXPONENTS:
        selected = summarize_selection(
            rows,
            lambda row, exponent=exponent: select_lattice(
                row["candidates"], exponent
            ),
        )
        rankings[f"alpha_{exponent:.2f}"] = {
            **strip_internal(selected),
            "paired_vs_seed_top1": paired(selected, seed_top1),
            "delta_edits_vs_seed_top1": selected["edits"] - seed_top1["edits"],
            "delta_cer_percentage_points_vs_seed_top1": (
                selected["micro_cer"] - seed_top1["micro_cer"]
            )
            * 100,
        }

    novel_per_row = [
        sum(not candidate["is_seed"] for candidate in row["candidates"])
        for row in rows
    ]
    terminal_counts = [len(row["candidates"]) for row in rows]
    union_oracle_better = paired(union_oracle, seed_oracle)
    structural_oracle_better = paired(structural_oracle, seed_oracle)
    structural_counts = [
        len(structural_candidates[row["utterance_id"]]) for row in rows
    ]
    structural_novel_counts = [count - 8 for count in structural_counts]
    elapsed = [float(row["search_and_lattice_elapsed_ms"]) for row in rows]
    return {
        "schema_version": 1,
        "condition": {
            "model": "reazonspeech_k2_v2",
            "precision": "float32",
            "provider": "cpu",
            "beam_size": 8,
            "search_normalization": "raw",
            "input_edge_silence_ms": 320,
            "lattice_state": "(emitted_token_count,last_two_tokens)",
            "arc_score": "representative alignment blanks followed by token",
            "arc_merge": "maximum",
            "accuracy_only": True,
            "timing_comparable": False,
        },
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "seed_top1": strip_internal(seed_top1),
        "lattice_ranking": rankings,
        "seed_oracle": strip_internal(seed_oracle),
        "lattice_oracle": strip_internal(lattice_oracle),
        "union_oracle": {
            **strip_internal(union_oracle),
            "paired_vs_seed_oracle": union_oracle_better,
        },
        "structural_lattice_oracle": {
            **strip_internal(structural_oracle),
            "paired_vs_seed_oracle": structural_oracle_better,
        },
        "pool": {
            "mean_terminal_candidates": sum(terminal_counts) / len(rows),
            "min_terminal_candidates": min(terminal_counts),
            "max_terminal_candidates": max(terminal_counts),
            "mean_novel_candidates": sum(novel_per_row) / len(rows),
            "utterances_with_novel_candidates": sum(count > 0 for count in novel_per_row),
            "total_novel_candidates": sum(novel_per_row),
        },
        "structural_pool": {
            "mean_candidates": sum(structural_counts) / len(rows),
            "min_candidates": min(structural_counts),
            "max_candidates": max(structural_counts),
            "mean_novel_candidates": sum(structural_novel_counts) / len(rows),
            "utterances_with_novel_candidates": sum(
                count > 0 for count in structural_novel_counts
            ),
            "total_novel_candidates": sum(structural_novel_counts),
        },
        "mean_search_and_lattice_elapsed_ms": sum(elapsed) / len(elapsed),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ReazonSpeech approximate time-free lattice",
        "",
        "JSUT BASIC5000, FP32 CPU, width-8 full-prefix raw seeds, with 320 ms silence on each edge.",
        "",
        "| selector | α=0 | α=.25 | α=.50 | α=.75 | α=1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rankings = summary["lattice_ranking"]
    cells = [
        f"{rankings[f'alpha_{exponent:.2f}']['micro_cer']:.4%}"
        for exponent in EXPONENTS
    ]
    lines.append("| lattice maximum | " + " | ".join(cells) + " |")
    seed = summary["seed_top1"]
    seed_oracle = summary["seed_oracle"]
    lattice_oracle = summary["lattice_oracle"]
    union_oracle = summary["union_oracle"]
    structural_oracle = summary["structural_lattice_oracle"]
    pool = summary["pool"]
    structural_pool = summary["structural_pool"]
    best_key, best = min(
        rankings.items(), key=lambda item: item[1]["micro_cer"]
    )
    lines.extend(
        [
            "",
            f"Seed top-1: {seed['micro_cer']:.4%} ({seed['edits']} edits), exact {seed['exact_rate']:.2%}.",
            f"Seed oracle: {seed_oracle['micro_cer']:.4%}; lattice-only oracle: {lattice_oracle['micro_cer']:.4%}; union oracle: {union_oracle['micro_cer']:.4%}.",
            f"All structurally possible time-free paths: {structural_oracle['micro_cer']:.4%} oracle, {structural_oracle['exact_rate']:.2%} exact.",
            "",
            "## Result",
            "",
            f"- Best lattice ranking was {best_key}: {best['micro_cer']:.4%}, {best['edits']} edits, {best['exact_rate']:.2%} exact ({best['paired_vs_seed_top1']['wins']} wins / {best['paired_vs_seed_top1']['losses']} losses versus seed top-1).",
            f"- Recombination produced {pool['total_novel_candidates']} novel terminal paths in {pool['utterances_with_novel_candidates']}/5000 utterances; mean terminal pool size was {pool['mean_terminal_candidates']:.2f}.",
            f"- Adding lattice paths to the original seeds changed seed-oracle errors by {union_oracle['edits'] - seed_oracle['edits']} ({union_oracle['paired_vs_seed_oracle']['wins']} better / {union_oracle['paired_vs_seed_oracle']['losses']} worse utterances).",
            f"- Ignoring scores and enumerating every structural path produced {structural_pool['total_novel_candidates']} novel paths in {structural_pool['utterances_with_novel_candidates']}/5000 utterances (mean {structural_pool['mean_candidates']:.2f}, maximum {structural_pool['max_candidates']} total paths).",
            f"- The structural oracle changed seed-oracle errors by {structural_oracle['edits'] - seed_oracle['edits']} ({structural_oracle['paired_vs_seed_oracle']['wins']} better / {structural_oracle['paired_vs_seed_oracle']['losses']} worse utterances). This is the useful upper bound for language-model selection.",
            "- Arc scores deliberately discard frame compatibility, so this is an optimistic recombination ablation rather than a probability-preserving lattice.",
            "- Hyperparameters were selected on this same JSUT set; timing is accuracy-only.",
            "",
            "## Decision",
            "",
            "- Do not use the approximate lattice score as the 1-best selector.",
            f"- Validate a second-pass language model on the original width-8 seeds first: their oracle leaves {seed['edits'] - seed_oracle['edits']} recoverable edits, while structural recombination adds only {seed_oracle['edits'] - structural_oracle['edits']} further oracle edits.",
            "- Then compare the same scorer on seeds plus every structural lattice path. The observed maximum of 64 paths per utterance is small enough for CPU second-pass scoring.",
            "- Start with a character 5-gram score and length reward; ablate low-order density-ratio subtraction with a character trigram because the stateless predictor already uses the previous two tokens.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--ort-runtime", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.inputs)
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
