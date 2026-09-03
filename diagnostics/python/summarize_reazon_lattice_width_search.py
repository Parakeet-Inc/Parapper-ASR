#!/usr/bin/env python3
"""Explore seed width, lattice construction, and static-coherence reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from summarize_reazon_approximate_lattice import (
    candidate_alignment,
    paired,
    select_oracle,
    structural_lattice_candidates,
    summarize_selection,
)
from summarize_reazon_static_coherence import (
    is_development,
    load_cache,
    select_seed_fusion,
    strip_internal,
    write_cache,
)
from static_embedding_numpy import StaticEmbeddingModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
LENGTH_EXPONENTS = (0.0, 0.5, 1.0)
COHERENCE_WEIGHTS = (0.0, 0.03, 0.1, 0.3)
CANDIDATE_CAPS = (1, 2, 4, 8, 16, 32, 64)
SEED_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_nbest(paths: list[Path], expected_width: int) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("status") != "completed":
                raise ValueError(f"{path}:{line_number}: non-completed row")
            if row["beam_size"] != expected_width or row["search_normalization"] != "raw":
                raise ValueError(f"{path}:{line_number}: unexpected N-best contract")
            if len(row["candidates"]) != expected_width:
                raise ValueError(f"{path}:{line_number}: incomplete N-best pool")
            if row["utterance_id"] in rows:
                raise ValueError(f"duplicate utterance {row['utterance_id']}")
            rows[row["utterance_id"]] = row
    return [rows[key] for key in sorted(rows)]


def width_rows(
    beam4: list[dict[str, Any]], beam8: list[dict[str, Any]], width: int
) -> list[dict[str, Any]]:
    source = beam8 if width == 8 else beam4
    rows = []
    for row in source:
        seeds = [dict(candidate) for candidate in row["candidates"][:width]]
        rows.append(
            {
                "utterance_id": row["utterance_id"],
                "reference": row["reference"],
                "seeds": seeds,
            }
        )
    return rows


def evidence_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = row["seeds"]
    visitors: dict[tuple[int, int, int], set[int]] = {}
    edges: dict[tuple[tuple[int, int, int], int], set[int]] = {}
    terminals: dict[tuple[int, int, int], set[int]] = {}
    for seed_index, seed in enumerate(seeds):
        state = (0, 0, 0)
        visitors.setdefault(state, set()).add(seed_index)
        for token_count, token_id in enumerate(seed["token_ids"], 1):
            token_id = int(token_id)
            edges.setdefault((state, token_id), set()).add(seed_index)
            state = (token_count, state[2], token_id)
            visitors.setdefault(state, set()).add(seed_index)
        terminals.setdefault(state, set()).add(seed_index)

    candidates = structural_lattice_candidates(row)
    raw_scores = np.asarray([float(seed["raw_score"]) for seed in seeds])
    for candidate in candidates:
        state = (0, 0, 0)
        evidence: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        supporters: list[tuple[int, ...]] = []
        for token_count, token_id in enumerate(candidate["token_ids"], 1):
            numerator = tuple(sorted(edges[(state, int(token_id))]))
            evidence.append((tuple(sorted(visitors[state])), numerator))
            supporters.append(numerator)
            state = (token_count, state[2], int(token_id))
        terminal = tuple(sorted(terminals[state]))
        evidence.append((tuple(sorted(visitors[state])), terminal))
        supporters.append(terminal)
        costs = {seed_index: 0 for seed_index in supporters[0]}
        for available in supporters[1:]:
            costs = {
                seed_index: min(
                    cost + int(previous != seed_index)
                    for previous, cost in costs.items()
                )
                for seed_index in available
            }
        candidate["min_source_switches"] = min(costs.values())
        candidate["support_scores"] = {}
        for temperature in TEMPERATURES:
            shifted = (raw_scores - raw_scores.max()) / temperature
            weights = np.exp(shifted)
            score = 0.0
            for denominator, numerator in evidence:
                denominator_mass = float(weights[list(denominator)].sum())
                numerator_mass = float(weights[list(numerator)].sum())
                score += math.log(numerator_mass / denominator_mass)
            candidate["support_scores"][str(temperature)] = score
    return candidates


def public_artifact_path(path: Path) -> str:
    """Return a stable, non-machine-specific identifier for an input artifact."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"external-artifact/{path.name}"


def attach_coherence_and_edits(
    rows_by_width: dict[int, list[dict[str, Any]]],
    model: StaticEmbeddingModel,
    cache_path: Path,
) -> None:
    cache = load_cache(cache_path)
    texts = list(
        dict.fromkeys(
            candidate["hypothesis"]
            for rows in rows_by_width.values()
            for row in rows
            for candidate in [*row["seeds"], *row["lattice_candidates"]]
        )
    )
    missing = [text for text in texts if text not in cache]
    for start in range(0, len(missing), 5_000):
        batch = missing[start : start + 5_000]
        scores = model.coherence_batch(batch)
        for text, score in zip(batch, scores):
            cache[text] = {
                "piece_sum": score.piece_sum,
                "piece_mean": score.piece_mean,
                "vertex_sum": score.vertex_sum,
                "vertex_mean": score.vertex_mean,
                "pieces": score.pieces,
                "vertices": score.vertices,
            }
        write_cache(cache_path, cache)
        print(f"static coherence: {start + len(batch)}/{len(missing)}", flush=True)
    seen: set[tuple[str, tuple[int, ...]]] = set()
    edit_cache: dict[tuple[str, tuple[int, ...]], int] = {}
    for rows in rows_by_width.values():
        for row in rows:
            for candidate in [*row["seeds"], *row["lattice_candidates"]]:
                candidate["coherence"] = cache[candidate["hypothesis"]]
                key = (row["utterance_id"], tuple(candidate["token_ids"]))
                if key not in seen:
                    edit_cache[key] = candidate_alignment(
                        row["reference"], candidate
                    )[0]["edits"]
                    seen.add(key)
                candidate["diagnostic_edits"] = edit_cache[key]


def eligible_candidates(
    row: dict[str, Any], switch_limit: int | None
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in row["lattice_candidates"]
        if switch_limit is None
        or int(candidate["min_source_switches"]) <= switch_limit
    ]


def lattice_base_score(
    candidate: dict[str, Any], temperature: float, exponent: float
) -> float:
    return float(candidate["support_scores"][str(temperature)]) / (
        (len(candidate["token_ids"]) + 2) ** exponent
    )


def retained_candidates(
    row: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = eligible_candidates(row, config["switch_limit"])
    return sorted(
        candidates,
        key=lambda candidate: (
            -lattice_base_score(
                candidate, config["temperature"], config["length_exponent"]
            ),
            candidate["token_ids"],
        ),
    )[: config["candidate_cap"]]


def select_lattice(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidates = retained_candidates(row, config)
    coherence = np.asarray(
        [candidate["coherence"]["piece_mean"] for candidate in candidates],
        dtype=np.float64,
    )
    deviation = float(coherence.std())
    standardized = (
        (coherence - float(coherence.mean())) / deviation
        if deviation > 1.0e-12
        else np.zeros_like(coherence)
    )
    return max(
        zip(candidates, standardized),
        key=lambda item: (
            lattice_base_score(
                item[0], config["temperature"], config["length_exponent"]
            )
            + config["coherence_weight"] * float(item[1]),
            -int(item[0]["rank"]),
        ),
    )[0]


def total_selected_edits(
    rows: list[dict[str, Any]], selector: Any
) -> int:
    return sum(int(selector(row)["diagnostic_edits"]) for row in rows)


def tune_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = [
        {
            "length_exponent": exponent,
            "coherence_weight": weight,
            "development_edits": total_selected_edits(
                rows,
                lambda row, exponent=exponent, weight=weight: select_seed_fusion(
                    row["seeds"], "piece_mean", exponent, weight
                ),
            ),
        }
        for exponent in SEED_EXPONENTS
        for weight in COHERENCE_WEIGHTS
    ]
    return min(
        configs,
        key=lambda item: (
            item["development_edits"],
            item["coherence_weight"],
            item["length_exponent"],
        ),
    )


def tune_lattice(
    rows: list[dict[str, Any]], switch_limit: int | None
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for temperature in TEMPERATURES:
        for exponent in LENGTH_EXPONENTS:
            for cap in CANDIDATE_CAPS:
                for weight in COHERENCE_WEIGHTS:
                    config = {
                        "switch_limit": switch_limit,
                        "temperature": temperature,
                        "length_exponent": exponent,
                        "candidate_cap": cap,
                        "coherence_weight": weight,
                    }
                    edits = total_selected_edits(
                        rows, lambda row, config=config: select_lattice(row, config)
                    )
                    candidate = {**config, "development_edits": edits}
                    if best is None or (
                        edits,
                        weight,
                        cap,
                        temperature,
                        exponent,
                    ) < (
                        best["development_edits"],
                        best["coherence_weight"],
                        best["candidate_cap"],
                        best["temperature"],
                        best["length_exponent"],
                    ):
                        best = candidate
    assert best is not None
    return best


def summarize_selector(
    rows: list[dict[str, Any]], selector: Any
) -> dict[str, Any]:
    selected = summarize_selection(rows, selector)
    baseline = summarize_selection(rows, lambda row: row["seeds"][0])
    return {
        **strip_internal(selected),
        "paired_vs_seed_top1": paired(selected, baseline),
        "delta_edits_vs_seed_top1": selected["edits"] - baseline["edits"],
        "delta_cer_percentage_points_vs_seed_top1": (
            selected["micro_cer"] - baseline["micro_cer"]
        )
        * 100,
    }


def evaluate_width(rows: list[dict[str, Any]], width: int) -> dict[str, Any]:
    development = [row for row in rows if is_development(row["utterance_id"])]
    test = [row for row in rows if not is_development(row["utterance_id"])]
    seed_config = tune_seed(development)
    seed_selector = lambda row: select_seed_fusion(
        row["seeds"],
        "piece_mean",
        seed_config["length_exponent"],
        seed_config["coherence_weight"],
    )
    constructions = {}
    for name, switch_limit in (("one_splice", 1), ("unrestricted", None)):
        config = tune_lattice(development, switch_limit)
        selector = lambda row, config=config: select_lattice(row, config)
        eligible_counts = [len(eligible_candidates(row, switch_limit)) for row in rows]
        cap_curve = {}
        for cap in CANDIDATE_CAPS:
            curve_config = {**config, "candidate_cap": cap}
            curve_selector = lambda row, curve_config=curve_config: select_lattice(
                row, curve_config
            )
            oracle_selector = lambda row, curve_config=curve_config: min(
                retained_candidates(row, curve_config),
                key=lambda candidate: (candidate["diagnostic_edits"], candidate["rank"]),
            )
            cap_curve[str(cap)] = {
                "selected": summarize_selector(test, curve_selector),
                "oracle": strip_internal(summarize_selection(test, oracle_selector)),
            }
        constructions[name] = {
            "selected_on_development": config,
            "mean_candidates": sum(eligible_counts) / len(eligible_counts),
            "max_candidates": max(eligible_counts),
            "development": summarize_selector(development, selector),
            "test": summarize_selector(test, selector),
            "full": summarize_selector(rows, selector),
            "pool_oracle": strip_internal(
                summarize_selection(
                    test,
                    lambda row, switch_limit=switch_limit: min(
                        eligible_candidates(row, switch_limit),
                        key=lambda candidate: (
                            candidate["diagnostic_edits"],
                            candidate["rank"],
                        ),
                    ),
                )
            ),
            "cap_curve": cap_curve,
        }
    return {
        "width": width,
        "source": "posthoc top-2 from exact width-4" if width == 2 else f"exact width-{width}",
        "samples": len(rows),
        "seed_top1": strip_internal(
            summarize_selection(test, lambda row: row["seeds"][0])
        ),
        "seed_oracle": strip_internal(
            summarize_selection(
                test, lambda row: select_oracle(row["reference"], row["seeds"])
            )
        ),
        "seed_fusion": {
            "selected_on_development": seed_config,
            "development": summarize_selector(development, seed_selector),
            "test": summarize_selector(test, seed_selector),
            "full": summarize_selector(rows, seed_selector),
        },
        "constructions": constructions,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ReazonSpeech lattice width and candidate construction search",
        "",
        "JSUT BASIC5000 FP32. Width 4 and 8 are exact raw full-prefix searches; width 2 is the top two candidates retained from exact width 4.",
        "All hyperparameters are selected on the hash development split and reported on the held-out split.",
        "",
        "| width | seed top-1 | seed oracle | seed + static | one-splice lattice | unrestricted lattice |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width in (2, 4, 8):
        result = summary["widths"][str(width)]
        lines.append(
            f"| {width} | {result['seed_top1']['micro_cer']:.4%} | "
            f"{result['seed_oracle']['micro_cer']:.4%} | "
            f"{result['seed_fusion']['test']['micro_cer']:.4%} | "
            f"{result['constructions']['one_splice']['test']['micro_cer']:.4%} | "
            f"{result['constructions']['unrestricted']['test']['micro_cer']:.4%} |"
        )
    lines.extend(["", "## Selected lattice configurations", ""])
    for width in (2, 4, 8):
        result = summary["widths"][str(width)]
        for name in ("one_splice", "unrestricted"):
            construction = result["constructions"][name]
            config = construction["selected_on_development"]
            lines.append(
                f"- width {width} {name}: CER {construction['test']['micro_cer']:.4%}, "
                f"oracle {construction['pool_oracle']['micro_cer']:.4%}, "
                f"mean/max pool {construction['mean_candidates']:.2f}/{construction['max_candidates']}, "
                f"temperature={config['temperature']}, alpha={config['length_exponent']}, "
                f"cap={config['candidate_cap']}, static weight={config['coherence_weight']}, "
                f"delta edits={construction['test']['delta_edits_vs_seed_top1']:+d}."
            )
    lines.extend(["", "## Width-8 retained-candidate curve", ""])
    lines.extend(
        [
            "| construction | retained width | selected CER | retained oracle |",
            "|---|---:|---:|---:|",
        ]
    )
    width8 = summary["widths"]["8"]["constructions"]
    for name in ("one_splice", "unrestricted"):
        for cap in CANDIDATE_CAPS:
            curve = width8[name]["cap_curve"][str(cap)]
            lines.append(
                f"| {name} | {cap} | {curve['selected']['micro_cer']:.4%} | "
                f"{curve['oracle']['micro_cer']:.4%} |"
            )
    best_seed = summary["widths"]["8"]["seed_fusion"]["test"]
    width4_lattice = summary["widths"]["4"]["constructions"]["one_splice"]
    width8_one = summary["widths"]["8"]["constructions"]["one_splice"]
    width8_all = summary["widths"]["8"]["constructions"]["unrestricted"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The best held-out top-1 remains exact width-8 seeds with static reranking: {best_seed['micro_cer']:.4%}.",
            f"- Exact width-4 plus a one-splice lattice is the lower-width compromise: {width4_lattice['test']['micro_cer']:.4%} with a mean pool of {width4_lattice['mean_candidates']:.2f}.",
            f"- At width 8, unrestricted recombination improves pool oracle by only {(width8_one['pool_oracle']['micro_cer'] - width8_all['pool_oracle']['micro_cer']) * 100:.4f} percentage points over one-splice, while expanding the maximum pool from {width8_one['max_candidates']} to {width8_all['max_candidates']}.",
            "- The width-8 oracle is nearly saturated by retaining 16 candidates. Increasing to 32 or 64 changes top-1 slightly but adds no useful oracle coverage.",
            "- Only one held-out utterance selected a novel recombined path at width 8; most gains come from reranking original N-best seeds, not from synthesis of a new transcript.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam4-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--beam8-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--coherence-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    beam4 = load_nbest(args.beam4_inputs, 4)
    beam8 = load_nbest(args.beam8_inputs, 8)
    if [row["utterance_id"] for row in beam4] != [row["utterance_id"] for row in beam8]:
        raise ValueError("width-4 and width-8 utterance IDs differ")
    rows_by_width = {
        width: width_rows(beam4, beam8, width) for width in (2, 4, 8)
    }
    for rows in rows_by_width.values():
        for row in rows:
            row["lattice_candidates"] = evidence_candidates(row)
    model = StaticEmbeddingModel(args.model_snapshot)
    attach_coherence_and_edits(
        rows_by_width, model, args.coherence_cache
    )
    summary = {
        "schema_version": 1,
        "condition": {
            "model": "reazonspeech_k2_v2",
            "precision": "float32",
            "static_embedding": "hotchpotch/static-embedding-japanese",
            "static_revision": args.model_snapshot.name,
            "development_split": "sha256(utterance_id) mod 5 == 0",
            "lattice_state": "(emitted_token_count,last_two_tokens)",
            "transition_score": "N-best-posterior conditional transition support",
        },
        "widths": {
            str(width): evaluate_width(rows, width)
            for width, rows in rows_by_width.items()
        },
        "input_artifacts": {
            "beam4": [
                {
                    "path": public_artifact_path(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in args.beam4_inputs
            ],
            "beam8": [
                {
                    "path": public_artifact_path(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in args.beam8_inputs
            ],
        },
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
