#!/usr/bin/env python3
"""Evaluate whole-sentence/static-token coherence on Reazon lattice paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from asr_eval_metrics import diagnostic_normalize
from summarize_reazon_approximate_lattice import (
    candidate_alignment,
    load_rows,
    paired,
    select_oracle,
    structural_lattice_candidates,
    summarize_selection,
)
from static_embedding_numpy import StaticEmbeddingModel


METRICS = ("piece_sum", "piece_mean", "vertex_sum", "vertex_mean")
EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
FUSION_WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_development(utterance_id: str) -> bool:
    digest = hashlib.sha256(utterance_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % 5 == 0


def select_embedding(
    candidates: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["coherence"][metric]),
            -int(candidate["rank"]),
        ),
    )


def select_seed_fusion(
    candidates: list[dict[str, Any]], metric: str, exponent: float, weight: float
) -> dict[str, Any]:
    coherence = np.asarray(
        [float(candidate["coherence"][metric]) for candidate in candidates],
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
            float(item[0]["raw_score"])
            / ((len(item[0]["token_ids"]) + 2) ** exponent)
            + weight * float(item[1]),
            float(item[0]["raw_score"]),
            -int(item[0]["rank"]),
        ),
    )[0]


def tuning_edits(
    rows: list[dict[str, Any]], metric: str, exponent: float, weight: float
) -> int:
    return sum(
        int(
            select_seed_fusion(row["seeds"], metric, exponent, weight)[
                "diagnostic_edits"
            ]
        )
        for row in rows
    )


def strip_internal(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "utterance_edits"}


def paired_bootstrap_delta_cer(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
    samples: int,
) -> list[float]:
    deltas = np.asarray(
        [
            candidate["utterance_edits"][row["utterance_id"]]
            - baseline["utterance_edits"][row["utterance_id"]]
            for row in rows
        ],
        dtype=np.int64,
    )
    reference_lengths = np.asarray(
        [len(diagnostic_normalize(row["reference"])) for row in rows],
        dtype=np.int64,
    )
    rng = np.random.default_rng(20_260_815)
    values = np.empty(samples, dtype=np.float64)
    chunk_size = 100
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(0, len(rows), size=(count, len(rows)))
        values[start : start + count] = (
            deltas[indices].sum(axis=1)
            / reference_lengths[indices].sum(axis=1)
            * 100
        )
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def load_cache(path: Path) -> dict[str, dict[str, float | int]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected static coherence cache schema")
    return payload["texts"]


def write_cache(path: Path, values: dict[str, dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "texts": values}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def attach_scores(
    rows: list[dict[str, Any]], model: StaticEmbeddingModel, cache_path: Path
) -> None:
    cache = load_cache(cache_path)
    texts = list(
        dict.fromkeys(
            candidate["hypothesis"]
            for row in rows
            for candidate in [*row["seeds"], *row["structural_candidates"]]
        )
    )
    missing = [text for text in texts if text not in cache]
    for start in range(0, len(missing), 5_000):
        batch_texts = missing[start : start + 5_000]
        scores = model.coherence_batch(batch_texts)
        for text, score in zip(batch_texts, scores):
            cache[text] = {
                metric: float(getattr(score, metric)) for metric in METRICS
            } | {"pieces": score.pieces, "vertices": score.vertices}
        print(
            f"static coherence: {start + len(batch_texts)}/{len(missing)}",
            flush=True,
        )
        write_cache(cache_path, cache)
    if missing:
        write_cache(cache_path, cache)
    for row in rows:
        for candidate in [*row["seeds"], *row["structural_candidates"]]:
            candidate["coherence"] = cache[candidate["hypothesis"]]
        for candidate in row["seeds"]:
            candidate["diagnostic_edits"] = candidate_alignment(
                row["reference"], candidate
            )[0]["edits"]


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_top1 = summarize_selection(rows, lambda row: row["seeds"][0])
    seed_oracle = summarize_selection(
        rows, lambda row: select_oracle(row["reference"], row["seeds"])
    )
    structural_oracle = summarize_selection(
        rows,
        lambda row: select_oracle(row["reference"], row["structural_candidates"]),
    )
    seed_embedding = {}
    structural_embedding = {}
    for metric in METRICS:
        seed_embedding[metric] = strip_internal(
            summarize_selection(
                rows,
                lambda row, metric=metric: select_embedding(row["seeds"], metric),
            )
        )
        structural_embedding[metric] = strip_internal(
            summarize_selection(
                rows,
                lambda row, metric=metric: select_embedding(
                    row["structural_candidates"], metric
                ),
            )
        )
    return {
        "samples": len(rows),
        "seed_top1": strip_internal(seed_top1),
        "seed_oracle": strip_internal(seed_oracle),
        "structural_oracle": strip_internal(structural_oracle),
        "embedding_only_seed": seed_embedding,
        "embedding_only_structural": structural_embedding,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.inputs)
    for row in rows:
        row["structural_candidates"] = structural_lattice_candidates(row)
    model = StaticEmbeddingModel(args.model_snapshot)
    attach_scores(rows, model, args.cache)
    development = [row for row in rows if is_development(row["utterance_id"])]
    test = [row for row in rows if not is_development(row["utterance_id"])]

    tuning = []
    for metric in METRICS:
        for exponent in EXPONENTS:
            for weight in FUSION_WEIGHTS:
                tuning.append(
                    {
                        "metric": metric,
                        "length_exponent": exponent,
                        "coherence_weight": weight,
                        "development_edits": tuning_edits(
                            development, metric, exponent, weight
                        ),
                    }
                )
    tuning.sort(
        key=lambda item: (
            item["development_edits"],
            item["coherence_weight"],
            item["length_exponent"],
            item["metric"],
        )
    )
    best = tuning[0]
    acoustic_only = min(
        (item for item in tuning if item["coherence_weight"] == 0.0),
        key=lambda item: (
            item["development_edits"],
            item["length_exponent"],
            item["metric"],
        ),
    )
    per_metric = {
        metric: min(
            (item for item in tuning if item["metric"] == metric),
            key=lambda item: (
                item["development_edits"],
                item["coherence_weight"],
                item["length_exponent"],
            ),
        )
        for metric in METRICS
    }

    def fusion_summary(
        subset: list[dict[str, Any]],
        parameters: dict[str, Any],
        bootstrap_samples: int = 0,
    ) -> dict[str, Any]:
        selected = summarize_selection(
            subset,
            lambda row: select_seed_fusion(
                row["seeds"],
                str(parameters["metric"]),
                float(parameters["length_exponent"]),
                float(parameters["coherence_weight"]),
            ),
        )
        baseline = summarize_selection(subset, lambda row: row["seeds"][0])
        result = {
            **strip_internal(selected),
            "paired_vs_seed_top1": paired(selected, baseline),
            "delta_edits_vs_seed_top1": selected["edits"] - baseline["edits"],
            "delta_cer_percentage_points_vs_seed_top1": (
                selected["micro_cer"] - baseline["micro_cer"]
            )
            * 100,
        }
        if bootstrap_samples:
            result["delta_cer_percentage_points_bootstrap_95_ci"] = (
                paired_bootstrap_delta_cer(
                    selected, baseline, subset, bootstrap_samples
                )
            )
        return result

    best_development = fusion_summary(development, best)
    best_test = fusion_summary(test, best, bootstrap_samples=10_000)
    best_full = fusion_summary(rows, best)

    return {
        "schema_version": 1,
        "condition": {
            "model": "hotchpotch/static-embedding-japanese",
            "revision": args.model_snapshot.name,
            "dimensions": model.dimensions,
            "sentence_embedding": "L2-normalized mean of static tokenizer pieces",
            "piece_score": "cosine(sentence_embedding, static token embedding)",
            "vertex_weight": "Unicode characters covered by each static token",
            "development_split": "sha256(utterance_id) mod 5 == 0",
        },
        "input_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.inputs
        ],
        "development": summarize_subset(development),
        "test": summarize_subset(test),
        "full": summarize_subset(rows),
        "fusion": {
            "selected_on_development": best,
            "development": best_development,
            "test": best_test,
            "full": best_full,
            "acoustic_only_selected_on_development": acoustic_only,
            "acoustic_only": {
                "development": fusion_summary(development, acoustic_only),
                "test": fusion_summary(test, acoustic_only),
                "full": fusion_summary(rows, acoustic_only),
            },
            "per_metric": {
                metric: {
                    "selected_on_development": parameters,
                    "development": fusion_summary(development, parameters),
                    "test": fusion_summary(test, parameters),
                    "full": fusion_summary(rows, parameters),
                }
                for metric, parameters in per_metric.items()
            },
            "grid": tuning,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ReazonSpeech static token-coherence reranking",
        "",
        "JSUT BASIC5000, FP32 width-8 seeds and all time-free structural paths.",
        "Hyperparameters are selected on a deterministic hash development split and reported on the held-out test split.",
        "",
        "| pool / selector | dev CER | held-out CER | full CER |",
        "|---|---:|---:|---:|",
    ]
    for pool, label in (
        ("embedding_only_seed", "width-8 / embedding only"),
        ("embedding_only_structural", "structural lattice / embedding only"),
    ):
        for metric in METRICS:
            lines.append(
                f"| {label} / {metric} | "
                f"{summary['development'][pool][metric]['micro_cer']:.4%} | "
                f"{summary['test'][pool][metric]['micro_cer']:.4%} | "
                f"{summary['full'][pool][metric]['micro_cer']:.4%} |"
            )
    best = summary["fusion"]["selected_on_development"]
    acoustic = summary["fusion"]["acoustic_only"]
    lines.append(
        f"| width-8 / tuned acoustic only | "
        f"{acoustic['development']['micro_cer']:.4%} | "
        f"{acoustic['test']['micro_cer']:.4%} | "
        f"{acoustic['full']['micro_cer']:.4%} |"
    )
    lines.append(
        f"| width-8 / acoustic + tuned coherence | "
        f"{summary['fusion']['development']['micro_cer']:.4%} | "
        f"{summary['fusion']['test']['micro_cer']:.4%} | "
        f"{summary['fusion']['full']['micro_cer']:.4%} |"
    )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            f"- Held-out width-8 top-1: {summary['test']['seed_top1']['micro_cer']:.4%}; seed oracle: {summary['test']['seed_oracle']['micro_cer']:.4%}; structural oracle: {summary['test']['structural_oracle']['micro_cer']:.4%}.",
            f"- Tuned fusion: metric={best['metric']}, acoustic length exponent={best['length_exponent']}, standardized coherence weight={best['coherence_weight']}.",
            f"- Held-out fusion versus seed top-1: {summary['fusion']['test']['paired_vs_seed_top1']['wins']} wins / {summary['fusion']['test']['paired_vs_seed_top1']['losses']} losses, {summary['fusion']['test']['delta_edits_vs_seed_top1']} edits.",
            f"- Held-out CER delta paired-bootstrap 95% CI: [{summary['fusion']['test']['delta_cer_percentage_points_bootstrap_95_ci'][0]:+.4f}, {summary['fusion']['test']['delta_cer_percentage_points_bootstrap_95_ci'][1]:+.4f}] percentage points (10,000 samples, fixed seed).",
            "",
            "## Fusion by token aggregation",
            "",
            "| metric | selected α | selected weight | held-out CER | delta edits vs seed top-1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        result = summary["fusion"]["per_metric"][metric]
        selected = result["selected_on_development"]
        lines.append(
            f"| {metric} | {selected['length_exponent']} | "
            f"{selected['coherence_weight']} | {result['test']['micro_cer']:.4%} | "
            f"{result['test']['delta_edits_vs_seed_top1']:+d} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
