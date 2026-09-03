#!/usr/bin/env python3
"""Summarize ReazonSpeech fixed-transcript monotonic alignment experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr_eval_metrics import (
    align_characters,
    diagnostic_normalize,
    summarize_alignment,
)


@dataclass(frozen=True)
class SelectionConfig:
    search_score_gap_max: float
    forward_score_gap_max: float | None
    head_time_gap_min: float
    tail_time_gap_min: float
    entropy_max: float | None
    timing: str
    require_nonoverlap: bool = False


SEARCH_TIMESTAMP_REFERENCE = SelectionConfig(5.0, None, 0.039, 1.199, None, "search")
POSTERIOR_EXPECTED_REFERENCE = SelectionConfig(5.0, None, 0.039, 1.199, None, "expected")
POSTERIOR_NONOVERLAP = SelectionConfig(5.0, None, 0.0, 0.0, None, "expected", True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def load_rows(paths: list[Path], expected_count: int | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("status") != "completed":
                raise ValueError(f"{path}:{line_number}: non-completed row")
            utterance_id = row["utterance_id"]
            if utterance_id in rows:
                raise ValueError(f"duplicate utterance_id {utterance_id}")
            if row["beam_size"] != 8 or row["search_normalization"] != "raw":
                raise ValueError(f"{path}:{line_number}: unexpected search contract")
            if len(row["candidates"]) != 8:
                raise ValueError(f"{path}:{line_number}: expected eight candidates")
            rows[utterance_id] = row
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(rows)}")
    return rows


def edit_summary(reference: str, hypothesis: str) -> dict[str, int | bool]:
    normalized_reference = diagnostic_normalize(reference)
    normalized_hypothesis = diagnostic_normalize(hypothesis)
    result = summarize_alignment(
        align_characters(normalized_reference, normalized_hypothesis)
    )
    result["edits"] = (
        result["substitutions"] + result["deletions"] + result["insertions"]
    )
    result["reference_characters"] = len(normalized_reference)
    result["exact"] = normalized_reference == normalized_hypothesis
    return result


def extension_features(
    top: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | str] | None:
    top_tokens = top["token_ids"]
    candidate_tokens = candidate["token_ids"]
    if not top_tokens or len(candidate_tokens) <= len(top_tokens):
        return None
    if candidate_tokens[-len(top_tokens) :] == top_tokens:
        direction = "head"
    elif candidate_tokens[: len(top_tokens)] == top_tokens:
        direction = "tail"
    else:
        return None
    if diagnostic_normalize(top["hypothesis"]) == diagnostic_normalize(
        candidate["hypothesis"]
    ):
        return None

    top_alignment = top["token_alignments"]
    candidate_alignment = candidate["token_alignments"]
    if direction == "head":
        search_gap = top["search_timestamps"][0] - candidate["search_timestamps"][0]
        expected_gap = (
            top_alignment[0]["expected_timestamp"]
            - candidate_alignment[0]["expected_timestamp"]
        )
        conservative_gap = (
            top_alignment[0]["posterior_lower_timestamp"]
            - candidate_alignment[0]["posterior_upper_timestamp"]
        )
        entropy = candidate_alignment[0]["entropy"]
    else:
        search_gap = candidate["search_timestamps"][-1] - top["search_timestamps"][-1]
        expected_gap = (
            candidate_alignment[-1]["expected_timestamp"]
            - top_alignment[-1]["expected_timestamp"]
        )
        conservative_gap = (
            candidate_alignment[-1]["posterior_lower_timestamp"]
            - top_alignment[-1]["posterior_upper_timestamp"]
        )
        entropy = candidate_alignment[-1]["entropy"]
    return {
        "direction": direction,
        "search_score_gap": float(top["search_raw_score"] - candidate["search_raw_score"]),
        "forward_score_gap": float(top["forward_score"] - candidate["forward_score"]),
        "search_time_gap": float(search_gap),
        "expected_time_gap": float(expected_gap),
        "conservative_time_gap": float(conservative_gap),
        "entropy": float(entropy),
    }


def select_extension(
    row: dict[str, Any], config: SelectionConfig
) -> tuple[dict[str, Any], dict[str, float | str] | None]:
    top = row["candidates"][0]
    eligible: list[tuple[dict[str, Any], dict[str, float | str]]] = []
    for candidate in row["candidates"][1:]:
        features = extension_features(top, candidate)
        if features is None:
            continue
        if config_allows(features, config):
            eligible.append((candidate, features))
    if not eligible:
        return top, None
    return max(eligible, key=lambda item: float(item[0]["search_raw_score"]))


def config_allows(
    features: dict[str, float | str], config: SelectionConfig
) -> bool:
    if features["search_score_gap"] > config.search_score_gap_max:
        return False
    if (
        config.forward_score_gap_max is not None
        and features["forward_score_gap"] > config.forward_score_gap_max
    ):
        return False
    time_gap = features[f"{config.timing}_time_gap"]
    minimum = (
        config.head_time_gap_min
        if features["direction"] == "head"
        else config.tail_time_gap_min
    )
    if time_gap < minimum:
        return False
    if config.require_nonoverlap and features["conservative_time_gap"] < 0.0:
        return False
    return config.entropy_max is None or features["entropy"] <= config.entropy_max


def summarize_selection(
    full_rows: dict[str, dict[str, Any]],
    aligned_rows: dict[str, dict[str, Any]],
    config: SelectionConfig | None,
) -> dict[str, Any]:
    operation_names = ("substitutions", "deletions", "insertions")
    boundary_names = ("leading_deletions", "trailing_deletions")
    totals = {name: 0 for name in operation_names + boundary_names}
    reference_characters = exact = 0
    changed = wins = losses = ties = 0
    utterance_deltas: list[int] = []
    utterance_reference_lengths: list[int] = []
    selected_examples: list[dict[str, Any]] = []
    for utterance_id, row in full_rows.items():
        top = row["candidates"][0]
        selected = top
        features = None
        if config is not None and utterance_id in aligned_rows:
            selected, features = select_extension(aligned_rows[utterance_id], config)
        baseline = edit_summary(row["reference"], top["hypothesis"])
        result = edit_summary(row["reference"], selected["hypothesis"])
        for name in totals:
            totals[name] += int(result[name])
        reference_characters += int(result["reference_characters"])
        exact += int(result["exact"])
        delta = int(result["edits"]) - int(baseline["edits"])
        utterance_deltas.append(delta)
        utterance_reference_lengths.append(int(result["reference_characters"]))
        if features is not None:
            changed += 1
            wins += delta < 0
            losses += delta > 0
            ties += delta == 0
            selected_examples.append(
                {
                    "utterance_id": utterance_id,
                    "reference": row["reference"],
                    "baseline": top["hypothesis"],
                    "selected": selected["hypothesis"],
                    "edit_delta": delta,
                    **features,
                }
            )
    edits = sum(totals[name] for name in operation_names)
    return {
        "samples": len(full_rows),
        "reference_characters": reference_characters,
        "edits": edits,
        "micro_cer": edits / reference_characters,
        "exact": exact,
        "exact_rate": exact / len(full_rows),
        **totals,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "selected_examples": selected_examples,
        "utterance_deltas": utterance_deltas,
        "utterance_reference_lengths": utterance_reference_lengths,
    }


def tune_posterior_config(rows: dict[str, dict[str, Any]]) -> SelectionConfig:
    candidates = [
        SelectionConfig(search, forward, head, tail, entropy, "expected")
        for search in (1.0, 2.0, 3.0, 5.0)
        for forward in (1.0, 2.0, 3.0, 5.0)
        for head in (0.04, 0.08, 0.2, 0.4)
        for tail in (0.4, 0.8, 1.2)
        for entropy in (1.0, 2.0, 3.0)
    ]
    prepared = []
    for row in rows.values():
        top = row["candidates"][0]
        baseline_edits = int(edit_summary(row["reference"], top["hypothesis"])["edits"])
        options = []
        for candidate in row["candidates"][1:]:
            features = extension_features(top, candidate)
            if features is not None:
                options.append(
                    (
                        candidate,
                        features,
                        int(
                            edit_summary(row["reference"], candidate["hypothesis"])[
                                "edits"
                            ]
                        ),
                    )
                )
        prepared.append((baseline_edits, options))
    scored = []
    for config in candidates:
        edits = wins = losses = changed = 0
        for baseline_edits, options in prepared:
            eligible = [option for option in options if config_allows(option[1], config)]
            if eligible:
                selected = max(
                    eligible, key=lambda item: float(item[0]["search_raw_score"])
                )
                selected_edits = selected[2]
                changed += 1
                wins += selected_edits < baseline_edits
                losses += selected_edits > baseline_edits
            else:
                selected_edits = baseline_edits
            edits += selected_edits
        scored.append(
            (
                edits,
                losses,
                -wins,
                changed,
                config.search_score_gap_max,
                config.forward_score_gap_max,
                config.head_time_gap_min,
                config.tail_time_gap_min,
                config.entropy_max,
                config,
            )
        )
    return min(scored)[-1]


def paired_bootstrap_delta_cer(
    deltas: list[int], reference_lengths: list[int], samples: int = 10_000
) -> list[float]:
    delta = np.asarray(deltas, dtype=np.float64)
    lengths = np.asarray(reference_lengths, dtype=np.float64)
    generator = np.random.default_rng(20260815)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 200):
        count = min(200, samples - start)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        estimates[start : start + count] = (
            delta[indices].sum(axis=1) / lengths[indices].sum(axis=1) * 100.0
        )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def strip_internal(summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    deltas = result.pop("utterance_deltas")
    lengths = result.pop("utterance_reference_lengths")
    result["delta_cer_percentage_points_bootstrap_95_ci"] = (
        paired_bootstrap_delta_cer(deltas, lengths)
    )
    return result


def validate_candidate_parity(
    full_rows: dict[str, dict[str, Any]], aligned_rows: dict[str, dict[str, Any]]
) -> None:
    if not aligned_rows.keys() <= full_rows.keys():
        raise ValueError("aligned validation IDs are absent from full search rows")
    for utterance_id, aligned in aligned_rows.items():
        full = full_rows[utterance_id]
        full_contract = [
            (candidate["hypothesis"], candidate["token_ids"])
            for candidate in full["candidates"]
        ]
        aligned_contract = [
            (candidate["hypothesis"], candidate["token_ids"])
            for candidate in aligned["candidates"]
        ]
        if aligned_contract != full_contract:
            raise ValueError(f"candidate contract changed for {utterance_id}")
    missing_prefilter_rows = []
    for utterance_id, row in full_rows.items():
        top = row["candidates"][0]
        top_tokens = top["token_ids"]
        if not top_tokens:
            continue
        for candidate in row["candidates"][1:]:
            candidate_tokens = candidate["token_ids"]
            strict_extension = len(candidate_tokens) > len(top_tokens) and (
                candidate_tokens[-len(top_tokens) :] == top_tokens
                or candidate_tokens[: len(top_tokens)] == top_tokens
            )
            diagnostic_change = diagnostic_normalize(
                top["hypothesis"]
            ) != diagnostic_normalize(candidate["hypothesis"])
            score_gap = top["search_raw_score"] - candidate["search_raw_score"]
            if strict_extension and diagnostic_change and score_gap <= 5.0:
                if utterance_id not in aligned_rows:
                    missing_prefilter_rows.append(utterance_id)
                break
    if missing_prefilter_rows:
        raise ValueError(
            "validation alignment prefilter omitted eligible IDs: "
            f"{missing_prefilter_rows}"
        )


def alignment_diagnostics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    search_viterbi = []
    search_expected = []
    interval_widths = []
    emission_sum_errors = []
    token_count = 0
    for row in rows.values():
        for candidate in row["candidates"]:
            alignments = candidate["token_alignments"]
            if len(alignments) != len(candidate["token_ids"]):
                raise ValueError("token/alignment lengths differ")
            if any(
                left["expected_timestamp"] > right["expected_timestamp"]
                for left, right in zip(alignments, alignments[1:])
            ):
                raise ValueError("posterior expected timestamps are not monotonic")
            for search_timestamp, alignment in zip(
                candidate["search_timestamps"], alignments
            ):
                search_viterbi.append(
                    abs(search_timestamp - alignment["viterbi_timestamp"])
                )
                search_expected.append(
                    abs(search_timestamp - alignment["expected_timestamp"])
                )
                interval_widths.append(
                    alignment["posterior_upper_timestamp"]
                    - alignment["posterior_lower_timestamp"]
                )
            token_count += len(alignments)
            emission_sum_errors.append(
                abs(sum(candidate["frame_emission_probabilities"]) - len(alignments))
            )
    return {
        "candidates": sum(len(row["candidates"]) for row in rows.values()),
        "tokens": token_count,
        "search_equals_viterbi_rate": sum(value < 1.0e-5 for value in search_viterbi)
        / len(search_viterbi),
        "mean_absolute_search_viterbi_seconds": float(np.mean(search_viterbi)),
        "mean_absolute_search_expected_seconds": float(np.mean(search_expected)),
        "mean_posterior_90_interval_seconds": float(np.mean(interval_widths)),
        "p95_posterior_90_interval_seconds": float(np.quantile(interval_widths, 0.95)),
        "max_emission_sum_error": max(emission_sum_errors),
        "mean_search_and_alignment_elapsed_ms": float(
            np.mean(
                [row["search_and_alignment_elapsed_ms"] for row in rows.values()]
            )
        ),
    }


def summarize(
    tuning_rows: dict[str, dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
    validation_aligned: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validate_candidate_parity(validation_rows, validation_aligned)
    tuned = tune_posterior_config(tuning_rows)
    configs = {
        "search_timestamp_reference": SEARCH_TIMESTAMP_REFERENCE,
        "posterior_expected_reference": POSTERIOR_EXPECTED_REFERENCE,
        "posterior_nonoverlap": POSTERIOR_NONOVERLAP,
        "cv_tuned_posterior": tuned,
    }
    datasets = {}
    for name, full, aligned in (
        ("common_voice_tuning", tuning_rows, tuning_rows),
        ("jsut_validation", validation_rows, validation_aligned),
    ):
        baseline = summarize_selection(full, aligned, None)
        selections = {}
        for config_name, config in configs.items():
            result = summarize_selection(full, aligned, config)
            result["delta_edits_vs_search_top1"] = result["edits"] - baseline["edits"]
            result["delta_cer_percentage_points_vs_search_top1"] = (
                result["micro_cer"] - baseline["micro_cer"]
            ) * 100.0
            selections[config_name] = strip_internal(result)
        datasets[name] = {
            "search_top1": strip_internal(baseline),
            "selections": selections,
        }
    return {
        "schema_version": 1,
        "condition": {
            "model": "reazonspeech_k2_v2",
            "precision": "float32",
            "provider": "cpu",
            "beam_size": 8,
            "search_normalization": "raw",
            "alignment_graph": "one_symbol_per_frame_forward_backward",
            "input_edge_silence_ms": 320,
            "accuracy_only": True,
        },
        "tuning_grid": {
            "dataset": "Common Voice ja 26 dev fixed hash1000",
            "objective": "minimum diagnostic corpus edits; tie-break losses, wins, changes, then stricter thresholds",
            "selected": asdict(tuned),
            "combinations": 4 * 4 * 4 * 3 * 3,
        },
        "configs": {name: asdict(config) for name, config in configs.items()},
        "datasets": datasets,
        "alignment_diagnostics_common_voice": alignment_diagnostics(tuning_rows),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    tuning = summary["datasets"]["common_voice_tuning"]
    validation = summary["datasets"]["jsut_validation"]
    lines = [
        "# ReazonSpeech monotonic alignment experiment",
        "",
        "Width-8 FP32 full-prefix candidates with 320 ms silence on each edge. Common Voice hash1000 is tuning data; JSUT BASIC5000 is held-out validation.",
        "",
        "| selector | CV edits | CV Δ | JSUT edits | JSUT Δ | JSUT changed (W/L/T) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "search_timestamp_reference": "search timestamp reference",
        "posterior_expected_reference": "posterior expected reference",
        "posterior_nonoverlap": "posterior 90% non-overlap",
        "cv_tuned_posterior": "CV-tuned posterior",
    }
    for name, label in labels.items():
        cv = tuning["selections"][name]
        jsut = validation["selections"][name]
        lines.append(
            f"| {label} | {cv['edits']} | {cv['delta_edits_vs_search_top1']:+d} | "
            f"{jsut['edits']} | {jsut['delta_edits_vs_search_top1']:+d} | "
            f"{jsut['changed']} ({jsut['wins']}/{jsut['losses']}/{jsut['ties']}) |"
        )
    base_cv = tuning["search_top1"]
    base_jsut = validation["search_top1"]
    tuned = validation["selections"]["cv_tuned_posterior"]
    reference = validation["selections"]["posterior_expected_reference"]
    diagnostics = summary["alignment_diagnostics_common_voice"]
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- Search top-1 baseline: CV {base_cv['edits']} edits ({base_cv['micro_cer']:.4%}), JSUT {base_jsut['edits']} edits ({base_jsut['micro_cer']:.4%}).",
            f"- The CV-tuned posterior rule improved tuning by {-tuning['selections']['cv_tuned_posterior']['delta_edits_vs_search_top1']} edits, but only {-tuned['delta_edits_vs_search_top1']} edit on JSUT. Its held-out bootstrap 95% CI is [{tuned['delta_cer_percentage_points_bootstrap_95_ci'][0]:+.4f}, {tuned['delta_cer_percentage_points_bootstrap_95_ci'][1]:+.4f}] percentage points.",
            f"- The fixed threshold posterior rule matched the representative search-timestamp result on JSUT: {reference['edits']} edits ({reference['delta_edits_vs_search_top1']:+d}), changing {reference['changed']} utterances ({reference['wins']} wins / {reference['losses']} loss / {reference['ties']} tie). It did not improve candidate selection beyond the existing timestamp heuristic.",
            f"- Exact alignment materially changes timestamps: search and fixed-sequence Viterbi agree for {diagnostics['search_equals_viterbi_rate']:.2%} of tokens; mean |search-expected| is {diagnostics['mean_absolute_search_expected_seconds'] * 1000:.1f} ms and the mean posterior 90% interval is {diagnostics['mean_posterior_90_interval_seconds'] * 1000:.1f} ms.",
            "- Requiring non-overlapping 90% intervals is harmful: a late low-probability extension can be acoustically distinct but linguistically wrong. Acoustic timing alone cannot decide whether suffixes such as `です` are valid.",
            "- Keep forward-backward alignment as a timestamp/confidence diagnostic. Do not enable acoustic-only boundary rescue in production; combine the aligned N-best set with a language model before revisiting selection.",
            "",
            "The fixed-text aligner cannot invent a missing token absent from width-8 N-best. It only redistributes probability over monotonic emission frames for each supplied transcript.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-search-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-alignment-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--tuning-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--ort-runtime", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    tuning = load_rows(args.tuning_inputs, 1000)
    validation = load_rows(args.validation_search_inputs, 5000)
    validation_aligned = load_rows(args.validation_alignment_inputs)
    summary = summarize(tuning, validation, validation_aligned)
    summary["artifacts"] = {
        "tuning_alignments": [
            {"path": display_path(path), "sha256": sha256_file(path)}
            for path in args.tuning_inputs
        ],
        "validation_search": [
            {"path": display_path(path), "sha256": sha256_file(path)}
            for path in args.validation_search_inputs
        ],
        "validation_alignments": [
            {"path": display_path(path), "sha256": sha256_file(path)}
            for path in args.validation_alignment_inputs
        ],
        "audit": {
            name: {"path": display_path(path), "sha256": sha256_file(path)}
            for name, path in (
                ("tuning_manifest", args.tuning_manifest),
                ("validation_manifest", args.validation_manifest),
                ("runner", args.runner),
                ("ort_runtime", args.ort_runtime),
            )
            if path is not None
        },
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
