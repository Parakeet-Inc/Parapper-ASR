#!/usr/bin/env python3
"""Summarize x1/x100 hotword width sweeps for ReazonSpeech and Parakeet TDT."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


PATTERNS = {
    "reazon": re.compile(
        r"reazon-fp32-beam(?P<beam>\d+)-hotword-fixed-score(?P<multiplier>1|100)$"
    ),
    "parakeet_tdt": re.compile(
        r"parakeet-tdt-fp32-dag-argmax-beam(?P<beam>\d+)-hotword-x(?P<multiplier>1|100)$"
    ),
}


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["status"] != "completed":
                raise ValueError(f"failed record in {path}: {record}")
            utterance_id = record["utterance_id"]
            if utterance_id in records:
                raise ValueError(f"duplicate utterance {utterance_id} in {path}")
            records[utterance_id] = record
    return records


def metrics(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    oracle: dict[str, list[str]],
    fixed_hotwords: list[str],
) -> dict:
    if baseline.keys() != candidate.keys():
        raise ValueError("x1 and x100 utterance IDs differ")
    edits = reference_characters = exact = 0
    raw_edits = raw_reference_characters = raw_exact = 0
    wins = losses = ties = changed = 0
    baseline_hits = hits = recovered = lost = total_terms = 0
    inference_ms = duration_samples = 0.0
    target = {"utterances": 0, "changed": 0, "wins": 0, "losses": 0, "ties": 0}
    non_target = {"utterances": 0, "changed": 0, "wins": 0, "losses": 0, "ties": 0}
    non_target_new_hotword_utterances = 0
    examples = []
    for utterance_id, baseline_record in baseline.items():
        record = candidate[utterance_id]
        if baseline_record["reference"] != record["reference"]:
            raise ValueError(f"reference mismatch for {utterance_id}")
        raw_reference = record["reference"]
        raw_baseline = baseline_record["hypothesis"]
        raw_hypothesis = record["hypothesis"]
        reference = normalize(raw_reference)
        baseline_hypothesis = normalize(raw_baseline)
        hypothesis = normalize(raw_hypothesis)
        baseline_error = edit_distance(reference, baseline_hypothesis)
        error = edit_distance(reference, hypothesis)
        edits += error
        reference_characters += len(reference)
        exact += hypothesis == reference
        raw_edits += edit_distance(raw_reference, raw_hypothesis)
        raw_reference_characters += len(raw_reference)
        raw_exact += raw_hypothesis == raw_reference
        wins += error < baseline_error
        losses += error > baseline_error
        ties += error == baseline_error
        changed += raw_hypothesis != raw_baseline
        group = target if utterance_id in oracle else non_target
        group["utterances"] += 1
        group["changed"] += raw_hypothesis != raw_baseline
        group["wins"] += error < baseline_error
        group["losses"] += error > baseline_error
        group["ties"] += error == baseline_error
        if utterance_id not in oracle and any(
            normalize(term) in hypothesis and normalize(term) not in baseline_hypothesis
            for term in fixed_hotwords
        ):
            non_target_new_hotword_utterances += 1
        inference_ms += record["inference_elapsed_ms"]
        duration_samples += record["duration_samples"]
        terms = oracle.get(utterance_id, [])
        for term in terms:
            normalized_term = normalize(term)
            baseline_hit = normalized_term in baseline_hypothesis
            hit = normalized_term in hypothesis
            baseline_hits += baseline_hit
            hits += hit
            recovered += hit and not baseline_hit
            lost += baseline_hit and not hit
            total_terms += 1
        if raw_hypothesis != raw_baseline:
            examples.append(
                {
                    "utterance_id": utterance_id,
                    "reference": raw_reference,
                    "x1": raw_baseline,
                    "x100": raw_hypothesis,
                    "terms": terms,
                    "diagnostic_edit_delta": error - baseline_error,
                }
            )
    count = len(candidate)
    return {
        "utterances": count,
        "raw_micro_cer": raw_edits / max(1, raw_reference_characters),
        "raw_exact_match_rate": raw_exact / max(1, count),
        "diagnostic_micro_cer": edits / max(1, reference_characters),
        "diagnostic_exact_match_rate": exact / max(1, count),
        "rtf": inference_ms / 1000.0 / max(1.0, duration_samples / 16000.0),
        "mean_inference_ms": inference_ms / max(1, count),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "changed_utterances": changed,
        "term_hits": hits,
        "x1_term_hits": baseline_hits,
        "term_total": total_terms,
        "term_recall": hits / max(1, total_terms),
        "recovered_terms": recovered,
        "lost_terms": lost,
        "target_breakdown": target,
        "non_target_breakdown": non_target,
        "non_target_new_hotword_utterances": non_target_new_hotword_utterances,
        "changed_examples": sorted(
            examples,
            key=lambda example: (
                example["diagnostic_edit_delta"],
                example["utterance_id"],
            ),
        ),
    }


def summarize(root: Path, fixture: dict) -> dict:
    result = {"schema_version": 1, "root": str(root), "models": {}}
    for model, pattern in PATTERNS.items():
        model_result = {}
        for corpus_dir in sorted((root / model).glob("*")):
            if not corpus_dir.is_dir():
                continue
            corpus = corpus_dir.name
            fixture_name = "common_voice" if corpus == "common_voice" else corpus
            corpus_fixture = fixture["corpora"][fixture_name]
            oracle = corpus_fixture["oracle_by_utterance"]
            fixed_hotwords = corpus_fixture["fixed_hotwords"]
            conditions: dict[tuple[int, int], tuple[Path, dict[str, dict]]] = {}
            for path in corpus_dir.glob("*.jsonl"):
                match = pattern.fullmatch(path.stem)
                if match:
                    key = (int(match["beam"]), int(match["multiplier"]))
                    conditions[key] = (path, load_jsonl(path))
            widths = {}
            for beam in sorted({key[0] for key in conditions}):
                if (beam, 1) not in conditions or (beam, 100) not in conditions:
                    continue
                x1_path, x1 = conditions[(beam, 1)]
                x100_path, x100 = conditions[(beam, 100)]
                x1_metrics = metrics(x1, x1, oracle, fixed_hotwords)
                x100_metrics = metrics(x1, x100, oracle, fixed_hotwords)
                widths[str(beam)] = {
                    "x1": {**x1_metrics, "output": str(x1_path)},
                    "x100": {**x100_metrics, "output": str(x100_path)},
                    "delta_diagnostic_cer_pp": 100.0
                    * (
                        x100_metrics["diagnostic_micro_cer"]
                        - x1_metrics["diagnostic_micro_cer"]
                    ),
                    "rtf_ratio_x100_over_x1": x100_metrics["rtf"]
                    / max(1.0e-12, x1_metrics["rtf"]),
                }
            model_result[corpus] = widths
        result["models"][model] = model_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hotword-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.hotword_config.read_text(encoding="utf-8"))
    result = summarize(args.root, fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
