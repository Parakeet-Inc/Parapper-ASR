#!/usr/bin/env python3
"""Summarize paired proper-noun hotword JSONL experiments."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
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
                raise ValueError(f"duplicate utterance_id {utterance_id} in {path}")
            records[utterance_id] = record
    return records


def summarize(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    oracle: dict[str, list[str]],
) -> dict:
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate utterance IDs differ")
    edits = 0
    reference_characters = 0
    macro_rates: list[float] = []
    exact = 0
    raw_edits = 0
    raw_reference_characters = 0
    raw_macro_rates: list[float] = []
    raw_exact = 0
    wins = losses = ties = 0
    term_hits = baseline_term_hits = 0
    recovered_terms = lost_terms = 0
    term_total = 0
    changed_examples: list[dict] = []
    for utterance_id, baseline_record in baseline.items():
        candidate_record = candidate[utterance_id]
        if baseline_record["reference"] != candidate_record["reference"]:
            raise ValueError(f"reference mismatch for {utterance_id}")
        raw_reference = candidate_record["reference"]
        raw_baseline_hypothesis = baseline_record["hypothesis"]
        raw_hypothesis = candidate_record["hypothesis"]
        reference = normalize(raw_reference)
        baseline_hypothesis = normalize(baseline_record["hypothesis"])
        hypothesis = normalize(candidate_record["hypothesis"])
        baseline_edits = edit_distance(reference, baseline_hypothesis)
        candidate_edits = edit_distance(reference, hypothesis)
        raw_candidate_edits = edit_distance(raw_reference, raw_hypothesis)
        edits += candidate_edits
        reference_characters += len(reference)
        macro_rates.append(candidate_edits / max(1, len(reference)))
        exact += hypothesis == reference
        raw_edits += raw_candidate_edits
        raw_reference_characters += len(raw_reference)
        raw_macro_rates.append(raw_candidate_edits / max(1, len(raw_reference)))
        raw_exact += raw_hypothesis == raw_reference
        wins += candidate_edits < baseline_edits
        losses += candidate_edits > baseline_edits
        ties += candidate_edits == baseline_edits
        utterance_changed = baseline_record["hypothesis"] != candidate_record["hypothesis"]
        terms = oracle.get(utterance_id, [])
        for term in terms:
            normalized_term = normalize(term)
            baseline_hit = normalized_term in baseline_hypothesis
            candidate_hit = normalized_term in hypothesis
            baseline_term_hits += baseline_hit
            term_hits += candidate_hit
            recovered_terms += candidate_hit and not baseline_hit
            lost_terms += baseline_hit and not candidate_hit
            term_total += 1
        if utterance_changed:
            changed_examples.append(
                {
                    "utterance_id": utterance_id,
                    "reference": candidate_record["reference"],
                    "baseline": baseline_record["hypothesis"],
                    "candidate": candidate_record["hypothesis"],
                    "terms": terms,
                    "edit_delta": candidate_edits - baseline_edits,
                }
            )
    return {
        "utterances": len(candidate),
        "raw_micro_cer": raw_edits / max(1, raw_reference_characters),
        "raw_macro_cer": sum(raw_macro_rates) / max(1, len(raw_macro_rates)),
        "raw_exact_match_rate": raw_exact / max(1, len(candidate)),
        "micro_cer": edits / max(1, reference_characters),
        "macro_cer": sum(macro_rates) / max(1, len(macro_rates)),
        "exact_match_rate": exact / max(1, len(candidate)),
        "diagnostic_micro_cer": edits / max(1, reference_characters),
        "diagnostic_macro_cer": sum(macro_rates) / max(1, len(macro_rates)),
        "diagnostic_exact_match_rate": exact / max(1, len(candidate)),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "term_hits": term_hits,
        "baseline_term_hits": baseline_term_hits,
        "term_total": term_total,
        "term_recall": term_hits / max(1, term_total),
        "recovered_terms": recovered_terms,
        "lost_terms": lost_terms,
        "changed_utterances": len(changed_examples),
        "changed_examples": sorted(
            changed_examples,
            key=lambda example: (example["edit_delta"], example["utterance_id"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--hotword-config", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.hotword_config.read_text(encoding="utf-8"))
    oracle = fixture["corpora"][args.corpus]["oracle_by_utterance"]
    baseline_path = args.input_dir / "reazon-fp32-beam4-baseline.jsonl"
    baseline = load_jsonl(baseline_path)
    conditions = {}
    for path in sorted(args.input_dir.glob("reazon-fp32-beam4-hotword-*.jsonl")):
        conditions[path.stem] = summarize(baseline, load_jsonl(path), oracle)
    result = {
        "schema_version": 1,
        "corpus": args.corpus,
        "input_dir": str(args.input_dir),
        "baseline": summarize(baseline, baseline, oracle),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
