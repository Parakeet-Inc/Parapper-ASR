"""Score Japanese Parakeet CTC vs TDT runs with okurigana-focused metrics.

Inputs are `run_parakeet_ja_tdt_sweep` JSONL outputs plus a
`dump_ja_morphology` tokenization of the diagnostic-normalized references.
Every reference character is assigned one orthography class:

- ``okurigana``: a hiragana character that follows at least one kanji inside
  one dictionary token (``買わ`` -> ``わ``, ``呼び起こす`` -> ``び``/``こ``/``す``).
  Particles and other independent kana tokens are excluded because the
  tokenizer separates them (``山が`` -> ``山`` + ``が``).
- ``kanji`` / ``hiragana_other`` / ``katakana`` / ``other``: by character.

Substitution and deletion errors are attributed to the reference character
they consume in the shared minimum-edit alignment; insertions have no
reference anchor and are reported separately. The okurigana hypothesis
("CTC is weaker than TDT on okurigana") is tested with per-utterance paired
statistics and a seeded bootstrap over utterances.

Usage:
  python summarize_parakeet_ja_ctc_vs_tdt.py \
    --dataset-dir <dir with <condition>.jsonl> \
    --conditions prod_ctc_greedy,onnx_asr_ctc_greedy,fused_tdt_greedy \
    --baseline-condition fused_tdt_greedy \
    --morphology <morphology.jsonl> \
    --output-dir <dir>

  # First pass: write the normalized-reference TSV consumed by
  # dump_ja_morphology, then re-run with --morphology.
  python summarize_parakeet_ja_ctc_vs_tdt.py \
    --dataset-dir <dir> --conditions <list> \
    --dump-references <references.tsv>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from asr_eval_metrics import align_characters, diagnostic_normalize  # noqa: E402

SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260816
CHAR_CLASSES = ("kanji", "okurigana", "hiragana_other", "katakana", "other")
ERROR_KEYS = ("substitutions", "deletions")


def is_kanji(character: str) -> bool:
    code_point = ord(character)
    return (
        0x4E00 <= code_point <= 0x9FFF
        or 0x3400 <= code_point <= 0x4DBF
        or 0xF900 <= code_point <= 0xFAFF
        or character in "々〆〇"
    )


def is_hiragana(character: str) -> bool:
    code_point = ord(character)
    return 0x3041 <= code_point <= 0x3096 or character in "ゝゞ"


def is_katakana(character: str) -> bool:
    code_point = ord(character)
    return 0x30A1 <= code_point <= 0x30FA or character in "ーヽヾ"


def okurigana_positions(tokens: Sequence[tuple[str, int]]) -> set[int]:
    """Return okurigana character offsets from ``(surface, start)`` tokens."""
    positions: set[int] = set()
    for surface, start in tokens:
        seen_kanji = False
        for offset, character in enumerate(surface):
            if is_kanji(character):
                seen_kanji = True
            elif seen_kanji and is_hiragana(character):
                positions.add(start + offset)
    return positions


def classify_reference(reference: str, okurigana: set[int]) -> list[str]:
    classes = []
    for index, character in enumerate(reference):
        if index in okurigana:
            classes.append("okurigana")
        elif is_kanji(character):
            classes.append("kanji")
        elif is_hiragana(character):
            classes.append("hiragana_other")
        elif is_katakana(character):
            classes.append("katakana")
        else:
            classes.append("other")
    return classes


def attribute_errors(
    reference: str, hypothesis: str, classes: Sequence[str]
) -> dict[str, object]:
    """Attribute alignment errors to reference character classes."""
    if len(classes) != len(reference):
        raise ValueError("classes must cover every reference character")
    operations = align_characters(reference, hypothesis)
    per_class = {
        char_class: {"substitutions": 0, "deletions": 0, "total": 0}
        for char_class in CHAR_CLASSES
    }
    for char_class in classes:
        per_class[char_class]["total"] += 1
    insertions = 0
    reference_index = 0
    for operation in operations:
        if operation == "insertion":
            insertions += 1
            continue
        char_class = classes[reference_index]
        if operation == "substitution":
            per_class[char_class]["substitutions"] += 1
        elif operation == "deletion":
            per_class[char_class]["deletions"] += 1
        reference_index += 1
    if reference_index != len(reference):
        raise RuntimeError("alignment did not consume the full reference")
    return {
        "per_class": per_class,
        "insertions": insertions,
        "edits": insertions
        + sum(
            per_class[char_class][key]
            for char_class in CHAR_CLASSES
            for key in ERROR_KEYS
        ),
    }


def load_runs(
    dataset_dir: Path, conditions: Sequence[str]
) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    """Load completed records per condition, keyed by utterance id."""
    runs: dict[str, dict[str, dict[str, str]]] = {}
    failed: set[str] = set()
    for condition in conditions:
        path = dataset_dir / f"{condition}.jsonl"
        records: dict[str, dict[str, str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                utterance_id = record["utterance_id"]
                if record.get("status") != "completed":
                    failed.add(utterance_id)
                    continue
                if utterance_id in records:
                    raise ValueError(
                        f"duplicate utterance {utterance_id} in {path}"
                    )
                records[utterance_id] = record
        runs[condition] = records
    utterance_sets = [set(records) for records in runs.values()]
    shared = set.intersection(*utterance_sets) - failed
    for condition, records in runs.items():
        missing = (set.union(*utterance_sets) - failed) - set(records)
        if missing:
            raise ValueError(
                f"{condition} is missing {len(missing)} shared utterances"
            )
    references: dict[str, str] = {}
    for records in runs.values():
        for utterance_id in shared:
            reference = records[utterance_id]["reference"]
            existing = references.setdefault(utterance_id, reference)
            if existing != reference:
                raise ValueError(
                    f"conditions disagree on the reference of {utterance_id}"
                )
    ordered = sorted(shared)
    return runs, ordered


def load_morphology(path: Path) -> dict[str, list[tuple[str, int]]]:
    tokens_by_id: dict[str, list[tuple[str, int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tokens_by_id[record["id"]] = [
                (token["surface"], token["start"]) for token in record["tokens"]
            ]
    return tokens_by_id


def score_condition(
    runs: dict[str, dict[str, dict[str, str]]],
    condition: str,
    ordered_ids: Sequence[str],
    classified: dict[str, tuple[str, list[str]]],
) -> dict[str, object]:
    per_utterance: dict[str, dict[str, object]] = {}
    for utterance_id in ordered_ids:
        reference, classes = classified[utterance_id]
        hypothesis = diagnostic_normalize(runs[condition][utterance_id]["hypothesis"])
        per_utterance[utterance_id] = attribute_errors(reference, hypothesis, classes)
    totals = {
        char_class: {"substitutions": 0, "deletions": 0, "total": 0}
        for char_class in CHAR_CLASSES
    }
    insertions = 0
    edits = 0
    reference_characters = 0
    exact = 0
    empty = 0
    for utterance_id in ordered_ids:
        attribution = per_utterance[utterance_id]
        for char_class in CHAR_CLASSES:
            for key in ("substitutions", "deletions", "total"):
                totals[char_class][key] += attribution["per_class"][char_class][key]
        insertions += attribution["insertions"]
        edits += attribution["edits"]
        reference, _ = classified[utterance_id]
        reference_characters += len(reference)
        if attribution["edits"] == 0:
            exact += 1
        hypothesis = diagnostic_normalize(runs[condition][utterance_id]["hypothesis"])
        if not hypothesis:
            empty += 1
    class_rates = {
        char_class: {
            "substitutions": totals[char_class]["substitutions"],
            "deletions": totals[char_class]["deletions"],
            "total": totals[char_class]["total"],
            "error_rate": (
                (
                    totals[char_class]["substitutions"]
                    + totals[char_class]["deletions"]
                )
                / totals[char_class]["total"]
            )
            if totals[char_class]["total"]
            else 0.0,
        }
        for char_class in CHAR_CLASSES
    }
    return {
        "per_utterance": per_utterance,
        "summary": {
            "utterances": len(ordered_ids),
            "reference_characters": reference_characters,
            "diagnostic_cer": edits / reference_characters,
            "edits": edits,
            "insertions": insertions,
            "exact": exact,
            "empty_hypotheses": empty,
            "classes": class_rates,
        },
    }


def pooled_class_rate(
    scored: dict[str, object], ids: Sequence[str], char_class: str
) -> float:
    errors = 0
    total = 0
    for utterance_id in ids:
        per_class = scored["per_utterance"][utterance_id]["per_class"][char_class]
        errors += per_class["substitutions"] + per_class["deletions"]
        total += per_class["total"]
    return errors / total if total else 0.0


def bootstrap_delta(
    scored_a: dict[str, object],
    scored_b: dict[str, object],
    ordered_ids: Sequence[str],
    char_class: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the pooled ``a - b`` class error-rate delta over utterances."""
    generator = random.Random(seed)
    deltas = []
    count = len(ordered_ids)
    for _ in range(samples):
        resample = [ordered_ids[generator.randrange(count)] for _ in range(count)]
        deltas.append(
            pooled_class_rate(scored_a, resample, char_class)
            - pooled_class_rate(scored_b, resample, char_class)
        )
    deltas.sort()
    lower = deltas[int(0.025 * samples)]
    upper = deltas[min(int(0.975 * samples), samples - 1)]
    return {
        "delta": pooled_class_rate(scored_a, ordered_ids, char_class)
        - pooled_class_rate(scored_b, ordered_ids, char_class),
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
    }


def paired_counts(
    scored_a: dict[str, object],
    scored_b: dict[str, object],
    ordered_ids: Sequence[str],
    char_class: str,
) -> dict[str, int]:
    a_worse = 0
    b_worse = 0
    ties = 0
    for utterance_id in ordered_ids:
        def errors(scored: dict[str, object]) -> int:
            per_class = scored["per_utterance"][utterance_id]["per_class"][char_class]
            return per_class["substitutions"] + per_class["deletions"]

        delta = errors(scored_a) - errors(scored_b)
        if delta > 0:
            a_worse += 1
        elif delta < 0:
            b_worse += 1
        else:
            ties += 1
    return {"a_worse": a_worse, "b_worse": b_worse, "ties": ties}


def mine_examples(
    runs: dict[str, dict[str, dict[str, str]]],
    scored: dict[str, dict[str, object]],
    condition: str,
    baseline: str,
    ordered_ids: Sequence[str],
    classified: dict[str, tuple[str, list[str]]],
    limit: int,
) -> list[dict[str, object]]:
    """Utterances where `condition` makes more okurigana errors than `baseline`."""

    def okurigana_errors(scored_condition: dict[str, object], utterance_id: str) -> int:
        per_class = scored_condition["per_utterance"][utterance_id]["per_class"][
            "okurigana"
        ]
        return per_class["substitutions"] + per_class["deletions"]

    ranked = sorted(
        ordered_ids,
        key=lambda utterance_id: (
            okurigana_errors(scored[baseline], utterance_id)
            - okurigana_errors(scored[condition], utterance_id),
            utterance_id,
        ),
    )
    examples = []
    for utterance_id in ranked[:limit]:
        gap = okurigana_errors(scored[condition], utterance_id) - okurigana_errors(
            scored[baseline], utterance_id
        )
        if gap <= 0:
            break
        reference, _ = classified[utterance_id]
        examples.append(
            {
                "utterance_id": utterance_id,
                "okurigana_error_gap": gap,
                "reference": reference,
                condition: diagnostic_normalize(
                    runs[condition][utterance_id]["hypothesis"]
                ),
                baseline: diagnostic_normalize(
                    runs[baseline][utterance_id]["hypothesis"]
                ),
            }
        )
    return examples


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(summary: dict[str, object]) -> str:
    lines = [
        "# Japanese Parakeet CTC vs TDT okurigana analysis",
        "",
        f"- split: `{summary['split_id']}`",
        f"- utterances scored: {summary['utterances']}",
        f"- baseline condition: `{summary['baseline_condition']}`",
        f"- bootstrap: {summary['parameters']['bootstrap_samples']} samples, "
        f"seed {summary['parameters']['seed']}",
        "",
        "## Per-condition diagnostic metrics",
        "",
        "| condition | diagnostic CER | edits | insertions | exact | empty |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, scored in summary["conditions"].items():
        condition_summary = scored
        lines.append(
            f"| {condition} | {condition_summary['diagnostic_cer']:.5f} "
            f"| {condition_summary['edits']} | {condition_summary['insertions']} "
            f"| {condition_summary['exact']} | {condition_summary['empty_hypotheses']} |"
        )
    lines += [
        "",
        "## Per-class reference error rates (substitutions + deletions)",
        "",
        "| condition | " + " | ".join(CHAR_CLASSES) + " |",
        "| --- |" + " ---: |" * len(CHAR_CLASSES),
    ]
    for condition, scored in summary["conditions"].items():
        cells = []
        for char_class in CHAR_CLASSES:
            rate = scored["classes"][char_class]["error_rate"]
            total = scored["classes"][char_class]["total"]
            cells.append(f"{rate:.5f} (n={total})")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons vs baseline", ""]
    for condition, comparison in summary["comparisons"].items():
        lines += [
            f"### {condition} - {summary['baseline_condition']}",
            "",
            "| class | delta | 95% bootstrap CI | condition worse | baseline worse | ties |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for char_class, stats in comparison["classes"].items():
            paired = comparison["paired"][char_class]
            lines.append(
                f"| {char_class} | {stats['delta']:+.5f} "
                f"| [{stats['bootstrap_lower']:+.5f}, {stats['bootstrap_upper']:+.5f}] "
                f"| {paired['a_worse']} | {paired['b_worse']} | {paired['ties']} |"
            )
        lines.append("")
    lines += ["## Example utterances (largest okurigana gaps vs baseline)", ""]
    for condition, examples in summary["examples"].items():
        lines.append(f"### {condition}")
        lines.append("")
        if not examples:
            lines.append("(no utterance with an okurigana gap)")
        for example in examples:
            lines += [
                f"- `{example['utterance_id']}` (gap {example['okurigana_error_gap']})",
                f"  - ref: {example['reference']}",
                f"  - {condition}: {example[condition]}",
                f"  - {summary['baseline_condition']}: "
                f"{example[summary['baseline_condition']]}",
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default="prod_ctc_greedy,onnx_asr_ctc_greedy,fused_tdt_greedy",
    )
    parser.add_argument("--baseline-condition", default="fused_tdt_greedy")
    parser.add_argument("--split-id", default="")
    parser.add_argument("--morphology", type=Path)
    parser.add_argument("--dump-references", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    conditions = [
        condition.strip()
        for condition in arguments.conditions.split(",")
        if condition.strip()
    ]
    runs, ordered_ids = load_runs(arguments.dataset_dir, conditions)
    normalized_references = {
        utterance_id: diagnostic_normalize(
            runs[conditions[0]][utterance_id]["reference"]
        )
        for utterance_id in ordered_ids
    }

    if arguments.dump_references is not None:
        with arguments.dump_references.open("w", encoding="utf-8", newline="\n") as out:
            for utterance_id in ordered_ids:
                out.write(f"{utterance_id}\t{normalized_references[utterance_id]}\n")
        print(f"wrote {len(ordered_ids)} references to {arguments.dump_references}")
        return

    if arguments.morphology is None or arguments.output_dir is None:
        raise SystemExit("--morphology and --output-dir are required for scoring")
    if arguments.baseline_condition not in conditions:
        raise SystemExit("--baseline-condition must be one of --conditions")
    tokens_by_id = load_morphology(arguments.morphology)
    classified: dict[str, tuple[str, list[str]]] = {}
    for utterance_id in ordered_ids:
        reference = normalized_references[utterance_id]
        tokens = tokens_by_id.get(utterance_id)
        if tokens is None:
            raise SystemExit(f"morphology is missing utterance {utterance_id}")
        surface = "".join(surface for surface, _ in tokens)
        if surface != reference:
            raise SystemExit(
                f"morphology does not cover the normalized reference of {utterance_id}"
            )
        classes = classify_reference(reference, okurigana_positions(tokens))
        classified[utterance_id] = (reference, classes)

    scored = {
        condition: score_condition(runs, condition, ordered_ids, classified)
        for condition in conditions
    }
    baseline = arguments.baseline_condition
    comparisons = {}
    examples = {}
    for condition in conditions:
        if condition == baseline:
            continue
        comparisons[condition] = {
            "classes": {
                char_class: bootstrap_delta(
                    scored[condition],
                    scored[baseline],
                    ordered_ids,
                    char_class,
                    arguments.bootstrap_samples,
                    BOOTSTRAP_SEED,
                )
                for char_class in CHAR_CLASSES
            },
            "paired": {
                char_class: paired_counts(
                    scored[condition], scored[baseline], ordered_ids, char_class
                )
                for char_class in CHAR_CLASSES
            },
        }
        examples[condition] = mine_examples(
            runs,
            scored,
            condition,
            baseline,
            ordered_ids,
            classified,
            arguments.examples,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "split_id": arguments.split_id,
        "utterances": len(ordered_ids),
        "baseline_condition": baseline,
        "parameters": {
            "bootstrap_samples": arguments.bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
            "morphology_sha256": sha256_of(arguments.morphology),
            "conditions": {
                condition: sha256_of(arguments.dataset_dir / f"{condition}.jsonl")
                for condition in conditions
            },
        },
        "conditions": {
            condition: scored[condition]["summary"] for condition in conditions
        },
        "comparisons": comparisons,
        "examples": examples,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = arguments.output_dir / "REPORT.md"
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"wrote {summary_path} and {report_path}")


if __name__ == "__main__":
    run()
