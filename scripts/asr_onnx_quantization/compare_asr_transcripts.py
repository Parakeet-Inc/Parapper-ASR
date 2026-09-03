#!/usr/bin/env python3
"""Compare ASR diagnostic JSON files against an FP32 reference."""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path
from typing import Any


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _run_key(run: dict[str, Any]) -> tuple[str, int]:
    raw_path = run.get("wav", run.get("path"))
    if raw_path is None:
        raise ValueError("run is missing wav/path")
    wav_name = str(raw_path).replace("\\", "/").rsplit("/", 1)[-1]
    repeat = run.get("repeat")
    if repeat is None:
        repeat = int(run.get("run_index", 0)) + 1
    return wav_name, int(repeat)


def _load_runs(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        runs = payload.get("rows")
    if not isinstance(runs, list):
        raise ValueError(f"{path}: expected a runs or rows array")
    keyed = {_run_key(run): run for run in runs}
    if len(keyed) != len(runs):
        raise ValueError(f"{path}: duplicate wav/repeat keys")
    return keyed


def _mean_rtf(runs: dict[tuple[str, int], dict[str, Any]]) -> float:
    values = [float(run["rtf"]) for run in runs.values()]
    return statistics.fmean(values) if values else 0.0


def compare_reports(
    reference_path: Path, candidates: dict[str, Path]
) -> dict[str, Any]:
    reference = _load_runs(reference_path)
    result: dict[str, Any] = {
        "reference": {
            "path": str(reference_path),
            "run_count": len(reference),
            "mean_rtf": _mean_rtf(reference),
        },
        "candidates": {},
    }
    for label, path in candidates.items():
        candidate = _load_runs(path)
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        if missing or extra:
            raise ValueError(f"{path}: key mismatch; missing={missing}, extra={extra}")

        exact = 0
        edit_distance = 0
        reference_characters = 0
        differences = []
        for key in sorted(reference):
            expected = _normalize_text(str(reference[key].get("text", "")))
            actual = _normalize_text(str(candidate[key].get("text", "")))
            distance = _edit_distance(expected, actual)
            reference_characters += len(expected)
            edit_distance += distance
            if distance == 0:
                exact += 1
            else:
                differences.append(
                    {
                        "wav": key[0],
                        "repeat": key[1],
                        "reference": expected,
                        "candidate": actual,
                        "edit_distance": distance,
                    }
                )

        result["candidates"][label] = {
            "path": str(path),
            "run_count": len(candidate),
            "exact_text_match_count": exact,
            "exact_text_match_rate": exact / len(reference) if reference else 1.0,
            "edit_distance": edit_distance,
            "reference_character_count": reference_characters,
            "character_error_rate_vs_reference": (
                edit_distance / reference_characters if reference_characters else 0.0
            ),
            "mean_rtf": _mean_rtf(candidate),
            "rtf_ratio_vs_reference": (
                _mean_rtf(candidate) / _mean_rtf(reference)
                if _mean_rtf(reference)
                else 0.0
            ),
            "differences": differences,
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate label and JSON path; may be repeated",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates: dict[str, Path] = {}
    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError(f"invalid --candidate {specification!r}; expected LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        candidates[label] = Path(raw_path)
    result = compare_reports(args.reference, candidates)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
