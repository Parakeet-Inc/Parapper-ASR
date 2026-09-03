#!/usr/bin/env python3
"""Run one offline ASR evaluation condition as K parallel ``run_asr_eval`` shards.

``run_asr_eval`` walks a ``RunnerManifestV1`` manifest strictly sequentially, so a
1000-utterance condition is bounded by one process' single-stream throughput. This
utility slices the manifest into K contiguous sub-manifests, runs one
``run_asr_eval`` process per shard concurrently, then concatenates the shard JSONL
outputs back into a single file in manifest order and verifies the result against
the source manifest.

Two contracts from the Rust binary drive the layout choices here:

* ``run_asr_eval`` resolves ``audio.relative_path`` against the *manifest file's*
  parent directory, so shard manifests must be written next to the source manifest.
* ``run_asr_eval`` refuses to overwrite an existing ``--output``, so a shard whose
  output already exists is either skipped (when it is provably complete) or the run
  aborts with the offending path named.

Timing fields (``inference_elapsed_ms``) in a sharded run are measured under
parallel CPU load and are therefore NOT comparable to a sequential run. Sharding is
for accuracy/throughput work only.

Only the standard library is used, so this runs under plain ``python``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = 1

# Actions returned by ``classify_existing_output``.
ACTION_RUN = "run"
ACTION_SKIP = "skip"
ACTION_ABORT = "abort"


class ShardingError(RuntimeError):
    """A precondition or post-run verification failure that must stop the run."""


# ---------------------------------------------------------------------------
# Manifest sharding (pure)
# ---------------------------------------------------------------------------


def shard_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    """Splits ``total`` items into ``shards`` contiguous half-open ranges.

    Sizes differ by at most one and the larger shards come first, so the ranges are
    a deterministic function of ``(total, shards)`` alone.
    """
    if shards < 1:
        raise ShardingError(f"--shards must be a positive integer, got {shards}")
    if total < 1:
        raise ShardingError("manifest contains no samples")
    if shards > total:
        raise ShardingError(
            f"--shards {shards} exceeds the {total} manifest samples; "
            "run_asr_eval rejects an empty manifest, so every shard must be non-empty"
        )

    base, remainder = divmod(total, shards)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def shard_label(index: int, shards: int) -> str:
    """Zero-padded ``shard-<i>-of-<K>`` stem fragment that sorts in shard order."""
    width = len(str(shards))
    return f"shard-{index:0{width}d}-of-{shards}"


def shard_manifest_path(manifest: Path, index: int, shards: int) -> Path:
    """Shard manifest path, always beside the source manifest (audio path anchor)."""
    return manifest.parent / f"{manifest.stem}.{shard_label(index, shards)}.json"


def shard_output_path(output: Path, index: int, shards: int) -> Path:
    return output.parent / f"{output.name}.{shard_label(index, shards)}.jsonl"


def shard_log_path(output: Path, index: int, shards: int) -> Path:
    return Path(f"{shard_output_path(output, index, shards)}.log")


def build_shard_manifest(document: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Copies ``document`` verbatim, replacing only the ``samples`` slice.

    Every other top-level key (``selection``, ``provenance``, ...) and every
    per-sample key (``source``, ``speaker_id``, ``sentence_id``, ``selection_hash``,
    ...) passes through untouched, including keys this script does not model.
    """
    if "samples" not in document:
        raise ShardingError("manifest has no 'samples' array")
    samples = document["samples"]
    if not isinstance(samples, list):
        raise ShardingError("manifest 'samples' is not a JSON array")
    shard = dict(document)  # preserves key order; 'samples' keeps its original slot
    shard["samples"] = samples[start:end]
    return shard


def serialize_manifest(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def manifest_utterance_ids(document: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for position, sample in enumerate(document.get("samples", [])):
        if not isinstance(sample, dict) or "utterance_id" not in sample:
            raise ShardingError(f"manifest sample {position} has no 'utterance_id'")
        ids.append(sample["utterance_id"])
    duplicates = _duplicates(ids)
    if duplicates:
        raise ShardingError(
            f"manifest contains duplicate utterance_id values: {sorted(duplicates)}"
        )
    return ids


def write_shard_manifest(path: Path, document: dict[str, Any]) -> str:
    """Writes a shard manifest idempotently.

    Returns ``"created"`` or ``"reused"``. An existing file whose decoded JSON
    differs from ``document`` aborts the run rather than being silently replaced:
    a stale shard manifest would otherwise evaluate a different sample set under
    the same output name.
    """
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ShardingError(
                f"shard manifest {path} exists but is not readable JSON ({error}); "
                "delete it and re-run"
            ) from error
        if existing == document:
            return "reused"
        raise ShardingError(
            f"shard manifest {path} exists with different content "
            "(different source manifest or shard count); delete it and re-run"
        )
    path.write_text(serialize_manifest(document), encoding="utf-8")
    return "created"


# ---------------------------------------------------------------------------
# Shard output inspection / resume (pure enough to test with real temp files)
# ---------------------------------------------------------------------------


def classify_existing_output(path: Path, expected_count: int) -> tuple[str, str]:
    """Decides what to do about a shard output that may already exist.

    Returns ``(action, detail)``:

    * ``("run", ...)``    -- no output yet, launch ``run_asr_eval``.
    * ``("skip", ...)``   -- a complete output exists, resume by reusing it.
    * ``("abort", ...)``  -- a partial/corrupt output exists. ``run_asr_eval``
      refuses to overwrite, so the operator must delete the named file; deleting
      it here would silently discard evaluation work.
    """
    if not path.exists():
        return ACTION_RUN, "no existing output"
    try:
        records = read_jsonl(path)
    except ShardingError as error:
        return ACTION_ABORT, f"existing output is not valid JSONL: {error}"
    if len(records) == expected_count:
        return ACTION_SKIP, f"existing output already has {expected_count} records"
    return (
        ACTION_ABORT,
        f"existing output has {len(records)} records but the shard covers "
        f"{expected_count} samples; delete {path} and re-run",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads a JSONL file, rejecting any line that is not a JSON object."""
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ShardingError(f"failed to read {path}: {error}") from error
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ShardingError(f"{path}:{number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ShardingError(f"{path}:{number} is not a JSON object")
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Concatenation and verification
# ---------------------------------------------------------------------------


def concatenate_jsonl(sources: Sequence[Path], destination: Path) -> int:
    """Appends the shard outputs in shard order into ``destination``.

    Returns the number of records written. Each shard's final line is newline
    terminated so a shard whose writer stopped mid-line cannot fuse two records.
    """
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as sink:
        for source in sources:
            if not source.exists():
                raise ShardingError(f"shard output is missing: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                sink.write(line)
                sink.write("\n")
                written += 1
    return written


def verify_records(
    records: Sequence[dict[str, Any]], expected_ids: Sequence[str]
) -> dict[str, int]:
    """Checks the concatenated records against the source manifest.

    Verified: record count, per-record ``utterance_id`` presence, absence of
    duplicates, and exact set equality with the manifest. Returns the
    completed/failed tally.
    """
    if len(records) != len(expected_ids):
        raise ShardingError(
            f"record count mismatch: manifest has {len(expected_ids)} samples "
            f"but the concatenated output has {len(records)} records"
        )

    seen: list[str] = []
    completed = 0
    failed = 0
    for position, record in enumerate(records):
        utterance_id = record.get("utterance_id")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise ShardingError(f"record {position} has no usable 'utterance_id'")
        seen.append(utterance_id)
        status = record.get("status")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        else:
            raise ShardingError(
                f"record {position} ({utterance_id}) has unknown status {status!r}"
            )

    duplicates = _duplicates(seen)
    if duplicates:
        raise ShardingError(
            f"duplicate utterance_id values in the output: {sorted(duplicates)}"
        )

    expected = set(expected_ids)
    observed = set(seen)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ShardingError(
            "utterance_id set mismatch between manifest and output "
            f"(missing={missing[:10]}, unexpected={unexpected[:10]})"
        )

    return {"total": len(records), "completed": completed, "failed": failed}


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


# ---------------------------------------------------------------------------
# Command construction and execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCondition:
    """The ``run_asr_eval`` arguments shared by every shard of one condition."""

    run_asr_eval: Path
    split_id: str
    model_dir: Path
    model: str
    precision: str
    decoding: str
    threads: int
    ort_dylib: Path


def build_command(
    condition: EvalCondition, manifest: Path, output: Path
) -> list[str]:
    return [
        str(condition.run_asr_eval),
        "--manifest",
        str(manifest),
        "--split-id",
        condition.split_id,
        "--model-dir",
        str(condition.model_dir),
        "--model",
        condition.model,
        "--precision",
        condition.precision,
        "--decoding",
        condition.decoding,
        "--threads",
        str(condition.threads),
        "--ort-dylib",
        str(condition.ort_dylib),
        "--output",
        str(output),
    ]


@dataclass
class ShardPlan:
    index: int
    shards: int
    start: int
    end: int
    utterance_ids: list[str]
    manifest_path: Path
    output_path: Path
    log_path: Path
    manifest_state: str = "created"
    action: str = ACTION_RUN
    detail: str = ""
    returncode: int | None = None

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        return shard_label(self.index, self.shards)


@dataclass
class ShardOutcome:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str], Path], ShardOutcome]


def run_shard_subprocess(command: Sequence[str], log_path: Path) -> ShardOutcome:
    """Default runner: one ``run_asr_eval`` process, all output captured to a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(  # noqa: S603 - command is built from validated CLI args
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(
        "\n".join(
            [
                f"$ {shlex.join(str(part) for part in command)}",
                f"exit_code: {completed.returncode}",
                "--- stdout ---",
                completed.stdout or "",
                "--- stderr ---",
                completed.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    return ShardOutcome(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def plan_shards(
    manifest_path: Path,
    document: dict[str, Any],
    shards: int,
    output: Path,
) -> list[ShardPlan]:
    """Materializes the shard manifests and resolves each shard's resume action."""
    utterance_ids = manifest_utterance_ids(document)
    plans: list[ShardPlan] = []
    for index, (start, end) in enumerate(shard_ranges(len(utterance_ids), shards)):
        plan = ShardPlan(
            index=index,
            shards=shards,
            start=start,
            end=end,
            utterance_ids=utterance_ids[start:end],
            manifest_path=shard_manifest_path(manifest_path, index, shards),
            output_path=shard_output_path(output, index, shards),
            log_path=shard_log_path(output, index, shards),
        )
        plan.manifest_state = write_shard_manifest(
            plan.manifest_path, build_shard_manifest(document, start, end)
        )
        plan.action, plan.detail = classify_existing_output(plan.output_path, plan.count)
        plans.append(plan)

    aborts = [plan for plan in plans if plan.action == ACTION_ABORT]
    if aborts:
        lines = [f"  {plan.label}: {plan.detail}" for plan in aborts]
        raise ShardingError(
            "cannot start: shard outputs exist but are not complete "
            "(run_asr_eval never overwrites, so delete the files below and re-run):\n"
            + "\n".join(lines)
        )
    return plans


def execute_shards(
    plans: Sequence[ShardPlan], condition: EvalCondition, runner: Runner
) -> list[ShardPlan]:
    """Runs every non-skipped shard concurrently. Returns the failed shards."""
    pending = [plan for plan in plans if plan.action == ACTION_RUN]
    if not pending:
        return []

    def execute(plan: ShardPlan) -> ShardPlan:
        command = build_command(condition, plan.manifest_path, plan.output_path)
        outcome = runner(command, plan.log_path)
        plan.returncode = outcome.returncode
        if outcome.returncode != 0:
            plan.detail = (outcome.stderr or outcome.stdout or "").strip()[-2000:]
        return plan

    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        for plan in pool.map(execute, pending):
            _ = plan
    return [plan for plan in pending if plan.returncode != 0]


def verify_shard_outputs(plans: Sequence[ShardPlan]) -> None:
    """Confirms every shard produced exactly its own slice before concatenating.

    A clean exit code is not proof of a complete shard, and attributing a shortfall
    to a specific shard is far more actionable than a total-count mismatch found
    after concatenation.
    """
    problems: list[str] = []
    for plan in plans:
        if not plan.output_path.exists():
            problems.append(
                f"{plan.label}: exited 0 but wrote no output at {plan.output_path}"
            )
            continue
        try:
            records = read_jsonl(plan.output_path)
        except ShardingError as error:
            problems.append(f"{plan.label}: {error}")
            continue
        if len(records) != plan.count:
            problems.append(
                f"{plan.label}: {len(records)} records for {plan.count} samples "
                f"({plan.output_path})"
            )
    if problems:
        raise ShardingError(
            "shard outputs are incomplete after a clean exit:\n"
            + "\n".join(f"  {problem}" for problem in problems)
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Shard a RunnerManifestV1 manifest and run K parallel run_asr_eval "
            "processes, then concatenate and verify the shard outputs."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--decoding", default="greedy")
    parser.add_argument("--threads-per-shard", type=int, required=True)
    parser.add_argument("--ort-dylib", type=Path, required=True)
    parser.add_argument("--run-asr-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--delete-shard-outputs",
        action="store_true",
        help=(
            "delete the per-shard JSONL outputs and logs after a successful "
            "verification (default: keep them for audit)"
        ),
    )
    return parser


def load_manifest_document(path: Path, split_id: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ShardingError(f"failed to read manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ShardingError(f"manifest {path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ShardingError(f"manifest {path} is not a JSON object")
    if document.get("split_id") != split_id:
        raise ShardingError(
            f"split ID mismatch: --split-id {split_id!r} but manifest contains "
            f"{document.get('split_id')!r}"
        )
    return document


def main(argv: Sequence[str] | None = None, runner: Runner | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    runner = runner or run_shard_subprocess

    manifest_path = arguments.manifest.resolve()
    output = arguments.output.resolve()

    try:
        if output.exists():
            raise ShardingError(
                f"--output already exists: {output} "
                "(delete it or choose another path; this tool never overwrites results)"
            )
        if arguments.threads_per_shard < 1:
            raise ShardingError("--threads-per-shard must be a positive integer")

        document = load_manifest_document(manifest_path, arguments.split_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        plans = plan_shards(manifest_path, document, arguments.shards, output)

        condition = EvalCondition(
            run_asr_eval=arguments.run_asr_eval,
            split_id=arguments.split_id,
            model_dir=arguments.model_dir,
            model=arguments.model,
            precision=arguments.precision,
            decoding=arguments.decoding,
            threads=arguments.threads_per_shard,
            ort_dylib=arguments.ort_dylib,
        )

        for plan in plans:
            print(
                f"[{plan.label}] samples {plan.start}..{plan.end} "
                f"({plan.count}) manifest={plan.manifest_state} action={plan.action}",
                file=sys.stderr,
            )

        failures = execute_shards(plans, condition, runner)
        if failures:
            for plan in failures:
                print(
                    f"[{plan.label}] FAILED exit={plan.returncode} log={plan.log_path}\n"
                    f"{plan.detail}",
                    file=sys.stderr,
                )
            print(
                "shard failures: "
                + ", ".join(f"{plan.label}(exit={plan.returncode})" for plan in failures),
                file=sys.stderr,
            )
            return 1

        verify_shard_outputs(plans)
        expected_ids = [
            utterance_id for plan in plans for utterance_id in plan.utterance_ids
        ]
        concatenate_jsonl([plan.output_path for plan in plans], output)
        tally = verify_records(read_jsonl(output), expected_ids)

        kept = not arguments.delete_shard_outputs
        if not kept:
            for plan in plans:
                plan.output_path.unlink(missing_ok=True)
                plan.log_path.unlink(missing_ok=True)

        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "split_id": arguments.split_id,
                    "shards": arguments.shards,
                    "total": tally["total"],
                    "completed": tally["completed"],
                    "failed": tally["failed"],
                    "output": str(output),
                    "shard_outputs_kept": kept,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except ShardingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
