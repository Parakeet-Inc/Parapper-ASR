from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from run_sharded_asr_eval import (  # noqa: E402
    ACTION_ABORT,
    ACTION_RUN,
    ACTION_SKIP,
    EvalCondition,
    ShardingError,
    ShardOutcome,
    build_command,
    build_shard_manifest,
    classify_existing_output,
    concatenate_jsonl,
    main,
    manifest_utterance_ids,
    plan_shards,
    read_jsonl,
    shard_label,
    shard_manifest_path,
    shard_output_path,
    shard_ranges,
    verify_records,
    write_shard_manifest,
)


def _sample(index: int) -> dict[str, Any]:
    """A sample carrying keys the Rust struct models plus keys it ignores."""
    return {
        "utterance_id": f"common_voice_ja_{index:03d}",
        "source": {"relative_path": f"{index}.mp3", "sha256": "0" * 64},
        "speaker_id": f"spk-{index % 3}",
        "sentence_id": f"sent-{index}",
        "selection_hash": f"{index:064x}",
        "audio": {
            "relative_path": f"wav/{index:03d}.wav",
            "sha256": "1" * 64,
            "duration_samples": 16000 + index,
        },
        "reference": {"raw": f"参照{index}", "normalized": f"参照{index}"},
    }


def _document(count: int, split_id: str = "split-v1") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split_id": split_id,
        "dataset": {
            "id": "common_voice_ja",
            "release": "26.0",
            "source_split": "dev",
            "language": "ja",
        },
        "selection": {"strategy": "hash", "seed": 7},
        "normalization": {"id": "identity_smoke", "version": "1"},
        "audio_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
        "samples": [_sample(index) for index in range(count)],
        "provenance": {"generated_by": "prepare_eval", "generated_at": "2026-08-17"},
    }


def _completed(utterance_id: str, hypothesis: str = "仮説") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed",
        "utterance_id": utterance_id,
        "reference": "参照",
        "hypothesis": hypothesis,
        "duration_samples": 16000,
        "inference_elapsed_ms": 12.5,
    }


def _failed(utterance_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "utterance_id": utterance_id,
        "stage": "audio",
        "message": "missing wav",
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Shard splitting
# ---------------------------------------------------------------------------


def test_shard_ranges_are_contiguous_and_differ_by_at_most_one_sample() -> None:
    assert shard_ranges(8, 2) == [(0, 4), (4, 8)]
    # 1000 / 6 -> four shards of 167 then two of 166; the remainder goes first.
    ranges = shard_ranges(1000, 6)
    assert [end - start for start, end in ranges] == [167, 167, 167, 167, 166, 166]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 1000
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    # A single shard degenerates to the whole manifest.
    assert shard_ranges(3, 1) == [(0, 3)]


def test_shard_ranges_reject_empty_shards_because_run_asr_eval_rejects_them() -> None:
    with pytest.raises(ShardingError, match="exceeds the 3 manifest samples"):
        shard_ranges(3, 4)
    with pytest.raises(ShardingError, match="positive integer"):
        shard_ranges(3, 0)
    with pytest.raises(ShardingError, match="no samples"):
        shard_ranges(0, 1)


def test_shard_naming_is_deterministic_and_zero_padded_to_sort_in_shard_order() -> None:
    assert shard_label(0, 6) == "shard-0-of-6"
    assert shard_label(3, 12) == "shard-03-of-12"
    labels = [shard_label(index, 12) for index in range(12)]
    assert labels == sorted(labels)

    manifest = Path("/data/split/manifest.json")
    assert shard_manifest_path(manifest, 1, 2) == Path(
        "/data/split/manifest.shard-1-of-2.json"
    )
    # Shard manifests MUST sit beside the source manifest: run_asr_eval resolves
    # audio.relative_path against the manifest file's parent directory.
    assert shard_manifest_path(manifest, 1, 2).parent == manifest.parent

    output = Path("/results/cond/parakeet.jsonl")
    assert shard_output_path(output, 0, 2) == Path(
        "/results/cond/parakeet.jsonl.shard-0-of-2.jsonl"
    )


def test_build_shard_manifest_preserves_every_unknown_field_verbatim() -> None:
    document = _document(5)
    shard = build_shard_manifest(document, 1, 3)

    assert list(shard.keys()) == list(document.keys())
    assert shard["selection"] == document["selection"]
    assert shard["provenance"] == document["provenance"]
    assert shard["split_id"] == "split-v1"
    assert [sample["utterance_id"] for sample in shard["samples"]] == [
        "common_voice_ja_001",
        "common_voice_ja_002",
    ]
    # Per-sample extras that the Rust struct does not model survive untouched.
    assert shard["samples"][0] == document["samples"][1]
    assert shard["samples"][0]["selection_hash"] == document["samples"][1]["selection_hash"]
    # The source document is not mutated.
    assert len(document["samples"]) == 5


def test_concatenating_all_shard_manifests_reproduces_the_source_sample_list() -> None:
    document = _document(7)
    rebuilt: list[dict[str, Any]] = []
    for start, end in shard_ranges(7, 3):
        rebuilt.extend(build_shard_manifest(document, start, end)["samples"])
    assert rebuilt == document["samples"]


def test_manifest_utterance_ids_reject_duplicates() -> None:
    document = _document(3)
    assert manifest_utterance_ids(document) == [
        "common_voice_ja_000",
        "common_voice_ja_001",
        "common_voice_ja_002",
    ]
    document["samples"][2]["utterance_id"] = "common_voice_ja_000"
    with pytest.raises(ShardingError, match="duplicate utterance_id"):
        manifest_utterance_ids(document)


def test_write_shard_manifest_reuses_identical_content_and_aborts_on_a_mismatch(
    tmp_path: Path,
) -> None:
    document = _document(4)
    target = tmp_path / "manifest.shard-0-of-2.json"

    assert write_shard_manifest(target, build_shard_manifest(document, 0, 2)) == "created"
    assert write_shard_manifest(target, build_shard_manifest(document, 0, 2)) == "reused"
    # A byte-different but JSON-equal file still counts as identical content.
    target.write_text(
        json.dumps(build_shard_manifest(document, 0, 2), ensure_ascii=False),
        encoding="utf-8",
    )
    assert write_shard_manifest(target, build_shard_manifest(document, 0, 2)) == "reused"

    with pytest.raises(ShardingError, match="different content"):
        write_shard_manifest(target, build_shard_manifest(document, 2, 4))


# ---------------------------------------------------------------------------
# Resume / skip decisions
# ---------------------------------------------------------------------------


def test_classify_existing_output_runs_skips_or_aborts(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jsonl"
    assert classify_existing_output(missing, 4)[0] == ACTION_RUN

    complete = _write_jsonl(
        tmp_path / "complete.jsonl", [_completed(f"u-{i}") for i in range(4)]
    )
    assert classify_existing_output(complete, 4)[0] == ACTION_SKIP

    partial = _write_jsonl(
        tmp_path / "partial.jsonl", [_completed(f"u-{i}") for i in range(2)]
    )
    action, detail = classify_existing_output(partial, 4)
    assert action == ACTION_ABORT
    assert "2 records" in detail
    assert str(partial) in detail

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text(json.dumps(_completed("u-0")) + '\n{"status": "comp',
        encoding="utf-8")
    assert classify_existing_output(truncated, 2)[0] == ACTION_ABORT


def test_plan_shards_marks_a_complete_shard_for_skip_and_aborts_on_a_partial_one(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    document = _document(6)
    manifest_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "results" / "cond.jsonl"

    plans = plan_shards(manifest_path, document, 3, output)
    assert [plan.count for plan in plans] == [2, 2, 2]
    assert [plan.action for plan in plans] == [ACTION_RUN] * 3
    assert all(plan.manifest_path.exists() for plan in plans)
    assert all(plan.manifest_state == "created" for plan in plans)

    # A complete shard output turns into a skip on the next plan.
    _write_jsonl(plans[1].output_path, [_completed(u) for u in plans[1].utterance_ids])
    replanned = plan_shards(manifest_path, document, 3, output)
    assert [plan.action for plan in replanned] == [ACTION_RUN, ACTION_SKIP, ACTION_RUN]
    assert all(plan.manifest_state == "reused" for plan in replanned)

    # A partial shard output aborts and names the file to delete.
    _write_jsonl(plans[2].output_path, [_completed(plans[2].utterance_ids[0])])
    with pytest.raises(ShardingError) as error:
        plan_shards(manifest_path, document, 3, output)
    assert str(plans[2].output_path) in str(error.value)
    assert "run_asr_eval never overwrites" in str(error.value)


# ---------------------------------------------------------------------------
# Concatenation + verification
# ---------------------------------------------------------------------------


def test_concatenate_jsonl_preserves_shard_order_and_terminates_every_line(
    tmp_path: Path,
) -> None:
    first = _write_jsonl(tmp_path / "a.jsonl", [_completed("u-0"), _completed("u-1")])
    # A shard whose writer stopped without a trailing newline must not fuse records.
    second = tmp_path / "b.jsonl"
    second.write_text(json.dumps(_completed("u-2"), ensure_ascii=False), encoding="utf-8")

    destination = tmp_path / "out" / "final.jsonl"
    assert concatenate_jsonl([first, second], destination) == 3
    assert [record["utterance_id"] for record in read_jsonl(destination)] == [
        "u-0",
        "u-1",
        "u-2",
    ]
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_verify_records_counts_completed_and_failed() -> None:
    expected = ["u-0", "u-1", "u-2"]
    records = [_completed("u-0"), _failed("u-1"), _completed("u-2")]
    assert verify_records(records, expected) == {"total": 3, "completed": 2, "failed": 1}


def test_verify_records_rejects_count_duplicate_and_set_mismatches() -> None:
    expected = ["u-0", "u-1", "u-2"]

    with pytest.raises(ShardingError, match="record count mismatch"):
        verify_records([_completed("u-0")], expected)

    with pytest.raises(ShardingError, match="duplicate utterance_id"):
        verify_records(
            [_completed("u-0"), _completed("u-1"), _completed("u-1")], expected
        )

    with pytest.raises(ShardingError, match="utterance_id set mismatch"):
        verify_records(
            [_completed("u-0"), _completed("u-1"), _completed("u-9")], expected
        )

    with pytest.raises(ShardingError, match="unknown status"):
        verify_records(
            [_completed("u-0"), _completed("u-1"), {"utterance_id": "u-2"}], expected
        )

    with pytest.raises(ShardingError, match="no usable 'utterance_id'"):
        verify_records(
            [_completed("u-0"), _completed("u-1"), {"status": "completed"}], expected
        )


def test_read_jsonl_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(_completed("u-0")) + "\n" + "not json\n", encoding="utf-8"
    )
    with pytest.raises(ShardingError, match=r"broken\.jsonl:2 is not valid JSON"):
        read_jsonl(path)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_build_command_matches_the_run_asr_eval_cli_contract() -> None:
    condition = EvalCondition(
        run_asr_eval=Path("target/release/run_asr_eval.exe"),
        split_id="cv26-ja-dev-smoke-8-v1",
        model_dir=Path("C:/models/parakeet"),
        model="nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
        precision="int8",
        decoding="greedy",
        threads=2,
        ort_dylib=Path("target/release/onnxruntime.dll"),
    )
    command = build_command(
        condition, Path("m.shard-0-of-2.json"), Path("out.jsonl.shard-0-of-2.jsonl")
    )
    # Paths are rendered with the host separator, so compare through Path.
    assert command[0] == str(Path("target/release/run_asr_eval.exe"))
    flags = dict(zip(command[1::2], command[2::2]))
    assert flags == {
        "--manifest": "m.shard-0-of-2.json",
        "--split-id": "cv26-ja-dev-smoke-8-v1",
        "--model-dir": str(Path("C:/models/parakeet")),
        "--model": "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
        "--precision": "int8",
        "--decoding": "greedy",
        "--threads": "2",
        "--ort-dylib": str(Path("target/release/onnxruntime.dll")),
        "--output": "out.jsonl.shard-0-of-2.jsonl",
    }
    # Every flag has a value and no flag is repeated.
    assert len(command) == 1 + 2 * len(flags)


# ---------------------------------------------------------------------------
# End-to-end with a faked runner (no Rust binary required)
# ---------------------------------------------------------------------------


class FakeRunner:
    """Stands in for run_asr_eval: reads the shard manifest, writes shard JSONL."""

    def __init__(self, fail_shards: Sequence[str] = (), skip_writes: bool = False):
        self.fail_shards = set(fail_shards)
        self.skip_writes = skip_writes
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str], log_path: Path) -> ShardOutcome:
        self.commands.append(list(command))
        flags = dict(zip(command[1::2], command[2::2]))
        manifest = Path(flags["--manifest"])
        output = Path(flags["--output"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"$ {' '.join(command)}\n", encoding="utf-8")

        if any(label in manifest.name for label in self.fail_shards):
            return ShardOutcome(returncode=1, stderr="Error: synthetic shard failure")

        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["split_id"] == flags["--split-id"]
        if not self.skip_writes:
            _write_jsonl(
                output,
                [
                    _completed(sample["utterance_id"], f"hyp {sample['utterance_id']}")
                    for sample in document["samples"]
                ],
            )
        return ShardOutcome(returncode=0, stdout=json.dumps({"total": len(document["samples"])}))


def _argv(manifest: Path, output: Path, shards: int, extra: Sequence[str] = ()) -> list[str]:
    return [
        "--manifest", str(manifest),
        "--split-id", "split-v1",
        "--shards", str(shards),
        "--model-dir", "C:/models/parakeet",
        "--model", "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
        "--precision", "int8",
        "--decoding", "greedy",
        "--threads-per-shard", "2",
        "--ort-dylib", "onnxruntime.dll",
        "--run-asr-eval", "run_asr_eval.exe",
        "--output", str(output),
        *extra,
    ]


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "split" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_document(8), ensure_ascii=False), encoding="utf-8")
    return path


def test_main_shards_runs_concatenates_and_verifies(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    runner = FakeRunner()

    assert main(_argv(manifest_file, output, 3), runner=runner) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "schema_version": 1,
        "split_id": "split-v1",
        "shards": 3,
        "total": 8,
        "completed": 8,
        "failed": 0,
        "output": str(output.resolve()),
        "shard_outputs_kept": True,
    }
    assert len(runner.commands) == 3
    # Manifest order is preserved across the shard boundary.
    assert [record["utterance_id"] for record in read_jsonl(output)] == [
        f"common_voice_ja_{index:03d}" for index in range(8)
    ]
    # Shard manifests land beside the source manifest; outputs are kept by default.
    assert (manifest_file.parent / "manifest.shard-0-of-3.json").exists()
    assert (output.parent / "cond.jsonl.shard-0-of-3.jsonl").exists()
    assert (output.parent / "cond.jsonl.shard-0-of-3.jsonl.log").exists()


def test_main_deletes_shard_outputs_on_request_but_keeps_shard_manifests(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    assert main(
        _argv(manifest_file, output, 2, ["--delete-shard-outputs"]), runner=FakeRunner()
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["shard_outputs_kept"] is False
    assert not (output.parent / "cond.jsonl.shard-0-of-2.jsonl").exists()
    assert not (output.parent / "cond.jsonl.shard-0-of-2.jsonl.log").exists()
    assert (manifest_file.parent / "manifest.shard-0-of-2.json").exists()
    assert len(read_jsonl(output)) == 8


def test_main_reports_failed_shards_and_leaves_no_final_output(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    runner = FakeRunner(fail_shards=["shard-1-of-2"])

    assert main(_argv(manifest_file, output, 2), runner=runner) == 1

    captured = capsys.readouterr()
    assert "shard-1-of-2(exit=1)" in captured.err
    assert "synthetic shard failure" in captured.err
    assert not output.exists()
    # The surviving shard output is retained so a re-run resumes it.
    assert (output.parent / "cond.jsonl.shard-0-of-2.jsonl").exists()


def test_main_resumes_by_skipping_the_shard_that_already_finished(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    assert main(_argv(manifest_file, output, 2), runner=FakeRunner(fail_shards=["shard-1-of-2"])) == 1
    capsys.readouterr()

    resumed = FakeRunner()
    assert main(_argv(manifest_file, output, 2), runner=resumed) == 0

    # Only the previously failed shard is re-run.
    assert len(resumed.commands) == 1
    assert "shard-1-of-2" in resumed.commands[0][resumed.commands[0].index("--manifest") + 1]
    assert json.loads(capsys.readouterr().out)["total"] == 8


def test_main_refuses_to_start_when_the_final_output_exists(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    _write_jsonl(output, [_completed("u-0")])
    runner = FakeRunner()

    assert main(_argv(manifest_file, output, 2), runner=runner) == 1
    assert "--output already exists" in capsys.readouterr().err
    assert runner.commands == []


def test_main_rejects_a_split_id_that_disagrees_with_the_manifest(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _argv(manifest_file, tmp_path / "results" / "cond.jsonl", 2)
    argv[argv.index("--split-id") + 1] = "other-split"
    runner = FakeRunner()

    assert main(argv, runner=runner) == 1
    assert "split ID mismatch" in capsys.readouterr().err
    assert runner.commands == []


def test_main_fails_verification_when_a_shard_writes_the_wrong_record_count(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"
    # A runner that exits 0 without writing anything: verification must catch the
    # shortfall and attribute it to the specific shards.
    assert main(_argv(manifest_file, output, 2), runner=FakeRunner(skip_writes=True)) == 1
    captured = capsys.readouterr().err
    assert "shard outputs are incomplete after a clean exit" in captured
    assert "shard-0-of-2: exited 0 but wrote no output" in captured
    assert not output.exists()


def test_main_fails_verification_when_a_shard_writes_a_short_output(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "cond.jsonl"

    class ShortRunner(FakeRunner):
        def __call__(self, command: Sequence[str], log_path: Path) -> ShardOutcome:
            outcome = super().__call__(command, log_path)
            flags = dict(zip(command[1::2], command[2::2]))
            target = Path(flags["--output"])
            records = read_jsonl(target)
            _write_jsonl(target, records[:-1])
            return outcome

    assert main(_argv(manifest_file, output, 2), runner=ShortRunner()) == 1
    captured = capsys.readouterr().err
    assert "3 records for 4 samples" in captured
    assert not output.exists()
