"""Consolidate the per-tensor external-data export into single-file layout.

Result (onnx-asr layout):
  encoder-model.onnx + encoder-model.onnx_data
  decoder_joint-model.onnx            (embedded weights)
  model.onnx + model.onnx_data        (CTC branch)
Removes the per-tensor external files and the *_tdt intermediate names,
then rewrites export-metadata.json with final hashes.
"""

import hashlib
import json
import sys
from pathlib import Path

import onnx


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    out = Path(sys.argv[1])
    keep = {"vocab.txt", "export-metadata.json"}

    # Load fully into memory before touching any files on disk.
    encoder = onnx.load(str(out / "encoder-model_tdt.onnx"), load_external_data=True)
    decoder_joint = onnx.load(
        str(out / "decoder_joint-model_tdt.onnx"), load_external_data=True
    )
    ctc = onnx.load(str(out / "model.onnx"), load_external_data=True)

    # Remove every old file except the keep-list so stale per-tensor blobs
    # cannot shadow the consolidated data files.
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in keep:
            path.unlink()

    onnx.save(
        encoder,
        str(out / "encoder-model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder-model.onnx_data",
    )
    onnx.save(decoder_joint, str(out / "decoder_joint-model.onnx"))
    onnx.save(
        ctc,
        str(out / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
    )
    print("consolidated", flush=True)

    import onnxruntime as ort

    for name, outputs in [
        ("encoder-model.onnx", ["outputs", "encoded_lengths"]),
        ("decoder_joint-model.onnx", ["outputs"]),
        ("model.onnx", ["logprobs"]),
    ]:
        session = ort.InferenceSession(
            str(out / name), providers=["CPUExecutionProvider"]
        )
        got = [o.name for o in session.get_outputs()]
        assert all(o in got for o in outputs), (name, got)
        print(f"session ok: {name} outputs={got}", flush=True)
        del session

    metadata_path = out / "export-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["consolidation"] = "single .onnx_data per graph via onnx.save all_tensors_to_one_file"
    metadata["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "export-metadata.json"
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("metadata updated", flush=True)


if __name__ == "__main__":
    main()
