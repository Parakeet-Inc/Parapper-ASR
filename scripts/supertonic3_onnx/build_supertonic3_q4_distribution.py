"""Build the adopted Supertonic 3 ONNX Q4 distribution.

The duration predictor and text encoder are copied byte-for-byte from the
audited upstream revision. Only the vector estimator and vocoder are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


SOURCE_REPOSITORY = "Supertone/supertonic-3"
SOURCE_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
SOURCE_LICENSE = "BigScience Open RAIL-M"
ONNX_MODEL_FILES = (
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
)
RUNTIME_FILES = ONNX_MODEL_FILES + (
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
)
PUBLICATION_FILES = (
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "MODIFICATIONS.md",
    "build-metadata.json",
    "quantization-report.json",
)
PAYLOAD_FILES = RUNTIME_FILES + PUBLICATION_FILES

SOURCE_FILE_INTEGRITY = {
    "onnx/duration_predictor.onnx": (3_700_147, "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"),
    "onnx/text_encoder.onnx": (36_416_150, "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff"),
    "onnx/tts.json": (8_253, "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09"),
    "onnx/unicode_indexer.json": (277_676, "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f"),
    "onnx/vector_estimator.onnx": (256_534_781, "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c"),
    "onnx/vocoder.onnx": (101_424_195, "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba"),
    "voice_styles/F1.json": (292_046, "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2"),
    "voice_styles/F2.json": (292_423, "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6"),
    "voice_styles/F3.json": (290_794, "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab"),
    "voice_styles/F4.json": (291_808, "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b"),
    "voice_styles/F5.json": (291_479, "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d"),
    "voice_styles/M1.json": (291_748, "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b"),
    "voice_styles/M2.json": (292_055, "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50"),
    "voice_styles/M3.json": (290_198, "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b"),
    "voice_styles/M4.json": (291_522, "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303"),
    "voice_styles/M5.json": (291_469, "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2"),
}

VECTOR_FINAL_FP32_NODE = "/vector_estimator/vector_field/proj_out/net/Conv"
VOCODER_D_FP32_NODES = {
    "/decoder/embed/net/Conv",
    "/decoder/convnext.9/pwconv1/Conv",
    "/decoder/convnext.9/pwconv2/Conv",
    "/decoder/head/layer1/net/Conv",
    "/decoder/head/layer2/Conv",
}
RAW_Q4_SHA256 = {
    "onnx/vector_estimator.onnx": "34ab76f0eda175700235a6d73e9ae3870f4059d392a785941353bc16c23abff5",
    "onnx/vocoder.onnx": "83017fe0a2552c52480b51f88d03e037e1c714778b9e4d47928523dada070d3c",
}


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_transform_kind(name: str) -> str:
    return {
        "onnx/duration_predictor.onnx": "copy_fp32",
        "onnx/text_encoder.onnx": "copy_fp32",
        "onnx/vector_estimator.onnx": "q4_block16",
        "onnx/vocoder.onnx": "q4_block16_vocoder_d",
    }[name]


def reject_unsafe_output(output_dir: Path, source_dir: Path) -> None:
    output_dir = output_dir.resolve()
    source_dir = source_dir.resolve()
    filesystem_root = Path(output_dir.anchor)
    if output_dir in {filesystem_root, Path.cwd().resolve(), source_dir}:
        raise BuildError(f"unsafe output directory: {output_dir}")
    if source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise BuildError("source and output directories must not contain each other")


def verify_source(source_dir: Path) -> None:
    problems = []
    for name, (expected_size, expected_hash) in SOURCE_FILE_INTEGRITY.items():
        path = source_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        if path.stat().st_size != expected_size or sha256(path) != expected_hash:
            problems.append(f"integrity mismatch for {name}")
    license_path = source_dir / "LICENSE"
    if not license_path.is_file():
        problems.append("missing LICENSE")
    elif "BigScience Open RAIL-M" not in license_path.read_text(encoding="utf-8"):
        problems.append("LICENSE is not the audited BigScience Open RAIL-M text")
    if problems:
        raise BuildError("invalid upstream snapshot: " + "; ".join(problems))


def load_quantizer(script_dir: Path):
    path = script_dir / "quantize_supertonic3_onnx.py"
    spec = importlib.util.spec_from_file_location("supertonic3_quantizer", path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load quantizer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_derivative_metadata(model: onnx.ModelProto, component: str) -> None:
    properties = {entry.key: entry.value for entry in model.metadata_props}
    properties.update(
        {
            "parapper.modified": "true",
            "parapper.component": component,
            "parapper.source_repository": SOURCE_REPOSITORY,
            "parapper.source_revision": SOURCE_REVISION,
            "parapper.quantization": "MatMulNBits Q4 asymmetric block_size=16",
            "parapper.derivative_notice": "Unofficial quantized derivative; not endorsed by Supertone",
        }
    )
    helper.set_model_props(model, properties)


def quantize_component(
    quantizer,
    source_path: Path,
    output_path: Path,
    relative_name: str,
) -> dict[str, object]:
    model = onnx.load(str(source_path))
    excluded = (
        {VECTOR_FINAL_FP32_NODE}
        if relative_name == "onnx/vector_estimator.onnx"
        else VOCODER_D_FP32_NODES
    )
    report = quantizer.lower_affine_nodes(model, excluded)
    quantizer.quantize_weights_q4_block16(model, excluded)
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    expected_q4 = 95 if relative_name == "onnx/vector_estimator.onnx" else 18
    if counts["MatMulNBits"] != expected_q4:
        raise BuildError(
            f"{relative_name}: expected {expected_q4} MatMulNBits nodes, "
            f"found {counts['MatMulNBits']}"
        )
    if set(report.excluded_final_layers) != excluded:
        raise BuildError(f"{relative_name}: FP32 exclusion contract changed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path))
    raw_hash = sha256(output_path)
    if raw_hash != RAW_Q4_SHA256[relative_name]:
        raise BuildError(
            f"{relative_name}: regenerated raw Q4 SHA-256 {raw_hash} does not "
            f"match adopted artifact {RAW_Q4_SHA256[relative_name]}"
        )
    set_derivative_metadata(model, Path(relative_name).stem)
    onnx.save_model(model, str(output_path))
    onnx.checker.check_model(onnx.load(str(output_path)))
    return {
        "source_sha256": sha256(source_path),
        "raw_q4_sha256": raw_hash,
        "distributed_sha256": sha256(output_path),
        "distributed_bytes": output_path.stat().st_size,
        "matmul_nbits": counts["MatMulNBits"],
        "bits": 4,
        "block_size": 16,
        "symmetric": False,
        "fp32_nodes": sorted(excluded),
        "op_counts": dict(sorted(counts.items())),
    }


def write_manifest(output_dir: Path) -> None:
    entries = {
        name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
        for name in PAYLOAD_FILES
    }
    manifest = {
        "schema_version": 1,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "files": entries,
    }
    (output_dir / "distribution-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [f"{entries[name]['sha256']}  {name}" for name in sorted(entries)]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def build_distribution(source_dir: Path, output_dir: Path, *, force: bool) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    reject_unsafe_output(output_dir, source_dir)
    verify_source(source_dir)
    if output_dir.exists():
        if not force:
            raise BuildError(f"output exists; pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for name in RUNTIME_FILES:
        if name in {"onnx/vector_estimator.onnx", "onnx/vocoder.onnx"}:
            continue
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / name, destination)

    script_dir = Path(__file__).resolve().parent
    quantizer = load_quantizer(script_dir)
    model_reports = {}
    for name in ("onnx/vector_estimator.onnx", "onnx/vocoder.onnx"):
        model_reports[name] = quantize_component(
            quantizer, source_dir / name, output_dir / name, name
        )

    shutil.copy2(source_dir / "LICENSE", output_dir / "LICENSE")
    assets = script_dir / "assets"
    for source_name, destination_name in (
        ("supertonic3-q4-model-card.md", "MODEL_CARD.md"),
        ("supertonic3-q4-third-party-notices.md", "THIRD_PARTY_NOTICES.md"),
        ("supertonic3-q4-modifications.md", "MODIFICATIONS.md"),
    ):
        shutil.copy2(assets / source_name, output_dir / destination_name)

    metadata = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
        },
        "distribution": {
            "name": "Supertonic 3 ONNX Q4",
            "unmodified_fp32_models": [
                "onnx/duration_predictor.onnx",
                "onnx/text_encoder.onnx",
            ],
            "modified_q4_models": [
                "onnx/vector_estimator.onnx",
                "onnx/vocoder.onnx",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "command": [
            "python",
            "build_supertonic3_q4_distribution.py",
            "<SOURCE_DIR>",
            "<OUTPUT_DIR>",
        ],
    }
    (output_dir / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    quantization_report = {
        "schema_version": 1,
        "source_revision": SOURCE_REVISION,
        "models": model_reports,
    }
    (output_dir / "quantization-report.json").write_text(
        json.dumps(quantization_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_manifest(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        build_distribution(args.source_dir, args.output_dir, force=args.force)
    except (BuildError, OSError, onnx.checker.ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Built Supertonic 3 Q4 distribution: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
