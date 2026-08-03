"""Verify the publishable Supertonic 3 ONNX Q4 distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


SOURCE_REPOSITORY = "Supertone/supertonic-3"
SOURCE_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
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
UNMODIFIED_INTEGRITY = {
    "onnx/duration_predictor.onnx": (3_700_147, "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db"),
    "onnx/text_encoder.onnx": (36_416_150, "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff"),
    "onnx/tts.json": (8_253, "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09"),
    "onnx/unicode_indexer.json": (277_676, "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f"),
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
FORBIDDEN_PUBLIC_CLAIMS = ("RTF", "faster", "speedup", "高速化")
LOCAL_PATH_PATTERN = re.compile(r"(?i)(?:[a-z]:[\\/]users[\\/]|/users/|/home/|appdata[\\/])")


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attributes(node: onnx.NodeProto) -> dict[str, int]:
    return {
        attribute.name: int(attribute.i)
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.INT
    }


def validate_model_graph(name: str, model: onnx.ModelProto) -> None:
    counts = Counter(node.op_type for node in model.graph.node)
    if name in {"onnx/duration_predictor.onnx", "onnx/text_encoder.onnx"}:
        if counts["MatMulNBits"] or counts["GatherBlockQuantized"]:
            raise VerificationError(f"{name} must remain FP32")
        return

    expected_q4 = 95 if name == "onnx/vector_estimator.onnx" else 18
    if counts["MatMulNBits"] != expected_q4:
        raise VerificationError(
            f"{name}: expected {expected_q4} MatMulNBits nodes, found {counts['MatMulNBits']}"
        )
    invalid = []
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attributes = _attributes(node)
        if attributes.get("bits") != 4 or attributes.get("block_size") != 16:
            invalid.append(node.name)
    if invalid:
        raise VerificationError(f"{name}: invalid Q4 attributes on {invalid}")
    fp32_names = {
        node.name for node in model.graph.node if node.op_type in {"Conv", "Gemm", "MatMul"}
    }
    expected_fp32 = (
        {VECTOR_FINAL_FP32_NODE}
        if name == "onnx/vector_estimator.onnx"
        else VOCODER_D_FP32_NODES
    )
    missing = expected_fp32 - fp32_names
    if missing:
        raise VerificationError(f"{name}: missing FP32 boundary nodes {sorted(missing)}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON in {path}: {error}") from error


def _validate_files(model_dir: Path) -> None:
    missing = [name for name in PAYLOAD_FILES if not (model_dir / name).is_file()]
    empty = [
        name
        for name in PAYLOAD_FILES
        if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0
    ]
    if missing or empty:
        raise VerificationError(f"missing={missing}; empty={empty}")
    for name, (expected_size, expected_hash) in UNMODIFIED_INTEGRITY.items():
        path = model_dir / name
        if path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise VerificationError(f"unmodified upstream file changed: {name}")


def _validate_documents(model_dir: Path) -> None:
    license_text = (model_dir / "LICENSE").read_text(encoding="utf-8")
    if "BigScience Open RAIL-M" not in license_text or "Attachment A" not in license_text:
        raise VerificationError("LICENSE is not the complete upstream OpenRAIL-M license")
    model_card = (model_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
    for marker in (SOURCE_REPOSITORY, SOURCE_REVISION, "Unofficial quantized derivative"):
        if marker not in model_card:
            raise VerificationError(f"MODEL_CARD.md is missing {marker!r}")
    for claim in FORBIDDEN_PUBLIC_CLAIMS:
        if claim.casefold() in model_card.casefold():
            raise VerificationError(f"MODEL_CARD.md contains forbidden performance claim {claim!r}")
    modifications = (model_dir / "MODIFICATIONS.md").read_text(encoding="utf-8")
    for marker in (
        "duration_predictor.onnx: unchanged FP32",
        "text_encoder.onnx: unchanged FP32",
        "vector_estimator.onnx: modified",
        "vocoder.onnx: modified",
    ):
        if marker not in modifications:
            raise VerificationError(f"MODIFICATIONS.md is missing {marker!r}")
    for name in (
        "MODEL_CARD.md",
        "THIRD_PARTY_NOTICES.md",
        "MODIFICATIONS.md",
        "build-metadata.json",
        "quantization-report.json",
    ):
        if LOCAL_PATH_PATTERN.search((model_dir / name).read_text(encoding="utf-8")):
            raise VerificationError(f"local path leaked into {name}")


def _validate_metadata(model_dir: Path) -> None:
    metadata = _read_json(model_dir / "build-metadata.json")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "license": "BigScience Open RAIL-M",
    }
    if metadata.get("source") != expected_source:
        raise VerificationError("build metadata source contract changed")
    distribution = metadata.get("distribution", {})
    if distribution.get("unmodified_fp32_models") != [
        "onnx/duration_predictor.onnx",
        "onnx/text_encoder.onnx",
    ]:
        raise VerificationError("FP32 model contract changed")
    if distribution.get("modified_q4_models") != [
        "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx",
    ]:
        raise VerificationError("Q4 model contract changed")


def _validate_models(model_dir: Path) -> None:
    for name in ONNX_MODEL_FILES:
        path = model_dir / name
        try:
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            validate_model_graph(name, model)
            if name in {"onnx/vector_estimator.onnx", "onnx/vocoder.onnx"}:
                properties = {entry.key: entry.value for entry in model.metadata_props}
                if properties.get("parapper.modified") != "true":
                    raise VerificationError(f"{name}: modification notice is missing")
                if properties.get("parapper.source_revision") != SOURCE_REVISION:
                    raise VerificationError(f"{name}: source revision metadata changed")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except VerificationError:
            raise
        except Exception as error:
            raise VerificationError(f"{name}: ONNX/runtime validation failed: {error}") from error


def _validate_manifest(model_dir: Path) -> None:
    manifest = _read_json(model_dir / "distribution-manifest.json")
    if manifest.get("source") != {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
    }:
        raise VerificationError("distribution manifest source changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
        raise VerificationError("distribution manifest file topology changed")
    expected_lines = []
    for name in sorted(PAYLOAD_FILES):
        path = model_dir / name
        expected = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if files.get(name) != expected:
            raise VerificationError(f"manifest integrity mismatch for {name}")
        expected_lines.append(f"{expected['sha256']}  {name}")
    actual_lines = (model_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if actual_lines != expected_lines:
        raise VerificationError("SHA256SUMS does not match the exact payload")


def verify_distribution(model_dir: Path) -> None:
    model_dir = model_dir.resolve()
    _validate_files(model_dir)
    _validate_documents(model_dir)
    _validate_metadata(model_dir)
    _validate_models(model_dir)
    _validate_manifest(model_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    try:
        verify_distribution(args.model_dir)
    except (VerificationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Verified Supertonic 3 Q4 distribution: {args.model_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
