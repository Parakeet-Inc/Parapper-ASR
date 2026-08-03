"""Validate the publishable CAT-Translate 0.8B Q4 k_quant distribution.

The default mode verifies an existing distribution manifest and SHA256SUMS.
Pass ``--write-manifest`` immediately after export to create both files, but
only after the runtime files, provenance, license, and adopted Q4 block16
embedding graph contract pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


SOURCE_REPOSITORY = "cyberagent/CAT-Translate-0.8b"
SOURCE_REVISION = "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9"
SOURCE_LICENSE = "MIT"

EXPECTED_PACKAGES = {
    "onnxruntime-genai": "0.14.1",
    "onnxruntime": "1.27.0",
    "onnx": "1.22.0",
    "onnx-ir": "0.2.1",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "torch": "2.12.1+cpu",
    "tokenizers": "0.22.2",
    "sentencepiece": "0.2.1",
}

RUNTIME_FILES = [
    "chat_template.jinja",
    "genai_config.json",
    "model_q4.onnx",
    "model_q4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

PUBLICATION_FILES = [
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "build-metadata.json",
]

PAYLOAD_FILES = RUNTIME_FILES + PUBLICATION_FILES
MANIFEST_NAME = "distribution-manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"

FORBIDDEN_METADATA_KEYS = {
    "api_key",
    "cache",
    "cache_dir",
    "credential",
    "credentials",
    "hf_token",
    "out_dir",
    "password",
    "python_path",
    "secret",
    "source_dir",
    "token",
}

LOCAL_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]users[\\/]|/users/|/home/|appdata[\\/]|\.cache[\\/])"
)
CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*bearer\s+\S+|hf_[a-z0-9]{20,}|"
    r"(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"]?[^\s,'\"]+)"
)
LEAK_SCAN_FILES = [
    "chat_template.jinja",
    "genai_config.json",
    "tokenizer_config.json",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "build-metadata.json",
]


class VerificationError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON in {path.name}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def _validate_no_local_paths_or_credentials(model_dir: Path, metadata: dict[str, Any]):
    violations: list[str] = []
    for key, value in _walk(metadata):
        if key is not None and key.casefold() in FORBIDDEN_METADATA_KEYS:
            violations.append(f"forbidden metadata key {key!r}")
        if isinstance(value, str) and (
            LOCAL_PATH_PATTERN.search(value) or CREDENTIAL_PATTERN.search(value)
        ):
            violations.append(f"sensitive value under {key!r}")

    # tokenizer.json is a source vocabulary and can legitimately contain strings
    # that look like shell fragments. Scan only generated/configuration text where
    # an export path or credential could actually leak.
    for name in LEAK_SCAN_FILES:
        try:
            text = (model_dir / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise VerificationError(f"cannot inspect text distribution file {name}: {error}") from error
        if LOCAL_PATH_PATTERN.search(text) or CREDENTIAL_PATTERN.search(text):
            violations.append(f"sensitive marker in {name}")
        if Path(name).suffix.casefold() == ".json":
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as error:
                raise VerificationError(f"invalid JSON in {name}: {error}") from error
            for _, value in _walk(decoded):
                if isinstance(value, str) and (
                    LOCAL_PATH_PATTERN.search(value) or CREDENTIAL_PATTERN.search(value)
                ):
                    violations.append(f"sensitive JSON value in {name}")

    if violations:
        raise VerificationError(
            "local path or credential marker found: " + "; ".join(sorted(set(violations)))
        )


def _validate_required_files(model_dir: Path):
    missing = [name for name in PAYLOAD_FILES if not (model_dir / name).is_file()]
    empty = [
        name
        for name in PAYLOAD_FILES
        if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0
    ]
    problems = []
    if missing:
        problems.append("missing required distribution file(s): " + ", ".join(missing))
    if empty:
        problems.append("empty required distribution file(s): " + ", ".join(empty))
    if problems:
        raise VerificationError("; ".join(problems))


def _validate_metadata(metadata: dict[str, Any]):
    if metadata.get("schema_version") != 1:
        raise VerificationError("build-metadata.json schema_version must be 1")

    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
    }
    if metadata.get("source") != expected_source:
        raise VerificationError(
            "source provenance mismatch; expected " + json.dumps(expected_source, sort_keys=True)
        )

    export = metadata.get("export")
    if not isinstance(export, dict):
        raise VerificationError("build metadata must contain an export object")
    expected_export = {
        "variant": "k_quant",
        "precision": "int4",
        "execution_provider": "cpu",
        "embedding": "groupwise_q4_block16",
    }
    mismatches = {
        key: export.get(key)
        for key, expected in expected_export.items()
        if export.get(key) != expected
    }
    if mismatches:
        raise VerificationError(
            "export contract mismatch for CAT publish candidate: "
            + json.dumps(mismatches, sort_keys=True)
        )
    expected_embedding_quantization = {
        "bits": 4,
        "block_size": 16,
        "is_symmetric": False,
        "operator": "GatherBlockQuantized",
        "command": [
            "python",
            "quantize_cat_embedding_gather.py",
            "<INTERMEDIATE_DIR>",
            "<OUT_DIR>",
            "--block-size",
            "16",
        ],
    }
    if export.get("embedding_quantization") != expected_embedding_quantization:
        raise VerificationError(
            "embedding_quantization does not match the adopted asymmetric "
            "Q4 block16 contract"
        )
    expected_command = [
        "python",
        "-m",
        "onnxruntime_genai.models.builder",
        "-i",
        "<SOURCE_DIR>",
        "-o",
        "<INTERMEDIATE_DIR>",
        "-c",
        "<CACHE_DIR>",
        "-p",
        "int4",
        "-e",
        "cpu",
        "--extra_options",
        "filename=model_q4.onnx",
        "hf_token=false",
        "hf_remote=false",
        "int4_algo_config=k_quant",
    ]
    if export.get("command") != expected_command:
        raise VerificationError(
            "export.command does not match the sanitized audited builder command"
        )
    duration = export.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration < 0:
        raise VerificationError("export.duration_seconds must be a non-negative number")

    environment = metadata.get("environment")
    if not isinstance(environment, dict):
        raise VerificationError("build metadata must contain an environment object")
    python_version = environment.get("python")
    if not isinstance(python_version, str) or not python_version.startswith("3.12."):
        raise VerificationError("export must run with pinned Python 3.12.x")
    packages = environment.get("packages")
    if packages != EXPECTED_PACKAGES:
        raise VerificationError(
            "dependency versions do not match the pinned export environment; expected "
            + json.dumps(EXPECTED_PACKAGES, sort_keys=True)
        )


def _validate_publication_documents(model_dir: Path):
    license_text = (model_dir / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text or "CyberAgent AI Lab" not in license_text:
        raise VerificationError(
            "LICENSE does not contain the source MIT license and CyberAgent notice"
        )

    model_card = (model_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
    required_model_card_markers = [SOURCE_REVISION, "k_quant", "Q4 block16"]
    missing_model_card = [
        marker for marker in required_model_card_markers if marker not in model_card
    ]
    if missing_model_card:
        raise VerificationError(
            "MODEL_CARD.md is missing publication contract marker(s): "
            + ", ".join(missing_model_card)
        )

    notices = (model_dir / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    required_notice_markers = [SOURCE_REPOSITORY, SOURCE_REVISION, "MIT"]
    missing_notices = [marker for marker in required_notice_markers if marker not in notices]
    if missing_notices:
        raise VerificationError(
            "THIRD_PARTY_NOTICES.md is missing source/license marker(s): "
            + ", ".join(missing_notices)
        )


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _validate_genai_config(model_dir: Path):
    config = _read_json(model_dir / "genai_config.json")
    model_paths = [value for value in _strings(config) if value.casefold().endswith(".onnx")]
    if "model_q4.onnx" not in model_paths:
        raise VerificationError("genai_config.json must reference model_q4.onnx")
    for value in model_paths:
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or ".." in Path(value).parts:
            raise VerificationError(f"genai_config.json contains non-portable model path: {value}")


def _validate_runtime_model_load(model_dir: Path):
    try:
        import onnxruntime_genai as og
    except ImportError as error:
        raise VerificationError(
            "onnxruntime-genai is required to validate external tensor data"
        ) from error

    try:
        model = og.Model(str(model_dir))
    except Exception as error:
        raise VerificationError(
            f"runtime model rejected model_q4.onnx external tensor data: {error}"
        ) from error
    del model


def _attribute_int(node: Any, name: str, default: int | None = None) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _inspect_graph(model_dir: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:
        raise VerificationError(
            "onnx is required for graph verification; install the pinned requirements"
        ) from error

    try:
        model = onnx.load(str(model_dir / "model_q4.onnx"), load_external_data=False)
    except Exception as error:
        raise VerificationError(f"cannot load model_q4.onnx: {error}") from error

    op_counts = Counter(node.op_type for node in model.graph.node)
    bit_counts = Counter(
        _attribute_int(node, "bits", 4)
        for node in model.graph.node
        if node.op_type == "MatMulNBits"
    )
    expected_bit_counts = Counter({4: 120, 8: 1})
    if bit_counts != expected_bit_counts:
        raise VerificationError(
            "expected MatMulNBits bit layout {4: 120, 8: 1} for k_quant, got "
            + repr(dict(sorted(bit_counts.items())))
        )
    if op_counts["GatherBlockQuantized"] != 1:
        raise VerificationError(
            "expected exactly one Q4 block16 embedding GatherBlockQuantized node"
        )
    if op_counts["Gather"] != 1:
        raise VerificationError(
            f"expected exactly 1 non-embedding Gather node, got {op_counts['Gather']}"
        )

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    embedding_nodes = [
        node for node in model.graph.node if node.op_type == "GatherBlockQuantized"
    ]
    embedding_node = embedding_nodes[0]
    block_size = _attribute_int(embedding_node, "block_size")
    gather_axis = _attribute_int(embedding_node, "gather_axis")
    quantize_axis = _attribute_int(embedding_node, "quantize_axis")
    if block_size != 16:
        raise VerificationError(
            f"Q4 embedding must use block_size=16, got {block_size}"
        )
    if gather_axis != 0 or quantize_axis != 1:
        raise VerificationError(
            "Q4 block16 embedding axes must be gather_axis=0 and quantize_axis=1"
        )
    if len(embedding_node.input) != 4:
        raise VerificationError(
            "Q4 block16 embedding must have weight, indices, scale, and zero-point inputs"
        )

    try:
        embedding = initializers[embedding_node.input[0]]
        scale = initializers[embedding_node.input[2]]
        zero_point = initializers[embedding_node.input[3]]
    except KeyError as error:
        raise VerificationError(
            f"Q4 block16 embedding initializer is missing: {error.args[0]}"
        ) from error

    expected_tensors = [
        ("embedding", embedding, onnx.TensorProto.UINT4, [102400, 1280]),
        ("embedding scale", scale, onnx.TensorProto.FLOAT, [102400, 80]),
        ("embedding zero-point", zero_point, onnx.TensorProto.UINT4, [102400, 80]),
    ]
    for label, tensor, data_type, shape in expected_tensors:
        if tensor.data_type != data_type or list(tensor.dims) != shape:
            raise VerificationError(
                f"{label} contract mismatch; expected type={data_type}, shape={shape}"
            )

    return {
        "mat_mul_n_bits": {str(key): value for key, value in sorted(bit_counts.items())},
        "gather": op_counts["Gather"],
        "gather_block_quantized": op_counts["GatherBlockQuantized"],
        "embedding": {
            "data_type": "UINT4",
            "shape": list(embedding.dims),
            "scale": {
                "data_type": "FLOAT",
                "shape": list(scale.dims),
            },
            "zero_point": {
                "data_type": "UINT4",
                "shape": list(zero_point.dims),
            },
            "block_size": block_size,
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
        },
    }


def _build_manifest(model_dir: Path, metadata: dict[str, Any], graph: dict[str, Any]):
    files = {
        name: {
            "size": (model_dir / name).stat().st_size,
            "sha256": _sha256(model_dir / name),
        }
        for name in PAYLOAD_FILES
    }
    return {
        "schema_version": 1,
        "model": "cat_translate_0_8b_q4_k_quant",
        "source": metadata["source"],
        "export": metadata["export"],
        "graph": graph,
        "files": files,
    }


def _write_manifest_and_checksums(model_dir: Path, manifest: dict[str, Any]):
    manifest_path = model_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_files = PAYLOAD_FILES + [MANIFEST_NAME]
    checksum_lines = [f"{_sha256(model_dir / name)}  {name}" for name in checksum_files]
    (model_dir / CHECKSUMS_NAME).write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"cannot read {CHECKSUMS_NAME}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if not match:
            raise VerificationError(f"invalid {CHECKSUMS_NAME} line {line_number}")
        digest, name = match.groups()
        if name in checksums:
            raise VerificationError(f"duplicate {CHECKSUMS_NAME} entry: {name}")
        checksums[name] = digest
    return checksums


def _verify_manifest_and_checksums(model_dir: Path, expected_manifest: dict[str, Any]):
    manifest_path = model_dir / MANIFEST_NAME
    checksums_path = model_dir / CHECKSUMS_NAME
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise VerificationError(
            f"{MANIFEST_NAME} and {CHECKSUMS_NAME} are required; run --write-manifest after export"
        )
    actual_manifest = _read_json(manifest_path)
    if actual_manifest != expected_manifest:
        raise VerificationError("distribution manifest does not match the current files or graph")

    expected_names = set(PAYLOAD_FILES + [MANIFEST_NAME])
    checksums = _read_checksums(checksums_path)
    if set(checksums) != expected_names:
        missing = sorted(expected_names - set(checksums))
        extra = sorted(set(checksums) - expected_names)
        raise VerificationError(
            f"{CHECKSUMS_NAME} file set mismatch; missing={missing}, extra={extra}"
        )
    mismatches = [
        name for name, expected in checksums.items() if _sha256(model_dir / name) != expected
    ]
    if mismatches:
        raise VerificationError("SHA-256 mismatch: " + ", ".join(sorted(mismatches)))


def validate_distribution(model_dir: Path, write_manifest: bool):
    if write_manifest:
        (model_dir / MANIFEST_NAME).unlink(missing_ok=True)
        (model_dir / CHECKSUMS_NAME).unlink(missing_ok=True)

    _validate_required_files(model_dir)
    metadata = _read_json(model_dir / "build-metadata.json")
    if not isinstance(metadata, dict):
        raise VerificationError("build-metadata.json must contain a JSON object")
    _validate_no_local_paths_or_credentials(model_dir, metadata)
    _validate_metadata(metadata)
    _validate_publication_documents(model_dir)
    _validate_genai_config(model_dir)
    _validate_runtime_model_load(model_dir)
    graph = _inspect_graph(model_dir)
    manifest = _build_manifest(model_dir, metadata, graph)

    if write_manifest:
        _write_manifest_and_checksums(model_dir, manifest)
        print(f"distribution manifest written: {model_dir / MANIFEST_NAME}")
        print(f"checksums written: {model_dir / CHECKSUMS_NAME}")
    else:
        _verify_manifest_and_checksums(model_dir, manifest)
        print(f"distribution verified: {model_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="create distribution-manifest.json and SHA256SUMS after validation",
    )
    args = parser.parse_args()
    try:
        validate_distribution(args.model_dir.resolve(), args.write_manifest)
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
