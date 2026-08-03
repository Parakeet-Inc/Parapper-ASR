"""Assemble one upload-ready Hugging Face folder for CAT-Translate.

The model payload is verified before and after staging.  The generated folder
is intentionally ignored by Git; the reproducible release tooling copied under
``release-tools/`` remains tracked in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


CORE_PAYLOAD_FILES = (
    "chat_template.jinja",
    "genai_config.json",
    "model_q4.onnx",
    "model_q4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "build-metadata.json",
)
CORE_METADATA_FILES = ("distribution-manifest.json", "SHA256SUMS")
RELEASE_TOOL_FILES = (
    "cat_export_environment.py",
    "export_cat_onnx_variants.ps1",
    "quantize_cat_embedding_gather.py",
    "requirements-cat-onnx.txt",
    "stage_cat_hf_release.py",
    "verify_cat_onnx_distribution.py",
    "verify_cat_source_snapshot.py",
)
RELEASE_ASSET_FILES = (
    "cat-translate-0.8b-q4-k-quant-model-card.md",
    "cat-translate-0.8b-q4-k-quant-third-party-notices.md",
)
HF_CHECKSUMS_NAME = "HF_UPLOAD_SHA256SUMS"


class StagingError(RuntimeError):
    pass


def _reject_unsafe_output(output_dir: Path, protected_paths: tuple[Path, ...]):
    filesystem_root = Path(output_dir.anchor)
    if output_dir == filesystem_root or output_dir == Path.cwd().resolve():
        raise StagingError(f"unsafe output directory: {output_dir}")
    for protected in protected_paths:
        protected = protected.resolve()
        if output_dir == protected or output_dir in protected.parents:
            raise StagingError(
                f"unsafe output directory contains required input {protected}: {output_dir}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_candidate_topology(candidate_dir: Path):
    manifest_path = candidate_dir / "distribution-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StagingError(f"cannot read distribution manifest: {error}") from error

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(CORE_PAYLOAD_FILES):
        raise StagingError(
            "distribution manifest must contain the exact audited CAT payload file set"
        )

    missing = [
        name
        for name in CORE_PAYLOAD_FILES + CORE_METADATA_FILES
        if not (candidate_dir / name).is_file()
    ]
    if missing:
        raise StagingError("candidate is missing required file(s): " + ", ".join(missing))


def _copy_release_tools(tool_source_dir: Path, documentation: Path, output_dir: Path):
    tools_dir = output_dir / "release-tools"
    assets_dir = tools_dir / "assets"
    tools_dir.mkdir()
    assets_dir.mkdir()

    for name in RELEASE_TOOL_FILES:
        source = tool_source_dir / name
        if not source.is_file():
            raise StagingError(f"release tool is missing: {source}")
        shutil.copy2(source, tools_dir / name)

    for name in RELEASE_ASSET_FILES:
        source = tool_source_dir / "assets" / name
        if not source.is_file():
            raise StagingError(f"release asset is missing: {source}")
        shutil.copy2(source, assets_dir / name)

    tool_license = tool_source_dir / "LICENSE"
    if not tool_license.is_file():
        tool_license = tool_source_dir.parents[2] / "LICENSE"
    if not tool_license.is_file():
        raise StagingError(f"Parakeet release-tool license is missing: {tool_license}")
    shutil.copy2(tool_license, tools_dir / "LICENSE")

    if not documentation.is_file():
        raise StagingError(f"release documentation is missing: {documentation}")
    procedure = documentation.read_text(encoding="utf-8")
    procedure = procedure.replace(
        ".\\scripts\\local-translation\\cat-translate\\",
        ".\\release-tools\\",
    ).replace(
        "./scripts/local-translation/cat-translate/",
        "./release-tools/",
    )
    (tools_dir / "RELEASE_PROCEDURE.md").write_text(
        procedure, encoding="utf-8", newline="\n"
    )


def _write_hf_checksums(output_dir: Path):
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != HF_CHECKSUMS_NAME
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files
    ]
    (output_dir / HF_CHECKSUMS_NAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def stage_release(
    candidate_dir: Path,
    output_dir: Path,
    tool_source_dir: Path,
    documentation: Path,
    verify: Callable[[Path], None],
    *,
    force: bool,
):
    candidate_dir = candidate_dir.resolve()
    output_dir = output_dir.resolve()
    _reject_unsafe_output(
        output_dir,
        (candidate_dir, tool_source_dir.resolve(), documentation.resolve()),
    )
    if candidate_dir == output_dir:
        raise StagingError("candidate and output directories must differ")
    if output_dir.exists() and not force:
        raise StagingError(f"output already exists; pass --force to replace it: {output_dir}")

    verify(candidate_dir)
    _validate_candidate_topology(candidate_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cat-stage-", dir=output_dir.parent
    ) as temporary:
        # Keep the temporary path short enough for Windows release-tool names.
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in CORE_PAYLOAD_FILES + CORE_METADATA_FILES:
            shutil.copy2(candidate_dir / name, staged / name)

        # Hugging Face renders README.md as the repository model card.  Keep the
        # canonical MODEL_CARD.md too because it is part of the app contract.
        shutil.copy2(candidate_dir / "MODEL_CARD.md", staged / "README.md")
        _copy_release_tools(tool_source_dir, documentation, staged)
        _write_hf_checksums(staged)
        verify(staged)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staged.replace(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    verifier = script_dir / "verify_cat_onnx_distribution.py"
    bundled_documentation = script_dir / "RELEASE_PROCEDURE.md"
    documentation = (
        bundled_documentation
        if bundled_documentation.is_file()
        else script_dir.parents[2]
        / "documents/developer/cat-translate-onnx-release.md"
    )

    def verify(model_dir: Path):
        result = subprocess.run(
            [str(args.python), str(verifier), str(model_dir)],
            check=False,
        )
        if result.returncode != 0:
            raise StagingError(f"distribution verification failed: {model_dir}")

    try:
        stage_release(
            args.candidate_dir,
            args.output_dir,
            script_dir,
            documentation,
            verify,
            force=args.force,
        )
    except (OSError, StagingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Hugging Face upload folder staged and verified: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
