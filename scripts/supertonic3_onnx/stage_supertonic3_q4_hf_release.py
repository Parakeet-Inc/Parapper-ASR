"""Stage one verified Supertonic 3 Q4 Hugging Face upload folder."""

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


RUNTIME_FILES = (
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
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
    "distribution-manifest.json",
    "SHA256SUMS",
)
CORE_DISTRIBUTION_FILES = RUNTIME_FILES + PUBLICATION_FILES
RELEASE_TOOL_FILES = (
    "build_supertonic3_q4_distribution.py",
    "quantize_supertonic3_onnx.py",
    "requirements-supertonic3-q4.txt",
    "stage_supertonic3_q4_hf_release.py",
    "verify_supertonic3_q4_distribution.py",
)
RELEASE_ASSET_FILES = (
    "supertonic3-q4-model-card.md",
    "supertonic3-q4-third-party-notices.md",
    "supertonic3-q4-modifications.md",
)
HF_CHECKSUMS_NAME = "HF_UPLOAD_SHA256SUMS"


class StagingError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_unsafe_output(output_dir: Path, protected_paths: tuple[Path, ...]) -> None:
    output_dir = output_dir.resolve()
    if output_dir in {Path(output_dir.anchor), Path.cwd().resolve()}:
        raise StagingError(f"unsafe output directory: {output_dir}")
    for protected in protected_paths:
        protected = protected.resolve()
        if (
            output_dir == protected
            or protected in output_dir.parents
            or output_dir in protected.parents
        ):
            raise StagingError(
                f"output and protected path must not contain each other: {protected}"
            )


def validate_candidate_topology(candidate_dir: Path) -> None:
    missing = [
        name for name in CORE_DISTRIBUTION_FILES if not (candidate_dir / name).is_file()
    ]
    if missing:
        raise StagingError("candidate is missing: " + ", ".join(missing))
    try:
        manifest = json.loads(
            (candidate_dir / "distribution-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StagingError(f"cannot read distribution manifest: {error}") from error
    expected = set(CORE_DISTRIBUTION_FILES) - {
        "distribution-manifest.json",
        "SHA256SUMS",
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise StagingError("distribution manifest has an unexpected file topology")


def copy_release_tools(
    tool_source_dir: Path, documentation: Path, output_dir: Path
) -> None:
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
    repository_root = tool_source_dir.parents[1]
    tool_license = repository_root / "LICENSE"
    if not tool_license.is_file():
        raise StagingError(f"release-tool license is missing: {tool_license}")
    shutil.copy2(tool_license, tools_dir / "LICENSE")
    if not documentation.is_file():
        raise StagingError(f"release documentation is missing: {documentation}")
    procedure = documentation.read_text(encoding="utf-8").replace(
        ".\\scripts\\supertonic3_onnx\\", ".\\release-tools\\"
    )
    (tools_dir / "RELEASE_PROCEDURE.md").write_text(
        procedure, encoding="utf-8", newline="\n"
    )


def write_hf_checksums(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != HF_CHECKSUMS_NAME
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files
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
) -> None:
    candidate_dir = candidate_dir.resolve()
    output_dir = output_dir.resolve()
    reject_unsafe_output(
        output_dir,
        (candidate_dir, tool_source_dir.resolve(), documentation.resolve()),
    )
    if output_dir.exists() and not force:
        raise StagingError(f"output exists; pass --force to replace it: {output_dir}")
    verify(candidate_dir)
    validate_candidate_topology(candidate_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".supertonic3-stage-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in CORE_DISTRIBUTION_FILES:
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_dir / name, destination)
        shutil.copy2(candidate_dir / "MODEL_CARD.md", staged / "README.md")
        copy_release_tools(tool_source_dir, documentation, staged)
        write_hf_checksums(staged)
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
    verifier = script_dir / "verify_supertonic3_q4_distribution.py"
    documentation = (
        script_dir.parents[1]
        / "documents/developer/supertonic3-onnx-q4-release.md"
    )

    def verify(model_dir: Path) -> None:
        result = subprocess.run(
            [str(args.python), str(verifier), str(model_dir)], check=False
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
    print(f"Staged Supertonic 3 Q4 Hugging Face folder: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
