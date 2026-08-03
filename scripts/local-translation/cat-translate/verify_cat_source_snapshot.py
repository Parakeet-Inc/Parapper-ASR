"""Verify the exact CAT-Translate source snapshot used for the v0.4.0 export."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


SOURCE_REVISION = "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9"
EXPECTED_SOURCE_FILES = {
    "LICENSE": (
        1_074,
        "b117dfdeb28b464adf227207c9129bdd6a1ec9de5852a01bd0ffaa6a7ab0d4f0",
    ),
    "README.md": (
        10_706,
        "65b7fd84f5509f2fc0428d595e9c6999ee06249dc3f285be6f2ac75e3cfbf765",
    ),
    "config.json": (
        691,
        "aac53a6493a42cf3467bf0179cbdcc37cfe9ef4a6cf11c846def55582b9125de",
    ),
    "model.safetensors": (
        1_586_121_792,
        "3375cdd5cb11dbc5036df4902cf20afbc5ffa560f51bb138bd4ffa8e8c0b10f9",
    ),
    "special_tokens_map.json": (
        968,
        "30bf8256f9a1eb3287af2a9b7940465e29a38aad4a459a3e773bb3a14bd34c0f",
    ),
    "tokenizer.json": (
        6_724_259,
        "26df73eb8312c8bc571cb27123198be013ff591c5d9b683e16ccb450ac3426de",
    ),
    "tokenizer.model": (
        1_831_879,
        "008293028e1a9d9a1038d9b63d989a2319797dfeaa03f171093a57b33a3a8277",
    ),
    "tokenizer_config.json": (
        6_449,
        "76b4610d6667bb74335ad383d79362cb43a27ac7aa51ae84942cd63b0cb60b97",
    ),
}


class SourceVerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(
    source_dir: Path,
    expected_files: dict[str, tuple[int, str]] = EXPECTED_SOURCE_FILES,
):
    source_dir = source_dir.resolve()
    if source_dir.name != SOURCE_REVISION:
        raise SourceVerificationError(
            f"source directory must be the pinned snapshot {SOURCE_REVISION}"
        )

    actual_names = {path.name for path in source_dir.iterdir() if path.is_file()}
    expected_names = set(expected_files)
    missing_names = expected_names - actual_names
    if missing_names:
        raise SourceVerificationError(
            "source snapshot is missing consumed file(s): "
            f"{sorted(missing_names)}"
        )

    for name, (expected_size, expected_hash) in expected_files.items():
        path = source_dir / name
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SourceVerificationError(
                f"{name} size mismatch: expected {expected_size}, found {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise SourceVerificationError(
                f"{name} SHA-256 mismatch: expected {expected_hash}, found {actual_hash}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    try:
        verify_snapshot(args.source_dir)
    except (OSError, SourceVerificationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"source snapshot verified: {args.source_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
