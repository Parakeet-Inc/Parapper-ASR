"""Fail unless the CAT ONNX export is running in the pinned environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys


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


def main() -> int:
    python_version = platform.python_version()
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python 3.12.x is required, found {python_version}")

    installed = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"missing package: {package}=={expected}")
            continue
        installed[package] = actual
        if actual != expected:
            problems.append(f"{package}: expected {expected}, found {actual}")

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {"python": python_version, "packages": installed},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
