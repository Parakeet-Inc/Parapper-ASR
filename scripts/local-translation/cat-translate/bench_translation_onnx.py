"""Run the audited CAT-Translate ONNX release case set.

Each positional model is ``label=model_dir``. The default gate uses the 28-case
extended set and sends every case twice through the same loaded model so that a
single successful request cannot hide a broken consecutive-request path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BASE_CASES = [
    ("ja_to_en_greeting", "Japanese", "English", "こんにちは。"),
    ("ja_to_en_settings", "Japanese", "English", "設定画面を開いてください。"),
    ("en_to_ja_greeting", "English", "Japanese", "Hello."),
    ("en_to_ja_help", "English", "Japanese", "Thank you for your help."),
    ("en_to_ja_meeting", "English", "Japanese", "I will join the meeting tomorrow."),
    ("en_to_ja_settings", "English", "Japanese", "Please open the settings screen."),
]

EXTENDED_CASES = BASE_CASES + [
    ("en_to_ja_good_morning", "English", "Japanese", "Good morning."),
    (
        "en_to_ja_save_before_close",
        "English",
        "Japanese",
        "Please save the file before closing the window.",
    ),
    (
        "en_to_ja_time_room",
        "English",
        "Japanese",
        "The meeting starts at 3:30 PM in room 204.",
    ),
    (
        "en_to_ja_server_error",
        "English",
        "Japanese",
        "I could not connect to the server.",
    ),
    (
        "en_to_ja_ui_button",
        "English",
        "Japanese",
        "The blue button on the top right opens the settings menu.",
    ),
    (
        "en_to_ja_local_feature",
        "English",
        "Japanese",
        "This translation feature runs locally on your computer.",
    ),
    (
        "en_to_ja_overwrite",
        "English",
        "Japanese",
        "Do you want to overwrite the existing file?",
    ),
    (
        "en_to_ja_product_name",
        "English",
        "Japanese",
        "Parakeet will start recording when speech is detected.",
    ),
    (
        "en_to_ja_cpu_usage",
        "English",
        "Japanese",
        "CPU usage may increase during the first model load.",
    ),
    (
        "en_to_ja_shortcut",
        "English",
        "Japanese",
        "Use Ctrl+C to copy the selected text.",
    ),
    (
        "en_to_ja_quote",
        "English",
        "Japanese",
        'She said, "I will be there soon."',
    ),
    ("ja_to_en_good_morning", "Japanese", "English", "おはようございます。"),
    (
        "ja_to_en_ui_button",
        "Japanese",
        "English",
        "右上の青いボタンを押してください。",
    ),
    (
        "ja_to_en_local_feature",
        "Japanese",
        "English",
        "この翻訳機能はインターネット接続なしで動作します。",
    ),
    (
        "ja_to_en_save_before_close",
        "Japanese",
        "English",
        "ファイルを保存してからウィンドウを閉じてください。",
    ),
    (
        "ja_to_en_server_error",
        "Japanese",
        "English",
        "サーバーに接続できませんでした。",
    ),
    (
        "ja_to_en_time_room",
        "Japanese",
        "English",
        "会議は午後3時30分に204号室で始まります。",
    ),
    (
        "ja_to_en_overwrite",
        "Japanese",
        "English",
        "既存のファイルを上書きしますか？",
    ),
    (
        "ja_to_en_product_name",
        "Japanese",
        "English",
        "音声が検出されるとParakeetが録音を開始します。",
    ),
    (
        "ja_to_en_cpu_usage",
        "Japanese",
        "English",
        "初回のモデル読み込み中はCPU使用率が上がる場合があります。",
    ),
    (
        "ja_to_en_shortcut",
        "Japanese",
        "English",
        "選択したテキストをコピーするにはCtrl+Cを使います。",
    ),
    (
        "ja_to_en_quote",
        "Japanese",
        "English",
        "彼女は「すぐに行きます」と言いました。",
    ),
]

CASE_SETS = {"base": BASE_CASES, "extended": EXTENDED_CASES}
JA_CHARS = re.compile(r"[぀-ヿ一-鿿]")
ASCII_ALPHA = re.compile(r"[A-Za-z]")


def build_prompt(style: str, source: str, target: str, text: str) -> str:
    if style == "legacy":
        return f"<|system|>Translate from {source} to {target}.</s><|user|>{text}</s><|assistant|>"
    if style == "official-leading-space":
        return (
            f"<|user|>Translate the following {source} text into {target}.\n\n {text}"
            f"</s><|assistant|>"
        )
    return (
        f"<|user|>Translate the following {source} text into {target}.\n\n{text}"
        f"</s><|assistant|>"
    )


def target_language_ok(target: str, output: str) -> bool:
    if target == "Japanese":
        return bool(JA_CHARS.search(output))
    return bool(ASCII_ALPHA.search(output)) and not JA_CHARS.search(output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    og: Any, model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> tuple[str, float]:
    input_ids = tokenizer.encode(prompt).ids
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(input_ids) + max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    start = time.perf_counter()
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)
    while not generator.is_done():
        generator.generate_next_token()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    sequence = generator.get_sequence(0)
    new_tokens = list(sequence[len(input_ids) :])
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del generator
    return text, elapsed_ms


def bench_model(
    og: Any,
    tokenizer_type: Any,
    label: str,
    model_dir: Path,
    cases: list[tuple[str, str, str, str]],
    repeats: int,
    max_new_tokens: int,
    prompt_style: str,
) -> dict[str, Any]:
    print(f"\n=== {label}")
    load_start = time.perf_counter()
    model = og.Model(str(model_dir))
    load_ms = (time.perf_counter() - load_start) * 1000.0
    tokenizer = tokenizer_type.from_file(str(model_dir / "tokenizer.json"))
    print(f"load time: {load_ms:.1f} ms")

    _, warm_ms = generate(
        og,
        model,
        tokenizer,
        build_prompt(prompt_style, *cases[0][1:]),
        max_new_tokens,
    )
    print(f"warmup: {warm_ms:.1f} ms")

    case_results = []
    all_times = []
    for case_id, source, target, text in cases:
        outputs = []
        elapsed_values = []
        target_checks = []
        echo_checks = []
        for _ in range(repeats):
            output, elapsed_ms = generate(
                og,
                model,
                tokenizer,
                build_prompt(prompt_style, source, target, text),
                max_new_tokens,
            )
            outputs.append(output)
            elapsed_values.append(round(elapsed_ms, 1))
            all_times.append(elapsed_ms)
            target_checks.append(target_language_ok(target, output))
            echo_checks.append(output.strip() == text.strip())

        flags = []
        if not all(target_checks):
            flags.append("LANG-FAIL")
        if any(echo_checks):
            flags.append("ECHO")
        if len(set(outputs)) != 1:
            flags.append("UNSTABLE")
        flag_text = f"  [{','.join(flags)}]" if flags else ""
        print(f"  {case_id}: {elapsed_values} ms  {outputs[0]!r}{flag_text}")
        case_results.append(
            {
                "case": case_id,
                "source_language": source,
                "target_language": target,
                "input": text,
                "outputs": outputs,
                "target_language_ok": target_checks,
                "echoed_input": echo_checks,
                "stable_output": len(set(outputs)) == 1,
                "elapsed_ms": elapsed_values,
            }
        )

    language_failures = [
        result["case"] for result in case_results if not all(result["target_language_ok"])
    ]
    manifest_path = model_dir / "distribution-manifest.json"
    summary = {
        "label": label,
        "distribution_manifest_sha256": sha256(manifest_path)
        if manifest_path.is_file()
        else None,
        "load_ms": round(load_ms, 1),
        "warmup_ms": round(warm_ms, 1),
        "mean_ms": round(statistics.mean(all_times), 1),
        "median_ms": round(statistics.median(all_times), 1),
        "language_failures": language_failures,
        "cases": case_results,
    }
    print(
        f"summary: mean {summary['mean_ms']} ms, median {summary['median_ms']} ms, "
        f"language failures: {language_failures or 'none'}"
    )
    del model
    return summary


def export_environment() -> tuple[dict[str, Any], list[str]]:
    from cat_export_environment import EXPECTED_PACKAGES

    versions = {}
    problems = []
    python_version = platform.python_version()
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python 3.12.x is required, found {python_version}")
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
        if versions[package] != expected:
            problems.append(
                f"{package}: expected {expected}, found {versions[package]}"
            )
    return {"python": python_version, "packages": versions}, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="label=model_dir entries")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="extended")
    parser.add_argument(
        "--prompt-style",
        choices=["official", "official-leading-space", "legacy"],
        default="official",
    )
    parser.add_argument(
        "--diagnostic-unverified",
        action="store_true",
        help="allow a diagnostic variant without publication manifest/checksum verification",
    )
    parser.add_argument(
        "--allow-single-request",
        action="store_true",
        help="allow --repeats 1 for diagnostics; release evidence requires at least 2",
    )
    parser.add_argument(
        "--allow-language-failures",
        action="store_true",
        help="write diagnostic results instead of failing on a target-language error",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.repeats < 2 and not args.allow_single_request:
        parser.error("release evidence requires --repeats >= 2")

    environment, environment_problems = export_environment()
    if environment_problems:
        for problem in environment_problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        import onnxruntime_genai as og
        from tokenizers import Tokenizer
    except ImportError as error:
        print(f"error: missing pinned benchmark dependency: {error}", file=sys.stderr)
        return 1

    if not args.diagnostic_unverified:
        try:
            from verify_cat_onnx_distribution import validate_distribution
        except ImportError as error:
            print(f"error: cannot load distribution verifier: {error}", file=sys.stderr)
            return 1
    else:
        validate_distribution = None

    cases = CASE_SETS[args.case_set]
    print(
        f"case set: {args.case_set} ({len(cases)} cases), "
        f"consecutive requests per case: {args.repeats}"
    )
    summaries = []
    for entry in args.models:
        label, separator, raw_model_dir = entry.partition("=")
        if not separator or not label or not raw_model_dir:
            parser.error(f"expected label=model_dir, got: {entry}")
        model_dir = Path(raw_model_dir).resolve()
        missing = [
            name
            for name in ("genai_config.json", "tokenizer.json", "model_q4.onnx")
            if not (model_dir / name).is_file()
        ]
        if missing:
            print(
                f"error: {label} is missing required runtime file(s): {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        if validate_distribution is not None:
            try:
                validate_distribution(model_dir, write_manifest=False)
            except Exception as error:
                print(f"error: {label} distribution verification failed: {error}", file=sys.stderr)
                return 1
        summaries.append(
            bench_model(
                og,
                Tokenizer,
                label,
                model_dir,
                cases,
                args.repeats,
                args.max_new_tokens,
                args.prompt_style,
            )
        )

    report = {
        "schema_version": 1,
        "case_set": args.case_set,
        "case_count": len(cases),
        "repeats": args.repeats,
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "environment": environment,
        "models": summaries,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("wrote JSON benchmark report")

    failures = {
        summary["label"]: summary["language_failures"]
        for summary in summaries
        if summary["language_failures"]
    }
    if failures and not args.allow_language_failures:
        print("error: target-language failure(s): " + json.dumps(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
