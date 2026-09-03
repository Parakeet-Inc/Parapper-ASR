from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "reference-manifest.json"
FIXTURES_DIR = ROOT / "fixtures"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("reference manifest schema_version must be 1")
    commit = manifest.get("nemo", {}).get("commit")
    if not is_hex(commit, 40):
        raise ValueError("NeMo commit must be a full 40-character SHA")

    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 8:
        raise ValueError("reference manifest must contain the eight supported ASR models")
    app_models = [model.get("app_model") for model in models]
    if len(app_models) != len(set(app_models)):
        raise ValueError("app_model values must be unique")

    for model in models:
        reference = model.get("reference", {})
        production = model.get("production_artifact", {})
        if not is_hex(reference.get("revision"), 40):
            raise ValueError(f"{model['app_model']}: reference revision must be a full SHA")
        if production.get("kind") == "huggingface_snapshot" and not is_hex(
            production.get("revision"), 40
        ):
            raise ValueError(f"{model['app_model']}: production snapshot must be pinned")
        digest = production.get("sha256")
        if production.get("kind") == "github_release_asset" and not is_hex(digest, 64):
            raise ValueError(f"{model['app_model']}: release asset must have SHA-256")


def validate_fixtures(manifest: dict[str, Any]) -> None:
    if not FIXTURES_DIR.exists():
        return
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("schema_version") != 1:
            raise ValueError(f"{path.name}: schema_version must be 1")
        if fixture.get("kind") not in {"algorithm", "tensor", "audio", "streaming"}:
            raise ValueError(f"{path.name}: invalid fixture kind")
        reference = fixture.get("reference", {})
        if reference.get("oracle") == "pinned_nvidia_python":
            if reference.get("nemo_commit") != manifest["nemo"]["commit"]:
                raise ValueError(f"{path.name}: NeMo commit does not match the manifest")
            if not is_hex(reference.get("model_revision"), 40):
                raise ValueError(f"{path.name}: model revision must be a full SHA")
        if not is_hex(fixture.get("input", {}).get("sha256"), 64):
            raise ValueError(f"{path.name}: input SHA-256 is missing or invalid")


def find_model(manifest: dict[str, Any], app_model: str) -> dict[str, Any]:
    for model in manifest["models"]:
        if model["app_model"] == app_model:
            return model
    raise ValueError(f"unknown app model: {app_model}")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def normalize_hypothesis(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value, "token_ids": [], "score": None, "timestamps": None}
    return {
        "text": getattr(value, "text", None),
        "token_ids": to_jsonable(getattr(value, "y_sequence", [])),
        "score": to_jsonable(getattr(value, "score", None)),
        "timestamps": to_jsonable(getattr(value, "timestamp", None)),
        "token_durations": to_jsonable(getattr(value, "token_duration", None)),
    }


def configure_decoding(model: Any, strategy: str, beam_size: int) -> None:
    if strategy == "model_default":
        return
    from omegaconf import OmegaConf, open_dict

    decoding = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(decoding):
        decoding.strategy = strategy
        if strategy == "greedy":
            decoding.greedy.max_symbols = 10
            decoding.tdt_include_token_duration = True
            decoding.compute_timestamps = False
        elif strategy == "beam":
            decoding.beam.beam_size = beam_size
            decoding.beam.return_best_hypothesis = False
            decoding.beam.score_norm = True
            decoding.compute_timestamps = False
            decoding.tdt_include_token_duration = False
    model.change_decoding_strategy(decoding)


def transcribe(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    model_contract = find_model(manifest, args.app_model)
    reference = model_contract["reference"]
    filename = reference.get("filename")
    if not filename:
        raise ValueError(f"{args.app_model} has no NVIDIA .nemo reference artifact")

    import nemo
    import soundfile
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    model_path = hf_hub_download(
        repo_id=reference["repository"],
        filename=filename,
        revision=reference["revision"],
    )
    device = torch.device(args.device)
    model = ASRModel.restore_from(model_path, map_location=device)
    model.eval()
    configure_decoding(model, args.decoding_strategy, args.beam_size)
    if "Prompt" in type(model).__name__:
        # Pinned NeMo's prompt-aware transcribe dataloader forwards
        # `default_lang` but omits `default_prompt_mode`. The dataset therefore
        # defaults to randomized `unified` mode and can try to resolve the
        # temporary manifest's None language. Force the requested production
        # contract (auto=101) without changing model code or weights.
        from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
            LhotseSpeechToTextBpeDatasetWithPromptIndex,
        )

        original_init = LhotseSpeechToTextBpeDatasetWithPromptIndex.__init__

        def prompt_dataset_init(instance: Any, *init_args: Any, **init_kwargs: Any) -> None:
            original_init(instance, *init_args, **init_kwargs)
            instance.default_prompt_mode = args.target_lang

        LhotseSpeechToTextBpeDatasetWithPromptIndex.__init__ = prompt_dataset_init
    prompt = {"target_lang": args.target_lang} if "Prompt" in type(model).__name__ else {}
    result = model.transcribe(
        [str(args.audio.resolve())], return_hypotheses=True, **prompt
    )
    first = result[0]
    if isinstance(first, tuple):
        first = first[0]

    audio_bytes = args.audio.read_bytes()
    audio_info = soundfile.info(args.audio)
    fixture = {
        "schema_version": 1,
        "kind": "audio",
        "reference": {
            "oracle": model_contract["oracle"],
            "nemo_commit": manifest["nemo"]["commit"],
            "model_repository": reference["repository"],
            "model_revision": reference["revision"],
            "decoding_strategy": args.decoding_strategy,
            "parameters": {},
        },
        "input": {
            "sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "sample_rate_hz": audio_info.samplerate,
            "samples": audio_info.frames,
            "provenance": args.provenance,
        },
        "output": normalize_hypothesis(first),
        "environment": {
            "python": platform.python_version(),
            "nemo": getattr(nemo, "__version__", "unknown"),
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--app-model", required=True)
    transcribe_parser.add_argument("--audio", type=Path, required=True)
    transcribe_parser.add_argument("--output", type=Path, required=True)
    transcribe_parser.add_argument("--device", default="cpu")
    transcribe_parser.add_argument(
        "--decoding-strategy", choices=("model_default", "greedy", "beam"), default="model_default"
    )
    transcribe_parser.add_argument("--beam-size", type=int, default=4)
    transcribe_parser.add_argument("--provenance")
    transcribe_parser.add_argument("--target-lang", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    validate_manifest(manifest)
    if args.command == "validate":
        validate_fixtures(manifest)
        print("reference manifest and fixtures are valid")
        return
    if args.command == "transcribe":
        transcribe(args, manifest)
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
