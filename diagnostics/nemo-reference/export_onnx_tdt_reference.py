from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile
import torch


NEMO_COMMIT = "b331a34885986ec879ca7acb62d955e7af71c015"
DURATIONS = [0, 1, 2, 3, 4]
MAX_SYMBOLS = 10


def pinned_nemo_frontend(samples: np.ndarray) -> tuple[np.ndarray, int]:
    audio = torch.from_numpy(samples.astype(np.float32, copy=False)).unsqueeze(0)
    valid_frames = audio.shape[1] // 160
    audio = torch.cat((audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]), dim=1)
    spectrum = torch.stft(
        audio,
        n_fft=512,
        hop_length=160,
        win_length=400,
        center=True,
        window=torch.hann_window(400, periodic=False),
        return_complex=True,
        pad_mode="constant",
    ).abs().pow(2.0)
    filters = torch.tensor(
        librosa.filters.mel(
            sr=16000,
            n_fft=512,
            n_mels=128,
            fmin=0,
            fmax=8000,
            norm="slaney",
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    features = torch.log(torch.matmul(filters, spectrum) + 2**-24)
    valid = features[:, :, :valid_frames]
    mean = valid.mean(dim=2, keepdim=True)
    std = torch.sqrt(((valid - mean) ** 2).sum(dim=2, keepdim=True) / (valid_frames - 1))
    features = (features - mean) / (std + 1.0e-5)
    features[:, :, valid_frames:] = 0.0
    return features.numpy(), valid_frames


def load_tokens(path: Path) -> list[str]:
    indexed: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        token, raw_id = line.rsplit(" ", 1)
        indexed[int(raw_id)] = token
    return [indexed[index] for index in range(len(indexed))]


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def decode(model_dir: Path, samples: np.ndarray) -> dict[str, object]:
    encoder = session(model_dir / "encoder.int8.onnx")
    decoder = session(model_dir / "decoder.int8.onnx")
    joiner = session(model_dir / "joiner.int8.onnx")
    tokens = load_tokens(model_dir / "tokens.txt")
    blank = len(tokens) - 1

    features, valid_frames = pinned_nemo_frontend(samples)
    encoded, encoded_lengths = encoder.run(
        ["outputs", "encoded_lengths"],
        {"audio_signal": features, "length": np.asarray([valid_frames], dtype=np.int64)},
    )
    frames = int(encoded_lengths[0])
    hidden = np.zeros((2, 1, 640), dtype=np.float32)
    cell = np.zeros((2, 1, 640), dtype=np.float32)
    last_token: int | None = None
    token_ids: list[int] = []
    timestamps: list[int] = []
    token_durations: list[int] = []
    score = 0.0
    time = 0
    while time < frames:
        symbols_added = 0
        skip = 0
        while symbols_added < MAX_SYMBOLS:
            label = blank if last_token is None else last_token
            prediction, _, next_hidden, next_cell = decoder.run(
                ["outputs", "prednet_lengths", "states", "162"],
                {
                    "targets": np.asarray([[label]], dtype=np.int32),
                    "target_length": np.asarray([1], dtype=np.int32),
                    "states.1": hidden,
                    "onnx::Slice_3": cell,
                },
            )
            logits = joiner.run(
                ["outputs"],
                {
                    "encoder_outputs": encoded[:, :, time : time + 1],
                    "decoder_outputs": prediction,
                },
            )[0].reshape(-1)
            token_logits = logits[: blank + 1]
            duration_logits = logits[blank + 1 :]
            token = int(np.argmax(token_logits))
            skip = DURATIONS[int(np.argmax(duration_logits))]
            if token != blank:
                token_ids.append(token)
                timestamps.append(time)
                token_durations.append(skip)
                score += float(token_logits[token])
                hidden = next_hidden
                cell = next_cell
                last_token = token
            symbols_added += 1
            time += skip
            if skip != 0:
                break
        if skip == 0 and symbols_added == MAX_SYMBOLS:
            time += 1

    token_texts = [tokens[index].replace("▁", " ") for index in token_ids]
    return {
        "text": "".join(token_texts).strip(),
        "token_ids": token_ids,
        "score": score,
        "timestamps": timestamps,
        "token_durations": token_durations,
        "encoded_frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--app-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples, sample_rate = soundfile.read(args.audio, dtype="float32")
    if sample_rate != 16000 or samples.ndim != 1:
        raise ValueError("TDT reference audio must be mono 16 kHz")
    output = decode(args.model_dir, samples)
    fixture = {
        "schema_version": 1,
        "kind": "audio",
        "reference": {
            "oracle": "pinned_nvidia_python",
            "nemo_commit": NEMO_COMMIT,
            "model_repository": args.app_model,
            "model_revision": args.model_revision,
            "runtime": f"onnxruntime-{ort.__version__}",
            "source_files": [
                "nemo/collections/asr/parts/preprocessing/features.py",
                "nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py",
            ],
            "decoding_strategy": "GreedyTDTInfer",
            "parameters": {
                "durations": DURATIONS,
                "max_symbols_per_step": MAX_SYMBOLS,
                "include_duration": True,
            },
        },
        "input": {
            "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
            "sample_rate_hz": sample_rate,
            "samples": len(samples),
            "model_dir": args.model_dir.name,
        },
        "output": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
