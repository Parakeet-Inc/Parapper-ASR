from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import librosa
import torch


# This is a narrow executable oracle for NVIDIA NeMo commit
# b331a34885986ec879ca7acb62d955e7af71c015, specifically
# nemo/collections/asr/parts/preprocessing/features.py. It intentionally uses
# torch/librosa operations from that source without importing the full NeMo
# package, whose wheel is not buildable on Windows due to path-length limits.
NEMO_COMMIT = "b331a34885986ec879ca7acb62d955e7af71c015"
OUTPUT = Path(__file__).resolve().parents[1] / "fixtures" / "nemo-ctc-frontend.json"


def deterministic_samples(length: int) -> torch.Tensor:
    return torch.tensor(
        [
            0.35 * math.sin(2.0 * math.pi * 233.0 * index / 16000.0)
            + 0.12 * math.cos(2.0 * math.pi * 911.0 * index / 16000.0)
            + ((index % 17) - 8) / 200.0
            for index in range(length)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def pinned_nemo_frontend(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    sample_length = samples.shape[1]
    sequence_length = sample_length // 160
    samples = torch.cat(
        (samples[:, 0].unsqueeze(1), samples[:, 1:] - 0.97 * samples[:, :-1]),
        dim=1,
    )
    window = torch.hann_window(400, periodic=False)
    spectrum = torch.stft(
        samples,
        n_fft=512,
        hop_length=160,
        win_length=400,
        center=True,
        window=window,
        return_complex=True,
        pad_mode="constant",
    ).abs().pow(2.0)
    filters = torch.tensor(
        librosa.filters.mel(
            sr=16000,
            n_fft=512,
            n_mels=80,
            fmin=0,
            fmax=8000,
            norm="slaney",
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    features = torch.matmul(filters, spectrum)
    features = torch.log(features + 2**-24)
    log_features = features.clone()
    valid = features[:, :, :sequence_length]
    mean = valid.mean(dim=2, keepdim=True)
    std = torch.sqrt(((valid - mean) ** 2).sum(dim=2, keepdim=True) / (sequence_length - 1))
    features = (features - mean) / (std + 1.0e-5)
    features[:, :, sequence_length:] = 0.0
    return features, log_features, sequence_length


def main() -> None:
    samples = deterministic_samples(640)
    features, log_features, valid_frames = pinned_nemo_frontend(samples)
    fixture = {
        "schema_version": 1,
        "kind": "tensor",
        "reference": {
            "oracle": "pinned_nvidia_python",
            "nemo_commit": NEMO_COMMIT,
            "model_repository": "nvidia/parakeet-tdt_ctc-0.6b-ja",
            "model_revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
            "source_file": "nemo/collections/asr/parts/preprocessing/features.py",
            "torch": torch.__version__,
            "librosa": librosa.__version__,
        },
        "input": {
            "sha256": hashlib.sha256(samples.numpy().tobytes()).hexdigest(),
            "generator": "deterministic_samples_v1",
            "sample_rate_hz": 16000,
            "sample_count": samples.shape[1],
            "samples": samples.flatten().tolist(),
        },
        "output": {
            "shape": list(features.shape),
            "valid_frames": valid_frames,
            "log_values": log_features.flatten().tolist(),
            "values": features.flatten().tolist(),
        },
        "tolerance": {"absolute": 2.0e-4, "relative": 2.0e-4},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
