from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile


NEMO_COMMIT = "b331a34885986ec879ca7acb62d955e7af71c015"
FRAME_LENGTH = 400
FRAME_SHIFT = 160
FFT_SIZE = 512
PREEMPHASIS = 0.97
MAX_SYMBOLS = 10


def load_tokens(path: Path) -> list[str]:
    indexed: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        token, raw_id = line.rsplit(" ", 1)
        indexed[int(raw_id)] = token
    return [indexed[index] for index in range(len(indexed))]


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def reflect(index: int, size: int) -> int:
    while index < 0 or index >= size:
        index = -index - 1 if index < 0 else 2 * size - 1 - index
    return index


class StreamingFrontend:
    def __init__(self, window_frames: int, shift_frames: int):
        self.window_frames = window_frames
        self.shift_frames = shift_frames
        self.samples = np.empty(0, dtype=np.float32)
        self.features: list[np.ndarray] = []
        self.processed = 0
        self.window = np.hanning(FRAME_LENGTH).astype(np.float32)
        self.mel = librosa.filters.mel(
            sr=16000,
            n_fft=FFT_SIZE,
            n_mels=128,
            fmin=0,
            fmax=8000,
            norm="slaney",
        ).astype(np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self.samples = np.concatenate((self.samples, samples.astype(np.float32, copy=False)))
        complete = 0 if len(self.samples) < 280 else 1 + (len(self.samples) - 280) // FRAME_SHIFT
        while len(self.features) < complete:
            frame_index = len(self.features)
            start = -120 + frame_index * FRAME_SHIFT
            frame = np.asarray(
                [self.samples[reflect(start + index, len(self.samples))] for index in range(FRAME_LENGTH)],
                dtype=np.float32,
            )
            frame[1:] -= PREEMPHASIS * frame[:-1].copy()
            frame[0] *= 1.0 - PREEMPHASIS
            spectrum = np.fft.rfft(frame * self.window, n=FFT_SIZE).astype(np.complex64)
            power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
            self.features.append(np.log(self.mel @ power + np.float32(2**-24)).astype(np.float32))
        windows = []
        while self.processed + self.window_frames < len(self.features):
            windows.append(
                np.stack(self.features[self.processed : self.processed + self.window_frames], axis=1)[None]
            )
            self.processed += self.shift_frames
        return windows


def visible_token(token: str) -> str | None:
    if token.startswith("<") and token.endswith(">"):
        return None
    return token.replace("▁", " ")


def remove_cjk_spaces(text: str) -> str:
    def cjk(character: str) -> bool:
        value = ord(character)
        return 0x3040 <= value <= 0x30FF or 0x3400 <= value <= 0x9FFF or 0xAC00 <= value <= 0xD7AF

    chars = list(text)
    return "".join(
        character
        for index, character in enumerate(chars)
        if not (
            character == " "
            and index > 0
            and index + 1 < len(chars)
            and cjk(chars[index - 1])
            and cjk(chars[index + 1])
        )
    ).strip()


def decode(model_dir: Path, samples: np.ndarray, chunk_samples: int) -> dict[str, object]:
    encoder = make_session(model_dir / "encoder.int8.onnx", 2)
    decoder = make_session(model_dir / "decoder.int8.onnx", 1)
    joiner = make_session(model_dir / "joiner.int8.onnx", 1)
    metadata = encoder.get_modelmeta().custom_metadata_map
    window_frames = int(metadata["window_size"])
    shift_frames = int(metadata["chunk_shift"])
    has_prompt = "auto_prompt_id" in metadata
    tokens = load_tokens(model_dir / "tokens.txt")
    blank = len(tokens) - 1
    frontend = StreamingFrontend(window_frames, shift_frames)
    cache_channel = np.zeros((1, 24, 70, 1024), dtype=np.float32)
    cache_time = np.zeros((1, 24, 1024, 8), dtype=np.float32)
    cache_len = np.asarray([0], dtype=np.int64)
    hidden = np.zeros((2, 1, 640), dtype=np.float32)
    cell = np.zeros((2, 1, 640), dtype=np.float32)
    last_token: int | None = None
    token_ids: list[int] = []
    timestamps: list[int] = []
    frame_offset = 0
    partials: list[str] = []

    for audio_chunk in np.array_split(samples, np.arange(chunk_samples, len(samples), chunk_samples)):
        for features in frontend.push(audio_chunk):
            inputs = {
                "audio_signal": features,
                "length": np.asarray([window_frames], dtype=np.int64),
                "cache_last_channel": cache_channel,
                "cache_last_time": cache_time,
                "cache_last_channel_len": cache_len,
            }
            if has_prompt:
                inputs["prompt_index"] = np.asarray([int(metadata["auto_prompt_id"])], dtype=np.int64)
            encoded, encoded_lengths, cache_channel, cache_time, cache_len = encoder.run(None, inputs)
            frames = int(encoded_lengths[0])
            for time in range(frames):
                for _ in range(MAX_SYMBOLS):
                    label = blank if last_token is None else last_token
                    prediction, _, next_hidden, next_cell = decoder.run(
                        None,
                        {
                            "targets": np.asarray([[label]], dtype=np.int32),
                            "target_length": np.asarray([1], dtype=np.int32),
                            "states.1": hidden,
                            "onnx::Slice_3": cell,
                        },
                    )
                    logits = joiner.run(
                        None,
                        {
                            "encoder_outputs": encoded[:, :, time : time + 1],
                            "decoder_outputs": prediction,
                        },
                    )[0].reshape(-1)
                    token = int(np.argmax(logits))
                    if token == blank:
                        break
                    token_ids.append(token)
                    timestamps.append(frame_offset + time)
                    hidden = next_hidden
                    cell = next_cell
                    last_token = token
            frame_offset += frames
        pieces = [piece for token in token_ids if (piece := visible_token(tokens[token])) is not None]
        partials.append(remove_cjk_spaces("".join(pieces)))

    pieces = [piece for token in token_ids if (piece := visible_token(tokens[token])) is not None]
    return {
        "text": remove_cjk_spaces("".join(pieces)),
        "token_ids": token_ids,
        "timestamps": timestamps,
        "partials": partials,
        "encoder_frames": frame_offset,
        "window_frames": window_frames,
        "shift_frames": shift_frames,
        "prompt_id": int(metadata["auto_prompt_id"]) if has_prompt else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--app-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-samples", type=int, default=2560)
    args = parser.parse_args()
    samples, rate = soundfile.read(args.audio, dtype="float32")
    if rate != 16000 or samples.ndim != 1:
        raise ValueError("Nemotron fixture audio must be mono 16 kHz")
    fixture = {
        "schema_version": 1,
        "kind": "streaming",
        "reference": {
            "oracle": "pinned_nvidia_python",
            "nemo_commit": NEMO_COMMIT,
            "model_repository": args.app_model,
            "model_revision": args.model_revision,
            "runtime": f"onnxruntime-{ort.__version__}",
            "decoding_strategy": "NVIDIA RNNT greedy, production streaming ONNX",
            "parameters": {"max_symbols_per_frame": MAX_SYMBOLS},
        },
        "input": {
            "sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
            "sample_rate_hz": rate,
            "samples": len(samples),
            "chunk_samples": args.chunk_samples,
            "model_dir": args.model_dir.name,
        },
        "output": decode(args.model_dir, samples, args.chunk_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
