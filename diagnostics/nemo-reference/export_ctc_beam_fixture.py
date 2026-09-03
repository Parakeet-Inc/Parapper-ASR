from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from nemo.collections.asr.parts.submodules.ctc_beam_decoding import BeamBatchedCTCInfer


NEMO_COMMIT = "b331a34885986ec879ca7acb62d955e7af71c015"
OUTPUT = Path(__file__).resolve().parent / "fixtures" / "nemo-ctc-batched-beam.json"


def main() -> None:
    logits = torch.tensor(
        [
            [2.3, 1.1, -0.4, 2.0],
            [1.7, 2.4, 0.2, 1.6],
            [1.6, 2.2, 0.9, 1.1],
            [0.1, 1.4, 2.5, 1.8],
            [1.3, 0.7, 2.0, 2.1],
            [2.4, 0.2, 1.3, 1.9],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    decoder = BeamBatchedCTCInfer(
        blank_index=3,
        beam_size=4,
        return_best_hypothesis=True,
        preserve_alignments=False,
        compute_timestamps=False,
        ngram_lm_alpha=1.0,
        beam_beta=0.0,
        beam_threshold=20.0,
        ngram_lm_model=None,
        allow_cuda_graphs=False,
    )
    hypotheses, = decoder(
        decoder_output=log_probs.unsqueeze(0),
        decoder_lengths=torch.tensor([log_probs.shape[0]]),
    )
    hypothesis = hypotheses[0]
    values = log_probs.flatten().tolist()
    fixture = {
        "schema_version": 1,
        "kind": "algorithm",
        "reference": {
            "oracle": "pinned_nvidia_python",
            "nemo_commit": NEMO_COMMIT,
            "model_repository": "nvidia/parakeet-tdt_ctc-0.6b-ja",
            "model_revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
            "decoding_strategy": "BeamBatchedCTCInfer",
            "parameters": {
                "blank_id": 3,
                "beam_size": 4,
                "beam_beta": 0.0,
                "beam_threshold": 20.0,
                "fusion_models": None,
            },
        },
        "input": {
            "sha256": hashlib.sha256(log_probs.numpy().tobytes()).hexdigest(),
            "shape": list(log_probs.shape),
            "log_probs": values,
        },
        "output": {
            "token_ids": hypothesis.y_sequence.tolist(),
            "score": float(hypothesis.score),
        },
        "environment": {"torch": torch.__version__},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(fixture, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
