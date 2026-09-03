"""Compare per-layer encoder activations between an FP32 and a quantized ONNX.

Promotes every Conformer block boundary tensor
(``/layers.{i}/norm_out/LayerNormalization_output_0``) plus the final
``outputs`` tensor to graph outputs on both models, feeds both the same
``audio_signal``/``length`` NPZ, and reports per-boundary cosine similarity,
relative L2, and max absolute error. The layer where relative error jumps is
the quantization choke point; feed those layer names into the quantizer's
``nodes_to_exclude`` and re-measure.

The augmented graph is written next to the source model (external-data
references are relative, so it must live in the same directory) and removed
afterwards unless ``--keep-augmented`` is passed.

Usage:
  python compare_encoder_activations.py \
    --fp32 <dir>/encoder-model.onnx \
    --candidate <dir>/encoder-model.int8.onnx \
    --features onnx-optimize/runs/inputs/parakeet-ctc-10s.npz \
    --threads 4 --output <report.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

BOUNDARY_TEMPLATE = "/layers.{index}/norm_out/LayerNormalization_output_0"
AUGMENTED_SUFFIX = ".activation-probe.onnx"


def boundary_tensors(model: onnx.ModelProto) -> list[str]:
    produced = {name for node in model.graph.node for name in node.output}
    names = []
    index = 0
    while True:
        candidate = BOUNDARY_TEMPLATE.format(index=index)
        if candidate not in produced:
            break
        names.append(candidate)
        index += 1
    if not names:
        raise SystemExit("no Conformer boundary tensors found; wrong model?")
    return names


def augment(model_path: Path) -> tuple[Path, list[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    boundaries = boundary_tensors(model)
    existing = {output.name for output in model.graph.output}
    for name in boundaries:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    augmented = model_path.with_name(model_path.name + AUGMENTED_SUFFIX)
    onnx.save(model, str(augmented))
    return augmented, boundaries


def run_probe(model_path: Path, feeds: dict[str, np.ndarray], threads: int) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(output_names, feeds)
    return dict(zip(output_names, values, strict=True))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64).ravel()
    candidate = candidate.astype(np.float64).ravel()
    diff = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "cosine": float(
            np.dot(reference, candidate)
            / ((reference_norm * float(np.linalg.norm(candidate))) + 1e-12)
        ),
        "relative_l2": float(np.linalg.norm(diff) / (reference_norm + 1e-12)),
        "max_abs": float(np.abs(diff).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-augmented", action="store_true")
    arguments = parser.parse_args()

    npz = np.load(arguments.features)
    feeds = {"audio_signal": npz["audio_signal"], "length": npz["length"]}

    augmented_fp32, boundaries = augment(arguments.fp32)
    augmented_candidate, candidate_boundaries = augment(arguments.candidate)
    if candidate_boundaries != boundaries:
        raise SystemExit("boundary tensors differ between the two models")
    try:
        fp32_values = run_probe(augmented_fp32, feeds, arguments.threads)
        candidate_values = run_probe(augmented_candidate, feeds, arguments.threads)
    finally:
        if not arguments.keep_augmented:
            augmented_fp32.unlink(missing_ok=True)
            augmented_candidate.unlink(missing_ok=True)

    report = {
        "fp32": str(arguments.fp32),
        "candidate": str(arguments.candidate),
        "features": str(arguments.features),
        "threads": arguments.threads,
        "boundaries": [],
    }
    print(f"{'boundary':<58} {'cosine':>10} {'rel_l2':>10} {'max_abs':>10}")
    for name in [*boundaries, "outputs"]:
        entry = {"tensor": name, **metrics(fp32_values[name], candidate_values[name])}
        report["boundaries"].append(entry)
        print(
            f"{name:<58} {entry['cosine']:>10.6f} {entry['relative_l2']:>10.5f} "
            f"{entry['max_abs']:>10.4f}"
        )
    worst = max(
        zip(report["boundaries"], [None, *report["boundaries"]], strict=False),
        key=lambda pair: pair[0]["relative_l2"] - (pair[1]["relative_l2"] if pair[1] else 0.0),
    )[0]
    report["largest_relative_l2_jump"] = worst["tensor"]
    print(f"largest relative-L2 jump at: {worst['tensor']}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
