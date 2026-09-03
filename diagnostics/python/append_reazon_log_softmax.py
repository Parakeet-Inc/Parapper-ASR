"""Create a diagnostic ReazonSpeech joiner with an appended LogSoftmax node."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import helper


def append_log_softmax(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    model = onnx.load(source, load_external_data=True)
    if len(model.graph.output) != 1 or model.graph.output[0].name != "logit":
        raise ValueError("expected one joiner output named 'logit'")

    producers = [
        (node, index)
        for node in model.graph.node
        for index, name in enumerate(node.output)
        if name == "logit"
    ]
    consumers = [node.name or node.op_type for node in model.graph.node if "logit" in node.input]
    if len(producers) != 1 or consumers:
        raise ValueError(
            "expected 'logit' to be produced once and have no graph-internal consumers; "
            f"producers={len(producers)}, consumers={consumers}"
        )

    raw_name = "parapper_raw_logit"
    producer, output_index = producers[0]
    producer.output[output_index] = raw_name
    model.graph.node.append(
        helper.make_node(
            "LogSoftmax",
            inputs=[raw_name],
            outputs=["logit"],
            axis=-1,
            name="parapper_log_softmax",
        )
    )
    metadata = model.metadata_props.add()
    metadata.key = "parapper.joiner_score_domain"
    metadata.value = "log_probabilities"
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    append_log_softmax(args.source, args.destination)
    print(f"source_sha256={sha256(args.source)}")
    print(f"destination_sha256={sha256(args.destination)}")


if __name__ == "__main__":
    main()
