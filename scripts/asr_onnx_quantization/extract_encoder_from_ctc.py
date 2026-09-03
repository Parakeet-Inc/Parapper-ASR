"""Extract a numerically-correct standalone encoder from the CTC full graph.

NeMo's standalone ConformerEncoder export is numerically wrong (cosine ~0.81 vs
torch), while the CTC single-graph export is faithful (max_abs ~9e-4). This
cuts the CTC graph at the ctc_decoder boundary:

  outputs          = Identity(/Transpose_2_output_0)   # [B, 1024, T'] NCT
  encoded_lengths  = Cast_int64(/Cast_output_0)        # subsampled lengths

and prunes the CTC head and any now-dead nodes/initializers.

It also extracts the inverse slice as ``ctc-head-model.onnx`` with the stable
contract ``encoder_outputs, encoded_lengths -> logprobs``.  The head receives
its own external-data file so it does not depend on the monolithic CTC graph's
``model.onnx_data``.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ENC_OUT = "/Transpose_2_output_0"
LEN_OUT = "/Cast_output_0"
HEAD_INPUT = "encoder_outputs"
HEAD_LENGTH_INPUT = "encoded_lengths"
HEAD_OUTPUT = "logprobs"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_value_info(value_info: onnx.ValueInfoProto, *, name: str) -> onnx.ValueInfoProto:
    """Copy a graph value-info while giving the boundary a stable public name."""

    copied = onnx.ValueInfoProto()
    copied.CopyFrom(value_info)
    copied.name = name
    return copied


def _find_value_info(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto | None:
    for value_info in [*graph.input, *graph.value_info, *graph.output]:
        if value_info.name == name:
            return value_info
    return None


def _head_output(graph: onnx.GraphProto, output_name: str) -> onnx.ValueInfoProto:
    value_info = next((item for item in graph.output if item.name == output_name), None)
    if value_info is None:
        raise ValueError(f"CTC output {output_name!r} is not a graph output")
    return _copy_value_info(value_info, name=HEAD_OUTPUT)


def _reverse_slice(
    graph: onnx.GraphProto,
    output_name: str,
    *,
    stop_tensors: set[str],
) -> tuple[list[onnx.NodeProto], set[str]]:
    """Return nodes needed for an output, stopping at the encoder boundary.

    Walking backwards from ``logprobs`` is important here: walking forward from
    the boundary can accidentally retain unrelated auxiliary outputs from a
    NeMo export.  A boundary tensor is treated as an input, so its producer is
    never copied into the head graph.
    """

    by_output = {output: node for node in graph.node for output in node.output if output}
    initializer_names = {item.name for item in graph.initializer}
    graph_inputs = {item.name for item in graph.input}
    kept_ids: set[int] = set()
    needed_tensors = set(stop_tensors)
    stack = [output_name]
    while stack:
        tensor = stack.pop()
        if not tensor or tensor in stop_tensors:
            continue
        node = by_output.get(tensor)
        if node is None:
            # Graph inputs and initializers do not have a producer in graph.node.
            continue
        node_id = id(node)
        if node_id in kept_ids:
            continue
        kept_ids.add(node_id)
        for input_name in node.input:
            if not input_name:
                continue
            needed_tensors.add(input_name)
            if (
                input_name not in stop_tensors
                and input_name not in initializer_names
                and input_name not in graph_inputs
            ):
                stack.append(input_name)

    return [node for node in graph.node if id(node) in kept_ids], needed_tensors


def extract_ctc_head(
    source_path: Path,
    output_path: Path,
    *,
    output_name: str | None = None,
) -> dict[str, object]:
    """Extract the small CTC decoder head from a faithful monolithic CTC graph.

    The resulting graph has the stable integration contract
    ``encoder_outputs [B,1024,T]`` and ``encoded_lengths [B]`` as inputs and
    ``logprobs`` as its only output.  ``encoded_lengths`` is intentionally part
    of the contract even for exports where the CTC logits do not consume it;
    the encoder owns length calculation and the Rust caller can pass it through
    unchanged to either decoder branch.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    model = onnx.load(str(source_path), load_external_data=True)
    graph = model.graph
    target_name = output_name or next(
        (item.name for item in graph.output if item.name == HEAD_OUTPUT),
        graph.output[0].name if graph.output else "",
    )
    if not target_name:
        raise ValueError("CTC model has no graph output to use as logprobs")

    kept_nodes, needed_tensors = _reverse_slice(
        graph,
        target_name,
        stop_tensors={ENC_OUT, LEN_OUT},
    )
    if not kept_nodes:
        raise ValueError(f"no CTC head nodes found upstream of output {target_name!r}")

    boundary_info = _find_value_info(graph, ENC_OUT)
    if boundary_info is not None and boundary_info.type.tensor_type.elem_type not in (
        0,
        TensorProto.FLOAT,
    ):
        raise ValueError(f"encoder boundary {ENC_OUT!r} is not FLOAT")
    encoder_input = helper.make_tensor_value_info(
        HEAD_INPUT, TensorProto.FLOAT, ["batch", 1024, "frames"]
    )
    lengths_input = helper.make_tensor_value_info(
        HEAD_LENGTH_INPUT, TensorProto.INT64, ["batch"]
    )
    output_info = _head_output(graph, target_name)
    output_info.name = HEAD_OUTPUT

    initializers = [
        initializer for initializer in graph.initializer if initializer.name in needed_tensors
    ]
    head_graph = helper.make_graph(
        kept_nodes,
        "parakeet_ja_ctc_head_extracted",
        [encoder_input, lengths_input],
        [output_info],
        initializers,
        # Keeping only reachable value-info prevents stale encoder metadata from
        # making the standalone head appear to have additional inputs.
        value_info=[
            _copy_value_info(value_info, name=value_info.name)
            for value_info in graph.value_info
            if value_info.name in needed_tensors
            and value_info.name not in {ENC_OUT, LEN_OUT}
        ],
    )
    head_model = helper.make_model(
        head_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name="Parapper-ASR extract_encoder_from_ctc.py",
    )

    # Replace the two monolithic boundary names with the public head inputs.
    for node in head_model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == ENC_OUT:
                node.input[index] = HEAD_INPUT
            elif input_name == LEN_OUT:
                node.input[index] = HEAD_LENGTH_INPUT

    onnx.checker.check_model(head_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        head_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + "_data",
    )
    # Re-open with external data enabled: this catches wrong relative locations
    # and validates that the head owns its initializer storage independently.
    saved = onnx.load(str(output_path), load_external_data=True)
    onnx.checker.check_model(saved)
    return {
        "output": str(output_path),
        "source_output": target_name,
        "nodes": len(kept_nodes),
        "initializers": len(initializers),
    }


def main() -> None:
    src_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src_dir / "model.onnx"), load_external_data=True)
    graph = model.graph

    by_output = {o: n for n in graph.node for o in n.output}
    initializer_names = {i.name for i in graph.initializer}
    graph_inputs = {i.name for i in graph.input}

    # Reverse reachability from the two boundary tensors.
    needed_tensors = {ENC_OUT, LEN_OUT}
    stack = [ENC_OUT, LEN_OUT]
    kept_nodes = set()
    while stack:
        tensor = stack.pop()
        node = by_output.get(tensor)
        if node is None or id(node) in kept_nodes:
            continue
        kept_nodes.add(id(node))
        for name in node.input:
            if name and name not in needed_tensors:
                needed_tensors.add(name)
                if name not in initializer_names and name not in graph_inputs:
                    stack.append(name)

    removed = [n.name for n in graph.node if id(n) not in kept_nodes]
    new_nodes = [n for n in graph.node if id(n) in kept_nodes]
    new_nodes.append(helper.make_node("Identity", [ENC_OUT], ["outputs"], name="/ExtractOutputs"))
    new_nodes.append(
        helper.make_node(
            "Cast",
            [LEN_OUT],
            ["encoded_lengths"],
            name="/ExtractEncodedLengths",
            to=TensorProto.INT64,
        )
    )

    kept_initializers = [i for i in graph.initializer if i.name in needed_tensors]
    outputs = [
        helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["batch", 1024, "frames"]),
        helper.make_tensor_value_info("encoded_lengths", TensorProto.INT64, ["batch"]),
    ]
    new_graph = helper.make_graph(
        new_nodes,
        "parakeet_ja_encoder_extracted",
        list(graph.input),
        outputs,
        kept_initializers,
        value_info=list(graph.value_info),
    )
    new_model = helper.make_model(
        new_graph, opset_imports=list(model.opset_import), ir_version=model.ir_version
    )
    onnx.save(
        new_model,
        str(out_dir / "encoder-model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder-model.onnx_data",
    )
    print(
        f"kept {len(new_nodes)} nodes, removed {len(removed)} "
        f"({[n for n in removed if 'decoder' in n or 'Softmax' in n or n.startswith('/Transpose_3')]}), "
        f"kept {len(kept_initializers)}/{len(graph.initializer)} initializers",
        flush=True,
    )

    head_report = extract_ctc_head(
        src_dir / "model.onnx", out_dir / "ctc-head-model.onnx"
    )
    print(
        f"ctc head written: {head_report['nodes']} nodes, "
        f"{head_report['initializers']} initializers",
        flush=True,
    )

    for name in ["decoder_joint-model.onnx", "model.onnx", "model.onnx_data", "vocab.txt"]:
        shutil.copyfile(src_dir / name, out_dir / name)

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(out_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"]
    )
    got = [o.name for o in session.get_outputs()]
    assert got == ["outputs", "encoded_lengths"], got
    print("session ok:", got, flush=True)

    source_metadata = json.loads((src_dir / "export-metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "date": source_metadata.get("date"),
        "derived_from": str(src_dir),
        "extraction": "encoder cut from the faithful CTC single graph at the ctc_decoder boundary"
        f" (outputs={ENC_OUT}, encoded_lengths=int64({LEN_OUT}));"
        " NeMo standalone ConformerEncoder export was numerically wrong (cosine ~0.81 vs torch)",
        "source_nemo": source_metadata.get("source_nemo"),
        "source_revision": source_metadata.get("source_revision"),
        "nemo_version": source_metadata.get("nemo_version"),
        "torch_version": source_metadata.get("torch_version"),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name != "export-metadata.json"
        },
    }
    (out_dir / "export-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("metadata written", flush=True)


if __name__ == "__main__":
    main()
