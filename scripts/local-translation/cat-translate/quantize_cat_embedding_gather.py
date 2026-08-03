"""Quantize the CAT token embedding to the adopted asymmetric Q4 block16 form.

The input must be the audited CAT ``k_quant`` graph with an FP32 embedding.
Only the embedding Gather is changed; the existing 120 Q4 and one Q8
``MatMulNBits`` nodes remain untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


MODEL_FILENAME = "model_q4.onnx"
ADOPTED_BLOCK_SIZE = 16
AUX_FILES = [
    "genai_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def _reject_unsafe_output(out_dir: Path, protected_paths: tuple[Path, ...]):
    if out_dir == Path(out_dir.anchor) or out_dir == Path.cwd().resolve():
        raise ValueError(f"unsafe out_dir: {out_dir}")
    for protected in protected_paths:
        protected = protected.resolve()
        if out_dir == protected or out_dir in protected.parents:
            raise ValueError(
                f"unsafe out_dir contains required input {protected}: {out_dir}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--block-size", type=int, default=ADOPTED_BLOCK_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        import onnx
        from onnxruntime.quantization import QuantFormat
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
    except ImportError as error:
        print(f"error: missing pinned quantization dependency: {error}", file=sys.stderr)
        return 1

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    try:
        _reject_unsafe_output(out_dir, (in_dir, Path(__file__).resolve()))
    except ValueError as error:
        parser.error(str(error))
    if in_dir == out_dir:
        parser.error("out_dir must differ from in_dir")
    if args.block_size != ADOPTED_BLOCK_SIZE:
        parser.error(
            f"the adopted CAT embedding contract requires --block-size {ADOPTED_BLOCK_SIZE}"
        )

    model_path = in_dir / MODEL_FILENAME
    if not model_path.is_file():
        parser.error(f"input is missing {MODEL_FILENAME}")
    missing_aux = [name for name in AUX_FILES if not (in_dir / name).is_file()]
    if missing_aux:
        parser.error("input is missing auxiliary file(s): " + ", ".join(missing_aux))
    if out_dir.exists():
        if not args.force:
            parser.error("out_dir already exists; choose a clean directory or pass --force")
        shutil.rmtree(out_dir)

    print("loading input model")
    model = onnx.load(str(model_path))
    input_counts = Counter(node.op_type for node in model.graph.node)
    if input_counts["MatMulNBits"] != 121 or input_counts["GatherBlockQuantized"] != 0:
        parser.error(
            "input must be the 121-node k_quant graph with an unquantized embedding"
        )

    quantizer = MatMulNBitsQuantizer(
        model=model,
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=ADOPTED_BLOCK_SIZE,
            is_symmetric=False,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("Gather",),
            bits=4,
        ),
    )
    quantizer.process()

    out_dir.mkdir(parents=True)
    out_path = out_dir / MODEL_FILENAME
    print("saving adopted Q4 block16 embedding model")
    onnx.save_model(
        quantizer.model.model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=MODEL_FILENAME + ".data",
        size_threshold=0,
    )
    for name in AUX_FILES:
        shutil.copy2(in_dir / name, out_dir / name)

    output_model = onnx.load(str(out_path), load_external_data=False)
    output_counts = Counter(node.op_type for node in output_model.graph.node)
    if output_counts["MatMulNBits"] != 121:
        print("error: MatMulNBits graph changed during embedding pass", file=sys.stderr)
        return 1
    if output_counts["GatherBlockQuantized"] != 1:
        print(
            "error: expected exactly one GatherBlockQuantized node in diagnostic output",
            file=sys.stderr,
        )
        return 1
    embedding_node = next(
        node
        for node in output_model.graph.node
        if node.op_type == "GatherBlockQuantized"
    )
    attributes = {attribute.name: int(attribute.i) for attribute in embedding_node.attribute}
    expected_attributes = {
        "block_size": ADOPTED_BLOCK_SIZE,
        "gather_axis": 0,
        "quantize_axis": 1,
    }
    if attributes != expected_attributes:
        print(
            "error: embedding quantization attributes do not match Q4 block16: "
            f"{attributes}",
            file=sys.stderr,
        )
        return 1

    print("Q4 block16 embedding output complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
