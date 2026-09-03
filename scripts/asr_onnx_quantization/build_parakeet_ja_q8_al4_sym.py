"""Build the adopted q8 MatMulNBits variant of the parakeet ja v2 fp32 export.

FastConformer pointwise lowering + MatMulNBitsQuantizer with
DefaultWeightOnlyQuantConfig(bits=8, block_size=32, is_symmetric=True,
accuracy_level=4, QOperator). Symmetric quantization removes zero points
entirely: on 8-bit this improves accuracy to fp32 parity (no zero-point
rounding bias; JSUT-1000 TDT diagnostic CER 5.849% vs fp32 5.845%) while the
accuracy_level=4 attribute routes the CPU kernel to the int8 (VNNI) path
(~1.2x faster than fp32 at 4 threads). Adopted as the accuracy-priority
variant.

Each graph is quantized in its own subprocess because two 2.4GB graphs in one
process exhaust memory during the lowering pass.

Usage:
  python build_parakeet_ja_q8_al4_sym.py <v2-fp32-dir> <output-dir>
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = r"""
import sys, pathlib
sys.path.insert(0, sys.argv[3])
import onnx
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer,
)
from quantize_hqq_w4a8 import lower_fastconformer_pointwise_conv_pairs_to_matmul

source, target = sys.argv[1], sys.argv[2]
model = onnx.load(source, load_external_data=True)
lower_fastconformer_pointwise_conv_pairs_to_matmul(model)
quantizer = MatMulNBitsQuantizer(
    model=model,
    algo_config=DefaultWeightOnlyQuantConfig(
        block_size=32, is_symmetric=True, accuracy_level=4,
        quant_format=QuantFormat.QOperator, op_types_to_quantize=("MatMul",), bits=8,
    ),
)
quantizer.process()
quantized = quantizer.model.model
touched = with_zp = 0
for node in quantized.graph.node:
    if node.op_type != "MatMulNBits":
        continue
    for attribute in node.attribute:
        if attribute.name == "accuracy_level":
            attribute.i = 4
            break
    else:
        node.attribute.append(onnx.helper.make_attribute("accuracy_level", 4))
    touched += 1
    if len(node.input) >= 4 and node.input[3]:
        with_zp += 1
if with_zp:
    raise SystemExit(f"symmetric build must not emit zero points, found {with_zp}")
onnx.save(quantized, target, save_as_external_data=True, all_tensors_to_one_file=True,
          location=pathlib.Path(target).name + ".data")
print(f"done {target} al4_nodes={touched}", flush=True)
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    v2 = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    script_dir = str(Path(__file__).resolve().parent)
    for source, target in [
        ("encoder-model.onnx", "encoder-model.q8.onnx"),
        ("model.onnx", "model.q8.onnx"),
    ]:
        subprocess.run(
            [sys.executable, "-c", SCRIPT, str(v2 / source), str(out / target), script_dir],
            check=True,
        )
        print(f"{source} -> {target}", flush=True)
    shutil.copyfile(v2 / "decoder_joint-model.onnx", out / "decoder_joint-model.onnx")
    shutil.copyfile(v2 / "vocab.txt", out / "vocab.txt")

    import onnxruntime

    metadata = {
        "variant": "q8-rtn-block32-al4-sym",
        "method": "pointwise lowering + DefaultWeightOnlyQuantConfig(bits=8,"
        " block_size=32, is_symmetric=True, accuracy_level=4, QOperator)",
        "source_dir": str(v2),
        "onnxruntime_version": onnxruntime.__version__,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "quantization-metadata.json"
        },
    }
    (out / "quantization-metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("metadata written", flush=True)


if __name__ == "__main__":
    main()
