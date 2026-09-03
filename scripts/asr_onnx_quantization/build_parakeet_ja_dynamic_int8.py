"""Build the adopted dynamic INT8 variant of the parakeet ja v2 fp32 export.

quantize_dynamic (QInt8, per-channel, MatMul only) on the extracted encoder and
the CTC single graph. decoder_joint stays fp32 (int8 was measured neutral).
Adopted as the speed-priority variant (1.8x faster than fp32 at 4 threads,
TDT diagnostic CER +0.09pp on JSUT-1000).

Usage:
  python build_parakeet_ja_dynamic_int8.py <v2-fp32-dir> <output-dir>
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


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
    for source, target in [
        ("encoder-model.onnx", "encoder-model.int8.onnx"),
        ("model.onnx", "model.int8.onnx"),
    ]:
        quantize_dynamic(
            model_input=str(v2 / source),
            model_output=str(out / target),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
            per_channel=True,
            use_external_data_format=True,
        )
        print(f"{source} -> {target}", flush=True)
    shutil.copyfile(v2 / "decoder_joint-model.onnx", out / "decoder_joint-model.onnx")
    shutil.copyfile(v2 / "vocab.txt", out / "vocab.txt")

    import onnxruntime

    metadata = {
        "variant": "dynamic-qint8-perchannel",
        "method": "onnxruntime.quantization.quantize_dynamic"
        " (QInt8, per_channel, op_types=[MatMul])",
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
