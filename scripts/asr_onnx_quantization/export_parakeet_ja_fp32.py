"""Export parakeet-tdt_ctc-0.6b-ja (.nemo) to fp32 ONNX, TDT and CTC branches.

Produces the onnx-asr file layout the diagnostics engines consume:
  <out>/encoder-model.onnx (+ .onnx_data)      TDT encoder
  <out>/decoder_joint-model.onnx (+ .onnx_data) fused prediction/joint
  <out>/model.onnx (+ .onnx_data)               CTC branch (single graph)
  <out>/vocab.txt                               copied token table
  <out>/export-metadata.json
"""

import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

NEMO_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--parakeet-tdt_ctc-0.6b-ja"
    / "snapshots/44edb27eea9317daf89333e75eb830db4b1cc298/parakeet-tdt_ctc-0.6b-ja.nemo"
)
VOCAB_SOURCE = (
    Path.home()
    / "AppData/Roaming/com.parakeet-inc.parapper/models"
    / "onnx-asr-nvidia-parakeet-tdt_ctc-0.6b-ja-4353e7b9/vocab.txt"
)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    import nemo
    import torch
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(str(NEMO_PATH), map_location="cpu")
    model.eval()

    # TDT branch (default cur_decoder=rnnt): writes encoder-model.onnx and
    # decoder_joint-model.onnx next to the given path.
    model.export(str(out / "model_tdt.onnx"), verbose=False)
    print("tdt export done", flush=True)

    # CTC branch: single graph.
    model.set_export_config({"decoder_type": "ctc"})
    model.export(str(out / "model.onnx"), verbose=False)
    print("ctc export done", flush=True)

    shutil.copyfile(VOCAB_SOURCE, out / "vocab.txt")

    files = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "export-metadata.json":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
    metadata = {
        "date": date.today().isoformat(),
        "source_nemo": str(NEMO_PATH),
        "source_revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
        "nemo_version": nemo.__version__,
        "torch_version": torch.__version__,
        "export_command": "export_parakeet_ja_fp32.py (TDT default export + set_export_config decoder_type=ctc)",
        "files": files,
    }
    (out / "export-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("metadata written", flush=True)


if __name__ == "__main__":
    main()
