# NVIDIA NeMo reference contract

This directory records reproducible inputs for checking the direct ONNX Runtime
ASR implementations.
For NVIDIA model families, the pinned NVIDIA NeMo Python implementation and model
revision are normative. The current sherpa-onnx output is a compatibility baseline,
not the decoder oracle.

## Files

- `reference-manifest.json`: supported application models, normative model revisions,
  current production artifacts, and oracle priority.
- `runtime-lock.json`: Rust and native ONNX Runtime versions used by the reference
  environment.
- `fixture.schema.json`: common envelope for algorithm, tensor, audio, and streaming
  reference fixtures.
- `export_reference.py`: validates the contract and captures pinned NeMo offline
  transcriptions without resolving `main` at run time.
- `export_ctc_beam_fixture.py`: captures the selected no-fusion
  `BeamBatchedCTCInfer` algorithm contract.
- `frontend/`: a small torch/librosa-only environment that exports frontend
  tensors without importing or building the complete NeMo package.
- `pyproject.toml` / `uv.lock`: the Python reference environment.

## Commands

Validate the checked-in contract without downloading a model:

```powershell
uv run --project diagnostics/nemo-reference python diagnostics/nemo-reference/export_reference.py validate
```

On Windows, use a short `UV_CACHE_DIR` when syncing the full NeMo environment;
the pinned NeMo source exceeds the legacy path limit when built below a long
repository cache path. `huggingface-hub[hf-xet]` is locked so the `.nemo`
artifact does not depend on the stalled regular-HTTP Xet fallback.

Regenerate the small frontend and CTC batched-beam fixtures:

```powershell
uv run --project diagnostics/nemo-reference/frontend --locked python `
  diagnostics/nemo-reference/frontend/export_frontend_fixture.py
uv run --project diagnostics/nemo-reference --locked python `
  diagnostics/nemo-reference/export_ctc_beam_fixture.py
```

Capture an offline reference from a pinned `.nemo` artifact:

```powershell
uv run --project diagnostics/nemo-reference python diagnostics/nemo-reference/export_reference.py transcribe `
  --app-model nemo_parakeet_tdt_0_6b_v2_int8 `
  --audio C:\path\to\16khz-mono.wav `
  --output diagnostics/nemo-reference/fixtures/parakeet-v2-audio.json
```

The command records the audio SHA-256, NeMo commit, model revision, Python package
versions, text, token IDs when exposed by NeMo, score, and timestamps. Do not commit
licensed or personal audio. Commit only fixtures whose audio provenance is documented.

Streaming partial/delta fixtures use a separate export path. The offline command
does not treat batch transcription as a streaming oracle.

## Update rules

1. Do not update `ort`, the native runtime, a model revision, and a decoder in one commit.
2. A revision change requires an explicit manifest diff and regenerated fixtures.
3. NVIDIA Python wins when it disagrees with current sherpa output, unless the pinned
   model cannot be loaded by the pinned NeMo commit; that is a broken reference setup.
4. ReazonSpeech is outside the NVIDIA oracle and has a separately recorded contract.
