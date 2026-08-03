# Supertonic 3 ONNX Q4 release procedure

This procedure builds and publishes the adopted Supertonic 3 ONNX Q4
distribution. The source is fixed to
`Supertone/supertonic-3@724fb5abbf5502583fb520898d45929e62f02c0b`.

The distribution contract is deliberately narrow:

- `duration_predictor.onnx`: copied from upstream without modification
- `text_encoder.onnx`: copied from upstream without modification
- `vector_estimator.onnx`: Q4 `MatMulNBits`, block size 16
- `vocoder.onnx`: adopted D boundary, Q4 `MatMulNBits`, block size 16

The complete upstream BigScience Open RAIL-M license and modification notices
must remain in the distribution.

## 1. Prepare the environment

Create an isolated Python 3.12 environment and install the pinned packages:

```powershell
uv venv .supertonic3-q4-venv --python 3.12
uv pip install --python .\.supertonic3-q4-venv\Scripts\python.exe `
  -r .\scripts\supertonic3_onnx\requirements-supertonic3-q4.txt
```

Download the audited upstream files into a clean directory:

```powershell
hf download Supertone/supertonic-3 `
  --revision 724fb5abbf5502583fb520898d45929e62f02c0b `
  --include "onnx/*" `
  --include "voice_styles/*" `
  --include "LICENSE" `
  --local-dir .\artifacts\supertonic3\upstream-724fb5a
```

Do not add custom or purchased voice styles.

## 2. Build and verify the candidate

```powershell
.\.supertonic3-q4-venv\Scripts\python.exe `
  .\scripts\supertonic3_onnx\build_supertonic3_q4_distribution.py `
  .\artifacts\supertonic3\upstream-724fb5a `
  .\artifacts\supertonic3\candidate

.\.supertonic3-q4-venv\Scripts\python.exe `
  .\scripts\supertonic3_onnx\verify_supertonic3_q4_distribution.py `
  .\artifacts\supertonic3\candidate
```

The builder rejects any upstream file whose size or SHA-256 differs from the
fixed revision. It also requires the regenerated pre-notice Q4 ONNX hashes to
match the adopted vector estimator and vocoder D artifacts. The verifier
checks the graph contract, the untouched FP32 files, modification metadata,
license and notices, manifest, checksums, and ONNX Runtime loading.

## 3. Verify the application runtime

Before publication, load all four ONNX files with the exact ONNX Runtime
libraries shipped by Parapper. Synthesize Japanese and English text through
the normal Rust pipeline, including consecutive requests and more than one
preset speaker. A failure is blocking; do not substitute another runtime or
model.

## 4. Stage and upload

```powershell
.\.supertonic3-q4-venv\Scripts\python.exe `
  .\scripts\supertonic3_onnx\stage_supertonic3_q4_hf_release.py `
  --candidate-dir .\artifacts\supertonic3\candidate `
  --output-dir .\artifacts\supertonic3\hf-upload `
  --python .\.supertonic3-q4-venv\Scripts\python.exe
```

Upload only the staged folder. Record the full publication commit returned by
Hugging Face. The application catalog must use
`/resolve/<40-character-commit>` and must not use `main` or another fallback.

## 5. Clean-download verification

Download the published full commit into a new directory. Verify
`HF_UPLOAD_SHA256SUMS`, then run
`verify_supertonic3_q4_distribution.py` again. Only after this succeeds may
the immutable URL and the published per-file integrity values be added to the
Parapper catalog.
