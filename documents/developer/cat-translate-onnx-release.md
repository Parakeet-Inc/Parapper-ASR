# CAT-Translate 0.8B ONNX export / release procedure

This procedure reproduces and verifies the CAT-Translate 0.8B Q4 `k_quant`
distribution with an asymmetric group-wise Q4 token embedding
(`block_size=16`). The app catalog currently points to an immutable published
revision. A replacement revision must pass every gate in this document before
the catalog, settings option, or in-app license entry is updated.

A source revision mismatch, dependency drift, incomplete payload, wrong graph,
local path/credential leakage, or checksum mismatch must stop the release. Do
not temporarily point the app at another model or add a fallback URL.

## Fixed inputs

| item                                         | value                                      |
| -------------------------------------------- | ------------------------------------------ |
| source                                       | `cyberagent/CAT-Translate-0.8b`            |
| source revision                              | `b555f93ef67846b6ed2773e0d2f16ceb0d30adb9` |
| source license                               | MIT                                        |
| Python                                       | 3.12.x                                     |
| `onnxruntime-genai`                          | 0.14.1                                     |
| `onnxruntime` (export / Python verification) | 1.27.0                                     |
| `onnx`                                       | 1.22.0                                     |
| `onnx-ir`                                    | 0.2.1                                      |
| `transformers`                               | 4.57.6                                     |
| `huggingface-hub`                            | 0.36.2                                     |
| `torch`                                      | 2.12.1+cpu                                 |
| `tokenizers`                                 | 0.22.2                                     |
| `sentencepiece`                              | 0.2.1                                      |

ONNX Runtime 1.27.0 in this table belongs only to the isolated Python export
and graph-verification environment. It is not the ONNX Runtime linked into the
app. The application runtime uses the ONNX Runtime 1.24.4 libraries bundled
with sherpa-onnx 1.13.3.

The pinned source revision must resolve from the public Hugging Face
repository. The upstream model card and `LICENSE` identify the model as MIT
licensed:

- https://huggingface.co/cyberagent/CAT-Translate-0.8b
- https://huggingface.co/cyberagent/CAT-Translate-0.8b/blob/b555f93ef67846b6ed2773e0d2f16ceb0d30adb9/LICENSE

Do not substitute `main` if the pinned revision cannot be downloaded. Stop and
resolve the source provenance first.

The exporter verifies the exact size and SHA-256 of all eight files in the
pinned source snapshot, including the 1,586,121,792-byte
`model.safetensors`. It then checks the source config (`LlamaForCausalLM`, 24
layers, hidden size 1280, vocabulary 102400, untied embeddings, and
`transformers_version=4.57.6`) before invoking the builder.

## 1. Prepare a clean Windows environment

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .cat-onnx-venv
.\.cat-onnx-venv\Scripts\python.exe -m pip install --upgrade pip
.\.cat-onnx-venv\Scripts\python.exe -m pip install `
  -r .\scripts\local-translation\cat-translate\requirements-cat-onnx.txt
```

Download the exact public snapshot with the same environment. `snapshot_download`
returns the revision-named snapshot directory expected by the exporter:

```powershell
$SourceDir = & .\.cat-onnx-venv\Scripts\python.exe -c `
  "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='cyberagent/CAT-Translate-0.8b', revision='b555f93ef67846b6ed2773e0d2f16ceb0d30adb9'))"
```

Do not place the resolved local source path, Hugging Face cache path, token, or
credential in committed logs or release metadata.

## 2. Export the publish candidate

Use clean output and builder-cache directories:

```powershell
.\scripts\local-translation\cat-translate\export_cat_onnx_variants.ps1 `
  -Variant k_quant `
  -SourceDir $SourceDir `
  -OutDir .\artifacts\local-translation\cat-translate-0.8b-onnx-q4-k-quant `
  -CacheDir .\artifacts\local-translation\cat-builder-cache `
  -PythonPath .\.cat-onnx-venv\Scripts\python.exe
```

`-Force` is required to replace an existing output directory. The script does
not infer an ignored `onnx-optimize` venv/cache and does not download a different
revision. Builder console paths are redacted and are not stored in the payload.

The exporter first creates the audited `k_quant` graph in an isolated
intermediate directory, then runs `quantize_cat_embedding_gather.py` to replace
only the FP32 token-embedding Gather with the adopted Q4 block16 operator. The
intermediate directory is removed after conversion.

The publish candidate must contain:

- `chat_template.jinja`
- `genai_config.json`
- `model_q4.onnx`
- `model_q4.onnx.data`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer.model`
- `tokenizer_config.json`
- `LICENSE`
- `MODEL_CARD.md`
- `THIRD_PARTY_NOTICES.md`
- `build-metadata.json`
- `distribution-manifest.json`
- `SHA256SUMS`

Before writing a manifest, the verifier loads the complete distribution
through `onnxruntime-genai`. This binds `model_q4.onnx` to its external tensor
data and rejects a truncated or unloadable `model_q4.onnx.data`. It then
requires this graph contract:

- `MatMulNBits(bits=4)`: 120 nodes
- `MatMulNBits(bits=8)`: 1 node (the k_quant output head)
- `Gather`: 1 non-embedding node
- `GatherBlockQuantized`: 1 token-embedding node
- embedding weight: UINT4 `[102400, 1280]`
- embedding scale: FLOAT `[102400, 80]`
- embedding zero-point: UINT4 `[102400, 80]`
- embedding attributes: `block_size=16`, `gather_axis=0`,
  `quantize_axis=1`

Those values are the required distribution graph contract.

## 3. Verify the app runtime on Windows

Do not copy the export environment's ONNX Runtime 1.27 DLLs into the app. Run
the Rust app path against the packaged ONNX Runtime 1.24.4 libraries from
sherpa-onnx:

```powershell
$env:PARAPPER_CAT_TRANSLATION_MODEL_DIR = `
  (Resolve-Path .\artifacts\local-translation\cat-translate-0.8b-onnx-q4-k-quant).Path
cargo test --manifest-path .\src-tauri\Cargo.toml `
  smoke_cat_translate_direction_table_returns_target_language `
  -- --ignored --nocapture
```

The direction-table test sends multiple requests through the same loaded
engine. Also run the packaged Windows app and confirm Japanese-to-English,
English-to-Japanese, and at least two consecutive requests through the normal UI
path.

## 4. Assemble the Hugging Face upload folder

After the candidate verification and application runtime checks pass, assemble one
directory for the manual Hugging Face upload:

```powershell
.\.cat-onnx-venv\Scripts\python.exe `
  .\scripts\local-translation\cat-translate\stage_cat_hf_release.py `
  --candidate-dir .\artifacts\local-translation\cat-translate-0.8b-onnx-q4-k-quant `
  --output-dir .\artifacts\local-translation\hf-upload\cat-translate-0.8b-onnx-q4-k-quant `
  --python .\.cat-onnx-venv\Scripts\python.exe
```

The staging command verifies the candidate before copying and verifies the
staged payload again. The folder contains the 14 core distribution files, an
HF-rendered `README.md`, the audited conversion/verification scripts under
`release-tools/`, a Parakeet MIT `release-tools/LICENSE`, a bundle-relative
`release-tools/RELEASE_PROCEDURE.md`, and `HF_UPLOAD_SHA256SUMS` covering the
complete upload. The entire `artifacts/local-translation/` tree is ignored by
Git; do not force-add the approximately 1 GB model data.

Upload the contents of that single folder to the user's Hugging Face model
repository. Record the immutable publication commit; do not use a mutable
branch as the app base URL.

## 5. Verify a clean download

Download the published files into a fresh directory, then run:

```powershell
.\.cat-onnx-venv\Scripts\python.exe `
  .\scripts\local-translation\cat-translate\verify_cat_onnx_distribution.py `
  .\artifacts\local-translation\cat-translate-published-download
```

This recalculates every size/SHA-256 value, reloads the ONNX graph without the
external tensor data, and compares the downloaded payload with
`distribution-manifest.json` and `SHA256SUMS`.

Also verify `HF_UPLOAD_SHA256SUMS` from the repository root so the uploaded
reproduction tools and model card are covered. The app catalog uses only the 14
core distribution files and their immutable Hugging Face revision URLs.

Only after this clean-download check and the Windows app check pass should
`src-tauri/src/model/catalog.rs` be updated to a new immutable base URL and the
same 14 required files. The settings option and in-app CAT-Translate MIT
license entry must be updated in the same change if the distribution identity
changes. Hash mismatch, missing notice/license, or a failed download is an
error; it must not fall back to another endpoint, port, model, or local cache.
