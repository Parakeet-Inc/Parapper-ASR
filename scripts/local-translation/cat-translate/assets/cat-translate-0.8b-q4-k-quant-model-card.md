---
language:
  - ja
  - en
license: mit
base_model: cyberagent/CAT-Translate-0.8b
library_name: onnxruntime
tags:
  - translation
  - onnx
  - int4
---

# CAT-Translate 0.8B ONNX Q4 k_quant for Parapper

This is an ONNX Runtime conversion of
[`cyberagent/CAT-Translate-0.8b`](https://huggingface.co/cyberagent/CAT-Translate-0.8b)
for Japanese-English bidirectional translation in Parapper.

## Provenance and export contract

- Source repository: `cyberagent/CAT-Translate-0.8b`
- Source revision: `b555f93ef67846b6ed2773e0d2f16ceb0d30adb9`
- Source license: MIT
- Exporter: `onnxruntime-genai==0.14.1`
- Quantization: `int4_algo_config=k_quant`
- Execution provider target: CPU
- Embedding: asymmetric group-wise Q4, `block_size=16`
- Embedding operator: `GatherBlockQuantized`
- Output model: `model_q4.onnx` + `model_q4.onnx.data`

`distribution-manifest.json` records the graph contract, exact build environment,
file sizes, and SHA-256 values. `SHA256SUMS` covers every payload file and the
manifest.

## Prompt

Use the upstream instruction format:

```text
Translate the following {source_language} text into {target_language}.

{text}
```

Parapper wraps it with the source model's chat tokens.

## Limitations

- Only Japanese-to-English and English-to-Japanese are supported.
- This is a quantized derivative. Wording and semantics can drift from the
  source BF16 model, so release validation includes the extended bidirectional
  case set.
- Generated translation should be reviewed before high-impact use.

## Reproduction

See `release-tools/RELEASE_PROCEDURE.md` and the scripts stored beside this
model. The export refuses an unpinned source snapshot, a mismatched Python
environment, an incomplete output, or a graph that does not match the adopted
Q4 k_quant + Q4 block16 embedding contract. ONNX Runtime 1.27.0 is pinned only for
export and graph verification; Parapper v0.4.0 validates the model through its
sherpa-onnx bundled ONNX Runtime 1.24.4 on Windows.

## License

The source model and this converted model are distributed under the MIT License.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
