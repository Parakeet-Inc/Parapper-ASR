---
pipeline_tag: text-to-speech
library_name: onnxruntime
license: openrail
base_model: Supertone/supertonic-3
base_model_relation: quantized
tags:
  - onnx
  - text-to-speech
  - quantized
  - int4
  - cpu
---

# Supertonic 3 ONNX Q4

This is an **Unofficial quantized derivative** of
[`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3).
It is not affiliated with or endorsed by Supertone Inc.

The source model is fixed to revision
`724fb5abbf5502583fb520898d45929e62f02c0b`.

## Distributed components

| Component | Distribution form |
|---|---|
| `duration_predictor.onnx` | Unchanged upstream FP32 |
| `text_encoder.onnx` | Unchanged upstream FP32 |
| `vector_estimator.onnx` | Asymmetric Q4 `MatMulNBits`, block size 16; final projection and depthwise convolutions remain FP32 |
| `vocoder.onnx` | Asymmetric Q4 `MatMulNBits`, block size 16; the five boundary layers listed in `MODIFICATIONS.md` remain FP32 |

The original `tts.json`, `unicode_indexer.json`, and the ten official preset
voice styles are included without modification. Custom or purchased voice
styles are not included.

## Reproduction and verification

The repository includes the conversion, staging, and verification tools under
`release-tools/`. The distribution manifest and checksum files cover every
runtime and publication file. The modified ONNX files also contain metadata
identifying the source revision and modification.

## Usage

The directory layout and ONNX input/output contracts match the upstream ONNX
package. A CPU build of ONNX Runtime with support for
`com.microsoft::MatMulNBits` is required.

Refer to the upstream model card for synthesis examples, supported languages,
model limitations, and responsible-use guidance.

## License

The model and this derivative are distributed under the included BigScience
Open RAIL-M license. Its use-based restrictions, including Attachment A,
continue to apply. See `THIRD_PARTY_NOTICES.md` and `MODIFICATIONS.md` for
provenance and modification details.
