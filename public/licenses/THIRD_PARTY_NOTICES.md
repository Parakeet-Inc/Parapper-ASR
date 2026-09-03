# Third-party notices

Parapper is licensed under the MIT License. The following notices cover upstream
software, reference implementations, and model artifacts used by or associated
with Parapper's ASR implementation. These notices do not change the license of
Parapper itself.

## NVIDIA Parakeet TDT CTC 0.6B Ja

Parapper's Japanese Parakeet ONNX distribution is a modified export of
[NVIDIA Parakeet TDT CTC 0.6B Ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja),
licensed under CC-BY-4.0. Parapper re-exported the model to ONNX, separated and
shared the encoder between its CTC and TDT decoding paths, and quantized selected
weights for CPU inference.

## NVIDIA NeMo

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo), licensed under Apache-2.0, is used
as the pinned behavioral reference for Parapper's independently implemented CTC,
TDT, and cache-aware streaming ASR paths. NeMo source code is not embedded in the
application.

## sherpa-onnx

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), licensed under Apache-2.0,
is the source of several model distributions used by Parapper and remains a
compatibility reference. Parapper executes ASR models directly with ONNX Runtime
and does not include sherpa-onnx as a runtime dependency.

## static-embedding-japanese

[static-embedding-japanese](https://huggingface.co/hotchpotch/static-embedding-japanese),
licensed under MIT, is downloaded when the ReazonSpeech high-accuracy reranker is
used.
