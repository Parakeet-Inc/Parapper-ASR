# recognition

音声入力から `RecognizedTextOutput` を作るまでの pipeline。

## 構造

```text
recognition/
  control/        orchestration、session state、driver priority
  segmentation/   VAD frame -> SegmentEvent
  transcription/  ASR request/result workflow、route / SLI
  turn/           TurnDraft mutation、continue/final/timeout、grammar boundary
```

## Production 経路

```text
RunningRecognitionInput
  -> AudioInputProcessor
  -> VAD
  -> RecognitionDriver
  -> SegmentationFlow
  -> RecognitionSession
  -> transcription::flow
  -> turn::{transcript, flow, boundary_flow}
  -> TurnOutputSink
  -> delivery
```

## 境界

- `control/` は全体の順序と state を持つ。ASR や Turn の判断そのものは stage module に置く。
- `segmentation/` は VAD 結果から ASR に投げる segment event を作る。
- `transcription/` は ASR を使った文字起こし workflow を持つ。ASR engine はこの stage の内側。
- `turn/` は transcript を Turn に反映し、open/continue/final/timeout/output を決める。

詳細は [documents/developer/architecture/02-recognition-modules.md](../../../documents/developer/architecture/02-recognition-modules.md) と [documents/developer/architecture/03-recognition-internals.md](../../../documents/developer/architecture/03-recognition-internals.md) を参照。
