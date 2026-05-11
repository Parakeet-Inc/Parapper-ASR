# `synthesis/` — テキストから音声を合成する層

`synthesis/` は TTS 専用層です。認識結果または翻訳結果のテキストから読み上げリクエストを作り、
YNC speech または local TTS に渡します。翻訳と再生はこの層の外に分離しています。

## 構成

```text
synthesis/
├── mod.rs             公開入口
├── request.rs         QueuedSpeechRequest の生成
├── queue.rs           stale 除去と送信順序
├── manager.rs         TTS worker
├── artifact.rs        TtsArtifact
├── local/             local TTS voice 別生成キュー
│   ├── mod.rs
│   ├── queue.rs       voice 別キュー / worker
│   ├── engine.rs      Sherpa / Supertonic adapter
│   ├── playback.rs    生成済み PCM を playback へ渡す
│   ├── audio.rs       生成済み音声 item
│   └── key.rs         キュー key
├── clients/
│   └── ync_speech.rs  YNC speech HTTP client
└── engines/
    ├── sherpa_onnx.rs  cfg(test) stub を内部に持つ
    └── supertonic_onnx.rs
```

## 方針

- `synthesis::submit_recognized_text` は認識結果から TTS request を作る
- `translation` から翻訳結果 TTS を起動する場合は `build_speech_requests_with_source_meta` と `spawn_speech_requests` を使う
- YNC speech は `clients/ync_speech.rs` で HTTP request を送る
- YNC speech は相手側 plugin のキューに渡すため、送信順だけを守り、ローカル再生完了は待たない
- local TTS は voice 別キューで並列生成する
- local TTS の生成後 PCM は `playback::PlaybackManager` へ渡し、再生は直列にする
- ローカル推論エンジンは `engines/`、外部 API は `clients/` に置く

## 所有しない責務

- 翻訳 request と translated-text event は `translation/`
- PCM のデバイス出力は `playback/`
- 認識結果をどの sink へ配送するかは `delivery/`
