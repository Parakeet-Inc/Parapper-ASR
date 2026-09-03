# `synthesis/` — テキストから音声を合成する層

`synthesis/` は TTS 専用層です。認識結果または翻訳結果のテキストから読み上げリクエストを作り、
YNC speech または local TTS に渡します。翻訳と再生はこの層の外に分離しています。

## 構成

```text
synthesis/
├── mod.rs             公開入口
├── request.rs         QueuedSpeechRequest の生成
├── queue.rs           stale 除去と送信順序
├── dispatch.rs        worker、YNC/local振り分け、Tauri event
└── local.rs           voice別生成キューとplayback接続
```

## 方針

- `synthesis::submit_recognized_text` は認識結果から TTS request を作る
- `translation` から翻訳結果 TTS を起動する場合は `build_speech_requests_with_source_meta` と `spawn_speech_requests` を使う
- YNC speech は `dispatch.rs` からplugin HTTP clientへ渡す
- YNC speech は相手側 plugin のキューに渡すため、送信順だけを守り、ローカル再生完了は待たない
- local TTS は voice 別キューで並列生成する
- local TTS の生成後 PCM は `playback::PlaybackManager` へ渡し、再生は直列にする
- Sherpa/SupertonicのSession選択、モデル固有の話者状態、生成PCM契約は
  `parapper-models::tts::LocalTtsEngine` が所有する
- `local.rs` はTauriのmodel path解決、voice別worker、再生キュー、event接続だけを所有する

## 所有しない責務

- 翻訳 request と translated-text event は `translation/`
- PCM のデバイス出力は `playback/`
- 認識結果をどの sink へ配送するかは `delivery/`
