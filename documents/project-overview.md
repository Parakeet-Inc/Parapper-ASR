# プロジェクト全体像

<!-- cspell:words parapper YNC -->

Parapper は Rust + TypeScript + Tauri で構成されています。
TypeScript の UI から Tauri command を呼び、Rust backend が音声入力、認識、翻訳、読み上げ、外部連携を処理します。

UI は設定と実行状態の表示に集中し、認識後の配送やキュー方針は Rust 側の各モジュールが持ちます。

## 全体像

```text
フロントエンド
  SettingsPanel
    ├─ 接続 / モデル設定 ───────────────┐
    ├─ VAD / ASR / ノイズ除去設定 ──────┼─ Tauri commands -> config / model / state
    ├─ 翻訳設定 ────────────────────────┤
    └─ 読み上げ設定 ────────────────────┘

  RuntimePanel
    ├─ 音声デバイス / 開始停止 ───────── commands -> state -> recognition
    ├─ RecognitionLog <──────────────── delivery::sinks::ui_event
    └─ TranslationSidePanel <─────────── translation::event

バックエンド
  音声入力
    -> recognition
    -> delivery
       ├─ UI への認識結果 event
       ├─ YNC text input
       ├─ translation
       │   ├─ 翻訳結果 event
       │   └─ 翻訳結果の読み上げを synthesis へ依頼
       └─ 認識結果の読み上げを synthesis へ依頼
           ├─ YNC speech
           └─ local TTS -> playback
```

バックエンドモジュールの詳細地図は [src-tauri/src/README.md](../src-tauri/src/README.md) に置きます。
大きい領域の詳細は [recognition/README.md](../src-tauri/src/recognition/README.md) と [synthesis/README.md](../src-tauri/src/synthesis/README.md) を参照します。

## UI とバックエンドの対応

### アプリ全体

| フロントエンド | ユーザーから見た役割 | バックエンド側の対応 |
| --- | --- | --- |
| `src/app.tsx` | 画面全体の構成、表示言語、モデル状態に応じた開始可否 | `commands::get_config_presets`、設定とモデル状態を扱う hook |
| `StatusBadges` | 実行状態、接続状態、モデル状態の表示 | `parapper://status`、`parapper://connection-state`、モデル状態 |
| `OnboardingModal` | 初回プリセット選択とモデルダウンロード | `config_preset`、command 経由の `model::ensure_models_downloaded` |

### 設定パネル

| UI tab / component | 設定または操作 | バックエンドの責務 |
| --- | --- | --- |
| `ConnectionSettings` | YNC text input port、plugin HTTP port、モデルのダウンロード状態 | `connect::ync::discovery`, `model/`, `commands.rs` |
| `NoiseCancellationSettings` | ノイズ除去モデルと有効/無効 | `config.rs`, `recognition::engines::noise_cancellation` |
| `VadSettings` | VAD 閾値、Turn Detector の途中経過表示 / 完了判定タイミング | `config.rs`, `recognition::segment_builder` |
| `AsrSettings` | ASR model、多言語 ASR、Namo TD、full ASR の再実行 | `config.rs`, `recognition::route`, `recognition::transcription`, `model/` |
| `TranslationSettings` | 翻訳の有効/無効、タイミング、source model から target language への mapping | `config.rs`, `delivery::common::mapping`, `translation/` |
| `SpeechSettings` | 読み上げ mapping、YNC talker、local TTS voice、出力デバイス、音量 | `config.rs`, `synthesis/request.rs`, `synthesis/local/`, `playback/` |
| `OtherSettings` | preset、log 上限、reset | `commands.rs`, `config_preset.rs`, `config.rs` |
| `LicenseSettings` | license 表示 | フロントエンド側の表示データと bundled license metadata |

### 実行パネル

| UI 領域 | バックエンドとのやり取り | イベント / コマンド |
| --- | --- | --- |
| 入力デバイス選択 | 入力デバイス設定を保存 | `save_config` |
| 認識の開始 / 停止 | 音声入力と認識 pipeline を開始または停止 | `start_recognition`, `stop_recognition`, `get_recognition_status` |
| 入力レベル表示 | 現在の入力音量 | `parapper://input-level` |
| VAD 状態表示 | 現在の speech/silence state | `parapper://vad-state` |
| `RecognitionLog` | 認識結果の interim/final upsert | `parapper://recognized-text` |
| `TranslationSidePanel` | 翻訳結果の target-language upsert | `parapper://translated-text` |
| 読み上げ停止ボタン | YNC plugin の speech queue を停止 | `neo_speech_stop` |
| 音声デバイス更新 | 入出力デバイス一覧を再取得 | `commands.rs` の audio device commands |

## 実行時の流れ

### 認識結果を UI に表示する流れ

```text
start_recognition command
  -> state.rs
  -> audio/input.rs が PCM を取得
  -> recognition::RecognitionPipeline
     -> VAD
     -> SegmentBuilder
     -> ASR worker
     -> Turn / TurnDraft / TurnConfirmed
  -> delivery::dispatch_recognized_text
  -> parapper://recognized-text
```

### 認識結果を外部出力へ配送する流れ

```text
delivery::dispatch_recognized_text
  -> translation::submit_recognized_text
  -> synthesis::submit_recognized_text
  -> connect::YncTextInputTransport
```

`delivery` は配送先への fan-out を持ちますが、翻訳キュー、読み上げキュー、PCM 再生は持ちません。

### 翻訳と読み上げの流れ

```text
translation
  -> YNC translate client
  -> parapper://translated-text
  -> synthesis for translated speech

synthesis
  ├─ YNC speech client
  └─ local TTS generation -> playback
```

YNC speech は外部プラグイン側にキューがあるため、Parapper 側では送信順を守ります。
ローカル TTS は voice 別に生成キューを持ち、PCM 再生だけを `playback/` で直列化します。

## Tauri 境界

### イベント

| イベント | 発行元 | UI の受け取り先 | 目的 |
| --- | --- | --- | --- |
| `parapper://input-level` | `audio/input.rs` | `RuntimePanel` | 入力レベル表示 |
| `parapper://vad-state` | `recognition/pipeline.rs` | runtime state | speech/silence state |
| `parapper://recognized-text` | `delivery/sinks/ui_event.rs` | `RecognitionLog` | ASR text upsert |
| `parapper://translated-text` | `translation/event.rs` | `TranslationSidePanel` | translated text upsert |
| `parapper://speech-request` | `synthesis/manager.rs` | runtime state | speech accepted/failure、delay warning |
| `parapper://connection-state` | delivery/connect checks | `StatusBadges` | 外部接続の availability |
| `parapper://model-download-progress` | `model/manager.rs` | onboarding/settings | モデルダウンロード進捗 |
| `parapper://asr-missing` | `recognition/transcription/worker_runtime.rs` | `RecognitionLog` warning | ASR route missing |
| `parapper://error` | `error_event.rs` | notifications/runtime state | structured warning/error |

### コマンド

| コマンド分類 | 例 | バックエンドの責務 |
| --- | --- | --- |
| 設定 | `get_config`, `save_config`, `reset_config`, preset commands | `config.rs`, `config_preset.rs`, `commands.rs` |
| 実行状態 | `start_recognition`, `stop_recognition`, `get_recognition_status` | `state.rs`, `recognition/` |
| モデル | `get_model_status`, `download_models`, `has_any_model_installed` | `model/` |
| 音声ファイル / log | `save_recognition_csv`, `save_asr_input_wav` | `commands.rs`, `audio/` |
| YNC plugin | `fetch_neo_voice_list`, `neo_speech_stop`, `neo_speech_test` | `connect::ync`, `synthesis` config values |

Commands は UI と backend の薄い境界です。
認識、翻訳、読み上げ、外部連携の業務方針は `commands.rs` ではなく、それぞれの責務を持つ module に置きます。
