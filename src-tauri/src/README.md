# `src/` - Rust バックエンド地図

このディレクトリは Tauri 側バックエンドの実装です。
フロントエンドを含むアプリ全体像と UI 対応は [`documents/project-overview.md`](../../documents/project-overview.md) にまとめます。

詳細ページ:

- [`recognition/`](recognition/) - 音声入力から `RecognizedTextOutput` を作る認識パイプライン
- [`synthesis/`](synthesis/) - 認識結果または翻訳結果から読み上げリクエストを作る TTS 層

## モジュール間の関係

| モジュール | 責務 | 呼び出すもの / 依存するもの | 持たない責務 |
| --- | --- | --- | --- |
| `audio/` | 入力ストリーム、リサンプル、デノイズ、入力レベル、PCM 再生補助 | `recognition`, `commands`, `playback` | VAD、ASR、Turn 状態、配送方針 |
| `recognition/` | 音声入力 worker、segmentation、transcription、Turn 状態 | `audio`, `model`, `delivery` | UI layout、翻訳、TTS |
| `delivery/` | `RecognizedTextOutput` を登録済み sink へ配送する | `translation`, `synthesis`, `connect`, Tauri emit | 翻訳キュー、TTS キュー、PCM 再生 |
| `translation/` | 翻訳リクエスト生成、古い interim の除去、YNC 翻訳 worker、翻訳結果 event | `connect::ync`, 翻訳結果読み上げ用の `synthesis` | 認識、local TTS 生成 |
| `synthesis/` | 読み上げリクエスト生成、YNC speech worker、local TTS 生成キュー | `connect::ync`, `playback`, local TTS engine | 翻訳 request/response、PCM device output |
| `playback/` | local PCM の直列再生キュー | `audio::play_mono_samples` | YNC speech、TTS 生成 |
| `connect/` | YNC、OSCQuery、registry、test server などの外部 I/O client | `delivery`, `translation`, `synthesis`, `commands` から利用 | 業務ルール、キュー方針 |
| `model/` | モデル catalog、モデル path、install/download status | `config`, `commands`, `recognition`, `synthesis` | 実行中 pipeline state |
| `config.rs` | 永続化設定と正規化 | `commands` 経由で UI と接続 | runtime worker |
| `state.rs` | アプリ全体の実行状態と start/stop wiring | `recognition`, `model`, `synthesis` prewarm | UI rendering |

依存方向はおおむね左から右に保ちます。

```text
commands/state
  -> model/config
  -> recognition
  -> audio
  -> delivery
  -> translation / synthesis
  -> connect / playback
```

例外は意図的に狭く保ちます。

- `translation` が `synthesis` を呼ぶのは、final 翻訳結果の読み上げを enqueue する場合だけ。
- `synthesis` が `playback` を呼ぶのは、local TTS の PCM を再生する場合だけ。
- `connect` は外部 I/O library とし、業務モジュールへ callback しない。

## 実行時データフロー

### 認識

```text
start_recognition command
  -> state.rs
  -> recognition::RunningRecognitionInput
     -> audio/input.rs が PCM を取得し、リサンプル/デノイズ済み chunk を返す
     -> AudioInputProcessor
     -> VadEngine
     -> RecognitionDriver
     -> SegmentationFlow
     -> RecognitionSession
     -> transcription::flow
     -> turn::{transcript, flow, boundary_flow}
     -> Turn / TurnDraft / TurnConfirmed
  -> delivery::dispatch_recognized_text
```

`recognition/` は認識結果だけを作ります。翻訳 mapping、TTS mapping、UI layout は決めません。

### 配送

```text
delivery::dispatch_recognized_text
  -> TranslationSink     -> translation::submit_recognized_text
  -> SynthesisSink       -> synthesis::submit_recognized_text
  -> UiEventSink         -> parapper://recognized-text
  -> YncTextSink         -> connect::YncTextInputTransport
```

`delivery` は fan-out 順序と `DispatchContext` を持ちます。
翻訳キュー、TTS キュー、再生は持ちません。

現在の sink 順序:

1. `translation`
2. `synthesis`
3. `ui_event`
4. `ync_text`

### 翻訳

```text
translation::submit_recognized_text
  -> request.rs が TranslationRequest を作る
  -> queue.rs が古い interim request を除去する
  -> manager.rs worker
  -> clients/ync_translate.rs POST /
  -> event.rs が parapper://translated-text を emit する
  -> final translated text は読み上げを synthesis へ enqueue できる
```

重要な方針:

- final 翻訳は、同じ構造化 source の pending interim job を破棄する。
- 翻訳 HTTP は単一 worker で直列処理する。
- 翻訳結果の読み上げは final translated text からだけ作る。

### 読み上げ合成

```text
synthesis::submit_recognized_text または translation::event
  -> request.rs が QueuedSpeechRequest を作る
  -> queue.rs が古い speech request を除去する
  -> manager.rs が backend を振り分ける
     ├─ clients/ync_speech.rs POST /              (外部 plugin queue)
     └─ local/
        ├─ queue.rs voice 別の生成キュー
        ├─ engine.rs Sherpa / Supertonic adapter
        └─ playback.rs -> playback::PlaybackManager
```

重要な方針:

- YNC speech は送信順を守るが、plugin 側の再生完了は待たない。
- local TTS 生成は voice 別キューに分ける。
- local PCM 再生は `playback/` で直列化する。

### 再生

```text
synthesis/local/playback.rs
  -> TtsArtifact
  -> PlaybackManager
  -> PlaybackQueue
  -> audio::play_mono_samples
  -> PlaybackEvent callback
```

`playback/` は local PCM 出力専用です。YNC speech は外部 plugin 側で処理され、このモジュールを通りません。

## 外部接続

| 外部接続先 | バックエンド path | 備考 |
| --- | --- | --- |
| YNC text input | `delivery/sinks/ync_text.rs` -> `connect::YncTextInputTransport` | `GET /api/input?text=...`、送信後は待たない |
| YNC translate | `translation/clients/ync_translate.rs` -> `connect::YncPluginClient` | plugin HTTP `POST /` |
| YNC speech | `synthesis/clients/ync_speech.rs` -> `connect::YncPluginClient` | plugin HTTP `POST /`、長めの timeout |
| YNC voice list / stop | `commands.rs` -> `connect::YncPluginClient` | ユーザー操作 command |
| VRChat mute | `delivery/sinks/vrchat_mute.rs` -> `connect::osc` | YNC text delivery の前に確認 |
| Windows registry | `connect::registry` / `connect::ync::discovery` | YNC ports を読む |

YNC plugin の port discovery は `HKCU\Software\YukarinetteConnectorNeo\TransServer` を読み、
見つけた port に `version` を投げて確認します。
ポート走査によるフォールバックは追加しません。誤ったローカルサービスへコマンドを送る危険があるためです。

## テスト配置

| 関心ごと | 推奨するテスト場所 |
| --- | --- |
| Segment の挙動 | `recognition/segmentation/segment/builder/tests.rs` |
| Turn / transcription / control の横断挙動 | `recognition/control/tests`, `recognition/transcription`, `recognition/turn` |
| Delivery mapping/timing | `delivery/tests.rs` |
| 翻訳キュー / stale policy | `translation/queue.rs` tests, `pipeline_tests.rs` |
| YNC request/response payload | mock HTTP server を使う `connect/ync/tests.rs` |
| 読み上げキュー / 順序 / local TTS | `synthesis/queue.rs`, `synthesis/local/queue.rs` tests |
| delivery -> translation -> synthesis の横断 | `pipeline_tests.rs` |
| UI type/build regression | `pnpm build` |

runtime bug を直すときは、望ましい挙動から mock または unit test を先に書き、その後に実装を変えます。

## ドキュメント方針

- このファイルは Rust backend のモジュール地図です。
- フロントエンドとバックエンドの対応、アプリ全体の流れは [`documents/project-overview.md`](../../documents/project-overview.md) に置きます。
- 詳細 README は大きい domain である `recognition/` と `synthesis/` にだけ残します。
- 長期の計画メモや完了済みレビュー文書は残しません。一般化できる作業ルールは root の [`AGENTS.md`](../../AGENTS.md)、人間向けの開発手順は [`documents/development-help.md`](../../documents/development-help.md) に移します。
- top-level backend module、Tauri event、command を追加するときは、この README も同じ変更で更新します。
