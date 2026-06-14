# 開発補助

<!-- cspell:words corpus dataset FLEURS ja_jp jvs parapper ReazonSpeech reqwest Silero sherpa OSCQuery CTC UniDic vibrato rkyv -->

Parapper の開発手順、再現テスト、モック、lint、配布、モデル仕様をまとめます。
アプリ全体像は [project-overview.md](project-overview.md) を参照します。

## 開発

依存関係を入れたあと、リポジトリ root で実行します。

```powershell
proto use
pnpm i
pnpm build
cargo test -p parapper
cargo check -p parapper
```

開発起動は以下を実行します。

```powershell
pnpm tauri dev
```

## 再現テストと診断

不具合を調べるときは、現象を再現できる単体テスト、統合寄りテスト、またはローカル診断コマンドを残しておくとレビューしやすくなります。特にゆかコネNEO連携、翻訳、読み上げ、ASR 入力のように外部プロセスやタイミングが絡む不具合は、外部アプリの設定不備と Parapper 側の遅延・キュー詰まりを切り分けられる形にします。

### ローカル dataset path

JVS / FLEURS-R を使う診断テストは、開発者ごとのローカル dataset path を git に含めないため、環境変数から参照します。リポジトリ root の [.env.example](../.env.example) を `.env` にコピーし、自分の環境に合わせて書き換えます。`.env` は `.gitignore` の対象です。

```powershell
Copy-Item .env.example .env
```

設定する値:

- `JVS_ROOT`: JVS corpus の root。直下に `jvs001`, `jvs002`, ... があるディレクトリを指定する。
- `FLEURS_R_ROOT`: FLEURS-R の root。直下に `ja_jp`, `en_us`, ... があるディレクトリを指定する。
- `PARAPPER_MODELS_ROOT`: 任意。診断コマンドや real ASR test が参照するモデル root。未指定なら通常の app data 配下を使う。
- `PARAPPER_ASR_MODEL_DIR`: 任意。`verify_jvs_asr` で使う ReazonSpeech ASR model directory を明示したい場合だけ指定する。
- `PARAPPER_VAD_MODEL`: 任意。`verify_jvs_asr` で使う Silero VAD ONNX file を明示したい場合だけ指定する。

`src-tauri/src/recognition/control/tests` 配下の診断テストは、プロセス環境変数を優先し、未設定なら root `.env` または `src-tauri/.env` を読みます。`src-tauri/tests/jvs_asr.rs` と `verify_jvs_asr` は `real-asr-tests` feature が必要です。`verify_jvs_asr` はプロセス環境変数の `JVS_ROOT` を読むか、CLI の `--jvs-root PATH` で指定します。

```powershell
$env:JVS_ROOT = ".\datasets\jvs\jvs_ver1"
$env:FLEURS_R_ROOT = ".\datasets\fleurs-r"
cargo test -p parapper --features real-asr-tests -- --ignored --nocapture
cargo run -p parapper-diagnostics --features real-asr-tests --bin verify_jvs_asr -- --jvs-root $env:JVS_ROOT --max-speakers 1
```

## テスト用モック

Rust の単体テストで HTTP 連携を確認するときは [src-tauri/src/connect/test_support.rs](../src-tauri/src/connect/test_support.rs) のモックを使います。

- `MockHttpServer`: ローカルの空きポートで HTTP サーバーを立て、受信したリクエスト本文を検証する。
- `TimedMockHttpServer`: 受信時刻つきでリクエストを記録する。遅いレスポンスを返しても、Parapper が次の翻訳・読み上げ POST を先に投げられているかを検証する。
- `json_response` / `text_response`: YNC プラグイン API や NEO text input API の簡易レスポンスを返す。
- `request_id_from_plugin_request`: YNC プラグイン形式の JSON から `id` を取り出し、連続リクエストの順序や取り違えを確認する。

期待したリクエスト数が重要なテストでは、受信できた 1 件だけで判断せず、`start_until_idle` や `try_recv_request` で余分な request が来ないことも確認します。Turn や翻訳/読み上げの順序を見るときは、表示用 id 文字列だけでなく `turn_id`、`turn_revision`、`segment_id` などの source metadata も検証対象にします。

外部プロセスとして手元で再現したい場合は、以下の診断用 bin を使います。

```powershell
cargo run -p parapper-diagnostics --bin mock_ync_server -- --port 18080 --translate-delay-ms 0 --speech-delay-ms 4000
cargo run -p parapper-diagnostics --bin replay_mock_recognition -- --port 18080 --count 4 --interval-ms 0
```

- `mock_ync_server`: YNC プラグイン API の簡易サーバー。`translate`, `translates`, `speech`, `speech.stop`, `speech.getvoicelist`, `version` を受け、受信時刻と operation/id/text/talker を標準出力に出す。`--translate-delay-ms` と `--speech-delay-ms` で遅いレスポンスを再現できる。
- `replay_mock_recognition`: 音声入力から ASR が返る前提をモックし、認識文、翻訳、読み上げ送信までをローカルで再生する。`--count` で連続発話、`--send-neo-text` で NEO text input API 送信、`--skip-translate` で翻訳なし読み上げを確認できる。
- `measure_ync_audio_latency`: 実際の YNC/読み上げ API に対して、送信から応答までの時間を測るための補助コマンド。

テストの配置目安:

- YNC プラグイン HTTP の送信形式、フォールバック禁止、遅い読み上げレスポンスの切り分け: [src-tauri/src/connect/ync/tests.rs](../src-tauri/src/connect/ync/tests.rs)
- 認識結果から NEO/翻訳/読み上げへ配送する条件、連続発話、Namo 中途確定の回帰: [src-tauri/src/delivery/tests.rs](../src-tauri/src/delivery/tests.rs)
- ASR 入力や request/result workflow: [src-tauri/src/recognition/transcription](../src-tauri/src/recognition/transcription)
- VAD/Turn Detector の区切り、完了、再開時のイベント順序: [src-tauri/src/recognition/segmentation/segment/builder/tests.rs](../src-tauri/src/recognition/segmentation/segment/builder/tests.rs)
- Turn の continue/final/timeout と grammar boundary: [src-tauri/src/recognition/turn](../src-tauri/src/recognition/turn)

## 整形 / 静的解析

TypeScript 側と Rust 側でそれぞれ用意してあります。CI でも format check を走らせているので、コミット前に通しておくのが無難です。

TypeScript / フロントエンド:

```powershell
pnpm format     # Prettier で src 配下と root の JSON を整形
pnpm lint       # ESLint (typescript-eslint, import, unused-imports)
pnpm spell      # cspell によるスペルチェック
```

Rust:

```powershell
pnpm rust:fmt   # cargo fmt -p parapper -- --check
pnpm rust:lint  # cargo clippy --all-targets --all-features -p parapper -- -D warnings
```

workspace の `[lints.clippy]` で `pedantic = "warn"` を有効化しているため、pedantic 系の指摘も `-D warnings` によりエラーになります。

設定ファイルは以下にあります。

- ESLint: [eslint.config.mjs](../eslint.config.mjs)
- Prettier: `package.json` の `prettier` 既定設定 + `.prettierignore` 等(必要に応じて追加)
- cspell: [cspell.json](../cspell.json)
- rustfmt / clippy: `cargo` 既定設定([rust-toolchain.toml](../rust-toolchain.toml) でツールチェインを固定)

## ビルドと配布

ローカルで MSI を作る場合は以下を実行します。

```powershell
pnpm build:msi
```

GitHub Actions の `Build` workflow は Windows MSI を作成します。`main` への push、pull request、手動実行では MSI を Actions artifact として保存します。`v*` タグを push した場合は GitHub Releases を作成し、生成した MSI を添付します。

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## モデル仕様

モデルはアプリの初回ダウンロード機能で、Tauri の app data 配下に保存します。

- VAD: Silero VAD ONNX / ort
- ASR: ReazonSpeech K2 v2 / sherpa-onnx
- ASR: NeMo Parakeet TDT CTC 0.6B Ja 35000 int8 / sherpa-onnx
- ASR: NeMo Parakeet TDT 0.6B v2 int8 / sherpa-onnx
- ASR: NeMo Parakeet TDT 0.6B v3 int8 / sherpa-onnx
- Turn Detector: Namo Turn Detector v1 (Japanese / English / Multilingual) / ort
- Japanese morph dictionary: UniDic CWJ 3.1.1 を vibrato rkyv 形式へ変換して grammar boundary 判定に使う

ReazonSpeech K2 v2 は `int8`, `int8-fp32`, `float32` を選択できます。デフォルトは `int8-fp32` です。
NeMo Parakeet TDT / TDT CTC は `int8` のみを使用します。
Turn Detector は `simple`, `morph`, `namo` から選択できます。`simple` は調整可能な無音時間で完了を判定します。`morph` は grammar boundary のみで判定します。`namo` は grammar boundary を優先し、`NormalEnd` や候補なしのときだけ Namo に最終判断させます。

## ライセンス生成

Rust 依存クレートのライセンス一覧は `pnpm generate-license` で `licenses/rust.json` に生成します。
