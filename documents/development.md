# 開発者向けドキュメント

<!-- cspell:words parapper Silero ReazonSpeech sherpa OSCQuery -->

Parapper は Rust + TypeScript + Tauri で構成されています。

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

## Lint / Format

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
- ASR: NeMo Parakeet TDT 0.6B v2 int8 / sherpa-onnx
- ASR: NeMo Parakeet TDT 0.6B v3 int8 / sherpa-onnx

ReazonSpeech K2 v2 は `int8`, `int8-fp32`, `float32` を選択できます。デフォルトは `int8-fp32` です。
NeMo Parakeet TDT は `int8` のみを使用します。

## ライセンス生成

Rust 依存クレートのライセンス一覧は `pnpm generate-license` で `licenses/rust.json` に生成します。
