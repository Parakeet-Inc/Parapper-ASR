# 開発者向け文書

Parapper の開発環境、実装構成、外部連携仕様への入口です。アプリを利用するだけの場合は [使い方](../how-to-use.md) を参照してください。

## 最初に読む

1. [開発・テスト・配布手順](./development-help.md) — 必要なツール、起動、検証コマンド
2. [プロジェクト全体像](./project-overview.md) — React UI、Tauri command、Rust backend の対応
3. [Rust backend のモジュール地図](./architecture/01-src-overview.md) — `src-tauri/src` の責務とデータフロー

認識パイプラインを変更する場合は、続けて [recognition モジュール俯瞰](./architecture/02-recognition-modules.md)、[recognition 内部詳細](./architecture/03-recognition-internals.md)、[日本語区切り規則](./japanese_separate_rule.md) の順に読みます。

## 外部アプリと接続する

1. [ストリーミング音声認識プロトコル v1](./streaming-recognition-protocol-v1.md) — WebSocket の制御 message、PCM、event、状態遷移
2. [セキュリティ上の注意](./security.md) — bind、Bearer 認証、TLS、送信先 URL
3. [外部アプリ連携のコード例](./example/README.md) — AIAvatarKit / AITuber-kit への組み込み
4. [接続のトラブルシューティング](../troubleshooting.md) — 起動状態、endpoint、port、model の確認

プロトコル実装を検証するときは [`protocol/fixtures/`](./protocol/fixtures/) の JSON fixture も使用します。

## モデルと外部サービスを保守する

- [CAT-Translate 0.8B ONNX export・公開手順](./cat-translate-onnx-release.md) — 固定した source から配布物を再現・検証する手順
- [ゆかコネNEOローカル翻訳サーバー確認ページ](./ync-neo-local-translation-server-check.html) — ブラウザから localhost の翻訳 API を確認する補助ページ

利用者向け概要は [ドキュメント一覧](../README.md) に戻って確認できます。
