# 外部アプリ連携の最小コード例

Parapper のストリーミング音声認識を既存アプリの音声認識と置き換える、最小構成の例です。
どちらも Parapper 固有の処理を入力境界に閉じ込め、確定した `turn.final.text` だけを既存の会話処理へ渡します。

## 接続方式

- **WebSocket入力**: 外部アプリがマイクを取得し、PCMをParapperへ送ります。Parapperの入力ソースをWebSocketにします。
- **HTTP(S) API出力**: Parapperがdesktopマイクを取得し、認識eventを外部APIへPOSTします。Parapperの接続modeをHTTPにします。

WebSocket方式の音声は16 kHz / mono / signed 16-bit little-endian PCMです。制御と認識結果の詳細は[ストリーミング音声認識プロトコル v1](../streaming-recognition-protocol-v1.md)を参照してください。

## コード例

- [AIAvatarKit の音声認識を置き換える](./aiavatarkit/README.md)
  - WebSocket入力とHTTP(S) API出力の両方を掲載しています。
- [AITuber-kit のブラウザ音声認識を置き換える](./aituber-kit/README.md)
  - Web Speech API の代わりに、ブラウザのマイク PCM を Parapper へ送り、既存の `onChatProcessStart` を呼びます。

## 制約

- WebSocket方式では、1本の接続が1つの認識sessionです。1回の接続から複数の `turn.final` が返る場合があります。
- `turn.partial` は表示専用です。会話処理へ送るのは `turn.final` だけにします。
- browser標準の `WebSocket` APIでは upgrade時の `Authorization` headerを指定できません。AITuber-kit例はloopback接続かつParapper側API keyなしを前提にします。
- WebSocket方式では、`session.stop`後に`session.done`を受け取るまで接続を閉じません。異常終了時だけ`session.cancel`を使用します。
- HTTP(S)方式では、受信APIは2秒以内に応答し、AIAvatarKitなどの下流処理は非同期で実行します。
