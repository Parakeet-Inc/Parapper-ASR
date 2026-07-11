# 開発者向け接続のトラブルシューティング

## WebSocketへ接続できない

1. 接続設定でWebSocket modeを選び、入力ソースもWebSocketにします。
2. ParapperでStartを押し、状態が`WaitingForClient`になったことを確認します。
3. `ws://<address>:<port>/ws/recognition`へ接続します。旧`/ws/stt`は存在しません。
4. LAN bindではBearer API keyが必須です。認証失敗はWebSocket messageではなくHTTP `401`になります。
5. desktop認識が動作中なら`recognition_busy`です。先に停止してください。

## 接続直後に切断される

- 最初のcontrol frameはversion 1の`session.start`である必要があります。
- audioは16 kHz、mono、signed 16-bit little-endianのbinary frameです。
- 1 frameは最大100 ms（3200 bytes）で、32 ms（1024 bytes）を推奨します。
- real timeより速く送り続けてqueueを溢れさせると、silent dropせず`audio_queue_overrun`で終了します。

## finalが返らない

- 正常終了には同じ`session_id`の`session.stop`を送ります。`session.cancel`とunexpected disconnectは未確定Turnを破棄します。
- `session.stop`後は`turn.final`と`session.done`までsocketを閉じないでください。
- VAD、Turn Detector、model不足はParapperのログと画面上のmodel statusを確認します。

## 翻訳HTTP listenerを開始できない

- modelが導入済みか確認してください。missing modelはlistener statusのErrorになります。
- 指定portが他processに使われていないか確認してください。別portへ自動fallbackしません。
- Stop後は同じportへ再度Startできます。Stopping中は完了を待ってください。
- Parapper起動だけではlistenerは開始されません。

## YNCへ届かない

- NEO text inputは本体API、翻訳/発話はpluginの`POST /`です。2つのportを取り違えないでください。
- 設定したportが誤っている場合、別portへのretry/fallbackはありません。
- macOSではYNC連携を利用できません。local translation/TTSを選択してください。

詳細なwire contractは[Streaming Recognition Protocol v1](developer/streaming-recognition-protocol-v1.md)を参照してください。
