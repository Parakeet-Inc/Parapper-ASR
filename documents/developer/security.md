# セキュリティ上の注意

Parapperの外部接続機能は、用途ごとに公開範囲と認証条件が異なります。

## ストリーミング音声認識WebSocket

- endpointは`/ws/recognition`です。音声はraw PCMなので、信頼できないネットワークへ平文で公開しないでください。
- `127.0.0.1`以外へbindする場合はAPI keyが必須です。upgrade時に`Authorization: Bearer <key>`を送ります。
- Parapper自身はTLSを終端しません。LANを越える場合は、TLS対応reverse proxyや安全なトンネルを利用してください。
- API key、音声内容、認識本文を診断ログへ意図せず貼り付けないでください。
- 同時認識sessionはdesktop/WebSocket合計1つです。これは資源競合を防ぎますが、認証の代わりではありません。

## 翻訳HTTP listener

- YNC Custom JSONの`POST /`とOpenAI互換の`POST /v1/chat/completions`を`127.0.0.1`でのみ待ち受けます。
- listenerには認証がありません。Parapper起動時には自動起動せず、接続画面のStart操作でのみ起動します。
- containerやport forwardによってloopback listenerを外部公開しないでください。
- `/api/input`、未知path、別portへのfallbackはありません。

## YNCと開発者向けHTTP出力

- YNC本体API、YNC plugin API、developer HTTP event送信は別契約です。送信先URL・portを信頼できるlocal serviceだけに設定してください。
- developer HTTP eventには認識本文とTurn metadataが含まれます。HTTPSでない外部URLへ送る場合は盗聴リスクがあります。
- Parapperは接続失敗時に別endpointや別portへ暗黙fallbackしません。

## secretと不具合報告

- `.env`、API key、個人の録音、認証情報を含む`config.json`などをGitへ追加しないでください。
- issueへログを添付する前に、認識本文、local path、host名、tokenを確認してマスクしてください。
