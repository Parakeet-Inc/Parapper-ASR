# Parapper ストリーミング音声認識プロトコル v1

状態: 実装済み。Rustのプロトコルテストと実socket契約テストで検証しています。

このプロトコルは、AIAvatarKit、AITuberKit、YNC固有のpayloadをRust coreへ持ち込まずに、Parapperの音声認識機能を外部へ公開します。

## 通信方式とendpoint

- WebSocket endpoint: `GET /ws/recognition`
- 制御messageと認識結果: UTF-8 JSON text frame
- 音声: raw PCMを格納したWebSocket binary frame
- 未知のHTTP pathには`404`を返します。他のParapper APIへfallbackしません。
- API keyを設定した場合、upgrade時に`Authorization: Bearer <key>`を送ります。
- 認証失敗はWebSocket upgrade前にHTTP `401`を返します。WebSocketの`error` messageは返しません。
- loopback以外のaddressへbindする場合、API keyが必須です。

すべてのJSON制御messageと結果messageは、数値の`"version": 1`を持ちます。versionがない、数値でない、または1以外の場合は`unsupported_version`です。

## 音声仕様

`session.start.audio`はv1で次の値に固定します。

| field | 値 |
| --- | --- |
| `encoding` | `pcm_s16le` |
| `sample_rate` | `16000` |
| `channels` | `1` |

binary frameはsigned 16-bit little-endian PCMです。空frameと奇数byte長は不正です。1 frameは最大100 ms（`3200` bytes）で、32 ms（`1024` bytes）を推奨します。アプリケーション側のqueueは約2秒分に制限されています。queue overflow時はfatal errorとし、音声を黙ってdropしません。

## clientから送る制御message

正式な`session.start` fixture:
[`fixtures/client-session-start-v1.json`](protocol/fixtures/client-session-start-v1.json)

```json
{"version":1,"type":"session.stop","session_id":"client-generated-id"}
```

```json
{"version":1,"type":"session.cancel","session_id":"client-generated-id"}
```

```json
{"version":1,"type":"ping","request_id":"ping-1"}
```

`session_id`はclientが生成する空でない文字列で、接続中は変更できません。`session.stop`と`session.cancel`には、active sessionと同じIDを指定します。v1では`flush`を公開しません。

## serverから返すmessage

正式なfixture:

- [`session.ready`](protocol/fixtures/server-session-ready-v1.json)
- [`turn.final`](protocol/fixtures/server-turn-final-v1.json)

その他のmessage:

```json
{"version":1,"type":"speech.started","session_id":"client-generated-id"}
```

```json
{"version":1,"type":"session.done","session_id":"client-generated-id"}
```

```json
{"version":1,"type":"session.cancelled","session_id":"client-generated-id"}
```

```json
{"version":1,"type":"pong","request_id":"ping-1"}
```

`turn.partial`は`turn.final`と同じidentity、`source_asr_model`、言語fieldを持ちますが、`audio_duration_ms`は持ちません。`session_id`はnetwork sessionを識別し、`turn_session_id`はParapper内部の構造化された認識sessionを識別します。同じTurnのrevisionは単調増加します。clientは古いpartialを新しいrevisionで置き換え、`turn.final`を変更不能な確定結果として扱います。

errorは機械処理できる形式で返し、内部Rust error文字列をcodeとして公開しません。

```json
{
  "version": 1,
  "type": "error",
  "session_id": "client-generated-id",
  "code": "invalid_state",
  "message": "binary audio is only accepted in an active session",
  "fatal": true
}
```

v1のprotocol errorはすべてfatalです。upgrade完了後は`error` messageを1件送り、接続を閉じます。session開始前のerrorでは`session_id`がnullの場合があります。

## session状態遷移

| 現在の状態 | 入力 | 処理 / 出力 | 次の状態 |
| --- | --- | --- | --- |
| `AwaitingStart` | `session.start` | 全体の認識ownershipを取得し、`session.ready` | `Active` |
| `AwaitingStart` | `ping` | `pong` | `AwaitingStart` |
| `AwaitingStart` | binary / stop / cancel | `invalid_state`を返して切断 | `ProtocolError` |
| `Active` | binary audio | 検証後、1回だけqueueへ追加 | `Active` |
| `Active` | `ping` | `pong` | `Active` |
| `Active` | `session.stop` | 音声受付を止め、SegmentをflushしてASR/Turnをdrain | `Draining` |
| `Active` | `session.cancel` | queueとopen Turnを破棄し、`session.cancelled` | `Cancelled` |
| `Active` | 2回目の`session.start` | `invalid_state`を返して切断 | `ProtocolError` |
| `Draining` | pipeline result | 残りのpartial/finalを送信 | `Draining` |
| `Draining` | drain完了 | `session.done`を送信して切断 | `Done` |
| `Draining` | `ping` | `pong` | `Draining` |
| `Draining` | binaryまたはsession制御 | `invalid_state`を返して切断 | `ProtocolError` |
| `AwaitingStart` / `Active` / `Draining` | 切断 | 未完了処理を破棄し、finalは出さない | `Disconnected` |

`session.stop`はgraceful stopです。受理済みの全音声を処理し、現在のSegmentをflushし、ASR出力をdrainし、通常のTurn規則に従ってopen Turnを確定します。その後、残りのfinalと`session.done`を送ります。

`session.cancel`と予期しない切断では、未確定Turnをfinalizeしません。Namo Continue後のactivityも省略せず、Segment activityの更新をproductionと同じ順序で処理します。

desktop入力とWebSocket入力を合わせて、同時に認識できるsessionは1つだけです。2つ目のownerには`recognition_busy`を返し、別のASR workerやmodel instanceを起動しません。

## 安定したerror code

| code | 条件 |
| --- | --- |
| `unsupported_version` | 制御messageのversionがない、または未対応 |
| `authentication_failed` | 機械処理用の予約code。実際のupgrade responseはHTTP 401 |
| `invalid_json` | 不正JSON、未知のtype、またはsession以外の必須field不足 |
| `invalid_state` | 現在の状態では受け付けられないmessage / frame |
| `session_id_required` | session IDがない、または空 |
| `session_id_mismatch` | stop/cancelのIDがactive sessionと異なる |
| `unsupported_audio_encoding` | encodingが`pcm_s16le`ではない |
| `unsupported_sample_rate` | sample rateが16000 Hzではない |
| `unsupported_channel_count` | channel数が1ではない |
| `invalid_audio_frame` | 空、または奇数byte長のbinary frame |
| `audio_frame_too_large` | binary frameが3200 bytesを超える |
| `audio_queue_overrun` | bounded application queueが満杯 |
| `recognition_busy` | desktopまたは別のWebSocket sessionが認識を所有中 |
| `model_unavailable` | 設定されたASR modelをloadできない |
| `recognition_failed` | session開始後に認識workerが失敗 |
| `drain_timeout` | graceful stopがdrain上限時間を超えた |
| `server_stopping` | アプリ終了処理中のため続行できない |

## 出力先

network入力の既定値は`WebSocketOnly`です。認識結果をdesktop UI、翻訳、TTS、YNCへ暗黙に送信しません。`WebSocketAndDesktop`を明示的に選んだ場合だけ、同じ構造化出力を各sinkへ1回ずつ送ります。

## fixtureとテスト

Rustのfixture testではclient controlをdeserializeし、server JSONを構造体全体で比較します。純粋なsession testで状態遷移を固定し、実socket testでHTTP upgrade、認証、binary PCM、stop/cancel、busy ownership、queue overrun、drain timeout、切断動作を検証します。WebSocket parserやsession state machineを迂回するmockは使用しません。

## 関連文書

- [セキュリティ上の注意](security.md)
- [トラブルシューティング](../troubleshooting.md)
- [開発者向け文書一覧](README.md)
