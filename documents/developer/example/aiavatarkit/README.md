# AIAvatarKit v0.8.19とParapperを接続する

<!-- cspell:words aiavatar aiavatarkit CERTFILE fullchain -->

[AIAvatarKit v0.8.19](https://github.com/uezo/aiavatarkit/releases/tag/v0.8.19)の
`ParapperStreamSpeechDetector`は、Parapperのストリーミング音声認識プロトコルv1に対応しています。

標準構成は`ParapperStreamSpeechDetector`をAIAvatarKitの`STSPipeline`へ
直接組み込む方式です。認識済みテキストを別のHTTP serverへ中継せず、
Parapperの`turn.final`がそのままLLM入力になります。

| 位置づけ | 構成                                                    | Parapperの接続mode  |
| -------- | ------------------------------------------------------- | ------------------- |
| 標準     | `ParapperStreamSpeechDetector`を`STSPipeline.vad`に指定 | WebSocket           |
| 補助     | bridgeから`AIAvatarHttpServer /chat`へ転送              | WebSocketまたはHTTP |

標準方式ではAIAvatarKit公式の`ParapperStreamSpeechDetector`を使用します。このdetectorは
`turn.partial`を途中経過、`turn.final`を唯一の発話終了として扱い、
クライアント側の無音timeoutでTurnを再分割しません。

`AIAvatarHttpServer`の`/chat`を利用するbridge例は補助方式として後半に掲載します。

> `ParapperStreamSpeechDetector`はv0.8.19時点でexperimentalです。
> AIAvatarKitの更新時はAPI変更の有無を確認してください。

## セットアップ

Python 3.11以降と[uv](https://docs.astral.sh/uv/)を使用します。
このexampleは`aiavatar==0.8.19`へ固定しています。

```powershell
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
uv sync --project documents/developer/example/aiavatarkit
```

別projectのvirtual environmentをactivateしたままだと、`uv`は
`VIRTUAL_ENV=... does not match the project environment path ...`と警告します。
上記の`Remove-Item`でその環境指定だけを解除します。以降は作成されたexample専用の
Pythonを明示して実行するため、別projectの環境には依存しません。

## 標準: AIAvatarKitのSTSへ直接組み込む

`ParapperStreamSpeechDetector`を`STSPipeline`の`vad`として渡します。
Parapperが認識済みテキストを返すため、この経路では別のSTTによる再認識や
`AIAvatarHttpServer`へのHTTP転送は行いません。

```python
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.vad.parapper_stream import ParapperStreamSpeechDetector

parapper = ParapperStreamSpeechDetector(
    url="ws://127.0.0.1:18082/ws/recognition",
    api_key=None,
)

pipeline = STSPipeline(
    vad=parapper,
    llm=llm,
    tts=tts,
)
```

Parapperの接続タブでは次のように設定します。

- 開発者向け接続: ON
- 接続mode: WebSocket
- bind address: `127.0.0.1`
- port: `18082`
- 入力ソース: WebSocket

ParapperでStartを押して`WaitingForClient`になってからAIAvatarKitを起動します。
使用するadapterから16kHz、mono、PCM s16leの音声をpipelineへ渡してください。
`turn.partial`は途中経過、`turn.final`は確定したLLM入力として扱われます。

<details>
<summary>開発者向け: protocol境界を確認する</summary>

[`check_parapper_integration.py`](./check_parapper_integration.py)は
LLM、TTS、AIAvatarKit `/chat`を起動せず、AIAvatarKit公式detectorとParapper間だけを確認します。
通常利用時に実行する必要はありません。protocol実装の変更時や、
AIAvatarKit更新後の互換性確認に使用します。

### 外部接続なしの自己テスト

同一process内に疑似Parapperサーバを起動し、次を検証します。

- protocol v1の`session.start`
- 16kHz、mono、PCM s16leの音声形式
- binary frameがParapperの上限3200 bytes以下
- `speech.started`と`turn.partial`のcallback
- `turn.final`だけがfinal callbackを起動すること
- `session.stop`と`session.done`による終了

```powershell
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/check_parapper_integration.py `
  --self-test
```

最後に`{"event": "self_test.pass", ...}`が表示されれば成功です。

### 実Parapperとのハンドシェイク

標準構成と同じ接続設定でParapperをStartし、
`WaitingForClient`になった後に次を実行します。

```powershell
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/check_parapper_integration.py
```

この確認は短い無音を送り、`session.stop`後の`session.done`まで待ちます。
認識結果は要求しません。`manual_check.pass`の`mode`が`handshake`なら成功です。

APIキーを設定している場合:

```powershell
$env:PARAPPER_API_KEY = "<Parapperと同じAPI key>"
```

接続先を変更する場合:

```powershell
$env:PARAPPER_URL = "ws://127.0.0.1:18082/ws/recognition"
```

別endpointや別portへの暗黙のfallbackは行いません。

### WAVで`turn.final`まで確認する

16kHz、mono、16-bit、非圧縮PCMのWAVを指定します。

```powershell
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/check_parapper_integration.py `
  --wav C:\path\to\speech-16k-mono.wav
```

`recording.started`、必要に応じて複数の`turn.partial`、1回の`turn.final`、
最後に`manual_check.pass`が表示されることを確認します。
WAVを送っても`turn.final`が来なければ終了コード1で失敗します。

</details>

<details>
<summary>補助: AIAvatarHttpServerの<code>/chat</code>を使うbridge</summary>

以下は既存アプリが`AIAvatarHttpServer`を起動している場合の補助例です。
標準のSTS直接組み込みでは使用しません。

### WebSocket入力を`AIAvatarHttpServer`へ転送する

[`websocket_bridge.py`](./websocket_bridge.py)はPythonでマイクを取得し、
PCMを`ParapperStreamSpeechDetector.process_samples()`へ渡します。
WebSocketのhandshake、frame分割、event受信、stop/drainはAIAvatarKit側の公式実装が担当します。

```text
microphone -> ParapperStreamSpeechDetector -> Parapper /ws/recognition
                                                |
                                                +-> turn.partial -> print only
                                                +-> turn.final   -> AIAvatarKit POST /chat
```

Parapperを前節と同じWebSocket入力でStartした後、実行します。

```powershell
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/websocket_bridge.py
```

起動時にまずParapperとの接続を確認し、成功してからマイクを開きます。
`Parapperに接続できません`と表示された場合は、ParapperをStartして
`WaitingForClient`になっていることと、設定したportが`PARAPPER_URL`のportと
一致することを確認してください。別endpointや別portへはfallbackしません。

Enterを押すとマイク入力を止め、AIAvatarKit公式detectorが`session.stop`を送り、
残りの`turn.final`と`session.done`を待って終了します。

AIAvatarKit serverは既定で`http://127.0.0.1:8000/chat`に接続します。
変更する場合:

```powershell
$env:AIAVATAR_CHAT_URL = "http://127.0.0.1:8000/chat"
```

このexampleは認識済みのfinal textを`/chat`へ渡すため、AIAvatarKit server側では
別のSTTを実行しないようにします。

```python
from aiavatar.adapter.http.server import AIAvatarHttpServer
from aiavatar.sts.stt import SpeechRecognizerDummy

aiavatar_app = AIAvatarHttpServer(
    llm=llm,
    stt=SpeechRecognizerDummy(),
    tts=tts,
    voice_recorder_enabled=False,
)
```

AIAvatarKitの`session_id`はbridgeの実行中固定し、serverから返された
`context_id`を次のTurnへ引き継ぎます。

### ParapperからHTTP(S)で認識eventを送る

Parapper desktopがマイクを取得し、認識eventをbridgeの`POST /api/events`へ送ります。
bridgeは`turn.partial`を表示だけに使い、`turn.final`をAIAvatarKitへ非同期転送します。
ParapperへのHTTP応答はAIAvatarKitの応答完了を待たず、`202 Accepted`を返します。

```text
microphone -> Parapper desktop -> POST /api/events -> https_api_bridge.py
                                                    |
                                                    +-> turn.partial -> print only
                                                    +-> turn.final   -> AIAvatarKit POST /chat
```

bridgeを起動します。

```powershell
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/https_api_bridge.py
```

Parapperの接続設定:

- 開発者向け接続: ON
- 接続mode: HTTP
- URL: `http://127.0.0.1:15522/api/events`
- 入力ソース: 使用するdesktopマイク

この方式ではbridgeからParapperへ接続しません。設定後にParapperでStartします。

### HTTPSで待ち受ける

```powershell
$env:BRIDGE_HOST = "0.0.0.0"
$env:BRIDGE_PORT = "15522"
$env:SSL_CERTFILE = "C:\path\to\fullchain.pem"
$env:SSL_KEYFILE = "C:\path\to\private-key.pem"
documents\developer\example\aiavatarkit\.venv\Scripts\python.exe `
  documents/developer/example/aiavatarkit/https_api_bridge.py
```

Parapper側には`https://<host>:15522/api/events`を設定します。
証明書はParapperのHTTP clientが検証できる信頼済みCAのchainにしてください。
未信頼のself-signed証明書へは接続できません。

developer HTTP出力は現在`Authorization` headerを送りません。LANやinternetへ公開する場合は、
private network、firewall、mTLS対応reverse proxyなどで接続元も制限してください。

### eventの扱い

| event            | exampleでの処理     | AIAvatarKit `/chat` |
| ---------------- | ------------------- | ------------------- |
| `speech.started` | 発話開始を表示      | 送らない            |
| `turn.partial`   | 途中表示            | 送らない            |
| `turn.final`     | immutableな確定結果 | 1回だけ送る         |

partialごとにLLMを開始すると重複応答になります。
`AIAvatarConversation.send_final()`を呼ぶのはfinal callbackだけです。

</details>

## この最小例に含めないもの

応答音声の再生、avatar制御、barge-inは含めていません。
追加する場合も、Parapperの`turn.final`だけをLLMへ渡す境界を維持してください。
