# AIAvatarKit の音声認識を Parapper で置き換える

Parapperと[AIAvatarKit](https://github.com/uezo/aiavatarkit)を接続する方法を、マイクの所有者ごとに2つ用意しています。どちらもAIAvatarKit側では音声認識を実行せず、認識済みの `text` を `POST /chat` へ送ります。

| 方法 | マイクを取得する側 | Parapperの接続mode | 使用するコード |
| --- | --- | --- | --- |
| WebSocket入力 | Python bridge | WebSocket | [`websocket_bridge.py`](./websocket_bridge.py) |
| HTTP(S) API出力 | Parapper desktop | HTTP | [`https_api_bridge.py`](./https_api_bridge.py) |

AIAvatarKit serverは `SpeechRecognizerDummy`、`voice_recorder_enabled=False` で起動します。

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

## 共通セットアップ

Python 3.11以降と[uv](https://docs.astral.sh/uv/)を使用します。

```powershell
uv sync --project documents/developer/example/aiavatarkit
```

AIAvatarKit serverは既定で `http://127.0.0.1:8000/chat` に接続します。変更する場合はbridge起動前に設定します。

```powershell
$env:AIAVATAR_CHAT_URL = "http://127.0.0.1:8000/chat"
```

AIAvatarKitの `session_id` はbridgeの実行中固定し、serverから返された `context_id` を次のTurnへ引き継ぎます。

## 途中経過と確定結果

両方の接続方式で、Parapperから受け取るeventを次のように分けます。

| event | exampleでの処理 | AIAvatarKit `/chat` |
| --- | --- | --- |
| `turn.partial` | `[partial] ...`として標準出力へprint | 送らない |
| `turn.final` | `[final] ...`として標準出力へprint | 1回だけ送る |

partialは同じTurnの新しいrevisionで置き換わる途中表示です。partialごとにLLMを開始すると重複応答になるため、`AIAvatarConversation.send_final()`を呼ぶのはfinal分岐だけにしています。

## 方法1: WebSocket入力

Python bridgeがマイクを取得し、16 kHz mono PCMをParapperの `/ws/recognition` へ送ります。

```text
microphone -> websocket_bridge.py -> Parapper /ws/recognition
                                      |
                                      +-> turn.partial -> print only
                                      +-> turn.final   -> AIAvatarKit POST /chat
```

Parapperの接続設定で次を選びます。

- 開発者向け接続: ON
- 接続mode: WebSocket
- bind address: `127.0.0.1`
- port: `18082`
- 入力ソース: WebSocket

ParapperでStartを押して `WaitingForClient` になった後、bridgeを実行します。

```powershell
uv run --project documents/developer/example/aiavatarkit python documents/developer/example/aiavatarkit/websocket_bridge.py
```

Enterを押すとマイク入力を止め、`session.stop`後の残りの`turn.final`と`session.done`を待って終了します。

```powershell
$env:PARAPPER_URL = "ws://127.0.0.1:18082/ws/recognition"
$env:PARAPPER_API_KEY = "<Parapper API key>" # 認証を設定した場合だけ
```

## 方法2: HTTP(S) API出力

Parapperが通常どおりdesktopのマイクを取得し、認識eventをbridgeの `POST /api/events` へ送ります。bridgeは `turn.partial` を表示だけに使い、`turn.final` をAIAvatarKitへ非同期転送します。ParapperへのHTTP応答はAIAvatarKitの応答完了を待たず、即座に`202 Accepted`を返します。

```text
microphone -> Parapper desktop -> POST /api/events -> https_api_bridge.py
                                                    |
                                                    +-> turn.partial -> print only
                                                    +-> turn.final   -> AIAvatarKit POST /chat
```

まずbridgeを起動します。

```powershell
uv run --project documents/developer/example/aiavatarkit python documents/developer/example/aiavatarkit/https_api_bridge.py
```

Parapperの接続設定で次を選びます。

- 開発者向け接続: ON
- 接続mode: HTTP
- URL: `http://127.0.0.1:15522/api/events`
- 入力ソース: 使用するdesktopマイク

その後ParapperでStartを押します。この方法ではbridgeがParapperへ接続するのではなく、Parapperから認識eventが送信されます。

### HTTPSで待ち受ける

証明書と秘密鍵を指定すると、Uvicornが直接HTTPSで待ち受けます。

```powershell
$env:BRIDGE_HOST = "0.0.0.0"
$env:BRIDGE_PORT = "15522"
$env:SSL_CERTFILE = "C:\path\to\fullchain.pem"
$env:SSL_KEYFILE = "C:\path\to\private-key.pem"
uv run --project documents/developer/example/aiavatarkit python documents/developer/example/aiavatarkit/https_api_bridge.py
```

Parapper側のURLには `https://<host>:15522/api/events` を設定します。証明書はParapperのHTTP clientが検証できる信頼済みCAのchainである必要があり、未信頼のself-signed証明書には接続できません。本番運用ではUvicornへ証明書を直接渡す代わりに、TLS reverse proxyの内側でbridgeをloopback待受にしても構いません。

developer HTTP出力は現在 `Authorization` headerを送らないため、HTTPSは通信の暗号化だけを提供します。LANやinternetへ公開する場合は、private network、firewall、mTLS対応reverse proxyなどで接続元も制限してください。

## この最小例に含めないもの

応答音声の再生、avatar制御、barge-inは含めていません。どちらの方式でも、`turn.final`だけをAIAvatarKitへ渡す境界は維持したまま追加してください。
