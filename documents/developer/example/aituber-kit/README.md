# AITuber-kit のブラウザ音声認識を Parapper で置き換える

この例は [`tegnike/aituber-kit`](https://github.com/tegnike/aituber-kit) のcommit `4de86c18d2ce26cbf45e0648a138a9634c32d9ed`（2026-07-10）で確認しています。

AITuber-kitのブラウザ認識経路は次の構成です。

```text
MessageInputContainer
  -> useVoiceRecognition
    -> useBrowserSpeechRecognition
      -> Web Speech API
```

[`useParapperSpeechRecognition.ts`](./useParapperSpeechRecognition.ts) は `useBrowserSpeechRecognition` と同じ主要な戻り値を持ち、Web Speech APIの代わりにマイクPCMをParapperへ送ります。Parapperが返した `turn.partial` を入力欄へ表示し、`turn.final`だけを既存の `onChatProcessStart` へ渡します。

## Parapper側の準備（GUI）

設定項目は認識中に変更できないため、Parapperが動作中の場合は先にメイン画面の「ストップ」を押してください。

1. Parapperを起動し、使用するASRモデルがダウンロード済みであることを確認します。初回の場合は初期設定で用途に合う設定プリセットを選び、「反映してモデルをダウンロード」を押します。
2. 設定パネルの「接続」タブを開きます。
3. 「HTTP / WebSocket接続をする（開発者向け）」をONにします。
4. 展開された設定を次のようにします。

| 項目             | 設定値          |
| ---------------- | --------------- |
| 接続モード       | `WebSocket`     |
| 待受アドレス     | `127.0.0.1`     |
| WebSocket port   | `18082`         |
| API key          | 空欄            |
| 認識結果の出力先 | `WebSocketのみ` |

5. 途中経過もAITuber-kitの入力欄へ表示する場合は、設定パネルの「VAD」タブで「途中経過を表示する」をONにします。確定結果だけでよければOFFでも動作します。
6. メイン画面へ戻り、「入力デバイス」から「ネットワーク入力」グループの「WebSocket (PCM 16 kHz)」を選びます。
7. メイン画面の「スタート」を押します。認識状態が「クライアント待機中」になればParapper側の準備は完了です。
8. AITuber-kitを起動し、マイクボタンを押します。接続されるとParapperの状態が「listening」に変わり、途中経過はAITuber-kitの入力欄へ表示され、確定結果だけがチャットへ送られます。常時認識ではAI発話中も同じ接続でマイク音声を送信し続けます。

接続できない場合は、Parapperが「クライアント待機中」になっていること、入力デバイスが通常のマイクではなく「WebSocket (PCM 16 kHz)」になっていること、AITuber-kitとParapperのportが一致していることを確認してください。別のデスクトップ認識やWebSocket sessionが動作中の場合は `recognition_busy` になるため、先にその認識を停止します。

この例ではbrowserから認証headerを送れないため、待受アドレスを `127.0.0.1` にしてAPI keyを空欄にします。`0.0.0.0`などでLANへ公開する設定にはしないでください。LAN越しに接続する場合は、後述の認証headerを付与できるserver-side proxyが必要です。

## 組み込み

1. ファイルをAITuber-kitへコピーします。

```text
documents/developer/example/aituber-kit/useParapperSpeechRecognition.ts
  -> src/hooks/useParapperSpeechRecognition.ts

documents/developer/example/aituber-kit/parapper-pcm-worklet.js
  -> public/parapper-pcm-worklet.js
```

2. `src/hooks/useVoiceRecognition.ts` のimportを追加します。

```ts
import { useParapperSpeechRecognition } from './useParapperSpeechRecognition'
```

3. browser用Hookの生成を1行だけ置き換えます。

```ts
// 変更前
const browserSpeech = useBrowserSpeechRecognition(onChatProcessStart)

// 変更後
const browserSpeech = useParapperSpeechRecognition(onChatProcessStart)
```

`src/hooks/useVoiceRecognition.ts`で必要な変更は、上記のimportとHook生成の2か所だけです。明示停止と内部ライフサイクル停止の違いは`useParapperSpeechRecognition`内で処理するため、AITuber-kit側へcancel用の分岐を追加する必要はありません。

既存の `MessageInputContainer`、マイクボタン、常時マイク入力設定、`onChatProcessStart` は変更しません。ParapperがTurn確定を行うため、AITuber-kit側の `useSilenceDetection` はこの経路では使用しません。

## 接続先

既定値は `ws://127.0.0.1:18082/ws/recognition` です。変更する場合はAITuber-kitのbuild時環境変数を設定します。

```dotenv
NEXT_PUBLIC_PARAPPER_URL=ws://127.0.0.1:18082/ws/recognition
```

browserの`WebSocket` constructorは任意のHTTP headerを指定できないため、この例ではParapperをloopbackで待ち受け、API keyを設定しません。LAN越しに利用する場合は、認証headerを付与できる同一originのserver-side proxyを用意してください。

## 音声処理

- browserの実sample rateから16 kHzへAudioWorklet内でdownsampleします。
- 512 samples（32 ms、1024 bytes）ごとにsigned 16-bit little-endian PCMを送ります。
- 1本のWebSocketとAudioContextを維持したまま、複数の`turn.final`を受信します。
- `turn.final`では`session.stop`を送らず、次の発話も同じsessionで認識します。
- マイクボタンで明示的に停止した場合だけ`session.stop`を送り、`session.done`までWebSocketを維持します。
- AI発話中のマイク入力を止める制御は行いません。必要なエコー抑制や発話制御はAITuber-kit側で行います。
