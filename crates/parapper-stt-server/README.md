# parapper-stt-server

`parapper-stt-server`は、Parapper STT engineのためのTauri非依存な
`/ws/recognition` WebSocket境界を提供するcrateです。

このcrateは次の責務を持ちます。

- WebSocketプロトコルとソケットサーバー
- 接続および認識セッションの状態機械
- PCM音声入力の形式とフレームサイズの契約
- bounded queueのbackpressureを通信エラーへ変換する処理
- graceful stop、cancel、切断、server shutdownを含むライフサイクル

実際の認識セッションは、このcrateが直接構築しません。ホスト側が
`RecognitionBackend`を実装し、認識セッション、音声入力、認識結果の出力を
提供します。現在はParapper desktopがこのbackendを実装しています。将来の
headless processは、Tauriへ依存せず別のbackendを実装できます。

このcrateは、`AppHandle`、`AppState`、desktop固有の出力処理、モデルパス、
モデルの構築方法を知りません。

## Decoder tuning

Parakeet TDT-DAGのbeam幅とCTC gateは、WebSocket clientの`session.start`では
受け取りません。headless hostは`StreamingRecognitionServerConfig.backend_config`
に`RecognitionBackendConfig { tdt_dag: Some(...) }`を指定して起動します。この値は
接続ごとに同じまま`RecognitionBackend::start`へ渡されます。

```rust
use std::num::NonZeroUsize;

use parapper_stt_server::{RecognitionBackendConfig, TdtDagDecodingConfig};

let backend_config = RecognitionBackendConfig {
    tdt_dag: Some(TdtDagDecodingConfig::new(
        NonZeroUsize::new(4).unwrap(),
        Some(-5.0),
    )?),
};
```

transport crateは`parapper-models`に依存しないため、この値を実際の
`ParakeetJaTdtDagConfig`へ変換してengineを構築するのはheadless hostの
`RecognitionBackend`実装です。desktopの既定backendは空の設定を渡し、アプリの
ASRモード設定をそのまま使用します。したがって、接続済みclientがdecoder設定を
上書きする経路はありません。

## 接続境界

```text
WebSocket client
        │
        ▼
parapper-stt-server
  protocol / socket
        │ RecognitionBackend
        ▼
Parapper desktop / headless host
        │
        ▼
parapper-stt-engine
```

Web GUI、CLI、タイピングゲームなど新しいクライアントを追加するときは、
`src-tauri`や`AppState`へ直接接続せず、このcrateのWebSocketプロトコルを
利用します。

headless serverなど新しいホストを追加するときは、`parapper-stt-engine`を
通信処理へ直接結合せず、`RecognitionBackend`を実装してこのcrateへ接続します。
これにより、STT engineは通信方式を知る必要がありません。

## 今後のテスト設計

現時点では単一ホストの都合を共通契約として固定しないため、将来を推測した汎用mockやcontract testは追加しません。
第二のクライアントまたは`RecognitionBackend`実装が現れた時点で、実際に共有される契約を比較してtest supportを抽出します。
その際はserver、backend adapter、client、結合の各層を分け、実際の通信とイベント順序を保ちながら不具合箇所を切り分けられる構成にします。

## 確認コマンド

```powershell
cargo test -p parapper-stt-server
cargo tree -p parapper-stt-server
```
