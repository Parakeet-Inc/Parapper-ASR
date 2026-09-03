# Parapperにおけるmodel / STT engine / serverの境界

## 依存方向

```text
src-tauri
├─ parapper-stt-server
│  └─ parapper-stt-engine
│     └─ parapper-models
├─ parapper-stt-engine
│  └─ parapper-models
└─ parapper-models
```

依存はhost/transportからdomain層とmodel層への一方向に限定する。再利用可能な3つのcrateはいずれもTauriへ依存しない。

## `parapper-models`

model crateはASR、MT、TTS、NC、VAD、TDの実装を所有する。TDにはNamoとJapanese Morphの両方を含む。一括ASR、通常のASRを使った反復的な部分推論、statefulなNemotron推論はいずれも`asr`に属し、WebSocket streamingは含まない。

model constructorは明示的なfilesystem pathを受け取る。このcrateは保存場所の選択、artifactのdownload、application eventの発行、host worker queueの所有を行わない。ORT Session、thread設定、Nemotronのencoder cache tensorなど、AIモデル固有のアルゴリズム状態は各モデル実装が所有する。共通ORT runtime helperはORTの初期化だけを担当し、session managerにはしない。

native実装はfeatureで制御する。default featureでは、`parapper-stt-engine`が必要とする軽量なcontractとpure algorithmだけを公開し、desktop側が`native-models`を有効にする。

## `parapper-stt-engine`

このengineは再利用可能なSTT coreであり、以下を所有する。

- `SttEngineConfig`、`RecognitionSession`、`RecognitionDriver`
- VAD frameからSegmentへの状態遷移
- ASR request identity、planning、streaming identity、stale resultのreduction
- 構築済みASR engineのregistry、STT Sessionとmodel Session IDの対応、明示的なstreaming lifecycle
- ASR requestに対するmodel選択、先頭padding、timestamp補正などの実行policy
- Turn draft/confirmed lifecycleとgrammar/Namo/silence/timeout policy
- activity epoch、pending work、runtime counter、structured output
- ASR、VAD、language identification、Turn decision、outputのport

MT/TTS、audio device、model storage、HTTP/WebSocket、window、`AppHandle`は所有しない。また、生のORT Sessionやmodel固有のcache tensorを直接管理しない。これらは`Box<dyn parapper_models::asr::AsrEngine>`の内部へカプセル化する。

driverは、`push_vad_frame`、`step`、`flush_input`、pending workの検査、drain後のfinalizationといったnon-blockingな状態遷移を公開する。worker thread、sleep、join timeout、shutdown deadlineはhost側に残す。

## configとpathの境界

`ParapperConfig`は、desktop applicationがflatな形式で永続化するconfigとして維持する。desktop側はSTT関連のfieldだけを`SttEngineConfig`へ変換するため、disk上の形式を変更する必要はない。翻訳、TTS、device、server bind設定、model storage pathをengine configへ入れない。

applicationのresource/data directoryを解決するときはTauriを使用してよい。ただし、そのpath解決はdesktop adapter内で完結させる。adapterは解決済みpathからASR、SLI、Namo、Morphの各実装を構築する。ASRについては構築済みengineを`parapper-stt-engine`のregistryへ渡し、SLI/Namo/Morphについてはhost-neutralなportとして注入する。CLIやprivate serverも、それぞれが明示的に解決したpathを使って同じ構築を行える。

## `parapper-stt-server`

server crateは`/ws/recognition`、protocol version 1のDTO/state、socket lifecycle、PCM変換、error mapping、host backend contractを所有する。engineのoutput型へ依存するが、modelを構築したりdesktop applicationのstateを参照したりしない。

`NetworkOutputMode`は`src-tauri`に残す。WebSocketだけへ出力するか、WebSocketとdesktopの両方へ出力するかはParapper applicationのpolicyであり、公開transportの規則ではない。

## desktop adapter

具体的なresourceとcompositionは`src-tauri`が所有する。

- desktop/network audio source adapterとresampling
- path解決、download、manifest、model validation
- concrete modelのpath解決と構築、host worker thread、application sessionの所有権
- `ParapperConfig`からmodel/engine用の狭いconfigへの変換
- Tauri command/event、desktop delivery、translation/synthesis routing
- `parapper-stt-server`が使用する`RecognitionBackend`実装

`src-tauri/src/recognition/driver.rs`は、crateのdriverを囲むdesktop固有のblocking shutdown policyだけを所有する。`session.rs`がcomposition rootとなり、`model_factory.rs`、`language_adapter.rs`、`turn_adapter.rs`で構築したportを接続する。以前重複していたtranscription flowとTurn flowは、application moduleとしては残さない。

`src-tauri`はASRのregistry、streaming Session対応、先頭padding、model選択policyを管理しない。これらの型と実装は`parapper-stt-engine`へ置き、desktop側からは構築とworker接続のAPIだけを見せる。

純粋なSegment/planner/reducer/route/Turnのテストは、実装とともに`parapper-stt-engine`へ置く。desktop側のrecognition testは、config/path変換、native worker、inputとshutdown lifecycle、Tauri event、desktop/WebSocket output compositionに限定する。

## config transactionに関する後続作業

recognitionの実行中にmodel sessionを生成・破棄してはならない。動的反映を許可するのはparameterの更新だけとし、session/thread/modelの変更には停止境界を必要とする。詳細なallowlist/reject transactionは引き続き独立したconfig変更として扱い、今回の抽出によって範囲を広げない。

## 依存関係のgate

```text
cargo check -p parapper-models --no-default-features
cargo test -p parapper-models --features native-models
cargo test -p parapper-stt-engine
cargo test -p parapper-stt-server
cargo tree -p parapper-stt-engine  # tauri / ort / sherpa / vibratoを含まない
cargo tree -p parapper-stt-server  # tauri / ort / sherpa / vibratoを含まない
cargo test -p parapper
```
