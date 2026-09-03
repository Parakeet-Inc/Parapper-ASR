# recognition desktop adapter

`src-tauri/src/recognition`はSTT coreそのものではなく、Parapper desktop
applicationを`parapper-stt-engine`へ接続するadapter/composition層である。

## Production経路

```text
RunningRecognitionInput (desktop lifecycle / queue)
  -> AudioInputProcessor / VAD adapter
  -> src-tauri RecognitionDriver (shutdown deadline adapter)
  -> parapper_stt_engine::RecognitionDriver
  -> parapper_stt_engine::RecognitionSession
  -> Segment / transcription / Turn flow
  -> RecognitionOutputSink
  -> desktop delivery / WebSocket output
```

## Tauri側に残すもの

- audio device、resampler、入力queueとworker lifecycle
- `ParapperConfig -> SttEngineConfig`変換
- `AppHandle`を使うresource/data path解決、model download/cache/validation
- 解決済みpathからのASR/SLI/Namo/Morph実装の構築
- Tauri event、missing-model/error通知、desktop delivery
- blocking sleep、shutdown deadline、worker join
- `parapper-stt-server`へ`AppState`を接続するbackend adapter

## Flat module layout

desktop adapterは責務の境界がファイル名から読めるように、階層shimを置かず
`recognition`直下へ配置する。

- `input.rs` / `input_source.rs`: 入力workerと入力source/channel
- `session.rs` / `driver.rs`: engine portの構築とdesktop lifecycle
- `model_factory.rs` / `asr_worker.rs`: ASR・SLI model構築とASR worker
- `language_adapter.rs` / `turn_adapter.rs`: SLI warning、Namo、Morph adapter
- `events.rs` / `output_sink.rs`: Tauri eventとdesktop/WebSocket出力
- `streaming.rs`: `parapper-stt-server`を`AppState`へ接続するserver adapter

旧`control`、`segmentation`、`transcription`、`turn`階層はengine内部の責務を
desktop側で再現してしまうため置かない。engine typeは`parapper-stt-engine`、
model typeは`parapper-models`から直接importする。

Tauriのpath APIはここで使用してよい。ただし`AppHandle`やTauriのpath型を
engineのtraitへ渡さず、構築済みのhost-neutral portを注入する。CLIやprivate
serverは別のpath resolverを使って同じengineを組み立てられる。

## Engine側へ移したもの

- `RecognitionSession`とruntime state
- Segment state machineとdriver priority
- ASR request planning、SLI route、stale result reduction
- ASR入力前処理
- Turn transcript、grammar/Namo/silence/timeout flow
- structured recognition output

## Test ownership

SegmentBuilder、planner/reducer、route、Turn state/finalizationなど、hostに
依存しない挙動のunit/regression testは`parapper-stt-engine`に置く。
`src-tauri`にはconfig/path変換、native model worker、audio/input lifecycle、
shutdown、Tauri event、desktop/WebSocket deliveryを接続するテストだけを置く。

`streaming.rs`は推論streamingではなくWebSocket serverのdesktop adapterである。
protocol/socket stateは`parapper-stt-server`が所有する。

詳細は[engine boundary](../../../documents/developer/architecture/05-parapper-engine-boundary.md)を参照。
