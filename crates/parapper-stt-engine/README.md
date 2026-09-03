# parapper-stt-engine

`parapper-stt-engine` owns Parapper's Tauri-free STT domain: NC + VAD + TD +
ASR orchestration, Segment and Turn lifecycle, request/revision state, activity
epochs, and structured recognition output.

It depends on `parapper-models` with native features disabled. Model sessions
are provided through ports or injected into `AsrModelRegistry`, so the engine
can be built without Tauri, an audio device, WebSocket, ORT, Sherpa, or Vibrato.

`SttEngineConfig` is the engine-owned runtime view. The desktop converts its
flat-persisted `ParapperConfig` into that type; the persistence shape is not an
engine API. `RecognitionDriver` and `RecognitionSession` own the complete
Segment -> ASR -> Turn orchestration. The host drives them through
`push_vad_frame`, `step`, `flush_input`, and drain methods.

`AsrExecutionRuntime` owns STT-level model selection, streaming Session ID
mapping, explicit start/push/cancel lifecycle, leading padding, and timestamp
correction. ORT Sessions and model-specific cache tensors remain hidden inside
the injected `parapper-models::asr::AsrEngine` implementations.

The desktop host still owns concrete audio capture and resampling, model path
resolution/downloads, model construction and worker threads, Tauri events, delivery,
and blocking shutdown deadlines. It resolves paths before constructing model
adapters, so `AppHandle` and filesystem policy never cross an engine port.

`parapper-stt-server` depends on this crate and adapts its output to the public
WebSocket protocol. MT and TTS are optional application paths and do not belong
to the STT engine dependency graph.

Pure Segment, planner/reducer, route, and Turn lifecycle regressions are owned
by this crate. Host tests should cover only adapter composition and external
contracts rather than rebuilding engine state-machine tests through a wrapper.

```powershell
cargo test -p parapper-stt-engine
cargo tree -p parapper-stt-engine
```
