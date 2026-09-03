# parapper-models

`parapper-models` contains the host-neutral model contracts and implementations
used by Parapper:

- `asr`: offline and streaming ASR, ECAPA language identification, feature extraction, CTC/TDT/transducer decoders
- `mt`: local translation models and session-local cache
- `tts`: Supertonic implementations
- `nc`, `vad`: noise cancellation and voice activity detection
- `td`: timestamp-aligned boundary candidates, Namo and Japanese Morph turn detectors

The crate does not resolve application paths, download models, create host
worker threads, emit UI events, or depend on Tauri. Constructors receive
explicit paths and model-specific parameters.

Native implementations are opt-in features. The default feature set exposes
ASR contracts, pure decoders, model metadata, VAD contracts, and TD domain
types without linking ONNX Runtime or Vibrato. The desktop enables
`native-models`; `parapper-stt-engine` intentionally uses no native features.

```powershell
cargo check -p parapper-models --no-default-features
cargo test -p parapper-models --features native-models
```

ASR streaming covers both repeated inference over partial audio and stateful
Nemotron inference. Network streaming is a transport concern and lives in
`parapper-stt-server`.

Each concrete model implementation owns its ORT sessions. The shared runtime
module only performs process-wide ORT initialization; it is not a session or
thread manager.

`tts::LocalTtsEngine` owns the Supertonic sessions and model-specific mutable
speaker state. It returns host-neutral PCM; playback,
device selection, queues, and Tauri events remain outside this crate.
