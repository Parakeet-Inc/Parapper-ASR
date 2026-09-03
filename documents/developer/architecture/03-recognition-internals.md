# recognition内部詳細

STTの状態遷移は`parapper-stt-engine`にあり、desktop側は`RecognitionDriver`を駆動する。`RecognitionSession`はstate holder、`RecognitionDriver`はevent order / step priority、`AsrExecutionRuntime`は構築済みASR modelの利用policyを持つ。

## Session状態

```mermaid
classDiagram
    class RecognitionDriver {
        -RecognitionSession runtime
        -SegmentationFlow segmentation_flow
        +push_vad_frame(samples, vad_result)
        +update_config(config)
        +step()
        +flush_input()
    }

    class RecognitionSession {
        -SttEngineConfig config
        -PendingRuntimeState pending
        -RuntimeIo io
        -TurnStore turn_store
        -RuntimeCounters counters
        -ActivityState activity
        -AsrRequestState requests
    }

    class RuntimeIo {
        +AsrRequestRunner asr_runner
        +TurnDecisionRunner turn_decision_runner
        +RecognitionOutputSink output_sink
        +LanguageDetector language_id
        +TranscriptBoundaryDetector boundary_detector
    }

    class AsrExecutionRuntime {
        -AsrModelRegistry models
        -Map~AsrStreamingSessionKey, AsrStreamingState~ streams
        +execute(config, request)
        +reset_streaming_sessions()
    }

    RecognitionDriver *-- RecognitionSession
    RecognitionSession *-- RuntimeIo
    RuntimeIo --> AsrExecutionRuntime : host worker経由
```

## step優先順位

desktop outer loopはfrontendからのconfig更新を取り出し、audio/VAD/driverへ必要な更新だけを渡す。engine driverの1 stepは次の優先順位を守る。

```mermaid
flowchart TD
    step["RecognitionDriver::step"] --> result{"ASR result ready?"}
    result -- yes --> apply["request一致・stale判定・result適用"]
    result -- no --> check{"pending turn check?"}
    check -- stale --> drop["stale checkを破棄"]
    check -- active --> silence["silence action"]
    check -- none --> timeout["timeout action"]
    silence --> next["rerecognition / final / next request"]
    timeout --> next
    apply --> next
```

activityは小さなbounded job queueに流さず、epochとして更新する。Namo Continue後に`SegmentStarted` / `SegmentExtended`が来た場合はtimeout起点を更新し、active speech中の誤finalを防ぐ。

## ASR request実行

```mermaid
sequenceDiagram
    participant Planner as transcription planner
    participant Worker as src-tauri ASR worker
    participant Runtime as AsrExecutionRuntime
    participant Model as parapper-models::AsrEngine

    Planner->>Worker: AsrRequest
    Worker->>Runtime: execute(SttAsrConfig, request)
    alt Nemotron streaming interim
        Runtime->>Model: start_stream(session)（初回のみ）
        Runtime->>Model: push_stream(session, delta)
    else completion / offline
        Runtime->>Model: cancel_stream(active sessions)
        Runtime->>Model: recognize(prepared source audio)
    end
    Model-->>Runtime: AsrTranscript
    Runtime->>Runtime: leading padding分のtimestamp補正
    Runtime-->>Worker: transcript / error
    Worker-->>Planner: AsrResult（elapsedとTauri warningを付加）
```

model registryとstreaming stateはengineに1つだけ置く。desktop側で同じSession keyのmapを重ねず、model APIのstart/push/cancelをengineから明示的に呼ぶ。ORT SessionやNemotron encoder cacheは`parapper-models`内部から出さない。

## ASR resultからoutputまで

```mermaid
sequenceDiagram
    participant Runner as AsrRequestRunner
    participant Transcription as transcription flow
    participant Turn as Turn flow
    participant Boundary as TranscriptBoundaryDetector
    participant Sink as RecognitionOutputSink

    Runner-->>Transcription: AsrResult
    Transcription->>Transcription: request match / stale check / reduce
    alt InterimDisplay
        Transcription->>Turn: segment transcriptを反映
        Turn->>Sink: 設定に応じてinterim
    else CompletionCheck
        Transcription->>Turn: completionを反映
        Turn->>Turn: rerecognizeまたはfinal
    else Rerecognition
        Transcription->>Turn: full turn transcriptへ置換
        Turn->>Boundary: grammar boundary候補
        Turn->>Sink: whole turnをfinal、またはopen維持
    else stale / mismatch / unusable
        Transcription->>Transcription: keep / drop / fallback
    end
```

## 読み方

- model固有推論: `crates/parapper-models/src/asr/`
- ASR実行policy: `crates/parapper-stt-engine/src/asr.rs`
- request planning/reduction/preprocessing: `crates/parapper-stt-engine/src/transcription/`
- Turn lifecycle: `crates/parapper-stt-engine/src/turn/`
- desktop composition/worker: `src-tauri/src/recognition/session.rs`と`asr_worker.rs`
- desktop入出力: `src-tauri/src/recognition/input.rs`と`output_sink.rs`
