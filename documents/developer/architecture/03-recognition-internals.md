# recognition 内部詳細

`RecognitionSession` は state holder、`RecognitionDriver` は event order / step priority を持つ。
個別 workflow は stage module の `flow.rs` に分散している。

## Session 状態

```mermaid
classDiagram
    class RecognitionDriver {
        -RecognitionSession runtime
        -SegmentationFlow segmentation_flow
        +push_vad_frame(samples, vad_result)
        +update_config(config)
        +step()
    }

    class SegmentationFlow {
        -SegmentBuilder segment_builder
        +push_vad_frame(samples, vad_result)
    }

    class RecognitionSession {
        -ParapperConfig config
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
        +TurnOutputSink output_sink
        +LanguageIdRuntime language_id_runtime
        +LanguageDetector language_id
        +JapaneseMorphAnalyzer japanese_morph
    }

    RecognitionDriver *-- SegmentationFlow
    RecognitionDriver *-- RecognitionSession
    RecognitionSession *-- RuntimeIo
```

## step 優先順位

`RecognitionOuterLoop` は各 step の先頭で frontend からの config 更新を 1 回だけ取り出す。dirty bit に応じて、audio-only 設定は `AudioInputProcessor` の参照 config だけを差し替え、VAD 閾値変更は `RecognitionVadStage`、ASR / turn / delivery に関わる変更は `RecognitionDriver::update_config` へ渡す。

```mermaid
flowchart TD
    outer[RecognitionOuterLoop::step] --> config{runtime config dirty?}
    config -- driver dirty --> update_driver[RecognitionDriver::update_config]
    config -- VAD dirty --> update_vad[RecognitionVadStage::update_config]
    config -- no / applied --> step[RecognitionDriver::step]
    update_driver --> step
    update_vad --> step

    step --> result{ASR result ready?}
    result -- yes --> transcription_result[transcription::flow\napply ASR result action]
    result -- no --> turn_check{pending turn check?}

    turn_check -- stale epoch --> drop_check[drop pending check]
    turn_check -- active --> turn_silence[turn::flow\nsilence action]
    turn_check -- none --> timeout[turn::flow\ntimeout action]

    turn_silence --> rerecognize[turn::flow\nrerecognition dispatch]
    turn_silence --> complete[turn::flow\nfinal without grammar]
    timeout --> timeout_final[turn::flow\ntimeout final or rerecognition]
    timeout --> dispatch[transcription::flow\ndispatch next ASR if idle]
```

## ASR result から output まで

```mermaid
sequenceDiagram
    participant Runner as AsrRequestRunner
    participant Transcription as transcription::flow
    participant Transcript as turn::transcript
    participant TurnFlow as turn::flow
    participant Boundary as turn::boundary_flow
    participant Sink as TurnOutputSink

    Runner-->>Transcription: AsrResult
    Transcription->>Transcription: match request / stale check / reduce

    alt InterimDisplay
        Transcription->>Transcript: apply segment transcript
        Transcription->>TurnFlow: emit interim when enabled
    else CompletionCheck
        Transcription->>Transcript: apply segment transcript
        Transcription->>TurnFlow: rerecognize or final
    else Rerecognition
        Transcription->>Transcript: replace full turn transcript
        Transcription->>Boundary: grammar boundary flow
        Boundary->>Sink: final whole turn or keep open
    else stale / mismatch / unusable
        Transcription->>Transcription: keep in-flight, drop, or fallback
    end
```

## 読み方

- ASR の engine / runner / task 型は `transcription/asr/`。
- request queue、in-flight、result action 適用は `transcription/flow.rs`。
- TurnDraft mutation は `turn/transcript.rs`。
- open turn lifecycle と timeout は `turn/flow.rs`。
- grammar boundary decision は `turn/boundary_flow.rs`。
- `PendingRuntimeState::turn_check` は単一 slot。stale な check を drain する queue として扱わない。
