# recognition モジュール俯瞰

`recognition/` は、AI 部品名ではなく pipeline stage で読む。

```text
recognition/
  control/
  segmentation/
  transcription/
  turn/
```

```mermaid
flowchart TD
    input[control::input\nouter loop] --> driver[control::driver\nRecognitionDriver]
    driver --> segmentation[segmentation::flow\nVAD frame -> SegmentEvent]
    segmentation --> session[control::session\nRecognitionSession]

    session --> transcription[transcription::flow\nASR request/result workflow]
    transcription --> planner[transcription::planner\nrequest planning]
    transcription --> reducer[transcription::reducer\nresult reduction]
    transcription --> asr[transcription::asr\nASR engine/runner/task]
    transcription --> route[transcription::route\nroute selection / SLI]

    transcription --> transcript[turn::transcript\nTurnDraft mutation]
    session --> turn_flow[turn::flow\nopen/continue/final/timeout/output]
    turn_flow --> boundary_flow[turn::boundary_flow\ngrammar boundary decision flow]

    boundary_flow --> boundary[turn::boundary\ncandidate generation]
    turn_flow --> policy[turn::policy\ncompletion/silence/timeout/grammar]
    turn_flow --> decision[turn::decision\nTD contract / Namo]
    turn_flow --> domain[turn::domain\nTurnDraft / TurnConfirmed]
    domain --> output[turn::port\nTurnOutputSink]
    output --> delivery[delivery\nrecognized text fanout]

    classDef control fill:#e8f2ff,stroke:#4d7fb8,color:#10243d
    classDef stage fill:#ecf8ee,stroke:#4f9a61,color:#103719
    classDef decision fill:#fff4df,stroke:#bd8733,color:#3b2705
    classDef io fill:#ffecec,stroke:#b85c5c,color:#451515

    class input,driver,session control
    class segmentation,transcription,transcript,turn_flow,boundary_flow stage
    class planner,reducer,route,boundary,policy,decision,domain decision
    class asr,output,delivery io
```

## 境界

- `control/`: production orchestration、session state、driver priority。runtime config は dirty bit で差分を分類し、audio / VAD / driver の必要な経路だけへ反映する。
- `segmentation/`: audio stream / VAD frame から segment event を作る。
- `transcription/`: segment / turn audio を ASR に投げ、result を workflow action として処理する。
- `turn/`: transcript を Turn 状態へ反映し、continue / final / output を決める。debug badge のような表示用 payload は runtime event contract に含めない。

## 不変条件

- ASR in-flight は 1 件だけ。
- pending turn check は queue ではなく 1 slot。新しい check は stale 判定用 epoch を持つ現在値として扱う。
- ASR result は request identity が一致してから適用する。
- stale ASR result / stale output は downstream へ流さない。
- Namo Continue 後の発話 activity 中は timeout final しない。
- grammar boundary は completion ASR の末尾候補だけを Turn 完了に使い、途中候補では Turn を open のまま維持する。

## 設計判断

- `RecognitionDriverHandle` は `control/input.rs` が使う boxed driver interface として残す。audio worker と recognition driver の境界を狭く保つため。
- `transcription/asr/task.rs` の request metadata は、workflow-level と engine-only に過剰分割しない。2 つ目以降の concrete engine 差分が明確になった時点で分ける。
- route / SLI の session tests は `control/tests` 配下に置く。`RecognitionSession` の state mutation と production harness 上の event order を合わせて見るため。
- `TurnDecision` は `is_end_of_turn` と `confidence` の最小契約にする。Namo response label や tokenizer details は engine 固有実装に閉じる。
- turn lifecycle から ASR rerecognition dispatch へ進む bridge は `RecognitionSession` に置く。turn stage が request queue の内部構造を直接所有しないため。
