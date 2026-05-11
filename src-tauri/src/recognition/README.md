# `recognition/` - 音声認識パイプライン

マイク入力の PCM サンプル列を、配信先へ渡せる `RecognizedTextOutput` に変換する層。
VAD とセグメンタはオーディオ入力側で軽く処理し、ASR / 言語識別 / Namo Turn Detector は
`AsrWorker` の別スレッドで実行する。

## 全体データフロー

```
マイク入力 (PCM 16kHz mono)
        |
        v
RecognitionPipeline.process_chunk()
  - VadEngine: チャンクを speech / silence に分類
  - SegmentBuilder: VAD chunk 列から ASR 用 Segment を作る
        |
        v
AsrWorker
  - SLI: 必要なときだけ言語識別して RecognitionRoute を選ぶ
  - ASR: SegmentClosed の full_audio を認識
  - Namo: 有効時だけテキストからターン終了を判定
  - Turn: recognition::turn の TurnDraft を更新
  - TurnConfirmed: immutable な発話結果を作る
        |
        v
delivery / Tauri events
```

## ユビキタス言語

| 用語 | 意味 |
| --- | --- |
| Segment | ASR に 1 回投げる音声単位。Turn を分割する意味ではなく、途中経過表示や長時間発話対策のために `SegmentBuilder` が VAD chunk 列から作る |
| Turn | `turn/` が定義する進行中の発話単位。内部に `TurnDraft` を持つ |
| TurnDraft | `Turn` の mutable な下書き。ASR 済み Segment の transcription を蓄積し、必要なら full audio から作り直す |
| Transcription | ASR の結果として得た文字起こし。`TurnDraft` 内で Segment ごとに追加される |
| TurnConfirmed | 完了した immutable な Turn。delivery へ渡した後は内容を変更しない |

責務の境界:

- `SegmentBuilder` は Segment を作る。Turn の完了判断や文字起こしは扱わない。
- `turn/` は `Turn / TurnDraft / TurnConfirmed` を定義する。
- `transcription/` は Segment を ASR し、`turn/` の `TurnDraft` に transcription を追加する。
- Turn が完了したら `TurnDraft` から `TurnConfirmed` を作り、以後は immutable な値として delivery に渡す。

## ファイル構成

```
recognition/
├── mod.rs                  公開 API (RecognitionPipeline / RecognitionStatus)
├── pipeline.rs             VAD + SegmentBuilder + AsrWorker の接続
├── route.rs                言語コードと設定から ASR / Turn Detector ルートを選ぶ
├── sli.rs                  言語識別の呼び出しと route 選択
├── events.rs               Tauri 向けイベント型
├── engine_cache.rs         ASR / Namo / SLI エンジンの lazy load とキャッシュ
├── engines/                VAD / ASR / Namo / SLI の trait と実装
├── segment_builder/        VAD 結果から ASR 用 Segment イベントを生成
├── turn/                   Turn / TurnDraft / TurnConfirmed
└── transcription/          AsrWorker、ジョブ、ワーカー実行ループ
```

## `pipeline.rs`

`RecognitionPipeline::process_chunk(samples)` が入口。

1. `VadEngine` で 512 サンプル前後の入力チャンクを speech / silence に分類する。
2. `SegmentBuilder` に渡して `SegmentBuilderEvent` を生成する。
3. `SegmentClosed` / `TurnCheckSilenceReached` を `AsrWorker` の bounded channel に投入する。

重い推論はワーカーに逃がすため、入力側では VAD と状態更新だけを行う。
`SegmentStarted` / `SegmentExtended` は ASR ジョブではなく activity epoch として記録し、
Namo が「まだ続く」と判断した open turn の timeout 起点を更新する。
worker は `recv_timeout` の待ち時間を使って、open turn が放置された場合だけ時間で確定させる。

## `segment_builder/`

VAD の時系列から、ASR に渡す Segment を切り出す。
`SegmentBuilder` は同期的な状態機械で、Turn Detector への問い合わせは行わない。
Turn の interim/final 判定は `transcription/worker_runtime.rs` に閉じ込める。

主要イベント:

| イベント | 用途 |
| --- | --- |
| `SegmentStarted { segment_id, audio_so_far }` | speech run が成立した時点の初期音声を通知 |
| `SegmentExtended { segment_id, new_audio }` | speech / short silence の追加音声を通知 |
| `SegmentClosed { segment_id, full_audio, reason }` | 途中経過表示、turn-end silence、最大長のいずれかで ASR 入力単位を閉じる |

`SegmentClosed.full_audio` が ASR の入力。`SegmentStarted` / `SegmentExtended` は `SegmentBuilder` の完全性を表す
イベントとして残しているが、現在の pipeline は ASR worker へ転送しない。

## `sli.rs` と `route.rs`

`sli.rs` は言語識別の呼び出し条件と route 選択をまとめる。

- `multilingual_asr_enabled == false` なら SLI は呼ばない。
- 1 秒未満の音声は SLI をスキップする。
- SLI の候補言語は `enabled_asr_models` から作る。
- SLI を呼ばない場合、または検出結果を有効な ASR ルートに変換できない場合は、
  現在の turn route があれば継続し、なければ設定済み言語を使う。

言語識別は ASR モデルを使い分けるためだけに使う。翻訳先、翻訳可否、delivery 側の
`target_lang` 推測には使わない。

## `transcription/`

`AsrWorker` は `AsrJob` を受け取り、Segment 単位の ASR と Turn 単位の確定を行う。
Turn の状態は `recognition::turn` の `Turn / TurnDraft / TurnConfirmed` を使う。

```rust
enum AsrJob {
    SegmentClosed {
        segment_id,
        previous_segment_id,
        full_audio,
        reason,
    },
    TurnCheckSilenceReached {
        previous_segment_id,
    },
}
```

`SegmentClosed` では次の順で処理する。

1. 既存の open turn があればその `turn_id` を使い、なければ `segment_id` を turn id にする。
2. `sli.rs` で `RecognitionRoute` を選ぶ。
3. ASR で `full_audio` をテキスト化する。
4. `recognition::turn::Turn` が持つ `TurnDraft` に音声・transcription・route・検出言語を追加する。
5. Namo が無効、または Namo が end-of-turn と判断したら `TurnConfirmed` を作って emit する。
6. Namo が「継続」と判断したら interim を emit し、次の `SegmentClosed` を同じ turn に追加する。

`interim_result_enabled` が有効な場合、`interim_result_silence_ms` まで短い無音が続いた時点で
途中経過表示用の `SegmentClosed` を出す。これは Turn の分割ではなく、そこまでの音声を ASR に渡して
同じ Turn のログ行を更新するための処理。無効な場合、この短い無音では ASR を走らせず、
`turn_check_silence_ms` に到達するまで待つ。

`turn_rerecognize_full_on_complete` が有効な場合、final emit の直前に `TurnDraft.full_audio`
全体を同じ route で再 ASR し、成功した結果で `TurnDraft` 内の transcription を作り直す。失敗または空文字の場合は、
それまでのセグメントごとの蓄積結果をそのまま使う。

open turn の timeout は worker の `recv_timeout` ごとに確認する。open turn 中に次の
`SegmentStarted` / `SegmentExtended` が来た場合は「発話が続いている」とみなし、activity epoch の変化で
timeout 起点を更新する。新しい segment activity がないまま `turn_check_silence_ms * 2` を超えたら、
Namo の継続判断を待たずに final を emit する。

## 出力方針

| 状況 | emit | `is_final` | `update_mode` |
| --- | --- | --- | --- |
| Simple turn / Namo end-of-turn | turn 確定時 | true | Replace |
| Namo が継続判断 | セグメントごと | false | Replace |
| Namo open turn の timeout | worker timeout check 時 | true | Replace |

現在は同じ turn id を Replace で上書きする設計。delivery 側は最新の turn 表示を受け取り、
final で確定テキストとして扱う。

## 設計のポイント

- 入力側は VAD とセグメント状態だけを扱い、重い推論をワーカーへ分離する。
- `SegmentBuilder` は VAD ベースの Segment 切り出しに専念し、Turn Detector の非同期状態を持たない。
- SLI は route 選択だけに限定し、翻訳設定とは結合しない。
- Namo の「まだ続く」は `AsrWorker` 内の `open_turn_id` で表現し、次の SegmentClosed を
  同じ `Turn` の `TurnDraft` に追加する。
- open turn 中の `SegmentStarted` / `SegmentExtended` は ASR 対象ではないが、timeout を止める activity epoch として
  `AsrWorker` が参照する。
- `TurnDraft` は mutable、`TurnConfirmed` は immutable として扱う。
- 古い turn は新しい final が出る前に `take_stale_turn_final_outputs` で確定させる。
