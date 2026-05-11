# AGENTS.md

## 作業姿勢

- 仕様が曖昧なまま実装しない。特に状態機械、非同期キュー、timeout、interim/final のような境界条件は、わからない場合は先に質問する。
- 「今の実装に合わせたテスト」ではなく、「ユーザーが期待する挙動から逆算したテスト」を先に書く。
- バグ修正では、可能な限り修正前に落ちるテストを確認する。落ちないテストだけ追加して完了扱いにしない。
- モックは壊れている実経路を迂回してはいけない。状態遷移を検証するときは最終 event だけでなく、その前の `SegmentStarted` / `SegmentExtended` や timeout check も含める。
- 高頻度の制御シグナルを、小さい bounded channel の通常ジョブとして流さない。必須ジョブを詰まらせるため、activity は epoch / atomic など別経路で扱う。
- 外部連携仕様は公開ドキュメントを確認してから実装する。本体内蔵 API とプラグイン個別 HTTP サーバの endpoint を混同しない。
- YNC plugin command は plugin HTTP server の `POST /`、NEO text input は本体 API の `/api/input` として扱う。接続失敗時に別 endpoint や別 port へ暗黙 fallback / retry しない。
- テスト名は「何を保証するか」が読める名前にする。例: `td_flow_continue_decision_then_active_speech_does_not_timeout_before_next_segment`。
- 既存の用語を曖昧に再利用しない。`Segment` は ASR に 1 回投げる音声単位、`Turn` はユーザー発話のまとまり、`TurnDraft` は mutable な途中状態、`TurnConfirmed` は immutable な確定状態。

## Recognition / Turn 周りの注意

- `segment_split_silence_ms` は途中経過表示用の短い無音であり、Turn を分割する条件ではない。
- `turn_check_silence_ms` は Turn 完了判定に進むための無音であり、Namo が Continue と判断した場合は open turn として保持する。
- Namo Continue 後に発話が続いた場合、次の `SegmentClosed` は同じ Turn に連結されるべき。
- Namo Continue 後、次の発話が active な間は timeout final してはいけない。`SegmentStarted` / `SegmentExtended` の activity で open turn の timeout 起点を更新する必要がある。
- Namo Continue 後に次の Segment activity が来ない場合だけ、`turn_check_silence_ms * 2` の timeout で final に倒す。

## 不具合修正と再現テスト

- 不具合修正では、実装修正を先に入れて通るテストだけを作らない。まず現象を再現するテストまたはローカル検証コマンドを用意し、修正前に失敗することを可能な範囲で確認する。
- ローカル手順でしか再現できない場合も、修正後に同じ条件を単体テスト、統合寄りテスト、診断コマンドのいずれかへ落とし込む。
- 受け取った音声、ASR 結果、翻訳リクエスト、読み上げリクエストのどこで遅れているかをログまたはモックで観測できるようにする。
- 回帰しやすい条件は Rust の単体テストに落とす。例: 連続発話、Namo Turn Detector の中途確定、翻訳後の読み上げ、遅いレスポンスを返す YNC モック。
- テストを通すこと自体を目的にしない。実際に起きる症状と同じ順序、同じ待ち時間、同じ送信先 API をなるべく再現する。
- 実装に都合のよい `is_final=true` や加工済みリクエストだけを手で作るテストで済ませない。segmenter -> transcription -> delivery のように問題が境界をまたぐ場合は、可能な限り上流のイベントや worker 由来の出力からモック送信までつなげて確認する。
- ポート番号や API パスのフォールバックを足して隠さない。設定値が間違っている場合は失敗させ、ログとテストで検出できるようにする。
- 外部連携の修正では、単発だけでなく 2 回目以降の連続発話も確認する。

テスト設計:

- まず「壊れている挙動」をテスト名に含める。例: Namo の判定待ち中に次の speech が入ったあと timeout しても panic しない、前 turn の final が replay より前に翻訳へ進む、など。
- 失敗原因がタイミング・順序・キュー・外部 HTTP のいずれにあるかを分けて観測できるようにする。順序が重要な場合は、受信した operation/id の集合だけでなく、必要に応じて発生順や時刻も検証する。
- 境界ごとに確認を分ける。SegmentBuilder は event の順序と音声 padding、transcription は turn final 出力と revision、delivery は翻訳・読み上げ request、connect は HTTP payload とポート/API path を検証する。
- 同じ不具合に対して「小さい再現テスト」と「モックを使う統合寄りテスト」を組み合わせる。小さいテストで panic や状態遷移を固定し、統合寄りテストで実際の送信先 API まで届くことを確認する。
- HTTP mock では期待 request が届くことだけで終わらせず、余分な request が来ないことも確認する。source identity は表示用 id 文字列だけでなく、`turn_session_id` / `turn_id` / `turn_revision` / `segment_id` などの構造化 metadata を優先して検証する。
- テストが通った後も、実際の症状を説明できない場合は修正完了とみなさない。テストが現象を再現できていない可能性を優先して疑う。

## テスト作成のアンチパターン

新しくテストを書くときは、以下のアンチパターンに該当しないか自己チェックする。

### A. 実装をなぞるだけのテスト (Implementation-mirror)

`fn join(a, b) = a + b` に対して `assert_eq!(join("a", "b"), "ab")` を書くだけのテスト。**実装を書き換えればテストも書き換わる**ため、リグレッション検出力ゼロ。

- 文字列フォーマット系の utility は、**ユーザー観点の挙動 (e.g. 「`...` で終わる interim final 化時に句点が重複しない」) を境界値ベースで table-driven test** にする。1 入力 1 出力の assertion を関数ごとに書かない。
- pure helper function は、callsite 側の挙動でカバーされていれば単体テストは不要。

### B. 実経路を迂回するモック (Mocking bypass)

「最終 event だけ直接流す」モックは、その手前の state 遷移バグを見落とす。

- モックが流す event 列は、**production の producer (segmenter / TD / route) が実際に発行する順序と一致**しているか、テスト前提条件として明記する。
- `SegmentClosed` だけ流すのではなく、`SegmentStarted -> SegmentExtended -> (timeout check) -> SegmentClosed` を流す。**timeout check の tick も省略しない**。
- モックを書くときは「production code の何行目から何行目を bypass しているか」を意識する。bypass がある時点で、そのコード経路はテストされていない。

### C. 恒真テスト (Tautological assertions)

- `assert!(result.is_ok())` だけで中身を見ない。
- `assert!(!output.is_empty())` で内容が重要なのに値を見ない。
- `assert_eq!(events.len() >= 1, true)` のような下限緩い assertion。
- 「panic しなければ OK」のテスト。
- bool config flag を変えて bool 結果を確認するだけのテスト (内部分岐ロジックを通っていない)。

**判定基準**: 「この assertion が成立しないコードを意図的に書けるか」。書けないなら恒真。

### D. 重複テスト (Redundant duplicates)

- recognition 版 / translation 版で同じロジックを 2 回テストする (e.g. `skip_namo_intermediate_recognition` と `skip_namo_intermediate_translation`)。
- locale や mapping 名だけ変えた cosmetic variation。

**統合方針**: parameterized test に統合し、**両 source_kind / 両 locale を for ループで回す**。片方を消すのではなく統合すると、対称性が型レベルで保証される (片方を将来追加した時に忘れない)。

### E. 長く脆い fixture (Brittle setup)

- `TimedMockHttpServer::start(8, ...)` のように **期待 request 数が hardcode** された helper。テストを 1 件追加するたびに `8` を `9` に直す必要がある。
- 50+ 行の `setup_*()` を共有する複数テスト。helper を変更するたび全 caller が壊れる。
- `OnceLock` / 環境変数 / global state を共有するテスト。実行順序に依存する。

**対策**: count を引数で取らず `recv_all() -> Vec` / `drain_until_timeout()` のような **unbounded API** にする。

### F. 13 フィールドの struct を 20 箇所で構築する (Field explosion)

`SpeechMapping { id, source_kind, target_lang, backend, talker, local_tts_voice: None, local_tts_language: None, local_tts_speaker_id: None, output_device_id: None, output_device_host: None, output_device_name: None, muted: false, volume: 1.0 }` のような **多フィールド struct を平叙文で複数箇所に書く**。

**問題**:

- フィールド追加時、20 箇所すべて編集が必要 → 「テストを増やすコスト」が高い → 結果として「コピペで済ます」誘因になる。
- 「このテストの mapping は何が特殊か」が `..` の中に埋もれる (テストの意図が読み取れない)。

**対策**: テストモジュール内に `fn speech_mapping(id, source_kind) -> SpeechMapping` のような薄いヘルパーを置き、各テストでは **変えるフィールドだけ struct update syntax で上書き** する。

```rust
let mapping = SpeechMapping {
    target_lang: Some("en_US".into()),
    muted: true,
    ..speech_mapping("speech-en", SpeechSourceKind::Translation)
};
```

**注意**: production の `Default` トレイト実装にしてはいけない。「テストの default」と「production の default」を混ぜないため、必ずテストモジュール内のヘルパー関数として置く。

### G. 検出範囲の不明な assertion 群

`assert_eq!(requests.len(), 1); assert_eq!(requests[0].id, "speech-foo"); assert_eq!(requests[0].text, "...");` のように **個別フィールドを順に assert** すると、新フィールドが追加されたときに「テストはそのフィールドを保証していない」状態になる。

**対策**: 完全一致を見たい時は `assert_eq!(requests, vec![expected_request])` で **構造体全体で比較** する。新フィールドが追加されると、完全一致の expected もコンパイルエラーで気付ける。

### H. 外部契約が曖昧なテスト (External contract ambiguity)

YNC 本体 API と YNC plugin HTTP API のように endpoint が違う外部契約を、同じ helper や曖昧なテスト名で混ぜない。

- plugin command は `POST /`、本体 text input は `/api/input` として fixture とテスト名で固定する。
- fallback / retry を禁止する仕様の regression は削らない。削る場合は、仕様変更として根拠をコードまたは README に残す。
- 外部依存の初期化や contract regression は、細粒度の unit test ではなく契約テストとして扱う。

### 自己チェックリスト

新しいテストを書いたら以下を自問する。

1. **検出力**: この assertion が破られるような production code 変更を 1 つ書けるか？書けないなら恒真テスト。
2. **イベント列**: 状態遷移バグなら、producer が発行する event 列を全部流しているか？
3. **モック整合**: モックが流す順序は production の producer が実際に流す順序と一致するか？
4. **重複**: 既存テストと "1 行だけ変えた" 関係になっていないか？parameterized 化すべきか？
5. **fixture 健康度**: 50 行超の helper に依存していないか？hardcode count はないか？
6. **意図の可読性**: テスト名から「何を保証するか」が読み取れるか？struct construction で「何が特殊か」が読み取れるか？
7. **修正前失敗確認**: バグ修正テストなら、修正前に確実に落ちることを確認したか？
8. **外部契約**: 外部 API の endpoint、payload、fallback 禁止が fixture と assertion に現れているか？
