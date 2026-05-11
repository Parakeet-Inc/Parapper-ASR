use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use super::worker_runtime::{
    PostSegmentAction, TurnProgressDecision, post_segment_action,
    refresh_open_turn_timeout_origin_if_activity_changed,
};
use crate::{
    config::{AsrLanguage, ParapperConfig},
    delivery::RecognizedTextOutput,
    recognition::{
        route::RecognitionRoute,
        segment_builder::SegmentCloseReason,
        turn::{
            Turn, TurnDraft, take_next_output_sequence, take_stale_turn_final_outputs,
            turn_event_id,
        },
    },
};

// These tests intentionally have two layers.
// TurnDraft and MockTurnFlow tests document turn state semantics in small, readable steps.
// Regression coverage for the real ASR worker path lives in worker_runtime tests; do not treat
// MockTurnFlow activity tests as proof that pipeline events reached the worker side-channel.

#[test]
fn turn_event_id_includes_session_id() {
    assert_ne!(turn_event_id(1, 1, 0), turn_event_id(2, 1, 0));
}

#[test]
fn turn_draft_appends_recognized_segment_audio_in_order() {
    let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
    let mut draft = TurnDraft::new("turn-1".to_string(), 0);

    draft.append_recognized_segment(1, None, &[1.0, 2.0], route, "今日は".to_string(), 10);
    draft.append_recognized_segment(2, Some(1), &[3.0], route, "晴れです".to_string(), 5);

    assert_eq!(draft.full_audio, vec![1.0, 2.0, 3.0]);
    assert_eq!(draft.combined_text, "今日は晴れです");
    assert_eq!(draft.processing_millis, 15);
    let source = draft.source_meta(7, 1, 3);
    assert_eq!(source.turn_session_id, 7);
    assert_eq!(source.turn_id, 1);
    assert_eq!(source.output_sequence, 3);
    assert_eq!(source.segment_id, 2);
    assert_eq!(source.previous_segment_id, Some(1));
}

#[test]
fn turn_draft_keeps_interim_result_in_the_same_turn() {
    let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
    let mut draft = TurnDraft::new("turn-1".to_string(), 0);

    draft.append_recognized_segment(
        1,
        None,
        &[1.0, 2.0, 0.0, 0.0, 0.0],
        route,
        "今日は".to_string(),
        10,
    );
    draft.append_recognized_segment(
        2,
        Some(1),
        &[0.0, 0.0, 0.0, 3.0, 4.0],
        route,
        "晴れです".to_string(),
        8,
    );

    assert_eq!(draft.combined_text, "今日は晴れです");
    assert_eq!(
        draft.full_audio,
        vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]
    );
    let source = draft.source_meta(7, 1, 2);
    assert_eq!(source.turn_id, 1);
    assert_eq!(source.segment_id, 2);
    assert_eq!(source.previous_segment_id, Some(1));
}

#[test]
fn stale_turns_before_new_final_are_promoted_to_final_outputs() {
    let config = ParapperConfig {
        asr_language: AsrLanguage::Japanese,
        ..ParapperConfig::default()
    };
    let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
    let mut turns = HashMap::from([
        (
            1,
            Turn::from_draft(TurnDraft {
                event_id: "turn-1".to_string(),
                segment_texts: vec!["今日は".to_string()],
                combined_text: "今日は".to_string(),
                full_audio: vec![1.0, 2.0],
                route: Some(route),
                detected_language: Some("ja".to_string()),
                processing_millis: 12,
                latest_segment_id: Some(1),
                latest_previous_segment_id: None,
                revision: 0,
            }),
        ),
        (
            3,
            Turn::from_draft(TurnDraft {
                event_id: "turn-3".to_string(),
                segment_texts: vec!["明日".to_string()],
                combined_text: "明日".to_string(),
                full_audio: vec![3.0],
                route: Some(route),
                detected_language: Some("ja".to_string()),
                processing_millis: 8,
                latest_segment_id: Some(3),
                latest_previous_segment_id: None,
                revision: 0,
            }),
        ),
    ]);
    let mut turn_revisions = HashMap::new();
    let mut next_output_sequence = 1;

    let outputs = take_stale_turn_final_outputs(
        &config,
        &mut turns,
        &mut turn_revisions,
        1,
        &mut next_output_sequence,
        3,
    );

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert_eq!(output.text, "今日は。");
    assert_eq!(output.phrase, vec![1.0, 2.0]);
    assert_eq!(output.detected_language.as_deref(), Some("ja"));
    assert_eq!(output.meta.source().output_sequence, 1);
    assert_eq!(turn_revisions.get(&1), Some(&1));
    assert!(!turns.contains_key(&1));
    assert!(turns.contains_key(&3));
}

#[test]
fn td_flow_short_speech_without_interim_result_finishes_one_turn() {
    let mut flow = MockTurnFlow::new();

    let output = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "短い発話",
    );

    assert!(output.is_final);
    assert_eq!(output.text, "短い発話。");
    assert_eq!(output.turn_id, 1);
    assert_eq!(output.segment_id, 1);
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_interim_result_emits_interim_without_closing_turn() {
    let mut flow = MockTurnFlow::new();

    let output = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        TurnProgressDecision::Complete,
        "途中経過",
    );

    assert!(!output.is_final);
    assert_eq!(output.text, "途中経過...");
    assert_eq!(output.turn_id, 1);
    assert_eq!(flow.open_turn_id, Some(1));
}

#[test]
fn td_flow_interim_result_then_speech_continues_and_finalizes_same_turn() {
    let mut flow = MockTurnFlow::new();

    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        TurnProgressDecision::Complete,
        "今日は",
    );
    let final_output = flow.segment_closed(
        2,
        Some(1),
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "晴れです",
    );

    assert!(!interim.is_final);
    assert_eq!(interim.turn_id, 1);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "今日は晴れです。");
    assert_eq!(final_output.turn_id, 1);
    assert_eq!(final_output.segment_id, 2);
    assert_eq!(final_output.previous_segment_id, Some(1));
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_turn_check_silence_uses_td_continue_instead_of_finalizing_open_turn() {
    let mut flow = MockTurnFlow::new();

    // App behavior:
    // 1. The short interim-result silence emits ASR text for display but does not ask TD yet.
    // 2. Silence then reaches turn_check_silence_ms, so the existing TurnDraft is checked by TD.
    // 3. TD Continue must keep the same turn open. There is no new ASR text at this event, so the
    //    worker should not emit another interim replacement.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        TurnProgressDecision::Complete,
        "まだ続く",
    );
    let turn_check_output = flow.turn_check_silence_reached(1, TurnProgressDecision::Continue);
    let timeout_final = flow
        .tick_after_timeout()
        .expect("open turn without later activity should still have fallback finalization");

    assert!(!interim.is_final);
    assert!(turn_check_output.is_none());
    assert_eq!(
        flow.outputs.len(),
        2,
        "TD Continue at turn-check silence should not emit a duplicate interim"
    );
    assert!(timeout_final.is_final);
    assert_eq!(timeout_final.text, "まだ続く。");
    assert_eq!(timeout_final.turn_id, 1);
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_turn_check_silence_uses_td_complete_to_finalize_open_turn() {
    let mut flow = MockTurnFlow::new();

    // App behavior:
    // - Interim-result silence only shows progress.
    // - When the longer turn-check silence arrives, TD Complete finalizes the same TurnDraft.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        TurnProgressDecision::Complete,
        "ここで終わり",
    );
    let final_output = flow
        .turn_check_silence_reached(1, TurnProgressDecision::Complete)
        .expect("TD Complete should finalize the open interim turn");

    assert!(!interim.is_final);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "ここで終わり。");
    assert_eq!(final_output.turn_id, 1);
    assert_eq!(final_output.segment_id, 1);
    assert_eq!(flow.outputs.len(), 2);
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_continue_decision_then_timeout_finalizes_same_turn() {
    let mut flow = MockTurnFlow::new();

    // App behavior:
    // 1. SegmentClosed reaches ASR after turn_check_silence_ms.
    // 2. Namo says "Continue", so ASR emits a non-final replacement output and keeps the turn open.
    // 3. No SegmentStarted/SegmentExtended activity arrives for the next speech.
    // 4. The worker timeout tick finalizes the same turn so the UI/Neo do not keep a draft forever.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Continue,
        "まだ続く",
    );
    let final_output = flow
        .tick_after_timeout()
        .expect("open turn without new activity should timeout");

    assert!(!interim.is_final);
    assert_eq!(interim.text, "まだ続く...");
    assert_eq!(interim.turn_id, 1);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "まだ続く。");
    assert_eq!(final_output.turn_id, 1);
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_continue_decision_then_speech_continues_same_turn() {
    let mut flow = MockTurnFlow::new();

    // App behavior:
    // - Namo Continue means the next SegmentClosed belongs to the still-open turn.
    // - SegmentBuilder may assign a new root segment id after EndSilenceReached; ASR still joins it
    //   to the open turn because open_turn_id is authoritative until finalization.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Continue,
        "これは",
    );
    let final_output = flow.segment_closed(
        2,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "続きです",
    );

    assert!(!interim.is_final);
    assert_eq!(flow.outputs[0].turn_id, 1);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "これは続きです。");
    assert_eq!(final_output.turn_id, 1);
    assert_eq!(final_output.segment_id, 2);
    assert_eq!(final_output.previous_segment_id, Some(1));
    assert!(flow.open_turn_id.is_none());
}

#[test]
fn td_flow_continue_decision_then_active_speech_does_not_timeout_before_next_segment() {
    let mut flow = MockTurnFlow::new();

    // App behavior:
    // 1. The first SegmentClosed has enough silence to ask Namo for a turn decision.
    // 2. Namo says "Continue", so ASR emits interim text and starts a timeout window.
    // 3. Before the next SegmentClosed is ready, SegmentStarted/SegmentExtended activity arrives
    //    through the atomic activity epoch. These are not ASR jobs and must not block behind
    //    bounded SegmentClosed jobs.
    // 4. A worker timeout tick after the old window would have elapsed must refresh the timeout
    //    origin instead of finalizing.
    // 5. The following SegmentClosed is appended to the same turn and emits the only final output.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Continue,
        "これは",
    );
    flow.segment_started_for_next_speech();
    flow.segment_extended_for_next_speech();
    let timeout_output = flow.tick_after_timeout();
    let final_output = flow.segment_closed(
        2,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "長い続きです",
    );

    assert!(!interim.is_final);
    assert!(
        timeout_output.is_none(),
        "active speech should refresh the open-turn timeout origin"
    );
    assert_eq!(
        flow.outputs.len(),
        2,
        "active speech must not emit timeout final"
    );
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "これは長い続きです。");
    assert_eq!(final_output.turn_id, 1);
}

#[test]
fn td_flow_continue_decision_without_activity_times_out_before_next_segment() {
    let mut flow = MockTurnFlow::new();

    // This is the negative control for the active-speech scenario above.
    // If no SegmentStarted/SegmentExtended activity happens after Namo Continue, the timeout tick
    // must finalize the draft. The next SegmentClosed then starts a new turn, yielding the
    // user-visible failure shape we want to avoid during active speech:
    // interim -> timeout final -> next segment final.
    let interim = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Continue,
        "これは",
    );
    let timeout_final = flow
        .tick_after_timeout()
        .expect("open turn without activity should final before the next segment");
    let next_final = flow.segment_closed(
        2,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "別の発話です",
    );

    assert!(!interim.is_final);
    assert!(timeout_final.is_final);
    assert!(next_final.is_final);
    assert_eq!(flow.outputs.len(), 3);
    assert_eq!(timeout_final.text, "これは。");
    assert_eq!(next_final.text, "別の発話です。");
    assert_eq!(timeout_final.turn_id, 1);
    assert_eq!(next_final.turn_id, 2);
}

#[test]
fn td_flow_complete_decision_then_next_speech_starts_new_turn() {
    let mut flow = MockTurnFlow::new();

    let first = flow.segment_closed(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "ここで完了",
    );
    let second = flow.segment_closed(
        2,
        None,
        SegmentCloseReason::EndSilenceReached,
        TurnProgressDecision::Complete,
        "次の発話",
    );

    assert!(first.is_final);
    assert!(second.is_final);
    assert_eq!(first.turn_id, 1);
    assert_eq!(second.turn_id, 2);
    assert_eq!(second.previous_segment_id, None);
    assert_eq!(first.text, "ここで完了。");
    assert_eq!(second.text, "次の発話。");
}

#[test]
fn post_segment_action_keeps_interim_result_independent_of_td_complete() {
    assert_eq!(
        post_segment_action(
            SegmentCloseReason::InterimResultSilenceReached,
            TurnProgressDecision::Complete
        ),
        PostSegmentAction::KeepOpenInterim
    );
    assert_eq!(
        post_segment_action(
            SegmentCloseReason::EndSilenceReached,
            TurnProgressDecision::Continue
        ),
        PostSegmentAction::KeepOpenInterim
    );
    assert_eq!(
        post_segment_action(
            SegmentCloseReason::EndSilenceReached,
            TurnProgressDecision::Complete
        ),
        PostSegmentAction::CloseFinal
    );
}

#[test]
fn open_turn_activity_epoch_refreshes_timeout_origin_only_when_activity_changed() {
    let mut open_since = Some(Instant::now());
    let mut seen_activity_epoch = 0;
    let next_activity_at = Instant::now();

    assert!(refresh_open_turn_timeout_origin_if_activity_changed(
        Some(1),
        &mut open_since,
        &mut seen_activity_epoch,
        1,
        next_activity_at
    ));
    assert_eq!(open_since, Some(next_activity_at));
    assert_eq!(seen_activity_epoch, 1);

    let duplicate_activity_at = Instant::now();
    assert!(!refresh_open_turn_timeout_origin_if_activity_changed(
        Some(1),
        &mut open_since,
        &mut seen_activity_epoch,
        1,
        duplicate_activity_at
    ));
    assert_eq!(open_since, Some(next_activity_at));

    assert!(!refresh_open_turn_timeout_origin_if_activity_changed(
        None,
        &mut open_since,
        &mut seen_activity_epoch,
        2,
        Instant::now()
    ));
    assert_eq!(open_since, Some(next_activity_at));
    assert_eq!(seen_activity_epoch, 1);
}

// Spec-only model for Turn behavior. It bypasses AsrWorker, Tauri event emission, and the
// RecognitionPipeline dispatcher, so high-risk timeout/activity regressions must also have a
// worker_runtime or pipeline test.
struct MockTurnFlow {
    turns: HashMap<u64, Turn>,
    open_turn_id: Option<u64>,
    open_turn_since: Option<Instant>,
    open_turn_activity_epoch: u64,
    segment_activity_epoch: u64,
    now: Instant,
    open_turn_timeout: Duration,
    next_output_sequence: u64,
    outputs: Vec<OutputSnapshot>,
}

impl MockTurnFlow {
    fn new() -> Self {
        Self {
            turns: HashMap::new(),
            open_turn_id: None,
            open_turn_since: None,
            open_turn_activity_epoch: 0,
            segment_activity_epoch: 0,
            now: Instant::now(),
            open_turn_timeout: Duration::from_millis(640),
            next_output_sequence: 1,
            outputs: Vec::new(),
        }
    }

    fn segment_closed(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        reason: SegmentCloseReason,
        td_decision: TurnProgressDecision,
        text: &str,
    ) -> OutputSnapshot {
        let turn_id = self.open_turn_id.unwrap_or(segment_id);
        let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
        let turn = self
            .turns
            .entry(turn_id)
            .or_insert_with(|| Turn::new(turn_event_id(1, turn_id, 0), 0));
        let sample = f32::from(u16::try_from(segment_id).expect("test segment id fits in u16"));
        turn.draft_mut().append_recognized_segment(
            segment_id,
            previous_segment_id,
            &[sample],
            route,
            text.to_string(),
            1,
        );

        match post_segment_action(reason, td_decision) {
            PostSegmentAction::KeepOpenInterim => {
                self.open_turn_id = Some(turn_id);
                self.open_turn_since = Some(self.now);
                self.open_turn_activity_epoch = self.segment_activity_epoch;
                self.emit(turn_id, false)
            }
            PostSegmentAction::CloseFinal => {
                self.open_turn_id = None;
                self.open_turn_since = None;
                self.open_turn_activity_epoch = self.segment_activity_epoch;
                self.emit(turn_id, true)
            }
        }
    }

    fn segment_started_for_next_speech(&mut self) {
        self.record_segment_activity();
    }

    fn segment_extended_for_next_speech(&mut self) {
        self.record_segment_activity();
    }

    fn turn_check_silence_reached(
        &mut self,
        previous_segment_id: u64,
        td_decision: TurnProgressDecision,
    ) -> Option<OutputSnapshot> {
        let turn_id = self.open_turn_id?;
        let turn = self
            .turns
            .get(&turn_id)
            .expect("turn-check silence requires an open turn");
        if turn.draft().latest_segment_id != Some(previous_segment_id) {
            return None;
        }

        match post_segment_action(SegmentCloseReason::EndSilenceReached, td_decision) {
            PostSegmentAction::KeepOpenInterim => {
                self.open_turn_since = Some(self.now);
                self.open_turn_activity_epoch = self.segment_activity_epoch;
                None
            }
            PostSegmentAction::CloseFinal => {
                self.open_turn_id = None;
                self.open_turn_since = None;
                self.open_turn_activity_epoch = self.segment_activity_epoch;
                Some(self.emit(turn_id, true))
            }
        }
    }

    fn record_segment_activity(&mut self) {
        self.segment_activity_epoch = self.segment_activity_epoch.saturating_add(1);
    }

    fn tick_after_timeout(&mut self) -> Option<OutputSnapshot> {
        self.tick_after(self.open_turn_timeout + Duration::from_millis(1))
    }

    fn tick_after(&mut self, duration: Duration) -> Option<OutputSnapshot> {
        self.now += duration;
        let turn_id = self.open_turn_id?;
        if refresh_open_turn_timeout_origin_if_activity_changed(
            Some(turn_id),
            &mut self.open_turn_since,
            &mut self.open_turn_activity_epoch,
            self.segment_activity_epoch,
            self.now,
        ) {
            return None;
        }
        let open_since = self.open_turn_since?;
        if self.now.duration_since(open_since) < self.open_turn_timeout {
            return None;
        }
        Some(self.timeout_open_turn())
    }

    fn timeout_open_turn(&mut self) -> OutputSnapshot {
        let turn_id = self
            .open_turn_id
            .expect("timeout requires an open turn created by TD continue");
        self.open_turn_id = None;
        self.open_turn_since = None;
        self.open_turn_activity_epoch = self.segment_activity_epoch;
        self.emit(turn_id, true)
    }

    fn emit(&mut self, turn_id: u64, is_final: bool) -> OutputSnapshot {
        let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
        let output_sequence = take_next_output_sequence(&mut self.next_output_sequence);
        let output = if is_final {
            let turn = self
                .turns
                .remove(&turn_id)
                .expect("final output requires an open turn");
            turn.into_draft()
                .confirm(1, turn_id, output_sequence, route)
                .expect("final output requires recognized text")
                .into_output()
        } else {
            self.turns
                .get(&turn_id)
                .expect("interim output requires an open turn")
                .draft()
                .interim_output(1, turn_id, output_sequence, route)
                .expect("interim output requires recognized text")
        };
        let snapshot = OutputSnapshot::from(&output);
        self.outputs.push(snapshot.clone());
        snapshot
    }
}

#[derive(Clone, Debug)]
struct OutputSnapshot {
    text: String,
    is_final: bool,
    turn_id: u64,
    segment_id: u64,
    previous_segment_id: Option<u64>,
}

impl OutputSnapshot {
    fn from(output: &RecognizedTextOutput) -> Self {
        let source = output.meta.source();
        Self {
            text: output.text.clone(),
            is_final: output.meta.is_final(),
            turn_id: source.turn_id,
            segment_id: source.segment_id,
            previous_segment_id: source.previous_segment_id,
        }
    }
}
