use crate::{
    RecognitionSession, SegmentCloseReason,
    ports::LanguageDetector,
    transcription::{
        planner::{
            PendingAsrSegment, drop_front_interim_segments_covered_by_completion,
            take_next_request_segment_plan,
        },
        reducer::{
            AsrRequestStaleInput, AsrResultAction, AsrResultCompletionAfterTranscript,
            AsrResultCompletionFailureAction, AsrResultReductionInput,
            AsrResultRerecognitionPurpose, reduce_asr_result_for_resolved_turn,
        },
        route::{
            AsrRouteInput, RecognitionRoute, RecognitionRouteSelection, TurnRouteInput,
            refresh_turn_route, select_asr_route,
        },
        task::{
            AsrInFlight, AsrRequest, AsrRequestId, AsrResult, AsrTaskKind, AudioRange,
            GlobalSampleIndex, VadFrameIndex,
        },
    },
    turn::RerecognitionPurpose,
};

use parapper_models::vad::VadResult;

impl RecognitionSession {
    #[expect(
        clippy::too_many_arguments,
        reason = "closed segment handling keeps ASR request audio and continuous turn-source audio separate"
    )]
    pub fn record_segment_closed_asr_candidate(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        vad_results: Vec<VadResult>,
        source_audio: Vec<f32>,
        source_vad_results: Vec<VadResult>,
        reason: SegmentCloseReason,
    ) {
        if full_audio.is_empty() {
            log::warn!("Ignoring empty ASR segment: segment_id={segment_id}");
            return;
        }
        let audio_len = full_audio.len() as u64;
        let end_sample = GlobalSampleIndex(self.counters.global_sample_cursor);
        let start_sample =
            GlobalSampleIndex(self.counters.global_sample_cursor.saturating_sub(audio_len));
        let segment = PendingAsrSegment {
            segment_id,
            previous_segment_id,
            audio: full_audio,
            vad_results,
            source_audio,
            source_vad_results,
            reason,
            range: AudioRange::new(start_sample, end_sample),
            created_at_frame: VadFrameIndex(self.counters.next_vad_frame_index),
        };
        if reason == SegmentCloseReason::InterimResultSilenceReached {
            let streaming_interim_enabled = self.streaming_interim_asr_enabled();
            if let Some(segment) = self
                .pending
                .interim_asr
                .interim_request(streaming_interim_enabled, segment)
            {
                self.pending.asr_segments.push_back(segment);
            }
        } else {
            self.pending.asr_segments.push_back(segment);
        }
    }

    pub fn record_interim_segment_started(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        audio_so_far: Vec<f32>,
        vad_results: Vec<VadResult>,
    ) {
        if !self.streaming_interim_asr_enabled() {
            self.pending.interim_asr.clear_streaming();
            return;
        }
        let ready = self.pending.interim_asr.start_streaming_segment(
            segment_id,
            previous_segment_id,
            audio_so_far,
            vad_results,
            GlobalSampleIndex(self.counters.global_sample_cursor),
            VadFrameIndex(self.counters.next_vad_frame_index),
        );
        self.pending.asr_segments.extend(ready);
    }

    pub fn record_interim_segment_extended(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        new_audio: Vec<f32>,
        vad_result: VadResult,
    ) {
        if !self.streaming_interim_asr_enabled() {
            self.pending.interim_asr.clear_streaming();
            return;
        }
        let ready = self.pending.interim_asr.extend_streaming_segment(
            segment_id,
            previous_segment_id,
            new_audio,
            vad_result,
            GlobalSampleIndex(self.counters.global_sample_cursor),
            VadFrameIndex(self.counters.next_vad_frame_index),
        );
        self.pending.asr_segments.extend(ready);
    }

    pub fn reset_interim_streaming_for_completion(&mut self, segment_id: u64) {
        let display_segment_id = self
            .pending
            .interim_asr
            .clear_streaming_if_segment(segment_id)
            .unwrap_or(segment_id);
        self.pending.asr_segments.retain(|segment| {
            segment.reason != SegmentCloseReason::InterimChunkReached
                || (segment.segment_id != display_segment_id && segment.segment_id != segment_id)
        });
        let source_session = self.source_session().clone();
        if source_session.source_id.as_str() == crate::SourceId::LEGACY_SINGLE_SOURCE {
            self.io.asr_runner.reset_streaming_sessions();
        } else {
            self.io
                .asr_runner
                .reset_streaming_sessions_for_source(&source_session);
        }
    }

    fn streaming_interim_asr_enabled(&self) -> bool {
        self.config.turn.interim_result_enabled
            && self
                .config
                .asr
                .interim_model
                .unwrap_or(self.config.asr.model)
                .is_nemotron()
    }

    pub fn take_next_request_id(&mut self) -> u64 {
        let request_id = self.counters.next_request_id;
        self.counters.next_request_id = self.counters.next_request_id.saturating_add(1);
        request_id
    }

    pub fn dispatch_next_asr_request_if_idle(&mut self) {
        if self.requests.in_flight_request.is_some() {
            return;
        }
        drop_front_interim_segments_covered_by_completion(&mut self.pending.asr_segments);
        let Some(request) = self.build_next_asr_request() else {
            return;
        };
        let in_flight = AsrInFlight::from(&request);
        if !self.io.asr_runner.submit(request.clone()) {
            log::warn!(
                "Dropping ASR request after submit failure: request_id={:?} kind={:?}",
                request.request_id,
                request.kind,
            );
            return;
        }
        self.requests.in_flight_request = Some(request);
        self.requests.last_dispatched = Some(in_flight);
    }

    fn build_next_asr_request(&mut self) -> Option<AsrRequest> {
        let planner_config = self.config.planner_config();
        loop {
            let plan =
                take_next_request_segment_plan(&planner_config, &mut self.pending.asr_segments)?;
            let range = plan.range();
            if range.end_sample <= self.turn_store.confirmed_until_sample {
                log::warn!(
                    "Dropping pending ASR segment plan already covered by confirmed audio: range={:?} confirmed_until={:?}",
                    range,
                    self.turn_store.confirmed_until_sample,
                );
                continue;
            }
            // Keep the segment-derived projection for legacy request
            // observers. It is not an allocation: `turn_target` remains New
            // until a usable transcript is applied.
            let projected_turn_id = plan.target_turn_id(
                &planner_config,
                self.turn_store.open_turn_id,
                self.turn_store.open_turn_accepts_root_segment,
            );
            let mut turn_target = plan.target_for_request(
                &planner_config,
                self.turn_store.open_turn_id,
                self.turn_store.open_turn_accepts_root_segment,
            );
            let target_turn_id = projected_turn_id;
            if matches!(turn_target, crate::transcription::task::AsrTurnTarget::New)
                && self.turn_store.turns.contains_key(&projected_turn_id)
            {
                turn_target = crate::transcription::task::AsrTurnTarget::Existing(crate::TurnId(
                    projected_turn_id,
                ));
            }
            let route_turn_id = match turn_target {
                crate::transcription::task::AsrTurnTarget::New => 0,
                crate::transcription::task::AsrTurnTarget::Existing(turn_id) => turn_id.0,
            };
            if self.turn_store.finalized_turns.contains(&target_turn_id) {
                log::warn!(
                    "Dropping pending ASR segment plan for finalized turn: turn_id={target_turn_id} range={range:?}",
                );
                continue;
            }
            let source_audio = plan.source_audio();
            let route_selection = self.route_selection_for_asr_request(
                route_turn_id,
                plan.kind,
                plan.first_reason(),
                source_audio.as_slice(),
            );
            let revision = *self.turn_store.revisions.get(&route_turn_id).unwrap_or(&0);
            let request_id = AsrRequestId(self.take_next_request_id());
            let mut request = plan.into_request(
                &planner_config,
                request_id,
                target_turn_id,
                revision,
                RecognitionRouteSelection {
                    route: route_selection.route,
                    detected_language: route_selection.detected_language,
                },
            );
            request.target.turn_target = turn_target;
            request.target.turn_id = crate::TurnId(target_turn_id);
            request
                .target
                .set_source_session(self.source_session().clone());
            return Some(request);
        }
    }

    fn route_selection_for_asr_request(
        &mut self,
        turn_id: u64,
        kind: AsrTaskKind,
        close_reason: SegmentCloseReason,
        request_audio: &[f32],
    ) -> RecognitionRouteSelection {
        let current_route = self.route_hint_for_request(turn_id);
        let draft_audio = self
            .turn_store
            .turns
            .get(&turn_id)
            .map(|turn| turn.draft().full_audio.as_slice());
        let language_id = self
            .io
            .language_id
            .as_mut()
            .map(|detector| detector.as_mut() as &mut dyn LanguageDetector);
        select_asr_route(
            AsrRouteInput {
                config: &self.config,
                warning_sink: self.io.language_warning_sink.as_deref(),
                kind,
                close_reason,
                current_route,
                fallback_route: RecognitionRoute::from_model(self.config.asr.model),
                draft_audio,
                request_audio,
            },
            language_id,
        )
    }

    fn route_hint_for_request(&self, turn_id: u64) -> Option<RecognitionRoute> {
        self.turn_store
            .turns
            .get(&turn_id)
            .and_then(|turn| turn.draft().route)
            .or(self.turn_store.last_recognition_route)
    }

    pub fn refresh_turn_route_with_sli(&mut self, turn_id: u64) {
        let Some((draft_route, full_audio)) = self
            .turn_store
            .turns
            .get(&turn_id)
            .map(|turn| (turn.draft().route, turn.draft().full_audio.clone()))
        else {
            return;
        };
        let language_id = self
            .io
            .language_id
            .as_mut()
            .map(|detector| detector.as_mut() as &mut dyn LanguageDetector);
        let Some(selection) = refresh_turn_route(
            TurnRouteInput {
                config: &self.config,
                warning_sink: self.io.language_warning_sink.as_deref(),
                current_route: draft_route.or(self.turn_store.last_recognition_route),
                full_audio: &full_audio,
            },
            language_id,
        ) else {
            return;
        };

        if let Some(turn) = self.turn_store.turns.get_mut(&turn_id) {
            let draft = turn.draft_mut();
            draft.route = Some(selection.route);
            draft.set_detected_language(selection.detected_language);
        }
    }

    pub fn apply_completed_asr_result_if_ready(&mut self) -> bool {
        let Some(result) = self.io.asr_runner.try_recv_result() else {
            return false;
        };
        let Some(request) = self.requests.in_flight_request.take() else {
            log::warn!(
                "Dropping ASR result without an in-flight request: request_id={:?} kind={:?}",
                result.request_id,
                result.kind,
            );
            return true;
        };
        let action = self.reduce_asr_result_for_runtime(&result, &request);
        if matches!(
            action,
            AsrResultAction::KeepInFlightForMismatchedResult { .. }
        ) {
            self.apply_asr_result_action(&request, action);
            self.requests.in_flight_request = Some(request);
            return true;
        }
        self.apply_asr_result_action(&request, action);
        true
    }

    fn reduce_asr_result_for_runtime(
        &self,
        result: &AsrResult,
        request: &AsrRequest,
    ) -> AsrResultAction {
        let resolved_turn_id = self.resolved_new_turn_id(request);
        reduce_asr_result_for_resolved_turn(
            result,
            request,
            AsrResultReductionInput {
                stale_input: self.stale_input_for_request(request, resolved_turn_id),
                completion_has_non_empty_draft: request.kind == AsrTaskKind::CompletionCheck
                    && self.turn_has_non_empty_draft(
                        resolved_turn_id.unwrap_or(request.target.turn_id.0),
                    ),
                completion_failure_action: self.completion_failure_action_for_request(),
                completion_rerecognition_purpose: self
                    .rerecognition_purpose_after_completion()
                    .map(result_purpose_from_runtime),
                pending_rerecognition_purpose: self
                    .requests
                    .pending_rerecognition_purpose
                    .map(result_purpose_from_runtime),
            },
            resolved_turn_id,
        )
    }

    fn resolved_new_turn_id(&self, request: &AsrRequest) -> Option<u64> {
        if !matches!(
            request.target.turn_target,
            crate::transcription::task::AsrTurnTarget::New
        ) {
            return None;
        }
        self.turn_store
            .pending_new_turns
            .get(&(
                request.target.source_session.clone(),
                request.target.first_segment_id,
            ))
            .copied()
    }

    fn apply_asr_result_action(&mut self, request: &AsrRequest, action: AsrResultAction) {
        match action {
            AsrResultAction::KeepInFlightForMismatchedResult {
                result_request_id,
                in_flight_request_id,
            } => {
                log::warn!(
                    "Ignoring ASR result that does not match the current in-flight request: result_id={result_request_id:?} in_flight_id={in_flight_request_id:?}",
                );
            }
            AsrResultAction::DropStaleResult
            | AsrResultAction::DropUnusableInterim
            | AsrResultAction::DropUnusableCompletionWithoutDraft => {}
            AsrResultAction::FallbackCompletionWithNamo { turn_id } => {
                self.complete_or_continue_turn_with_namo(turn_id);
            }
            AsrResultAction::FallbackCompletionWithoutGrammar { turn_id } => {
                self.complete_turn_without_grammar(turn_id);
            }
            AsrResultAction::FallbackCompletionKeepOpen { turn_id } => {
                self.keep_turn_open(turn_id, true);
            }
            AsrResultAction::FallbackRerecognition { turn_id, purpose } => {
                self.requests.pending_rerecognition_purpose.take();
                self.apply_rerecognition_follow_up(turn_id, purpose);
            }
            AsrResultAction::ApplyInterimTranscript {
                transcript,
                elapsed_millis,
            } => {
                let turn_id = self.apply_segment_transcript(request, transcript, elapsed_millis);
                self.emit_turn_output(turn_id, false);
                let previous_open_turn_id = self.turn_store.open_turn_id;
                if self
                    .turn_store
                    .open_turn_id
                    .is_none_or(|open_turn_id| open_turn_id <= turn_id)
                {
                    self.turn_store.open_turn_id = Some(turn_id);
                    if previous_open_turn_id != Some(turn_id) {
                        self.turn_store.open_turn_accepts_root_segment = false;
                    }
                }
            }
            AsrResultAction::ApplyCompletionTranscript {
                transcript,
                elapsed_millis,
                after_transcript,
            } => {
                let turn_id = self.apply_segment_transcript(request, transcript, elapsed_millis);
                match after_transcript {
                    AsrResultCompletionAfterTranscript::RerecognizeIfIdle(purpose) => {
                        if self.dispatch_rerecognition_for_turn_if_idle(
                            turn_id,
                            runtime_purpose_from_result(purpose),
                        ) {
                            return;
                        }
                    }
                    AsrResultCompletionAfterTranscript::CompleteWithoutGrammar => {}
                }
                self.complete_turn_without_grammar(turn_id);
            }
            AsrResultAction::ApplyRerecognitionTranscript {
                transcript,
                elapsed_millis,
                purpose,
            } => {
                self.requests.pending_rerecognition_purpose.take();
                self.apply_rerecognition_transcript(
                    request,
                    transcript,
                    elapsed_millis,
                    purpose == AsrResultRerecognitionPurpose::GrammarAfterCompletion,
                );
                self.apply_rerecognition_follow_up(request.target.turn_id.0, purpose);
            }
        }
    }

    fn stale_input_for_request(
        &self,
        request: &AsrRequest,
        resolved_turn_id: Option<u64>,
    ) -> AsrRequestStaleInput {
        let is_new = matches!(
            request.target.turn_target,
            crate::transcription::task::AsrTurnTarget::New
        );
        let effective_turn_id = resolved_turn_id.unwrap_or(request.target.turn_id.0);
        let is_unresolved_new = is_new && resolved_turn_id.is_none();
        AsrRequestStaleInput {
            current_revision: if is_unresolved_new {
                0
            } else {
                *self
                    .turn_store
                    .revisions
                    .get(&effective_turn_id)
                    .unwrap_or(&0)
            },
            confirmed_until_sample: self.turn_store.confirmed_until_sample,
            target_turn_is_finalized: !is_unresolved_new
                && self.turn_store.finalized_turns.contains(&effective_turn_id),
            turn_route: if is_unresolved_new {
                None
            } else {
                self.turn_store
                    .turns
                    .get(&effective_turn_id)
                    .and_then(|turn| turn.draft().route)
                    .filter(|route| {
                        request.kind == AsrTaskKind::InterimDisplay
                            || !route.model.is_interim_only()
                    })
            },
            last_recognition_route: self.turn_store.last_recognition_route,
            default_route: RecognitionRoute::from_model(self.config.asr.model),
            split_route: crate::transcription::route::configured_split_route(
                &self.config,
                request.kind,
            ),
        }
    }

    fn apply_rerecognition_follow_up(
        &mut self,
        turn_id: u64,
        purpose: AsrResultRerecognitionPurpose,
    ) {
        match purpose {
            AsrResultRerecognitionPurpose::GrammarAfterCompletion => {
                self.process_grammar_boundaries_after_rerecognition(turn_id);
            }
            AsrResultRerecognitionPurpose::SimpleTurnCheckFinal => {
                self.complete_turn_without_grammar(turn_id);
            }
            AsrResultRerecognitionPurpose::TimeoutFinal => {
                self.finalize_timeout_turn_after_rerecognition(turn_id);
            }
        }
    }

    fn turn_has_non_empty_draft(&self, turn_id: u64) -> bool {
        self.turn_store
            .turns
            .get(&turn_id)
            .is_some_and(|turn| !turn.draft().combined_text.trim().is_empty())
    }

    fn completion_failure_action_for_request(&self) -> AsrResultCompletionFailureAction {
        match self.config.turn.detector {
            crate::turn::TurnDetector::Namo => AsrResultCompletionFailureAction::DecideWithNamo,
            crate::turn::TurnDetector::Morph => AsrResultCompletionFailureAction::KeepOpen,
            crate::turn::TurnDetector::Simple => {
                AsrResultCompletionFailureAction::CompleteWithoutGrammar
            }
        }
    }
}

#[cfg(test)]
fn is_stale_asr_request_for_runtime(runtime: &RecognitionSession, request: &AsrRequest) -> bool {
    crate::transcription::reducer::is_stale_asr_request(
        request,
        runtime.stale_input_for_request(request, None),
    )
}

fn result_purpose_from_runtime(purpose: RerecognitionPurpose) -> AsrResultRerecognitionPurpose {
    match purpose {
        RerecognitionPurpose::GrammarAfterCompletion => {
            AsrResultRerecognitionPurpose::GrammarAfterCompletion
        }
        RerecognitionPurpose::SimpleTurnCheckFinal => {
            AsrResultRerecognitionPurpose::SimpleTurnCheckFinal
        }
        RerecognitionPurpose::TimeoutFinal => AsrResultRerecognitionPurpose::TimeoutFinal,
    }
}

fn runtime_purpose_from_result(purpose: AsrResultRerecognitionPurpose) -> RerecognitionPurpose {
    match purpose {
        AsrResultRerecognitionPurpose::GrammarAfterCompletion => {
            RerecognitionPurpose::GrammarAfterCompletion
        }
        AsrResultRerecognitionPurpose::SimpleTurnCheckFinal => {
            RerecognitionPurpose::SimpleTurnCheckFinal
        }
        AsrResultRerecognitionPurpose::TimeoutFinal => RerecognitionPurpose::TimeoutFinal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AsrLanguage, AsrModel, SegmentId, SttEngineConfig, TurnId,
        transcription::task::{AsrTarget, TurnRevision},
        turn::{Turn, TurnDetector},
    };

    macro_rules! parapper_config {
        (turn_detector: $turn_detector:expr, ..$base:expr) => {{
            let mut config = $base;
            config.turn.detector = $turn_detector;
            config
        }};
    }

    #[test]
    fn dispatch_next_asr_request_if_idle_leaves_empty_queue_without_test_only_side_effects() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());

        runtime.dispatch_next_asr_request_if_idle();

        assert!(runtime.requests.in_flight_request.is_none());
        assert!(runtime.requests.last_dispatched.is_none());
        assert!(runtime.pending.asr_segments.is_empty());
    }

    #[test]
    fn record_segment_closed_asr_candidate_ignores_empty_audio_without_panic() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());

        runtime.record_segment_closed_asr_candidate(
            1,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            SegmentCloseReason::EndSilenceReached,
        );

        assert!(
            runtime.pending.asr_segments.is_empty(),
            "empty closed segments must not create a zero-length ASR range"
        );
    }

    #[test]
    fn take_following_interim_segments_stops_at_non_contiguous_segment() {
        let mut runtime = RecognitionSession::new(&parapper_config! {
            turn_detector: TurnDetector::Namo,
            ..SttEngineConfig::default()
        });
        runtime.pending.asr_segments.push_back(pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            0..10,
        ));
        runtime.pending.asr_segments.push_back(pending_segment(
            2,
            Some(99),
            SegmentCloseReason::InterimResultSilenceReached,
            10..20,
        ));

        runtime.dispatch_next_asr_request_if_idle();

        let request = runtime
            .requests
            .in_flight_request
            .as_ref()
            .expect("first interim request should be dispatched");
        assert_eq!(request.target.first_segment_id, Some(SegmentId(1)));
        assert_eq!(request.target.last_segment_id, Some(SegmentId(1)));
        assert_eq!(
            request.source_audio,
            vec![1.0; 10],
            "request-level ASR padding must not alter the pending segment source audio"
        );
        assert_eq!(runtime.pending.asr_segments.len(), 1);
        assert_eq!(
            runtime
                .pending
                .asr_segments
                .front()
                .expect("non-contiguous segment should remain queued")
                .segment_id,
            2
        );
    }

    #[test]
    fn drop_front_interim_segments_covered_by_completion_promotes_covering_completion() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        runtime.pending.asr_segments.push_back(pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            0..10,
        ));
        runtime.pending.asr_segments.push_back(pending_segment(
            2,
            Some(1),
            SegmentCloseReason::InterimResultSilenceReached,
            10..20,
        ));
        runtime.pending.asr_segments.push_back(pending_segment(
            2,
            Some(1),
            SegmentCloseReason::EndSilenceReached,
            0..20,
        ));

        runtime.dispatch_next_asr_request_if_idle();

        let request = runtime
            .requests
            .in_flight_request
            .as_ref()
            .expect("covering completion should be dispatched first");
        assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(request.target.first_segment_id, Some(SegmentId(1)));
        assert_eq!(request.target.last_segment_id, Some(SegmentId(2)));
        assert_eq!(request.audio, vec![2.0; 20]);
        assert!(
            runtime.pending.asr_segments.is_empty(),
            "covered interim segments must be removed instead of replayed after completion"
        );
    }

    #[test]
    fn turn_detector_can_connect_interim_after_completion_controls_request_merging() {
        for (turn_detector, expected_audio_len, expected_remaining) in
            [(TurnDetector::Namo, 20, 0), (TurnDetector::Simple, 10, 1)]
        {
            let mut runtime = RecognitionSession::new(&parapper_config! {
                turn_detector: turn_detector,
                ..SttEngineConfig::default()
            });
            runtime.pending.asr_segments.push_back(pending_segment(
                1,
                None,
                SegmentCloseReason::EndSilenceReached,
                0..10,
            ));
            runtime.pending.asr_segments.push_back(pending_segment(
                2,
                Some(1),
                SegmentCloseReason::InterimResultSilenceReached,
                10..20,
            ));

            runtime.dispatch_next_asr_request_if_idle();

            let request = runtime
                .requests
                .in_flight_request
                .as_ref()
                .expect("completion request should be dispatched");
            assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
            assert_eq!(
                request.audio.len(),
                expected_audio_len,
                "turn_detector={turn_detector:?}"
            );
            assert_eq!(
                runtime.pending.asr_segments.len(),
                expected_remaining,
                "turn_detector={turn_detector:?}"
            );
        }
    }

    #[test]
    fn stale_asr_request_detects_turn_revision_change() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        runtime.turn_store.revisions.insert(1, 1);

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::InterimDisplay,
                RecognitionRoute::from_model(SttEngineConfig::default().asr.model),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn stale_asr_request_detects_audio_range_already_confirmed() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        runtime.turn_store.confirmed_until_sample = GlobalSampleIndex(10);

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::InterimDisplay,
                RecognitionRoute::from_model(SttEngineConfig::default().asr.model),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn dispatch_next_asr_request_if_idle_drops_confirmed_pending_segment_before_asr_submit() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        runtime.turn_store.confirmed_until_sample = GlobalSampleIndex(10);
        runtime.pending.asr_segments.push_back(pending_segment(
            1,
            None,
            SegmentCloseReason::EndSilenceReached,
            0..10,
        ));

        runtime.dispatch_next_asr_request_if_idle();

        assert!(
            runtime.requests.in_flight_request.is_none(),
            "a pending segment whose range is already confirmed must not consume an ASR cycle"
        );
        assert!(runtime.requests.last_dispatched.is_none());
        assert!(runtime.pending.asr_segments.is_empty());
    }

    #[test]
    fn stale_asr_request_detects_existing_turn_route_mismatch() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
        turn.draft_mut().append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            RecognitionRoute::from_language(AsrLanguage::English),
            "hello".to_string(),
            0,
        );
        runtime.turn_store.turns.insert(1, turn);

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::InterimDisplay,
                RecognitionRoute::from_language(AsrLanguage::Japanese),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn stale_asr_request_accepts_sli_selected_route_even_without_cached_last_route() {
        let runtime = RecognitionSession::new(&SttEngineConfig::default());

        assert!(!is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::CompletionCheck,
                RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
                Some("en".to_string()),
                0..10,
            )
        ));
    }

    #[test]
    fn stale_asr_request_rejects_non_default_route_without_sli_or_cached_last_route() {
        let runtime = RecognitionSession::new(&SttEngineConfig::default());

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::CompletionCheck,
                RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn stale_asr_request_accepts_cached_last_recognition_route() {
        let mut runtime = RecognitionSession::new(&SttEngineConfig::default());
        let route = RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8);
        runtime.turn_store.last_recognition_route = Some(route);

        assert!(!is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10,)
        ));
    }

    fn pending_segment(
        segment_id: u64,
        previous_segment_id: Option<u64>,
        reason: SegmentCloseReason,
        range: std::ops::Range<u64>,
    ) -> PendingAsrSegment {
        let sample_value =
            f32::from(u16::try_from(segment_id).expect("test segment id should fit u16"));
        let audio = vec![
            sample_value;
            usize::try_from(range.end - range.start)
                .expect("test range should fit usize")
        ];
        let vad_results = vec![vad(true)];
        PendingAsrSegment {
            segment_id,
            previous_segment_id,
            source_audio: audio.clone(),
            source_vad_results: vad_results.clone(),
            audio,
            vad_results,
            reason,
            range: AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
            created_at_frame: VadFrameIndex(segment_id),
        }
    }

    fn asr_request(
        kind: AsrTaskKind,
        route: RecognitionRoute,
        detected_language: Option<String>,
        range: std::ops::Range<u64>,
    ) -> AsrRequest {
        AsrRequest {
            request_id: AsrRequestId(1),
            kind,
            target: AsrTarget::new(
                TurnId(1),
                TurnRevision(0),
                AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
                Some(SegmentId(1)),
                Some(SegmentId(1)),
            ),
            route,
            detected_language,
            audio: vec![1.0; usize::try_from(range.end - range.start).unwrap()],
            vad_results: vec![vad(true)],
            source_audio: vec![1.0; usize::try_from(range.end - range.start).unwrap()],
            source_vad_results: vec![vad(true)],
            close_reason: Some(SegmentCloseReason::EndSilenceReached),
            created_at_frame: VadFrameIndex(1),
        }
    }

    fn vad(is_speech: bool) -> VadResult {
        VadResult {
            probability: if is_speech { 0.9 } else { 0.1 },
            is_speech,
        }
    }
}
