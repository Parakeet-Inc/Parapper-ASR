use crate::recognition::{
    control::{RecognitionSession, RerecognitionPurpose},
    segmentation::segment::builder::SegmentCloseReason,
    segmentation::vad::engine::VadResult,
    transcription::{
        asr::task::{
            AsrInFlight, AsrRequest, AsrRequestId, AsrResult, AsrTaskKind, AudioRange,
            GlobalSampleIndex, TurnId, VadFrameIndex,
        },
        planner::{
            PendingAsrSegment, drop_front_interim_segments_covered_by_completion,
            take_next_request_segment_plan,
        },
        reducer::{
            AsrRequestStaleInput, AsrResultAction, AsrResultCompletionAfterTranscript,
            AsrResultCompletionFailureAction, AsrResultReductionInput,
            AsrResultRerecognitionPurpose, reduce_asr_result,
        },
        route::{
            RecognitionRoute, RecognitionRouteSelection,
            language_id::LanguageDetector,
            selection::{AsrInput, TurnInput, refresh_turn, select_asr},
        },
    },
};

impl RecognitionSession {
    #[expect(
        clippy::too_many_arguments,
        reason = "closed segment handling keeps ASR request audio and continuous turn-source audio separate"
    )]
    pub(in crate::recognition) fn record_segment_closed_asr_candidate(
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
        self.pending.asr_segments.push_back(PendingAsrSegment {
            segment_id,
            previous_segment_id,
            audio: full_audio,
            vad_results,
            source_audio,
            source_vad_results,
            reason,
            range: AudioRange::new(start_sample, end_sample),
            created_at_frame: VadFrameIndex(self.counters.next_vad_frame_index),
        });
    }

    pub(in crate::recognition) fn take_next_request_id(&mut self) -> u64 {
        let request_id = self.counters.next_request_id;
        self.counters.next_request_id = self.counters.next_request_id.saturating_add(1);
        request_id
    }

    pub(in crate::recognition) fn dispatch_next_asr_request_if_idle(&mut self) {
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
        loop {
            let plan =
                take_next_request_segment_plan(&self.config, &mut self.pending.asr_segments)?;
            let range = plan.range();
            if range.end_sample <= self.turn_store.confirmed_until_sample {
                log::warn!(
                    "Dropping pending ASR segment plan already covered by confirmed audio: range={:?} confirmed_until={:?}",
                    range,
                    self.turn_store.confirmed_until_sample,
                );
                continue;
            }
            let target_turn_id = plan.target_turn_id(&self.config, self.turn_store.open_turn_id);
            let source_audio = plan.source_audio();
            let route_selection = self.route_selection_for_asr_request(
                target_turn_id,
                plan.kind,
                plan.first_reason(),
                source_audio.as_slice(),
            );
            let revision = *self.turn_store.revisions.get(&target_turn_id).unwrap_or(&0);
            let request_id = AsrRequestId(self.take_next_request_id());
            return Some(plan.into_request(
                &self.config,
                request_id,
                target_turn_id,
                revision,
                RecognitionRouteSelection {
                    route: route_selection.route,
                    detected_language: route_selection.detected_language,
                },
            ));
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
        select_asr(
            AsrInput {
                config: &self.config,
                warning_sink: self
                    .io
                    .language_id_runtime
                    .as_deref()
                    .map(|runtime| runtime as &dyn crate::recognition::transcription::route::language_id::LanguageDetectionWarningSink),
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

    pub(in crate::recognition) fn refresh_turn_route_with_sli(&mut self, turn_id: u64) {
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
        let Some(selection) = refresh_turn(
            TurnInput {
                config: &self.config,
                warning_sink: self
                    .io
                    .language_id_runtime
                    .as_deref()
                    .map(|runtime| runtime as &dyn crate::recognition::transcription::route::language_id::LanguageDetectionWarningSink),
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

    pub(in crate::recognition) fn apply_completed_asr_result_if_ready(&mut self) -> bool {
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
        reduce_asr_result(
            result,
            request,
            AsrResultReductionInput {
                stale_input: self.stale_input_for_request(request),
                interim_is_covered_by_completion: request.kind == AsrTaskKind::InterimDisplay
                    && self.interim_result_is_already_covered_by_completion(
                        request.target.turn_id.0,
                        request
                            .target
                            .last_segment_id
                            .map(|segment_id| segment_id.0),
                        request.target.range,
                    ),
                completion_has_non_empty_draft: request.kind == AsrTaskKind::CompletionCheck
                    && self.turn_has_non_empty_draft(request.target.turn_id.0),
                completion_failure_action: self.completion_failure_action_for_request(),
                completion_rerecognition_purpose: self
                    .rerecognition_purpose_after_completion()
                    .map(result_purpose_from_runtime),
                pending_rerecognition_purpose: self
                    .requests
                    .pending_rerecognition_purpose
                    .map(result_purpose_from_runtime),
            },
        )
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
                emit_interim,
            } => {
                let turn_id = self.apply_segment_transcript(request, transcript, elapsed_millis);
                if emit_interim {
                    self.emit_turn_output(turn_id, false);
                }
                if self
                    .turn_store
                    .open_turn_id
                    .is_none_or(|open_turn_id| open_turn_id <= turn_id)
                {
                    self.turn_store.open_turn_id = Some(turn_id);
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

    fn interim_result_is_already_covered_by_completion(
        &self,
        turn_id: u64,
        last_segment_id: Option<u64>,
        range: AudioRange,
    ) -> bool {
        if last_segment_id.is_some_and(|segment_id| {
            self.pending.turn_check.is_some_and(|turn_check| {
                turn_check.previous_segment_id == segment_id
                    && turn_check.activity_epoch == self.activity.segment_activity_epoch
            })
        }) {
            return true;
        }

        self.pending.asr_segments.iter().any(|segment| {
            segment.kind() == AsrTaskKind::CompletionCheck
                && segment.turn_id() == TurnId(turn_id)
                && segment.range.contains(range)
        })
    }

    fn stale_input_for_request(&self, request: &AsrRequest) -> AsrRequestStaleInput {
        AsrRequestStaleInput {
            current_revision: *self
                .turn_store
                .revisions
                .get(&request.target.turn_id.0)
                .unwrap_or(&0),
            confirmed_until_sample: self.turn_store.confirmed_until_sample,
            turn_route: self
                .turn_store
                .turns
                .get(&request.target.turn_id.0)
                .and_then(|turn| turn.draft().route),
            last_recognition_route: self.turn_store.last_recognition_route,
            default_route: RecognitionRoute::from_model(self.config.asr.model),
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
            crate::config::TurnDetector::Namo => AsrResultCompletionFailureAction::DecideWithNamo,
            crate::config::TurnDetector::Morph => AsrResultCompletionFailureAction::KeepOpen,
            crate::config::TurnDetector::Simple => {
                AsrResultCompletionFailureAction::CompleteWithoutGrammar
            }
        }
    }
}

#[cfg(test)]
fn is_stale_asr_request_for_runtime(runtime: &RecognitionSession, request: &AsrRequest) -> bool {
    crate::recognition::transcription::reducer::is_stale_asr_request(
        request,
        runtime.stale_input_for_request(request),
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
        config::{AsrLanguage, AsrModel, ParapperConfig, TurnDetector},
        recognition::{
            transcription::asr::task::{AsrTarget, SegmentId, TurnRevision},
            turn::Turn,
        },
    };

    #[test]
    fn dispatch_next_asr_request_if_idle_leaves_empty_queue_without_test_only_side_effects() {
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());

        runtime.dispatch_next_asr_request_if_idle();

        assert!(runtime.requests.in_flight_request.is_none());
        assert!(runtime.requests.last_dispatched.is_none());
        assert!(runtime.pending.asr_segments.is_empty());
    }

    #[test]
    fn record_segment_closed_asr_candidate_ignores_empty_audio_without_panic() {
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());

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
            ..ParapperConfig::default()
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
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
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
                ..ParapperConfig::default()
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
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
        runtime.turn_store.revisions.insert(1, 1);

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::InterimDisplay,
                RecognitionRoute::from_model(ParapperConfig::default().asr.model),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn stale_asr_request_detects_audio_range_already_confirmed() {
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
        runtime.turn_store.confirmed_until_sample = GlobalSampleIndex(10);

        assert!(is_stale_asr_request_for_runtime(
            &runtime,
            &asr_request(
                AsrTaskKind::InterimDisplay,
                RecognitionRoute::from_model(ParapperConfig::default().asr.model),
                None,
                0..10,
            )
        ));
    }

    #[test]
    fn dispatch_next_asr_request_if_idle_drops_confirmed_pending_segment_before_asr_submit() {
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
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
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
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
        let runtime = RecognitionSession::new(&ParapperConfig::default());

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
        let runtime = RecognitionSession::new(&ParapperConfig::default());

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
        let mut runtime = RecognitionSession::new(&ParapperConfig::default());
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
