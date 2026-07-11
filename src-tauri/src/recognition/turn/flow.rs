use crate::recognition::{
    control::{PendingFinalization, RecognitionSession, RerecognitionPurpose},
    segmentation::segment::builder::SegmentCloseReason,
    transcription::{
        asr::{
            input::{AsrRequestEdgePadding, ensure_asr_request_edge_silence},
            task::{
                AsrInFlight, AsrRequest, AsrRequestId, AsrTarget, AsrTaskKind, AudioRange,
                GlobalSampleIndex, SegmentId, TurnId, TurnRevision, VadFrameIndex,
            },
        },
        route::{RecognitionRoute, selection::configured_split_route},
    },
    turn::{
        policy::{completion, namo, silence, timeout},
        take_next_output_sequence,
    },
};

impl RecognitionSession {
    pub(in crate::recognition) fn complete_turn_without_grammar(&mut self, turn_id: u64) {
        if self.defer_finalization_if_blocked(PendingFinalization::new(turn_id)) {
            return;
        }
        self.finalize_turn_now(turn_id);
    }

    fn finalize_turn_now(&mut self, turn_id: u64) {
        self.emit_stale_turn_finals(turn_id);
        self.emit_turn_output(turn_id, true);
        self.clear_open_turn();
    }

    pub(in crate::recognition) fn rerecognition_purpose_after_completion(
        &self,
    ) -> Option<RerecognitionPurpose> {
        completion::rerecognition_purpose(&self.config)
    }

    pub(in crate::recognition) fn dispatch_rerecognition_for_turn_if_idle(
        &mut self,
        turn_id: u64,
        purpose: RerecognitionPurpose,
    ) -> bool {
        if self.requests.in_flight_request.is_some() {
            return false;
        }
        let Some((
            draft_route,
            mut detected_language,
            mut audio,
            mut vad_results,
            first_segment_id,
            latest_segment_id,
        )) = self.turn_store.turns.get(&turn_id).and_then(|turn| {
            let draft = turn.draft();
            Some((
                draft.route?,
                draft.detected_language.clone(),
                draft.full_audio.clone(),
                draft.vad_results.clone(),
                draft.segment_ids.first().copied(),
                draft.latest_segment_id,
            ))
        })
        else {
            return false;
        };
        let route = configured_split_route(&self.config, AsrTaskKind::Rerecognition)
            .unwrap_or_else(|| {
                if draft_route.model.is_interim_only() {
                    RecognitionRoute::from_model(self.config.asr.model)
                } else {
                    draft_route
                }
            });
        if route != draft_route {
            detected_language = None;
        }
        if audio.is_empty() {
            return false;
        }
        let source_audio_len = audio.len();
        ensure_asr_request_edge_silence(
            &self.config,
            &mut audio,
            &mut vad_results,
            AsrRequestEdgePadding::TrailingOnly,
        );
        let range = self
            .turn_store
            .audio_ranges
            .get(&turn_id)
            .copied()
            .unwrap_or_else(|| {
                AudioRange::new(
                    GlobalSampleIndex(0),
                    GlobalSampleIndex(source_audio_len as u64),
                )
            });
        let target = AsrTarget::new(
            TurnId(turn_id),
            TurnRevision(*self.turn_store.revisions.get(&turn_id).unwrap_or(&0)),
            range,
            first_segment_id.map(SegmentId),
            latest_segment_id.map(SegmentId),
        );
        let request = AsrRequest {
            request_id: AsrRequestId(self.take_next_request_id()),
            kind: AsrTaskKind::Rerecognition,
            target,
            route,
            detected_language,
            audio,
            vad_results,
            source_audio: Vec::new(),
            source_vad_results: Vec::new(),
            close_reason: None,
            created_at_frame: VadFrameIndex(self.counters.next_vad_frame_index),
        };
        let in_flight = AsrInFlight::from(&request);
        if !self.io.asr_runner.submit(request.clone()) {
            log::warn!(
                "Dropping rerecognition ASR request after submit failure: request_id={:?} turn_id={turn_id}",
                request.request_id,
            );
            return false;
        }
        self.requests.in_flight_request = Some(request);
        self.requests.pending_rerecognition_purpose = Some(purpose);
        self.requests.last_dispatched = Some(in_flight);
        true
    }

    pub(in crate::recognition) fn complete_or_continue_turn_with_namo(&mut self, turn_id: u64) {
        match namo::action(self.namo_final_decision(turn_id)) {
            namo::Action::Complete => {
                self.complete_turn_without_grammar(turn_id);
            }
            namo::Action::Continue { emit_interim } => {
                self.keep_turn_open(turn_id, emit_interim);
            }
        }
    }

    fn namo_final_decision(&mut self, turn_id: u64) -> bool {
        let Some(text) = self
            .turn_store
            .turns
            .get(&turn_id)
            .map(|turn| turn.draft().combined_text.clone())
        else {
            return false;
        };
        self.namo_final_decision_for_text(turn_id, &text)
    }

    pub(in crate::recognition) fn namo_final_decision_for_text(
        &mut self,
        turn_id: u64,
        text: &str,
    ) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let Some(route) = self
            .turn_store
            .turns
            .get(&turn_id)
            .and_then(|turn| turn.draft().route)
        else {
            return false;
        };
        self.io
            .turn_decision_runner
            .decide(route, text, self.config.turn.namo_context_max_tokens)
            .is_ok_and(|decision| {
                decision.is_end_of_turn
                    && decision.confidence >= self.config.turn.namo_confidence_threshold
            })
    }

    pub(in crate::recognition) fn keep_turn_open(&mut self, turn_id: u64, emit_interim: bool) {
        self.turn_store.open_turn_id = Some(turn_id);
        self.turn_store.open_turn_accepts_root_segment = true;
        self.reset_open_turn_timeout_origin();
        if emit_interim && self.config.turn.interim_result_enabled {
            self.emit_turn_output(turn_id, false);
        }
    }

    pub(in crate::recognition) fn finalize_timeout_turn_after_rerecognition(
        &mut self,
        turn_id: u64,
    ) {
        if self.defer_finalization_if_blocked(PendingFinalization::new(turn_id)) {
            return;
        }
        self.finalize_turn_now(turn_id);
    }

    pub(in crate::recognition) fn clear_open_turn(&mut self) {
        self.turn_store.open_turn_id = None;
        self.turn_store.open_turn_accepts_root_segment = false;
        self.activity.open_turn_since_tick = None;
    }

    fn reset_open_turn_timeout_origin(&mut self) {
        self.activity.open_turn_activity_epoch = self.activity.segment_activity_epoch;
        self.activity.open_turn_since_tick = Some(self.counters.next_runtime_tick);
    }

    pub(in crate::recognition) fn handle_turn_check_silence_reached(
        &mut self,
        previous_segment_id: u64,
    ) -> bool {
        let open_turn_id = self.turn_store.open_turn_id;
        let turn = open_turn_id.and_then(|turn_id| self.turn_store.turns.get(&turn_id));
        let open_turn = match (open_turn_id, turn) {
            (None, _) => silence::OpenTurn::None,
            (Some(_), None) => silence::OpenTurn::Missing,
            (Some(turn_id), Some(turn)) => silence::OpenTurn::Present {
                turn_id,
                latest_segment_id: turn.draft().latest_segment_id,
            },
        };
        let action = silence::action(silence::Input {
            open_turn,
            previous_segment_id,
            asr_state: silence::asr_state(self.requests.in_flight_request.is_some()),
            pending_interim: silence::pending_interim(
                self.can_promote_pending_interim_to_completion(previous_segment_id),
            ),
            completion_strategy: silence::completion_strategy(
                self.config.turn.detector,
                self.config.turn.rerecognize_full_on_complete,
            ),
        });
        match action {
            silence::Action::WaitForBusyAsr => false,
            silence::Action::Ignore => true,
            silence::Action::PromotePendingInterim => {
                self.promote_pending_interim_to_completion_for_turn_check(previous_segment_id)
            }
            silence::Action::RefreshRouteThenDispatchRerecognition {
                turn_id,
                purpose,
                fallback_complete_without_grammar,
            } => {
                self.refresh_turn_route_with_sli(turn_id);
                if self.dispatch_rerecognition_for_turn_if_idle(turn_id, purpose) {
                    true
                } else if fallback_complete_without_grammar {
                    self.complete_turn_without_grammar(turn_id);
                    true
                } else {
                    false
                }
            }
            silence::Action::CompleteWithoutGrammar { turn_id } => {
                self.complete_turn_without_grammar(turn_id);
                true
            }
        }
    }

    fn can_promote_pending_interim_to_completion(&self, previous_segment_id: u64) -> bool {
        self.promotable_pending_interim_index(previous_segment_id)
            .is_some()
    }

    fn promote_pending_interim_to_completion_for_turn_check(
        &mut self,
        previous_segment_id: u64,
    ) -> bool {
        let Some(index) = self.promotable_pending_interim_index(previous_segment_id) else {
            return false;
        };
        for _ in 0..index {
            self.pending.asr_segments.pop_front();
        }

        let Some(segment) = self.pending.asr_segments.front_mut() else {
            return false;
        };
        segment.reason = SegmentCloseReason::EndSilenceReached;
        self.dispatch_next_asr_request_if_idle();
        self.requests.in_flight_request.is_some()
    }

    fn promotable_pending_interim_index(&self, previous_segment_id: u64) -> Option<usize> {
        let candidate_index = self.pending.asr_segments.iter().position(|segment| {
            segment.reason == SegmentCloseReason::InterimResultSilenceReached
                && segment.last_segment_id().0 == previous_segment_id
        })?;
        let candidate = self.pending.asr_segments.get(candidate_index)?;
        let preceding_segments_are_covered = self
            .pending
            .asr_segments
            .iter()
            .take(candidate_index)
            .all(|segment| {
                segment.kind() == AsrTaskKind::InterimDisplay
                    && segment.turn_id() == candidate.turn_id()
                    && candidate.range.contains(segment.range)
            });
        preceding_segments_are_covered.then_some(candidate_index)
    }

    pub(in crate::recognition) fn emit_stale_turn_finals(&mut self, before_turn_id: u64) {
        let mut stale_turn_ids = self
            .turn_store
            .turns
            .keys()
            .copied()
            .filter(|turn_id| *turn_id < before_turn_id)
            .collect::<Vec<_>>();
        stale_turn_ids.extend(
            self.turn_store
                .audio_ranges
                .keys()
                .copied()
                .filter(|turn_id| {
                    *turn_id < before_turn_id && !self.turn_store.turns.contains_key(turn_id)
                }),
        );
        stale_turn_ids.sort_unstable();
        stale_turn_ids.dedup();

        for turn_id in stale_turn_ids {
            if self.turn_store.turns.contains_key(&turn_id) {
                self.emit_turn_output(turn_id, true);
            } else {
                self.finalize_turn_audio_range(turn_id);
            }
        }
    }

    pub(in crate::recognition) fn emit_turn_output(&mut self, turn_id: u64, is_final: bool) {
        let Some(turn) = self.turn_store.turns.get(&turn_id) else {
            if is_final {
                self.finalize_turn_audio_range(turn_id);
            }
            return;
        };
        let draft = turn.draft();
        if draft.combined_text.is_empty() {
            if is_final {
                self.cleanup_final_turn_state(turn_id);
            }
            return;
        }
        let route = draft
            .route
            .unwrap_or_else(|| RecognitionRoute::from_model(self.config.asr.model));
        if is_final {
            let output_sequence =
                take_next_output_sequence(&mut self.counters.next_output_sequence);
            self.finalize_turn_audio_range(turn_id);
            self.turn_store.finalized_turns.insert(turn_id);
            self.turn_store.streaming_interim_ranges.remove(&turn_id);
            let Some(turn) = self.turn_store.turns.remove(&turn_id) else {
                return;
            };
            let Some(confirmed) = turn.into_draft().confirm(
                self.counters.turn_session_id,
                turn_id,
                output_sequence,
                route,
            ) else {
                return;
            };
            *self.turn_store.revisions.entry(turn_id).or_insert(0) += 1;
            self.io.output_sink.emit(confirmed.into_output());
            return;
        }

        // Only emit an interim when the turn text changed since the last emitted interim;
        // skip without consuming an output sequence when it is unchanged.
        if draft.last_emitted_interim_text.as_deref() == Some(draft.combined_text.as_str()) {
            return;
        }
        let combined_text = draft.combined_text.clone();
        let output_sequence = take_next_output_sequence(&mut self.counters.next_output_sequence);
        let Some(output) = draft.interim_output(
            self.counters.turn_session_id,
            turn_id,
            output_sequence,
            route,
        ) else {
            return;
        };
        if let Some(turn) = self.turn_store.turns.get_mut(&turn_id) {
            turn.draft_mut().last_emitted_interim_text = Some(combined_text);
        }
        self.io.output_sink.emit(output);
    }

    fn cleanup_final_turn_state(&mut self, turn_id: u64) {
        self.turn_store.finalized_turns.insert(turn_id);
        self.turn_store.streaming_interim_ranges.remove(&turn_id);
        self.finalize_turn_audio_range(turn_id);
        self.turn_store.turns.remove(&turn_id);
    }

    fn finalize_turn_audio_range(&mut self, turn_id: u64) {
        if let Some(range) = self.turn_store.audio_ranges.remove(&turn_id) {
            self.turn_store.confirmed_until_sample =
                self.turn_store.confirmed_until_sample.max(range.end_sample);
        }
    }

    pub(in crate::recognition) fn handle_open_turn_timeout(&mut self) -> bool {
        let action = timeout::action(timeout::Input {
            open_turn_id: self.turn_store.open_turn_id,
            open_turn_activity_epoch: self.activity.open_turn_activity_epoch,
            segment_activity_epoch: self.activity.segment_activity_epoch,
            open_turn_since_tick: self.activity.open_turn_since_tick,
            next_runtime_tick: self.counters.next_runtime_tick,
            timeout_ticks: self.timeout_ticks(),
        });
        let turn_id = match action {
            timeout::Action::NoOpenTurn | timeout::Action::Waiting => return false,
            timeout::Action::ResetTimeoutOrigin => {
                self.reset_open_turn_timeout_origin();
                return false;
            }
            timeout::Action::Timeout { turn_id } => turn_id,
        };
        if self.config.uses_deferred_turn_completion() && {
            self.refresh_turn_route_with_sli(turn_id);
            self.dispatch_rerecognition_for_turn_if_idle(
                turn_id,
                RerecognitionPurpose::TimeoutFinal,
            )
        } {
            return true;
        }
        if self.defer_finalization_if_blocked(PendingFinalization::new(turn_id)) {
            return true;
        }
        self.finalize_timeout_turn_after_rerecognition(turn_id);
        true
    }

    pub(in crate::recognition) fn timeout_ticks(&self) -> u64 {
        timeout::ticks(&self.config)
    }
}

impl RecognitionSession {
    pub(in crate::recognition) fn process_pending_finalization_if_ready(&mut self) -> bool {
        let Some(action) = self.pending.finalization else {
            return false;
        };
        if self.finalization_is_blocked(action.turn_id()) {
            return false;
        }
        self.pending.finalization = None;
        self.finalize_turn_now(action.turn_id());
        true
    }

    fn defer_finalization_if_blocked(&mut self, action: PendingFinalization) -> bool {
        if !self.finalization_is_blocked(action.turn_id()) {
            return false;
        }
        // Finalization is intentionally deferred globally, not just during shutdown:
        // late ASR for the same or an older turn can still extend the text and saved audio.
        // Keep the newest blocked turn because finalizing it also sweeps stale older turns.
        if self
            .pending
            .finalization
            .is_some_and(|pending| pending.turn_id() >= action.turn_id())
        {
            return true;
        }
        self.pending.finalization = Some(action);
        true
    }

    fn finalization_is_blocked(&self, turn_id: u64) -> bool {
        self.requests
            .in_flight_request
            .as_ref()
            .is_some_and(|request| request.target.turn_id.0 <= turn_id)
            || self
                .pending
                .asr_segments
                .iter()
                .any(|segment| segment.turn_id().0 <= turn_id)
    }
}
