use crate::recognition::{
    control::RecognitionSession,
    transcription::asr::{
        engine::AsrTranscript,
        task::{AsrRequest, AudioRange},
    },
    turn::{Turn, boundary::candidates_for_transcript, turn_event_id},
};

impl RecognitionSession {
    pub(in crate::recognition) fn apply_segment_transcript(
        &mut self,
        request: &AsrRequest,
        transcript: AsrTranscript,
        elapsed_millis: u128,
    ) -> u64 {
        let turn_id = request.target.turn_id.0;
        self.counters.next_turn_id = self.counters.next_turn_id.max(turn_id.saturating_add(1));
        self.merge_turn_audio_range(turn_id, request.target.range);
        let revision = *self.turn_store.revisions.entry(turn_id).or_insert(0);
        let turn = self.turn_store.turns.entry(turn_id).or_insert_with(|| {
            Turn::new(
                turn_event_id(self.counters.turn_session_id, turn_id, revision),
                revision,
            )
        });
        let draft = turn.draft_mut();
        draft.set_detected_language(request.detected_language.clone());
        draft.append_recognized_segment(
            request
                .target
                .last_segment_id
                .map_or(request.target.turn_id.0, |segment_id| segment_id.0),
            request.target.first_segment_id.and_then(|segment_id| {
                (Some(segment_id) != request.target.last_segment_id).then_some(segment_id.0)
            }),
            &request.source_audio,
            &request.source_vad_results,
            request.route,
            transcript.text,
            elapsed_millis,
        );
        self.turn_store.last_recognition_route = Some(request.route);
        turn_id
    }

    fn merge_turn_audio_range(&mut self, turn_id: u64, range: AudioRange) {
        self.turn_store
            .audio_ranges
            .entry(turn_id)
            .and_modify(|current| *current = current.merge(range))
            .or_insert(range);
    }

    pub(in crate::recognition) fn apply_rerecognition_transcript(
        &mut self,
        request: &AsrRequest,
        transcript: AsrTranscript,
        elapsed_millis: u128,
        refresh_boundary_candidates: bool,
    ) {
        let turn_id = request.target.turn_id.0;
        let candidates = refresh_boundary_candidates.then(|| {
            candidates_for_transcript(
                request.route.language,
                &transcript,
                &request.audio,
                &request.vad_results,
                self.io.japanese_morph.as_ref(),
            )
        });
        if let Some(turn) = self.turn_store.turns.get_mut(&turn_id) {
            let draft = turn.draft_mut();
            draft.set_detected_language(request.detected_language.clone());
            draft.replace_text_preserving_sources(request.route, transcript.text, elapsed_millis);
            if let Some(candidates) = candidates {
                draft.boundary_candidates = candidates;
            }
            self.turn_store.last_recognition_route = Some(request.route);
        }
    }
}
