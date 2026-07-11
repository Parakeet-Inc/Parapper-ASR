use crate::{
    delivery::{
        RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, continuing_turn_text,
        finalize_turn_text, join_turn_segments,
    },
    recognition::{segmentation::vad::engine::VadResult, transcription::route::RecognitionRoute},
};

pub(crate) struct Turn {
    draft: TurnDraft,
}

impl Turn {
    pub(crate) fn new(event_id: String, revision: u64) -> Self {
        Self {
            draft: TurnDraft::new(event_id, revision),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_draft(draft: TurnDraft) -> Self {
        Self { draft }
    }

    pub(crate) fn draft(&self) -> &TurnDraft {
        &self.draft
    }

    pub(crate) fn draft_mut(&mut self) -> &mut TurnDraft {
        &mut self.draft
    }

    pub(crate) fn into_draft(self) -> TurnDraft {
        self.draft
    }
}

#[derive(Default)]
pub(crate) struct TurnDraft {
    pub(crate) event_id: String,
    pub(crate) segment_texts: Vec<String>,
    pub(crate) segment_ids: Vec<u64>,
    pub(crate) segment_audio_lens: Vec<usize>,
    pub(crate) segment_vad_lens: Vec<usize>,
    pub(crate) boundary_candidates: Vec<TurnBoundaryCandidate>,
    pub(crate) vad_results: Vec<VadResult>,
    pub(crate) combined_text: String,
    pub(crate) full_audio: Vec<f32>,
    pub(crate) route: Option<RecognitionRoute>,
    pub(crate) detected_language: Option<String>,
    pub(crate) processing_millis: u128,
    pub(crate) latest_segment_id: Option<u64>,
    pub(crate) latest_previous_segment_id: Option<u64>,
    pub(crate) revision: u64,
    pub(crate) last_emitted_interim_text: Option<String>,
}

impl TurnDraft {
    pub(crate) fn new(event_id: String, revision: u64) -> Self {
        Self {
            event_id,
            revision,
            ..Self::default()
        }
    }

    pub(crate) fn set_detected_language(&mut self, detected_language: Option<String>) {
        if let Some(detected_language) = detected_language {
            self.detected_language = Some(detected_language);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TurnDraft keeps the ASR segment text, audio, VAD, route, and source metadata together."
    )]
    pub(crate) fn append_recognized_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: &[f32],
        vad_results: &[VadResult],
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.record_recognized_segment(
            segment_id,
            previous_segment_id,
            full_audio,
            vad_results,
            route,
            text,
            elapsed_millis,
            false,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TurnDraft keeps the replacement ASR segment text, audio, VAD, route, and source metadata together."
    )]
    pub(crate) fn replace_latest_recognized_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: &[f32],
        vad_results: &[VadResult],
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.record_recognized_segment(
            segment_id,
            previous_segment_id,
            full_audio,
            vad_results,
            route,
            text,
            elapsed_millis,
            true,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TurnDraft keeps the ASR segment text, audio, VAD, route, and source metadata together."
    )]
    fn record_recognized_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: &[f32],
        vad_results: &[VadResult],
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
        replace_latest_segment: bool,
    ) {
        let replacing_latest_segment = replace_latest_segment && !self.segment_ids.is_empty();
        let previous_latest_segment_id = self.latest_segment_id;
        let previous_latest_previous_segment_id = self.latest_previous_segment_id;
        if replacing_latest_segment {
            self.segment_texts.pop();
            self.segment_ids.pop();
            if let Some(audio_len) = self.segment_audio_lens.pop() {
                self.full_audio
                    .truncate(self.full_audio.len().saturating_sub(audio_len));
            }
            if let Some(vad_len) = self.segment_vad_lens.pop() {
                self.vad_results
                    .truncate(self.vad_results.len().saturating_sub(vad_len));
            }
        }

        self.latest_previous_segment_id = previous_segment_id.or(if replacing_latest_segment {
            previous_latest_previous_segment_id
        } else {
            previous_latest_segment_id
        });
        self.latest_segment_id = Some(segment_id);
        self.full_audio.extend_from_slice(full_audio);
        self.vad_results.extend_from_slice(vad_results);
        self.route = Some(route);
        self.segment_texts.push(text);
        self.segment_ids.push(segment_id);
        self.segment_audio_lens.push(full_audio.len());
        self.segment_vad_lens.push(vad_results.len());
        self.combined_text = join_turn_segments(&self.segment_texts, route.language);
        self.processing_millis += elapsed_millis;
    }

    #[cfg(test)]
    pub(crate) fn replace_with_full_turn_transcription(
        &mut self,
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.route = Some(route);
        self.segment_texts.clear();
        self.segment_texts.push(text);
        self.segment_ids.clear();
        self.segment_audio_lens.clear();
        self.segment_vad_lens.clear();
        self.boundary_candidates.clear();
        self.combined_text = join_turn_segments(&self.segment_texts, route.language);
        self.processing_millis += elapsed_millis;
    }

    pub(crate) fn replace_text_preserving_sources(
        &mut self,
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.route = Some(route);
        self.segment_texts.clear();
        self.segment_texts.push(text);
        self.boundary_candidates.clear();
        self.combined_text = join_turn_segments(&self.segment_texts, route.language);
        self.processing_millis += elapsed_millis;
    }

    #[cfg(test)]
    pub(crate) fn spans_multiple_source_segments(&self) -> bool {
        let Some(first_segment_id) = self.segment_ids.first() else {
            return false;
        };
        self.segment_ids
            .iter()
            .any(|segment_id| segment_id != first_segment_id)
    }

    pub(crate) fn source_meta(
        &self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            turn_session_id,
            turn_id,
            turn_revision: self.revision,
            output_sequence,
            segment_id: self
                .latest_segment_id
                .expect("turn source meta requires at least one segment"),
            previous_segment_id: self.latest_previous_segment_id,
        }
    }

    pub(crate) fn interim_output(
        &self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<RecognizedTextOutput> {
        if self.combined_text.is_empty() {
            return None;
        }

        let source = self.source_meta(turn_session_id, turn_id, output_sequence);
        let meta = RecognizedTextMeta::replace_turn_output(self.event_id.clone(), source, false);
        Some(RecognizedTextOutput::from_route(
            self.full_audio.clone(),
            continuing_turn_text(&self.combined_text),
            route,
            self.detected_language.clone(),
            meta,
            self.processing_millis,
        ))
    }

    pub(crate) fn confirm(
        self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<TurnConfirmed> {
        if self.combined_text.is_empty() {
            return None;
        }

        let source = self.source_meta(turn_session_id, turn_id, output_sequence);
        let meta = RecognizedTextMeta::replace_turn_output(self.event_id, source, true);
        Some(TurnConfirmed {
            full_audio: self.full_audio,
            text: finalize_turn_text(&self.combined_text, route.language),
            route,
            detected_language: self.detected_language,
            meta,
            processing_millis: self.processing_millis,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnBoundaryCandidate {
    pub(crate) char_end: usize,
    pub(crate) sample_end: usize,
    pub(crate) prefix_audio_end: usize,
    pub(crate) suffix_audio_start: usize,
    pub(crate) class: GrammarBoundaryClass,
}

#[cfg(test)]
impl TurnBoundaryCandidate {
    pub(crate) fn offset_by(self, char_offset: usize, audio_offset: usize) -> Self {
        Self {
            char_end: char_offset + self.char_end,
            sample_end: audio_offset + self.sample_end,
            prefix_audio_end: audio_offset + self.prefix_audio_end,
            suffix_audio_start: audio_offset + self.suffix_audio_start,
            class: self.class,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrammarBoundaryClass {
    StrongEnd,
    PredicateEnd,
    NormalEnd,
    Reject,
    ClauseWeak,
}

pub(crate) struct TurnConfirmed {
    full_audio: Vec<f32>,
    text: String,
    route: RecognitionRoute,
    detected_language: Option<String>,
    meta: RecognizedTextMeta,
    processing_millis: u128,
}

impl TurnConfirmed {
    pub(crate) fn into_output(self) -> RecognizedTextOutput {
        RecognizedTextOutput::from_route(
            self.full_audio,
            self.text,
            self.route,
            self.detected_language,
            self.meta,
            self.processing_millis,
        )
    }
}
