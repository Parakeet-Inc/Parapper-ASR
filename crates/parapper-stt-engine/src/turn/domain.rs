use crate::{
    RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, SourceIdentitySnapshot,
    VadResult, continuing_turn_text, finalize_turn_text, join_turn_segments,
    transcription::route::RecognitionRoute, turn::TurnBoundaryCandidate,
};

pub struct Turn {
    draft: TurnDraft,
}

impl Turn {
    #[must_use]
    pub fn new(event_id: String, revision: u64) -> Self {
        Self {
            draft: TurnDraft::new(event_id, revision),
        }
    }

    #[must_use]
    pub fn from_draft(draft: TurnDraft) -> Self {
        Self { draft }
    }

    #[must_use]
    pub const fn draft(&self) -> &TurnDraft {
        &self.draft
    }

    pub const fn draft_mut(&mut self) -> &mut TurnDraft {
        &mut self.draft
    }

    #[must_use]
    pub fn into_draft(self) -> TurnDraft {
        self.draft
    }
}

#[derive(Default)]
pub struct TurnDraft {
    pub event_id: String,
    pub segment_texts: Vec<String>,
    pub segment_ids: Vec<u64>,
    pub segment_audio_lens: Vec<usize>,
    pub segment_vad_lens: Vec<usize>,
    pub boundary_candidates: Vec<TurnBoundaryCandidate>,
    pub vad_results: Vec<VadResult>,
    pub combined_text: String,
    pub full_audio: Vec<f32>,
    pub route: Option<RecognitionRoute>,
    pub detected_language: Option<String>,
    pub processing_millis: u128,
    pub latest_segment_id: Option<u64>,
    pub latest_previous_segment_id: Option<u64>,
    pub revision: u64,
    pub last_emitted_interim_text: Option<String>,
}

impl TurnDraft {
    #[must_use]
    pub fn new(event_id: String, revision: u64) -> Self {
        Self {
            event_id,
            revision,
            ..Self::default()
        }
    }

    pub fn set_detected_language(&mut self, detected_language: Option<String>) {
        if let Some(detected_language) = detected_language {
            self.detected_language = Some(detected_language);
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "TurnDraft keeps the ASR segment text, audio, VAD, route, and source metadata together."
    )]
    pub fn append_recognized_segment(
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
    pub fn replace_latest_recognized_segment(
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

    pub fn replace_with_full_turn_transcription(
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

    pub fn replace_text_preserving_sources(
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

    #[must_use]
    pub fn spans_multiple_source_segments(&self) -> bool {
        let Some(first_segment_id) = self.segment_ids.first() else {
            return false;
        };
        self.segment_ids
            .iter()
            .any(|segment_id| segment_id != first_segment_id)
    }

    #[must_use]
    /// Builds source metadata for a draft that already contains a segment.
    ///
    /// # Panics
    ///
    /// Panics if no segment has been attached to the draft.
    pub fn source_meta(
        &self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        let source_identity = SourceIdentitySnapshot::legacy_single_source();
        self.source_meta_for_source(&source_identity, turn_session_id, turn_id, output_sequence)
    }

    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics if no segment has been attached to the draft.
    pub fn source_meta_for_source(
        &self,
        source_identity: &SourceIdentitySnapshot,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            identity: source_identity.clone(),
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

    #[must_use]
    pub fn interim_output(
        &self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<RecognizedTextOutput> {
        let source_identity = SourceIdentitySnapshot::legacy_single_source();
        self.interim_output_for_source(
            &source_identity,
            turn_session_id,
            turn_id,
            output_sequence,
            route,
        )
    }

    #[must_use]
    pub fn interim_output_for_source(
        &self,
        source_identity: &SourceIdentitySnapshot,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<RecognizedTextOutput> {
        if self.combined_text.is_empty() {
            return None;
        }

        let source =
            self.source_meta_for_source(source_identity, turn_session_id, turn_id, output_sequence);
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

    #[must_use]
    pub fn confirm(
        self,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<TurnConfirmed> {
        let source_identity = SourceIdentitySnapshot::legacy_single_source();
        self.confirm_for_source(
            &source_identity,
            turn_session_id,
            turn_id,
            output_sequence,
            route,
        )
    }

    #[must_use]
    pub fn confirm_for_source(
        self,
        source_identity: &SourceIdentitySnapshot,
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
        route: RecognitionRoute,
    ) -> Option<TurnConfirmed> {
        if self.combined_text.is_empty() {
            return None;
        }

        let source =
            self.source_meta_for_source(source_identity, turn_session_id, turn_id, output_sequence);
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

pub struct TurnConfirmed {
    full_audio: Vec<f32>,
    text: String,
    route: RecognitionRoute,
    detected_language: Option<String>,
    meta: RecognizedTextMeta,
    processing_millis: u128,
}

impl TurnConfirmed {
    #[must_use]
    pub fn into_output(self) -> RecognizedTextOutput {
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

#[cfg(test)]
mod tests {
    use parapper_models::asr::AsrLanguage;

    use super::*;

    #[test]
    fn repeated_interim_for_one_segment_replaces_audio_text_and_vad_without_duplication() {
        let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
        let mut draft = TurnDraft::new("turn-1".to_owned(), 0);
        let speech = VadResult {
            probability: 0.9,
            is_speech: true,
        };
        draft.append_recognized_segment(
            1,
            None,
            &[1.0, 2.0],
            &[speech],
            route,
            "最初".to_owned(),
            10,
        );
        draft.replace_latest_recognized_segment(
            1,
            None,
            &[1.0, 2.0, 3.0, 4.0],
            &[speech, speech],
            route,
            "最初の続き".to_owned(),
            20,
        );

        assert_eq!(draft.segment_ids, vec![1]);
        assert_eq!(draft.segment_texts, vec!["最初の続き"]);
        assert_eq!(draft.full_audio, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(draft.vad_results, vec![speech, speech]);
    }

    #[test]
    fn confirmed_turn_preserves_structured_source_identity_for_any_host() {
        let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
        let mut draft = TurnDraft::new("turn-9-4-2".to_owned(), 2);
        draft.append_recognized_segment(8, Some(7), &[1.0], &[], route, "完了".to_owned(), 5);

        let output = draft.confirm(9, 4, 3, route).unwrap().into_output();

        assert_eq!(output.text, "完了。");
        assert_eq!(
            output.meta.source,
            RecognitionSourceMeta {
                identity: SourceIdentitySnapshot::legacy_single_source(),
                turn_session_id: 9,
                turn_id: 4,
                turn_revision: 2,
                output_sequence: 3,
                segment_id: 8,
                previous_segment_id: Some(7),
            }
        );
    }
}
