use crate::{
    delivery::{
        RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, continuing_turn_text,
        finalize_turn_text, join_turn_segments,
    },
    recognition::route::RecognitionRoute,
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
    pub(crate) combined_text: String,
    pub(crate) full_audio: Vec<f32>,
    pub(crate) route: Option<RecognitionRoute>,
    pub(crate) detected_language: Option<String>,
    pub(crate) processing_millis: u128,
    pub(crate) latest_segment_id: Option<u64>,
    pub(crate) latest_previous_segment_id: Option<u64>,
    pub(crate) revision: u64,
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

    pub(crate) fn append_recognized_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: &[f32],
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.latest_previous_segment_id = previous_segment_id.or(self.latest_segment_id);
        self.latest_segment_id = Some(segment_id);
        self.full_audio.extend_from_slice(full_audio);
        self.route = Some(route);
        self.segment_texts.push(text);
        self.combined_text = join_turn_segments(&self.segment_texts, route.language);
        self.processing_millis += elapsed_millis;
    }

    pub(crate) fn replace_with_full_turn_transcription(
        &mut self,
        route: RecognitionRoute,
        text: String,
        elapsed_millis: u128,
    ) {
        self.route = Some(route);
        self.segment_texts.clear();
        self.segment_texts.push(text);
        self.combined_text = join_turn_segments(&self.segment_texts, route.language);
        self.processing_millis += elapsed_millis;
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
        let meta = RecognizedTextMeta::replace_turn(self.event_id.clone(), source, false);
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
        let meta = RecognizedTextMeta::replace_turn(self.event_id, source, true);
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
