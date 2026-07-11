use crate::{config::SpeechSourceKind, delivery::RecognitionSourceMeta};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProcessingContext {
    pub(crate) turn_session_id: u64,
    pub(crate) turn_id: u64,
    pub(crate) turn_revision: u64,
    pub(crate) segment_id: u64,
    pub(crate) source_kind: SpeechSourceKind,
    pub(crate) source_language: Option<String>,
}

impl ProcessingContext {
    pub(crate) fn from_source(
        source: &RecognitionSourceMeta,
        source_kind: SpeechSourceKind,
        source_language: Option<String>,
    ) -> Self {
        Self {
            turn_session_id: source.turn_session_id,
            turn_id: source.turn_id,
            turn_revision: source.turn_revision,
            segment_id: source.segment_id,
            source_kind,
            source_language,
        }
    }
}
