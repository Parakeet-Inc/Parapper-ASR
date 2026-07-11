use crate::{
    config::{AsrLanguage, AsrModel},
    recognition::{
        control::events::{RecognitionSourceMeta, RecognizedTextUpdateMode},
        transcription::route::RecognitionRoute,
    },
};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecognizedTextMeta {
    pub(crate) id: String,
    pub(crate) is_final: bool,
    pub(crate) update_mode: RecognizedTextUpdateMode,
    pub(crate) source: RecognitionSourceMeta,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecognizedTextOutput {
    pub(crate) phrase: Arc<[f32]>,
    pub(crate) text: String,
    pub(crate) source_asr_model: AsrModel,
    pub(crate) source_language: AsrLanguage,
    pub(crate) detected_language: Option<String>,
    pub(crate) meta: RecognizedTextMeta,
    pub(crate) elapsed_millis: u128,
}

impl RecognizedTextOutput {
    pub(crate) fn new(
        phrase: Vec<f32>,
        text: String,
        source_asr_model: AsrModel,
        source_language: AsrLanguage,
        detected_language: Option<String>,
        meta: RecognizedTextMeta,
        elapsed_millis: u128,
    ) -> Self {
        Self {
            phrase: phrase.into(),
            text,
            source_asr_model,
            source_language,
            detected_language,
            meta,
            elapsed_millis,
        }
    }

    pub(crate) fn from_route(
        phrase: Vec<f32>,
        text: String,
        route: RecognitionRoute,
        detected_language: Option<String>,
        meta: RecognizedTextMeta,
        elapsed_millis: u128,
    ) -> Self {
        Self::new(
            phrase,
            text,
            route.model,
            route.language,
            detected_language,
            meta,
            elapsed_millis,
        )
    }
}

impl RecognizedTextMeta {
    #[cfg(test)]
    pub(crate) fn replace_turn(id: String, source: RecognitionSourceMeta, is_final: bool) -> Self {
        Self::replace_turn_output(id, source, is_final)
    }

    pub(crate) fn replace_turn_output(
        id: String,
        source: RecognitionSourceMeta,
        is_final: bool,
    ) -> Self {
        Self {
            id,
            is_final,
            update_mode: RecognizedTextUpdateMode::Replace,
            source,
        }
    }

    pub(crate) fn source(&self) -> &RecognitionSourceMeta {
        &self.source
    }

    pub(crate) fn is_final(&self) -> bool {
        self.is_final
    }

    pub(crate) fn update_mode(&self) -> RecognizedTextUpdateMode {
        self.update_mode
    }
}
