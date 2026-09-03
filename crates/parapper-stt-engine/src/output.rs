use std::sync::Arc;

use parapper_models::asr::{AsrLanguage, AsrModel};
use serde::{Deserialize, Serialize};

use crate::{SourceIdentitySnapshot, SourceSessionKey, transcription::route::RecognitionRoute};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecognitionSourceMeta {
    pub identity: SourceIdentitySnapshot,
    pub turn_session_id: u64,
    pub turn_id: u64,
    pub turn_revision: u64,
    pub output_sequence: u64,
    pub segment_id: u64,
    pub previous_segment_id: Option<u64>,
}

impl RecognitionSourceMeta {
    #[must_use]
    pub fn source_session_key(&self) -> SourceSessionKey {
        SourceSessionKey::new(self.turn_session_id, self.identity.source_id.clone())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecognizedTextUpdateMode {
    Append,
    Replace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecognizedTextMeta {
    pub id: String,
    pub is_final: bool,
    pub update_mode: RecognizedTextUpdateMode,
    pub source: RecognitionSourceMeta,
}

/// Host-neutral recognition output consumed by desktop or future server hosts.
#[derive(Debug, Clone, PartialEq)]
pub struct RecognizedTurn {
    pub phrase: Arc<[f32]>,
    pub text: String,
    pub source_asr_model: AsrModel,
    pub source_language: AsrLanguage,
    pub detected_language: Option<String>,
    pub meta: RecognizedTextMeta,
    pub elapsed_millis: u128,
}

pub type RecognizedTextOutput = RecognizedTurn;

impl RecognizedTurn {
    #[must_use]
    pub fn new(
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

    #[must_use]
    pub fn from_route(
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
    #[must_use]
    pub fn replace_turn(id: String, source: RecognitionSourceMeta, is_final: bool) -> Self {
        Self::replace_turn_output(id, source, is_final)
    }

    #[must_use]
    pub fn replace_turn_output(id: String, source: RecognitionSourceMeta, is_final: bool) -> Self {
        Self {
            id,
            is_final,
            update_mode: RecognizedTextUpdateMode::Replace,
            source,
        }
    }

    #[must_use]
    pub const fn source(&self) -> &RecognitionSourceMeta {
        &self.source
    }

    #[must_use]
    pub const fn is_final(&self) -> bool {
        self.is_final
    }

    #[must_use]
    pub const fn update_mode(&self) -> RecognizedTextUpdateMode {
        self.update_mode
    }
}

#[must_use]
pub fn join_turn_segments(segments: &[String], language: AsrLanguage) -> String {
    let mut text = String::new();
    for segment in segments
        .iter()
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
    {
        if text.is_empty() {
            text.push_str(segment);
            continue;
        }
        if language == AsrLanguage::Japanese {
            text.push_str(segment);
        } else {
            text.push(' ');
            text.push_str(segment);
        }
    }
    text
}

#[must_use]
pub fn continuing_turn_text(text: &str) -> String {
    let text = trim_continuation_marker(text.trim());
    if text.is_empty() {
        return String::new();
    }
    format!("{text}...")
}

#[must_use]
pub fn finalize_turn_text(text: &str, language: AsrLanguage) -> String {
    let text = trim_continuation_marker(text.trim());
    if language != AsrLanguage::Japanese || text.is_empty() || has_japanese_sentence_end(text) {
        return text.to_string();
    }
    format!("{text}。")
}

#[must_use]
pub fn trim_continuation_marker(text: &str) -> &str {
    text.trim_end_matches("...")
}

fn has_japanese_sentence_end(text: &str) -> bool {
    text.chars()
        .last()
        .is_some_and(|character| matches!(character, '。' | '！' | '？'))
}

#[must_use]
pub fn turn_event_id(session_id: u64, turn_id: u64, revision: u64) -> String {
    format!("turn-{session_id}-{turn_id}-{revision}")
}

pub fn take_next_output_sequence(next_output_sequence: &mut u64) -> u64 {
    let output_sequence = *next_output_sequence;
    *next_output_sequence = next_output_sequence.saturating_add(1);
    output_sequence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interim_then_final_formatting_never_duplicates_continuation_or_sentence_markers() {
        let cases = [
            ("途中", "途中...", "途中。"),
            ("途中...", "途中...", "途中。"),
            ("完了。", "完了。...", "完了。"),
        ];
        for (input, expected_interim, expected_final) in cases {
            assert_eq!(continuing_turn_text(input), expected_interim);
            assert_eq!(
                finalize_turn_text(input, AsrLanguage::Japanese),
                expected_final
            );
        }
    }

    #[test]
    fn segment_joining_follows_language_word_boundary_contract() {
        let segments = vec![" hello ".to_owned(), String::new(), "world".to_owned()];
        assert_eq!(
            join_turn_segments(&segments, AsrLanguage::English),
            "hello world"
        );
        assert_eq!(
            join_turn_segments(&segments, AsrLanguage::Japanese),
            "helloworld"
        );
    }
}
