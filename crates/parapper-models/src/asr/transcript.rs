use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrToken {
    pub text: String,
    pub start_sec: Option<f32>,
    pub duration_sec: Option<f32>,
    pub char_range: Option<Range<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrTranscript {
    pub text: String,
    pub tokens: Vec<AsrToken>,
}

impl AsrTranscript {
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_parts(
        text: impl Into<String>,
        token_texts: Vec<String>,
        timestamps: Option<&[f32]>,
        durations: Option<&[f32]>,
    ) -> Self {
        let text = text.into().trim().to_string();
        let token_ranges = token_char_ranges_relative_to_trimmed_text(&token_texts);
        let tokens = token_texts
            .into_iter()
            .enumerate()
            .map(|(index, token_text)| AsrToken {
                text: token_text,
                start_sec: timestamps.and_then(|values| values.get(index)).copied(),
                duration_sec: durations.and_then(|values| values.get(index)).copied(),
                char_range: token_ranges.get(index).cloned().flatten(),
            })
            .collect();
        Self { text, tokens }
    }
}

fn token_char_ranges_relative_to_trimmed_text(token_texts: &[String]) -> Vec<Option<Range<usize>>> {
    let joined = token_texts.concat();
    let trimmed_start_bytes = joined.len() - joined.trim_start().len();
    let trimmed_end_bytes = joined.len().saturating_sub(joined.trim_end().len());
    let trimmed_start = joined[..trimmed_start_bytes].chars().count();
    let visible_end_bytes = joined.len().saturating_sub(trimmed_end_bytes);
    let trimmed_end = joined[..visible_end_bytes].chars().count();

    let mut cursor = 0;
    token_texts
        .iter()
        .map(|token| {
            let start = cursor;
            let end = start + token.chars().count();
            cursor = end;
            let visible_start = start.max(trimmed_start);
            let visible_end = end.min(trimmed_end);
            (visible_start < visible_end)
                .then(|| visible_start - trimmed_start..visible_end - trimmed_start)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::AsrTranscript;

    #[test]
    fn transcript_tokens_keep_ranges_relative_to_trimmed_text() {
        let transcript = AsrTranscript::from_parts(
            "Well, I don't.",
            vec![" We", "ll", ",", " I", " don", "'", "t", "."]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Some(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            None,
        );

        assert_eq!(transcript.text, "Well, I don't.");
        assert_eq!(transcript.tokens[0].char_range, Some(0..2));
        assert_eq!(transcript.tokens[2].char_range, Some(4..5));
        assert_eq!(transcript.tokens[7].char_range, Some(13..14));
        assert_eq!(transcript.tokens[7].start_sec, Some(0.7));
    }

    #[test]
    fn transcript_tokens_keep_character_ranges_for_multibyte_text() {
        let transcript = AsrTranscript::from_parts(
            "漢字🙂かな",
            ["  漢", "字", "🙂", "か", "な  "]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Some(&[0.0, 0.1, 0.2, 0.3, 0.4]),
            None,
        );

        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.char_range.clone())
                .collect::<Vec<_>>(),
            vec![Some(0..1), Some(1..2), Some(2..3), Some(3..4), Some(4..5)]
        );
    }
}
