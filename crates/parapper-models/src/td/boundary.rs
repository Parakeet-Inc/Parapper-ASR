use crate::{
    SAMPLE_RATE_HZ,
    asr::{AsrLanguage, AsrToken, AsrTranscript},
    td::{GrammarBoundaryClass, TurnBoundaryCandidate, audio_window::audio_window_for_boundary},
    vad::VadResult,
};

/// Finds timestamp-aligned punctuation boundaries without loading a
/// morphology model.
#[must_use]
pub fn candidates_for_transcript_without_morph(
    language: AsrLanguage,
    transcript: &AsrTranscript,
    audio: &[f32],
    vad_results: &[VadResult],
) -> Vec<TurnBoundaryCandidate> {
    if transcript.tokens.is_empty() || !tokens_have_aligned_timestamps(&transcript.tokens) {
        return Vec::new();
    }

    match language {
        AsrLanguage::Japanese => sentence_punctuation_candidates_from_text(
            transcript,
            audio.len(),
            vad_results,
            |character| matches!(character, '。' | '？' | '！'),
        ),
        AsrLanguage::English | AsrLanguage::EuropeanMultilingual | AsrLanguage::Multilingual => {
            sentence_punctuation_candidates_from_text(
                transcript,
                audio.len(),
                vad_results,
                |character| matches!(character, '.' | '?' | '!'),
            )
        }
    }
}

fn tokens_have_aligned_timestamps(tokens: &[AsrToken]) -> bool {
    tokens
        .iter()
        .filter(|token| token.char_range.is_some())
        .all(|token| token.start_sec.is_some())
}

fn sentence_punctuation_candidates_from_text(
    transcript: &AsrTranscript,
    audio_len: usize,
    vad_results: &[VadResult],
    is_sentence_punctuation: impl Fn(char) -> bool,
) -> Vec<TurnBoundaryCandidate> {
    transcript
        .text
        .chars()
        .enumerate()
        .filter(|(_, character)| is_sentence_punctuation(*character))
        .filter_map(|(index, _)| {
            let char_end = index + 1;
            let sample_end = sample_end_for_char_end(transcript, char_end, audio_len)?;
            let audio_window = audio_window_for_boundary(audio_len, vad_results, sample_end);
            Some(TurnBoundaryCandidate {
                char_end,
                sample_end,
                prefix_audio_end: audio_window.prefix_audio_end,
                suffix_audio_start: audio_window.suffix_audio_start,
                class: GrammarBoundaryClass::StrongEnd,
            })
        })
        .collect()
}

pub(super) fn sample_end_for_char_end(
    transcript: &AsrTranscript,
    char_end: usize,
    audio_len: usize,
) -> Option<usize> {
    transcript
        .tokens
        .iter()
        .enumerate()
        .find(|(_, token)| {
            token
                .char_range
                .as_ref()
                .is_some_and(|range| range.end >= char_end)
        })
        .and_then(|(index, _)| token_end_sample(transcript, index, audio_len))
}

fn token_end_sample(
    transcript: &AsrTranscript,
    token_index: usize,
    audio_len: usize,
) -> Option<usize> {
    let token = transcript.tokens.get(token_index)?;
    let start_sec = token.start_sec?;
    let end_sec = token
        .duration_sec
        .filter(|duration| *duration > 0.0)
        .map_or_else(
            || {
                transcript
                    .tokens
                    .iter()
                    .skip(token_index + 1)
                    .find_map(|next| next.start_sec)
                    .unwrap_or(start_sec)
            },
            |duration| start_sec + duration,
        );
    Some(seconds_to_sample(end_sec, audio_len))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(super) fn seconds_to_sample(seconds: f32, audio_len: usize) -> usize {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * SAMPLE_RATE_HZ as f32)
        .round()
        .clamp(0.0, audio_len as f32) as usize
}
