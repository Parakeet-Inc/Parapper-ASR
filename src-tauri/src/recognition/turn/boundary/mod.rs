use std::ops::Range;

use crate::{
    audio::ASR_SAMPLE_RATE,
    config::AsrLanguage,
    recognition::{
        segmentation::vad::engine::VadResult,
        transcription::asr::engine::{AsrToken, AsrTranscript},
        turn::{GrammarBoundaryClass, TurnBoundaryCandidate},
    },
};

mod audio_window;
mod japanese;

use audio_window::audio_window_for_boundary;
use japanese::japanese_morph_candidates;
pub(crate) use japanese::{JapaneseMorphAnalyzer, load_japanese_morph_analyzer};

pub(crate) fn candidates_for_transcript(
    language: AsrLanguage,
    transcript: &AsrTranscript,
    audio: &[f32],
    vad_results: &[VadResult],
    japanese_morph: Option<&JapaneseMorphAnalyzer>,
) -> Vec<TurnBoundaryCandidate> {
    if transcript.tokens.is_empty() || !tokens_have_aligned_timestamps(&transcript.tokens) {
        return Vec::new();
    }

    let mut candidates = match language {
        AsrLanguage::Japanese => {
            japanese_candidates_from_tokens(transcript, audio.len(), vad_results)
        }
        AsrLanguage::English | AsrLanguage::EuropeanMultilingual => {
            english_candidates_from_tokens(transcript, audio.len(), vad_results)
        }
    };

    if language == AsrLanguage::Japanese
        && let Some(analyzer) = japanese_morph
    {
        candidates.extend(japanese_morph_candidates(
            transcript,
            audio.len(),
            vad_results,
            &analyzer.analyze(&transcript.text),
        ));
    }

    candidates.sort_by_key(|candidate| candidate.char_end);
    candidates.dedup_by_key(|candidate| candidate.char_end);
    candidates
}

fn tokens_have_aligned_timestamps(tokens: &[AsrToken]) -> bool {
    tokens
        .iter()
        .filter(|token| token.char_range.is_some())
        .all(|token| token.start_sec.is_some())
}

fn english_candidates_from_tokens(
    transcript: &AsrTranscript,
    audio_len: usize,
    vad_results: &[VadResult],
) -> Vec<TurnBoundaryCandidate> {
    sentence_punctuation_candidates_from_text(
        transcript,
        audio_len,
        vad_results,
        |character| matches!(character, '.' | '?' | '!'),
        GrammarBoundaryClass::StrongEnd,
    )
}

fn japanese_candidates_from_tokens(
    transcript: &AsrTranscript,
    audio_len: usize,
    vad_results: &[VadResult],
) -> Vec<TurnBoundaryCandidate> {
    sentence_punctuation_candidates_from_text(
        transcript,
        audio_len,
        vad_results,
        |character| matches!(character, '。' | '？' | '！'),
        GrammarBoundaryClass::StrongEnd,
    )
}

fn sentence_punctuation_candidates_from_text(
    transcript: &AsrTranscript,
    audio_len: usize,
    vad_results: &[VadResult],
    is_sentence_punctuation: impl Fn(char) -> bool,
    class: GrammarBoundaryClass,
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
                class,
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
fn seconds_to_sample(seconds: f32, audio_len: usize) -> usize {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * ASR_SAMPLE_RATE as f32)
        .round()
        .clamp(0.0, audio_len as f32) as usize
}

pub(crate) fn slice_chars(text: &str, range: Range<usize>) -> String {
    text.chars()
        .skip(range.start)
        .take(range.end.saturating_sub(range.start))
        .collect()
}

#[cfg(test)]
mod tests;
