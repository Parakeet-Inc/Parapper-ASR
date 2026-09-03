use std::ops::Range;

use crate::{
    asr::{AsrLanguage, AsrTranscript},
    td::{TurnBoundaryCandidate, boundary::candidates_for_transcript_without_morph},
    vad::VadResult,
};

mod japanese;

#[cfg(test)]
use crate::td::boundary::{sample_end_for_char_end, seconds_to_sample};
pub use japanese::JapaneseMorphAnalyzer;
use japanese::japanese_morph_candidates;

#[must_use]
pub fn candidates_for_transcript(
    language: AsrLanguage,
    transcript: &AsrTranscript,
    audio: &[f32],
    vad_results: &[VadResult],
    japanese_morph: Option<&JapaneseMorphAnalyzer>,
) -> Vec<TurnBoundaryCandidate> {
    let mut candidates =
        candidates_for_transcript_without_morph(language, transcript, audio, vad_results);

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

#[must_use]
pub fn slice_chars(text: &str, range: Range<usize>) -> String {
    text.chars()
        .skip(range.start)
        .take(range.end.saturating_sub(range.start))
        .collect()
}

#[cfg(test)]
mod tests;
