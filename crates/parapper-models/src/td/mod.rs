//! Turn detector implementations.

mod audio_window;
mod boundary;
#[cfg(feature = "td-morph")]
pub mod morph;
#[cfg(feature = "td-namo-ort")]
pub mod namo;

use serde::{Deserialize, Serialize};

pub use boundary::candidates_for_transcript_without_morph;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarBoundaryClass {
    StrongEnd,
    PredicateEnd,
    NormalEnd,
    Reject,
    ClauseWeak,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnBoundaryCandidate {
    pub char_end: usize,
    pub sample_end: usize,
    pub prefix_audio_end: usize,
    pub suffix_audio_start: usize,
    pub class: GrammarBoundaryClass,
}

impl TurnBoundaryCandidate {
    #[must_use]
    pub const fn offset_by(self, char_offset: usize, audio_offset: usize) -> Self {
        Self {
            char_end: char_offset + self.char_end,
            sample_end: audio_offset + self.sample_end,
            prefix_audio_end: audio_offset + self.prefix_audio_end,
            suffix_audio_start: audio_offset + self.suffix_audio_start,
            class: self.class,
        }
    }
}
