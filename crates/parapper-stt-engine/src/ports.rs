//! Host-provided recognition capabilities.
//!
//! Desktop adapters may own Tauri handles, model paths, worker threads, or
//! event emitters internally. None of those host types cross these contracts.

use crate::{
    RecognizedTurn,
    transcription::{
        route::RecognitionRoute,
        task::{AsrRequest, AsrResult},
    },
    turn::TurnDecision,
};

use parapper_models::{
    asr::{AsrLanguage, AsrTranscript},
    td::TurnBoundaryCandidate,
    vad::VadResult,
};

pub trait AsrRequestRunner {
    fn set_normalize_input_audio(&mut self, _enabled: bool) {}
    fn reset_streaming_sessions(&mut self) {}
    fn reset_streaming_sessions_for_source(&mut self, _source: &crate::SourceSessionKey) {
        // A host that cannot identify source-scoped decoder state must not
        // reset every source as an implicit fallback.
    }
    fn submit(&mut self, request: AsrRequest) -> bool;
    fn try_recv_result(&mut self) -> Option<AsrResult>;
    fn shutdown(&mut self) {}
}

pub trait TurnDecisionRunner {
    /// Decides whether the current transcript completes a turn.
    ///
    /// # Errors
    ///
    /// Returns an error when the host detector cannot evaluate the transcript.
    fn decide(
        &mut self,
        route: RecognitionRoute,
        text: &str,
        max_context_tokens: u32,
    ) -> anyhow::Result<TurnDecision>;
}

pub trait RecognitionOutputSink: Send {
    fn emit(&mut self, output: RecognizedTurn);
}

pub trait LanguageDetector {
    /// Detects the spoken language from audio.
    ///
    /// # Errors
    ///
    /// Returns an error when the host detector cannot process the audio.
    fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> anyhow::Result<String>;
}

pub trait LanguageDetectionWarningSink {
    fn emit_language_detection_warning(&self, error: &anyhow::Error);
}

/// Host-provided grammar boundary analysis.
///
/// The desktop host may back this with Vibrato and model files resolved through
/// Tauri. The engine only consumes the resulting domain candidates.
pub trait TranscriptBoundaryDetector {
    fn candidates_for_transcript(
        &self,
        language: AsrLanguage,
        transcript: &AsrTranscript,
        audio: &[f32],
        vad_results: &[VadResult],
    ) -> Vec<TurnBoundaryCandidate>;
}

pub use parapper_models::vad::VadEngine;
