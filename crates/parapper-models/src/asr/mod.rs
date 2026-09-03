//! Automatic speech recognition models and decoders.

mod engine;
#[cfg(feature = "asr-ort")]
mod language_id;
mod model;
mod session;
mod transcript;

#[cfg(feature = "asr-ort")]
pub mod backend;
pub mod decoder;
#[cfg(feature = "asr-ort")]
pub mod frontend;

pub use engine::{AsrEngine, AsrSpeechRangeSamples, AsrStreamConfig};
#[cfg(feature = "asr-ort")]
pub use language_id::SpokenLanguageIdentificationEngine;
pub use model::{
    AsrLanguage, AsrModel, AsrModelCapability, AsrModelImplementation, AsrModelInfo, AsrPrecision,
    AsrStreamLanguage,
};
pub use session::StreamingSessionId;
pub use transcript::{AsrToken, AsrTranscript};

pub const SAMPLE_RATE_HZ: u32 = 16_000;
