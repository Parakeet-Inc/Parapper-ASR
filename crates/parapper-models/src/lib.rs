//! Host-neutral model contracts and implementations used by Parapper.

pub mod asr;
pub mod mt;
#[cfg(feature = "nc-ort")]
pub mod nc;
pub mod td;
pub mod tts;
pub mod vad;

#[cfg(feature = "native-ort")]
mod runtime;

#[cfg(feature = "native-ort")]
pub use runtime::init_onnx_runtime;

pub use asr::decoder;
pub use asr::{
    AsrEngine, AsrLanguage, AsrModel, AsrModelCapability, AsrModelImplementation, AsrModelInfo,
    AsrPrecision, AsrSpeechRangeSamples, AsrStreamConfig, AsrStreamLanguage, AsrToken,
    AsrTranscript, SAMPLE_RATE_HZ, StreamingSessionId,
};
#[cfg(feature = "asr-ort")]
pub use asr::{backend, frontend};
