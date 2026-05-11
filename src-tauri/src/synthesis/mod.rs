//! Text-to-speech synthesis queues.

mod artifact;
mod clients;
mod engines;
mod local;
mod manager;
mod queue;
mod request;

pub(crate) use local::prewarm_local_tts_engines;
pub(crate) use manager::{
    build_speech_requests_with_source_meta, spawn_speech_requests, submit_recognized_text,
};
#[cfg(test)]
pub(crate) use request::QueuedSpeechRequest;
#[cfg(test)]
pub(crate) use request::build_speech_requests;
