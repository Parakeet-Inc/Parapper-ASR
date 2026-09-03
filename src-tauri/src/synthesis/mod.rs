//! Text-to-speech synthesis queues.

mod dispatch;
mod local;
mod queue;
mod request;

pub(crate) use dispatch::{
    build_speech_requests_with_source_meta, spawn_speech_requests, submit_recognized_text,
};
pub(crate) use local::prewarm_local_tts_engines;
#[cfg(test)]
pub(crate) use request::QueuedSpeechRequest;
#[cfg(test)]
pub(crate) use request::build_speech_requests;
