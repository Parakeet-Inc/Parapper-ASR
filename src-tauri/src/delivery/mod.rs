//! Delivery of recognized text and derived side effects to UI and integrations.

pub(crate) mod common;
mod dispatch;
mod sinks;
mod types;

pub(crate) use crate::recognition::control::events::RecognitionSourceMeta;
#[cfg(test)]
pub(crate) use crate::synthesis::{
    QueuedSpeechRequest, build_speech_requests, spawn_speech_requests,
};
#[cfg(test)]
pub(crate) use crate::translation::{
    build_translation_request, translate_and_spawn_speech_for_test,
};
#[cfg(test)]
pub(crate) use common::{
    SpeechTextSource, speech_mapping_matches, translation_targets_for_mappings,
    translation_timing_allows,
};
pub(crate) use common::{continuing_turn_text, finalize_turn_text, join_turn_segments};
pub(crate) use dispatch::dispatch_recognized_text;
pub(crate) use sinks::vrchat_mute::spawn_mute_check_if_needed;
#[cfg(test)]
pub(crate) use sinks::ync_text::should_send_to_neo;
pub(crate) use types::{RecognizedTextMeta, RecognizedTextOutput};

#[cfg(test)]
mod tests;
