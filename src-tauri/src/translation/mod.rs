mod clients;
mod event;
mod local;
mod manager;
mod provider;
mod queue;
mod request;

pub(crate) use local::TranslationHttpListener;
pub(crate) use manager::submit_recognized_text;
#[cfg(test)]
pub(crate) use manager::{spawn_translation_if_needed, translate_and_spawn_speech_for_test};
#[cfg(test)]
pub(crate) use request::build_translation_request;
