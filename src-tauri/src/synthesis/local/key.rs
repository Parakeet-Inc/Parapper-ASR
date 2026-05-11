use crate::{config::LocalTtsVoice, synthesis::request::QueuedSpeechRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct LocalTtsQueueKey {
    pub(super) voice: Option<LocalTtsVoice>,
}

pub(super) fn local_tts_queue_key(request: &QueuedSpeechRequest) -> LocalTtsQueueKey {
    LocalTtsQueueKey {
        voice: request.local_tts_voice,
    }
}
