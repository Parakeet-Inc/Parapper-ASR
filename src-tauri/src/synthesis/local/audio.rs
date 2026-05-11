use tauri::AppHandle;

use crate::synthesis::request::QueuedSpeechRequest;

pub(super) struct GeneratedLocalTtsItem {
    pub(super) handle: Option<AppHandle>,
    pub(super) request: QueuedSpeechRequest,
    pub(super) audio: GeneratedLocalTtsAudio,
}

pub(super) struct GeneratedLocalTtsAudio {
    pub(super) samples: Vec<f32>,
    pub(super) sample_rate: i32,
}
