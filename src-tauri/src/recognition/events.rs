use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::config::{AsrLanguage, AsrModel, SpeechSourceKind};
pub(crate) use parapper_stt_engine::{RecognitionSourceMeta, RecognizedTextUpdateMode};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum RecognitionStatus {
    #[default]
    Idle,
    WaitingForClient,
    Listening,
    Draining,
    Stopped,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VadState {
    Speech,
    Silence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VadStateEvent {
    pub state: VadState,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RecognizedTextEvent {
    pub id: String,
    pub source: RecognitionSourceMeta,
    pub is_final: bool,
    pub update_mode: RecognizedTextUpdateMode,
    pub text: String,
    pub source_asr_model: AsrModel,
    pub source_language: AsrLanguage,
    pub detected_language: Option<String>,
    pub recognized_at_millis: u64,
    pub audio_seconds: f64,
    pub elapsed_millis: u128,
    pub audio_frames: usize,
    pub debug_asr_audio_sample_rate: Option<u32>,
    pub debug_asr_audio_samples: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TranslationTextEvent {
    pub id: String,
    pub source_recognition_id: String,
    pub source: RecognitionSourceMeta,
    pub source_asr_model: AsrModel,
    pub source_text: String,
    pub source_detected_language: Option<String>,
    pub target_lang: String,
    pub translated_text: String,
    pub is_final: bool,
    pub update_mode: RecognizedTextUpdateMode,
    pub translated_at_millis: u64,
    pub elapsed_millis: u128,
    pub status: TranslationTextStatus,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SpeechRequestEvent {
    pub id: String,
    pub source_event_id: String,
    pub source: RecognitionSourceMeta,
    pub source_kind: SpeechSourceKind,
    pub target_lang: Option<String>,
    pub elapsed_millis: u128,
    pub status: SpeechRequestStatus,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechRequestStatus {
    Accepted,
    Failure,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TranslationTextStatus {
    Success,
    Failure,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MissingModelKind {
    Asr,
    LanguageId,
    TurnDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MissingModelEvent {
    pub kind: MissingModelKind,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OscMuteStateEvent {
    pub muted: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConnectionTarget {
    Neo,
    Vrchat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ConnectionStateEvent {
    pub target: ConnectionTarget,
    pub found: bool,
    pub detail: Option<String>,
}

pub(in crate::recognition) fn emit_missing_model_event(
    handle: &AppHandle,
    kind: MissingModelKind,
    reason: String,
) {
    let _ = handle.emit("parapper://asr-missing", MissingModelEvent { kind, reason });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    use tauri::Listener;

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn emit_missing_model_event_dispatches_kind_and_reason_to_ui_channel() {
        let handle = crate::recognition::tests::tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<MissingModelEvent>();
        let _event_id = handle.listen("parapper://asr-missing", move |event| {
            let payload = serde_json::from_str::<MissingModelEvent>(event.payload())
                .expect("missing model payload should decode");
            sender
                .send(payload)
                .expect("missing model event should be recorded");
        });

        emit_missing_model_event(
            &handle,
            MissingModelKind::TurnDetector,
            "missing model".to_string(),
        );

        let event = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("missing model event should be emitted");
        assert_eq!(event.kind, MissingModelKind::TurnDetector);
        assert_eq!(event.reason, "missing model");
    }
}
