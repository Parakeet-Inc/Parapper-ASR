use serde::{Deserialize, Serialize};

use crate::config::{AsrModel, SpeechSourceKind};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecognitionStatus {
    Idle,
    Listening,
    Stopped,
    Error,
}

impl Default for RecognitionStatus {
    fn default() -> Self {
        Self::Idle
    }
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
    pub detected_language: Option<String>,
    pub recognized_at_millis: u64,
    pub audio_seconds: f64,
    pub elapsed_millis: u128,
    pub audio_frames: usize,
    pub debug_asr_audio_sample_rate: Option<u32>,
    pub debug_asr_audio_samples: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RecognitionSourceMeta {
    pub turn_session_id: u64,
    pub turn_id: u64,
    pub turn_revision: u64,
    pub output_sequence: u64,
    pub segment_id: u64,
    pub previous_segment_id: Option<u64>,
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
pub(crate) enum RecognizedTextUpdateMode {
    Append,
    Replace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AsrMissingEvent {
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
