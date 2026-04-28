use serde::{Deserialize, Serialize};

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
    pub text: String,
    pub recognized_at_millis: u64,
    pub audio_seconds: f64,
    pub elapsed_millis: u128,
    pub audio_frames: usize,
    pub debug_asr_audio_sample_rate: Option<u32>,
    pub debug_asr_audio_samples: Option<Vec<f32>>,
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
