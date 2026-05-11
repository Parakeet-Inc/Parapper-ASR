use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    YncPluginClient,
    protocol::{CommandRequest, IdParams, ensure_sended, ensure_sended_or_success},
};

#[derive(Debug, Clone, Copy)]
pub struct SpeechRequest<'a> {
    pub id: &'a str,
    pub text: &'a str,
    pub talker: &'a str,
    pub volume: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechResponse {
    pub id: String,
}

#[derive(Debug, Serialize)]
struct SpeechParams<'a> {
    id: &'a str,
    text: &'a str,
    talker: &'a str,
    volume: f32,
}

#[derive(Debug, Deserialize)]
struct RawSpeechResponse {
    operation: String,
    status: String,
    id: String,
}

#[derive(Debug, Deserialize)]
struct RawVoiceListResponse {
    operation: String,
    status: String,
    voice: Option<Vec<String>>,
}

impl YncPluginClient {
    pub fn speech(&mut self, request: SpeechRequest<'_>) -> Result<SpeechResponse> {
        let command = CommandRequest {
            operation: "speech",
            params: vec![SpeechParams {
                id: request.id,
                text: request.text,
                talker: request.talker,
                volume: request.volume,
            }],
        };
        let response = self.post_command::<_, RawSpeechResponse>(&command)?;
        ensure_sended(&response.operation, &response.status, "speech")?;
        Ok(SpeechResponse { id: response.id })
    }

    pub fn voice_list(&mut self, id: &str) -> Result<Vec<String>> {
        let command = CommandRequest {
            operation: "speech.getvoicelist",
            params: vec![IdParams { id }],
        };
        let response = self.post_command::<_, RawVoiceListResponse>(&command)?;
        ensure_sended_or_success(&response.operation, &response.status, "speech.getvoicelist")?;
        Ok(response.voice.unwrap_or_default())
    }

    pub fn speech_stop(&mut self, id: &str) -> Result<()> {
        let command = CommandRequest {
            operation: "speech.stop",
            params: vec![IdParams { id }],
        };
        let response = self.post_command::<_, RawSpeechResponse>(&command)?;
        ensure_sended(&response.operation, &response.status, "speech.stop")
    }
}
