use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    YncPluginClient,
    protocol::{CommandRequest, ensure_success},
};

#[derive(Debug, Clone, Deserialize)]
pub struct TranslateResponse {
    pub text: String,
}

#[derive(Debug, Serialize)]
struct TranslateParams<'a> {
    id: &'a str,
    lang: &'a str,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct RawTranslateResponse {
    operation: String,
    status: String,
    text: String,
}

impl YncPluginClient {
    pub fn translate(&mut self, id: &str, lang: &str, text: &str) -> Result<TranslateResponse> {
        let request = CommandRequest {
            operation: "translate",
            params: vec![TranslateParams { id, lang, text }],
        };
        let response = self.post_command::<_, RawTranslateResponse>(&request)?;
        ensure_success(&response.operation, &response.status, "translate")?;
        Ok(TranslateResponse {
            text: response.text,
        })
    }
}
