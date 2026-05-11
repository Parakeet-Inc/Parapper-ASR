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

#[derive(Debug, Clone, Deserialize)]
pub struct TranslatedText {
    pub lang: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslatesResponse {
    pub result: Vec<TranslatedText>,
}

#[derive(Debug, Serialize)]
struct TranslateParams<'a> {
    id: &'a str,
    lang: &'a str,
    text: &'a str,
}

#[derive(Debug, Serialize)]
struct TranslatesParams<'a> {
    id: &'a str,
    lang: &'a [String],
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct RawTranslateResponse {
    operation: String,
    status: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct RawTranslatesResponse {
    operation: String,
    status: String,
    result: Vec<TranslatedText>,
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

    pub fn translates(
        &mut self,
        id: &str,
        langs: &[String],
        text: &str,
    ) -> Result<TranslatesResponse> {
        let request = CommandRequest {
            operation: "translates",
            params: vec![TranslatesParams {
                id,
                lang: langs,
                text,
            }],
        };
        let response = self.post_command::<_, RawTranslatesResponse>(&request)?;
        ensure_success(&response.operation, &response.status, "translates")?;
        Ok(TranslatesResponse {
            result: response.result,
        })
    }
}
