use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::{
    DEFAULT_HOST, HTTP_TIMEOUT, PLUGIN_COMMAND_PATH, SPEECH_HTTP_TIMEOUT,
    protocol::{CommandRequest, IdParams, RawVersionResponse, ensure_success},
};

#[derive(Debug, Clone)]
pub struct YncPluginClient {
    host: String,
    configured_port: u16,
    client: reqwest::blocking::Client,
}

impl YncPluginClient {
    pub fn for_command(port: u16) -> Result<Self> {
        Self::localhost_with_timeout(port, HTTP_TIMEOUT)
    }

    pub fn for_speech(port: u16) -> Result<Self> {
        Self::localhost_with_timeout(port, SPEECH_HTTP_TIMEOUT)
    }

    fn localhost_with_timeout(port: u16, timeout: Duration) -> Result<Self> {
        Ok(Self {
            host: DEFAULT_HOST.to_string(),
            configured_port: port,
            client: reqwest::blocking::Client::builder()
                .timeout(timeout)
                .build()
                .context("Failed to build YNC plugin HTTP client")?,
        })
    }

    pub(super) fn post_command<TRequest, TResponse>(
        &mut self,
        request: &TRequest,
    ) -> Result<TResponse>
    where
        TRequest: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        self.post_command_to_port(self.configured_port, request)
    }

    fn post_command_to_port<TRequest, TResponse>(
        &self,
        port: u16,
        request: &TRequest,
    ) -> Result<TResponse>
    where
        TRequest: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        let url = format!("http://{}{}", self.endpoint(port), PLUGIN_COMMAND_PATH);
        self.post_command_to_url(&url, request)
    }

    fn post_command_to_url<TRequest, TResponse>(
        &self,
        url: &str,
        request: &TRequest,
    ) -> Result<TResponse>
    where
        TRequest: Serialize,
        TResponse: for<'de> Deserialize<'de>,
    {
        let response = self
            .client
            .post(url)
            .json(request)
            .send()
            .with_context(|| format!("Failed to send YNC plugin command: {url}"))?;
        let status = response.status();
        let body = response
            .text()
            .context("Failed to read YNC plugin response body")?;
        if body.trim().is_empty() {
            return Err(anyhow!("YNC plugin returned an empty response"));
        }
        if !status.is_success() {
            return Err(anyhow!("YNC plugin returned HTTP {status}"));
        }
        serde_json::from_str(&body)
            .with_context(|| "YNC plugin response is not valid JSON".to_string())
    }

    fn endpoint(&self, port: u16) -> String {
        format!("{}:{port}", self.host)
    }

    pub(super) fn probe_plugin_port(&self, port: u16) -> Result<()> {
        let command = CommandRequest {
            operation: "version",
            params: vec![IdParams {
                id: "plugin-port-probe",
            }],
        };
        let response = self.post_command_to_port::<_, RawVersionResponse>(port, &command)?;
        ensure_success(&response.operation, &response.status, "version")
    }
}
