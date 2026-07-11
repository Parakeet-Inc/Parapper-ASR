use std::{
    io::Write,
    net::{Shutdown, SocketAddr, TcpStream},
    time::Duration,
};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::connect::{TextInputPayload, TextTransport};

#[cfg(windows)]
use crate::connect::registry::detect_dword_value_u16;

#[derive(Serialize)]
struct TextInputRequestBody<'a> {
    #[serde(rename = "Text")]
    text: &'a str,
    #[serde(rename = "fixedText")]
    fixed_text: bool,
    #[serde(rename = "textID")]
    text_id: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YncTextInputTransport {
    host: String,
    configured_port: u16,
    pub(super) timeout: Duration,
}

impl YncTextInputTransport {
    pub fn localhost(port: u16) -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            configured_port: port,
            timeout: Duration::from_secs(2),
        }
    }

    fn endpoint(&self, port: u16) -> String {
        format!("{}:{}", self.host, port)
    }

    pub(super) fn send_text_to_port(&self, port: u16, payload: TextInputPayload<'_>) -> Result<()> {
        let body = serde_json::to_string(&TextInputRequestBody {
            text: payload.text,
            fixed_text: payload.is_final,
            text_id: payload.text_id,
        })
        .context("Failed to serialize YNC text input request body")?;
        let endpoint = self.endpoint(port);
        let address = endpoint
            .parse::<SocketAddr>()
            .with_context(|| format!("Invalid YNC text input API endpoint: {endpoint}"))?;
        let mut stream = TcpStream::connect_timeout(&address, self.timeout)
            .with_context(|| format!("Failed to connect to YNC text input API: {endpoint}"))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .context("Failed to set YNC text input write timeout")?;
        let request = format!(
            "POST /api/input HTTP/1.1\r\nHost: {endpoint}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .with_context(|| format!("Failed to send text to YNC text input API: {endpoint}"))?;
        let _ = stream.shutdown(Shutdown::Write);
        Ok(())
    }
}

impl TextTransport for YncTextInputTransport {
    fn send_text(&mut self, payload: TextInputPayload<'_>) -> Result<()> {
        self.send_text_to_port(self.configured_port, payload)
    }
}

pub fn ync_text_input_http_available(port: u16) -> bool {
    let address = ([127, 0, 0, 1], port).into();
    TcpStream::connect_timeout(&address, Duration::from_millis(500)).is_ok()
}

#[cfg(windows)]
pub fn detect_ync_text_input_http_port() -> Option<u16> {
    detect_dword_value_u16(r"HKCU\Software\YukarinetteConnectorNeo", "HTTP")
}

#[cfg(not(windows))]
pub fn detect_ync_text_input_http_port() -> Option<u16> {
    None
}
