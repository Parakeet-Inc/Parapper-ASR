use std::{
    io::Write,
    net::{Shutdown, SocketAddr, TcpStream},
    time::Duration,
};

use anyhow::{Context, Result};

use crate::connect::{TextTransport, registry::detect_dword_value_u16};

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

    pub(super) fn send_text_to_port(&self, port: u16, text: &str) -> Result<()> {
        let encoded_text = percent_encode_query_component(text);
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
            "GET /api/input?text={encoded_text} HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\n\r\n"
        );
        stream
            .write_all(request.as_bytes())
            .with_context(|| format!("Failed to send text to YNC text input API: {endpoint}"))?;
        let _ = stream.shutdown(Shutdown::Write);
        Ok(())
    }
}

impl TextTransport for YncTextInputTransport {
    fn send_text(&mut self, text: &str) -> Result<()> {
        self.send_text_to_port(self.configured_port, text)
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

pub(super) fn percent_encode_query_component(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push_str("%20"),
            _ => {
                encoded.push('%');
                encoded.push(HEX[(byte >> 4) as usize] as char);
                encoded.push(HEX[(byte & 0x0f) as usize] as char);
            }
        }
    }
    encoded
}
