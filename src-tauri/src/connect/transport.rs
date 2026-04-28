use std::{net::TcpStream, time::Duration};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
use std::process::Command;

use anyhow::{Context, Result, anyhow};

pub trait TextTransport: Send {
    fn send_text(&mut self, text: &str) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeoHttpTextTransport {
    host: String,
    configured_port: u16,
    detected_port: Option<u16>,
    timeout: Duration,
}

impl NeoHttpTextTransport {
    pub fn localhost(port: u16) -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            configured_port: port,
            detected_port: None,
            timeout: Duration::from_secs(2),
        }
    }

    fn endpoint(&self, port: u16) -> String {
        format!("{}:{}", self.host, port)
    }

    fn send_text_to_port(&self, port: u16, text: &str) -> Result<()> {
        let encoded_text = percent_encode_query_component(text);
        let endpoint = self.endpoint(port);
        let url = format!("http://{endpoint}/api/input?text={encoded_text}");
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .context("Failed to build NEO API HTTP client")?;

        match client.get(url).send() {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => Err(anyhow!("NEO API returned an error: {}", response.status())),
            Err(err) if is_empty_neo_response(&err) => Ok(()),
            Err(err) => Err(err).context(format!("Failed to send text to NEO API: {endpoint}")),
        }
    }
}

pub fn neo_http_available(port: u16) -> bool {
    let address = ([127, 0, 0, 1], port).into();
    TcpStream::connect_timeout(&address, Duration::from_millis(500)).is_ok()
}

fn is_empty_neo_response(err: &reqwest::Error) -> bool {
    let message = format!("{err:?}");
    message.contains("connection closed before message completed")
        || message.contains("IncompleteMessage")
        || message.contains("end of file before message length reached")
}

fn percent_encode_query_component(text: &str) -> String {
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

impl TextTransport for NeoHttpTextTransport {
    fn send_text(&mut self, text: &str) -> Result<()> {
        let port = self.detected_port.unwrap_or(self.configured_port);
        match self.send_text_to_port(port, text) {
            Ok(()) => Ok(()),
            Err(first_err) => {
                let Some(detected_port) = detect_neo_http_port() else {
                    return Err(first_err);
                };
                if detected_port == port {
                    return Err(first_err);
                }
                self.send_text_to_port(detected_port, text)
                    .with_context(|| {
                        format!(
                            "{first_err}; also failed after retrying detected NEO API port {detected_port}"
                        )
                    })?;
                self.detected_port = Some(detected_port);
                Ok(())
            }
        }
    }
}

#[cfg(windows)]
pub fn detect_neo_http_port() -> Option<u16> {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let mut command = Command::new("reg");
    command
        .args([
            "query",
            r"HKCU\Software\YukarinetteConnectorNeo",
            "/v",
            "HTTP",
        ])
        .creation_flags(CREATE_NO_WINDOW);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_reg_dword)
}

#[cfg(not(windows))]
pub fn detect_neo_http_port() -> Option<u16> {
    None
}

#[cfg(windows)]
fn parse_reg_dword(line: &str) -> Option<u16> {
    let mut parts = line.split_whitespace();
    if !parts.any(|part| part == "HTTP") {
        return None;
    }
    let value = parts.find(|part| part.starts_with("0x"))?;
    u16::from_str_radix(value.trim_start_matches("0x"), 16).ok()
}

#[cfg(test)]
mod tests {
    use super::{NeoHttpTextTransport, TextTransport};
    use std::io::{Read, Write};

    #[test]
    fn neo_http_transport_posts_text_to_input_api() {
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut buffer = [0_u8; 2048];
            let len = stream.read(&mut buffer).unwrap();
            let request = String::from_utf8_lossy(&buffer[..len]).to_string();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Length: 2\r\n\
                      Connection: close\r\n\
                      \r\n\
                      ok",
                )
                .unwrap();
            request
        });

        let mut transport = NeoHttpTextTransport::localhost(port);
        transport.send_text("こんにちは").unwrap();
        let request = handle.join().unwrap();

        assert!(request.starts_with(
            "GET /api/input?text=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF HTTP/1.1\r\n"
        ));
    }

    #[test]
    fn neo_http_transport_accepts_keep_alive_response() {
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut buffer = [0_u8; 2048];
            let _ = stream.read(&mut buffer).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Length: 2\r\n\
                      Connection: keep-alive\r\n\
                      \r\n\
                      ok",
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(200));
        });

        let mut transport = NeoHttpTextTransport::localhost(port);
        transport.send_text("こんにちは").unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn neo_http_transport_accepts_empty_response_after_request() {
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut buffer = [0_u8; 2048];
            let _ = stream.read(&mut buffer).unwrap();
        });

        let mut transport = NeoHttpTextTransport::localhost(port);
        transport.send_text("こんにちは").unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn neo_http_transport_rejects_response_timeout() {
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = server.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut buffer = [0_u8; 2048];
            let _ = stream.read(&mut buffer).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(300));
        });

        let mut transport = NeoHttpTextTransport::localhost(port);
        transport.timeout = std::time::Duration::from_millis(100);
        let result = transport.send_text_to_port(port, "こんにちは");

        assert!(result.is_err());
        handle.join().unwrap();
    }

    #[test]
    fn percent_encode_query_component_encodes_utf8_and_reserved_chars() {
        assert_eq!(
            super::percent_encode_query_component("a b+c=こんにちは"),
            "a%20b%2Bc%3D%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF"
        );
    }

    #[cfg(windows)]
    #[test]
    fn parse_reg_dword_reads_http_port() {
        let line = "    HTTP    REG_DWORD    0x3ca0";

        assert_eq!(super::parse_reg_dword(line), Some(15520));
    }
}
