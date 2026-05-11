use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let started_at = Instant::now();
    println!(
        "mock_recognition_replay_start port={} neo_port={} send_neo_text={} http_client={} count={} asr_text_chars={} local_time={}",
        args.port,
        args.neo_port,
        args.send_neo_text,
        args.http_client.as_str(),
        args.count,
        args.asr_text.chars().count(),
        local_time()
    );

    for index in 1..=args.count {
        if index > 1 && args.interval_ms > 0 {
            thread::sleep(Duration::from_millis(args.interval_ms));
        }
        let audio = fake_audio(index);
        let recognized_text = mock_transcribe(&audio, args.asr_text_for(index), started_at, index);
        let recognition_id = format!("mock-recognition-{index}");
        if args.send_neo_text {
            send_neo_text(args.neo_port, &recognized_text, started_at, index)?;
        }
        let translated_text = if args.skip_translate {
            recognized_text.clone()
        } else {
            send_translate(
                args.port,
                args.http_client,
                &recognition_id,
                &args.target_lang,
                &recognized_text,
            )?
        };
        println!(
            "mock_replay_translation_ready index={} elapsed_ms={} id={} text_chars={}",
            index,
            started_at.elapsed().as_millis(),
            recognition_id,
            translated_text.chars().count()
        );
        let speech_id = format!("speech-{recognition_id}|{}-mock", args.target_lang);
        if args.blocking_speech {
            send_speech(
                args.port,
                args.http_client,
                &speech_id,
                &translated_text,
                &args.talker,
                started_at,
                index,
            )?;
        } else {
            let talker = args.talker.clone();
            let http_client = args.http_client;
            thread::spawn(move || {
                if let Err(err) = send_speech(
                    args.port,
                    http_client,
                    &speech_id,
                    &translated_text,
                    &talker,
                    started_at,
                    index,
                ) {
                    eprintln!(
                        "mock_replay_speech_error index={} elapsed_ms={} error={err:#}",
                        index,
                        started_at.elapsed().as_millis()
                    );
                }
            });
        }
    }
    if !args.blocking_speech {
        thread::sleep(Duration::from_millis(args.wait_after_ms));
    }
    Ok(())
}

struct Args {
    port: u16,
    neo_port: u16,
    send_neo_text: bool,
    http_client: HttpClientMode,
    count: usize,
    asr_text: String,
    target_lang: String,
    talker: String,
    blocking_speech: bool,
    wait_after_ms: u64,
    interval_ms: u64,
    speech_texts: Vec<String>,
    skip_translate: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            port: 18080,
            neo_port: 15520,
            send_neo_text: false,
            http_client: HttpClientMode::Raw,
            count: 4,
            asr_text: "モック音声認識です".to_string(),
            target_lang: "en_US".to_string(),
            talker: "Microsoft Zira Desktop/SAPI5".to_string(),
            blocking_speech: false,
            wait_after_ms: 7000,
            interval_ms: 0,
            speech_texts: Vec::new(),
            skip_translate: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--port" => parsed.port = next_value(&mut args, "--port")?.parse()?,
                "--neo-port" => parsed.neo_port = next_value(&mut args, "--neo-port")?.parse()?,
                "--send-neo-text" => parsed.send_neo_text = true,
                "--http-client" => {
                    parsed.http_client =
                        HttpClientMode::parse(&next_value(&mut args, "--http-client")?)?;
                }
                "--count" => parsed.count = next_value(&mut args, "--count")?.parse()?,
                "--asr-text" => parsed.asr_text = next_value(&mut args, "--asr-text")?,
                "--target-lang" => parsed.target_lang = next_value(&mut args, "--target-lang")?,
                "--talker" => parsed.talker = next_value(&mut args, "--talker")?,
                "--blocking-speech" => parsed.blocking_speech = true,
                "--wait-after-ms" => {
                    parsed.wait_after_ms = next_value(&mut args, "--wait-after-ms")?.parse()?;
                }
                "--interval-ms" => {
                    parsed.interval_ms = next_value(&mut args, "--interval-ms")?.parse()?;
                }
                "--speech-texts" => {
                    parsed.speech_texts = next_value(&mut args, "--speech-texts")?
                        .split('|')
                        .map(str::to_string)
                        .collect();
                }
                "--skip-translate" => parsed.skip_translate = true,
                _ => bail!("unknown argument: {arg}"),
            }
        }
        Ok(parsed)
    }

    fn asr_text_for(&self, index: usize) -> &str {
        self.speech_texts
            .get(index.saturating_sub(1))
            .map_or(&self.asr_text, String::as_str)
    }
}

#[derive(Clone, Copy)]
enum HttpClientMode {
    Raw,
    Reqwest,
}

impl HttpClientMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "reqwest" => Ok(Self::Reqwest),
            _ => bail!("--http-client must be raw or reqwest, got {value}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Reqwest => "reqwest",
        }
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{name} requires a value"))
}

fn fake_audio(index: usize) -> Vec<f32> {
    (0..16_000)
        .map(|sample| {
            let phase = u16::try_from((sample + index) % 64).expect("phase modulo 64 fits in u16");
            (f32::from(phase) / 64.0) * 0.2 - 0.1
        })
        .collect()
}

fn mock_transcribe(audio: &[f32], text: &str, started_at: Instant, index: usize) -> String {
    println!(
        "mock_asr_receive index={} elapsed_ms={} audio_samples={} text_chars={}",
        index,
        started_at.elapsed().as_millis(),
        audio.len(),
        text.chars().count()
    );
    text.to_string()
}

fn send_translate(
    port: u16,
    http_client: HttpClientMode,
    id: &str,
    target_lang: &str,
    text: &str,
) -> Result<String> {
    let body = json!({
        "operation": "translate",
        "params": [{
            "id": id,
            "lang": target_lang,
            "text": text
        }]
    });
    let response = post_json(port, http_client, Duration::from_secs(2), &body)?;
    let status = response
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    if status != "success" {
        bail!("translate returned status {status}: {response}");
    }
    response
        .get("text")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .context("translate response did not contain text")
}

fn send_neo_text(port: u16, text: &str, started_at: Instant, index: usize) -> Result<()> {
    println!(
        "mock_replay_neo_send_start index={} elapsed_ms={} text_chars={}",
        index,
        started_at.elapsed().as_millis(),
        text.chars().count()
    );
    let send_started_at = Instant::now();
    let endpoint = format!("127.0.0.1:{port}");
    let address = endpoint.parse::<SocketAddr>()?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .with_context(|| format!("failed to connect to NEO text input API at {endpoint}"))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let encoded_text = percent_encode_query_component(text);
    write!(
        stream,
        "GET /api/input?text={encoded_text} HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\n\r\n"
    )?;
    let _ = stream.shutdown(Shutdown::Write);
    println!(
        "mock_replay_neo_send_done index={} elapsed_ms={} response_wait_ms=0 send_ms={}",
        index,
        started_at.elapsed().as_millis(),
        send_started_at.elapsed().as_millis()
    );
    Ok(())
}

fn send_speech(
    port: u16,
    http_client: HttpClientMode,
    id: &str,
    text: &str,
    talker: &str,
    started_at: Instant,
    index: usize,
) -> Result<()> {
    println!(
        "mock_replay_speech_send_start index={} elapsed_ms={} id={} text_chars={}",
        index,
        started_at.elapsed().as_millis(),
        id,
        text.chars().count()
    );
    let send_started_at = Instant::now();
    let body = json!({
        "operation": "speech",
        "params": [{
            "id": id,
            "text": text,
            "talker": talker,
            "volume": 1.0
        }]
    });
    let response = post_json(port, http_client, Duration::from_secs(90), &body)?;
    println!(
        "mock_replay_speech_response index={} elapsed_ms={} response_ms={} id={} body={}",
        index,
        started_at.elapsed().as_millis(),
        send_started_at.elapsed().as_millis(),
        id,
        response
    );
    Ok(())
}

fn post_json(
    port: u16,
    http_client: HttpClientMode,
    timeout: Duration,
    body: &Value,
) -> Result<Value> {
    match http_client {
        HttpClientMode::Raw => post_json_raw(port, body),
        HttpClientMode::Reqwest => post_json_reqwest(port, timeout, body),
    }
}

fn post_json_reqwest(port: u16, timeout: Duration, body: &Value) -> Result<Value> {
    let url = format!("http://127.0.0.1:{port}/");
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build reqwest client")?;
    let response = client
        .post(&url)
        .json(body)
        .send()
        .with_context(|| format!("failed to send YNC command with reqwest: {url}"))?;
    let status = response.status();
    let body = response
        .text()
        .context("failed to read YNC response body")?;
    if body.trim().is_empty() {
        bail!("YNC response body is empty; status={status}");
    }
    if !status.is_success() {
        bail!("YNC returned HTTP {status}; body={body:?}");
    }
    serde_json::from_str(&body).with_context(|| format!("YNC response body is not JSON: {body:?}"))
}

fn post_json_raw(port: u16, body: &Value) -> Result<Value> {
    let endpoint = format!("127.0.0.1:{port}");
    let address = endpoint.parse::<SocketAddr>()?;
    let body = body.to_string();
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .with_context(|| format!("failed to connect to mock YNC at {endpoint}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    write!(
        stream,
        "POST / HTTP/1.1\r\nHost: {endpoint}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    let _ = stream.shutdown(Shutdown::Write);
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .context("HTTP response did not contain a header terminator")?;
    let headers = String::from_utf8_lossy(&response[..header_end]);
    let body = &response[header_end + 4..];
    serde_json::from_slice(body).with_context(|| {
        format!(
            "HTTP response body is not JSON; headers={headers:?} body={:?}",
            String::from_utf8_lossy(body)
        )
    })
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

fn local_time() -> String {
    chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S%.3f")
        .to_string()
}
