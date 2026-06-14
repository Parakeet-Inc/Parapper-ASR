use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let listener = TcpListener::bind(("127.0.0.1", args.port))
        .with_context(|| format!("failed to bind 127.0.0.1:{}", args.port))?;
    let bound_port = listener.local_addr()?.port();
    let started_at = Instant::now();
    let sequence = Arc::new(AtomicUsize::new(0));
    println!(
        "mock_ync_start port={} translate_delay_ms={} speech_delay_ms={} local_time={}",
        bound_port,
        args.translate_delay_ms,
        args.speech_delay_ms,
        local_time()
    );

    for stream in listener.incoming() {
        let stream = stream?;
        let sequence = sequence.clone();
        let translate_delay_ms = args.translate_delay_ms;
        let speech_delay_ms = args.speech_delay_ms;
        thread::spawn(move || {
            let index = sequence.fetch_add(1, Ordering::SeqCst) + 1;
            if let Err(err) = handle_client(
                stream,
                index,
                started_at,
                translate_delay_ms,
                speech_delay_ms,
            ) {
                eprintln!(
                    "mock_ync_error index={} elapsed_ms={} error={err:#}",
                    index,
                    started_at.elapsed().as_millis()
                );
            }
        });
    }
    Ok(())
}

struct Args {
    port: u16,
    translate_delay_ms: u64,
    speech_delay_ms: u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            port: 18080,
            translate_delay_ms: 0,
            speech_delay_ms: 0,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--port" => parsed.port = next_value(&mut args, "--port")?.parse()?,
                "--translate-delay-ms" => {
                    parsed.translate_delay_ms =
                        next_value(&mut args, "--translate-delay-ms")?.parse()?;
                }
                "--speech-delay-ms" => {
                    parsed.speech_delay_ms = next_value(&mut args, "--speech-delay-ms")?.parse()?;
                }
                _ => bail!("unknown argument: {arg}"),
            }
        }
        Ok(parsed)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{name} requires a value"))
}

fn handle_client(
    mut stream: TcpStream,
    index: usize,
    started_at: Instant,
    translate_delay_ms: u64,
    speech_delay_ms: u64,
) -> Result<()> {
    let request = read_http_request(&mut stream)?;
    if !is_plugin_command_request(&request) {
        write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )?;
        bail!("unexpected plugin request path: {}", request_line(&request));
    }
    let body = request_body(&request)?;
    let value: Value = serde_json::from_slice(body).context("request body is not JSON")?;
    let operation = value
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let param = value
        .get("params")
        .and_then(Value::as_array)
        .and_then(|params| params.first())
        .unwrap_or(&Value::Null);
    let id = param
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let text = param.get("text").and_then(Value::as_str).unwrap_or("");
    let talker = param.get("talker").and_then(Value::as_str).unwrap_or("");
    let lang = param.get("lang").and_then(Value::as_str).unwrap_or("");
    println!(
        "mock_ync_receive index={} elapsed_ms={} local_time={} operation={} id={} lang={} talker={} text_chars={}",
        index,
        started_at.elapsed().as_millis(),
        local_time(),
        operation,
        id,
        lang,
        talker,
        text.chars().count()
    );

    let response_body = match operation {
        "translate" => translate_response(id, lang, translate_delay_ms),
        "translates" => translates_response(param, id, translate_delay_ms),
        "speech" => speech_response(id, text, talker, speech_delay_ms),
        "version" => version_response(id),
        "speech.getvoicelist" => voice_list_response(id),
        "speech.stop" => speech_stop_response(id),
        _ => unknown_operation_response(operation, id),
    };
    let response_text = response_body.to_string();
    let status_line = if operation == "<missing>" {
        "HTTP/1.1 400 Bad Request"
    } else {
        "HTTP/1.1 200 OK"
    };
    write!(
        stream,
        "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response_text.len(),
        response_text
    )?;
    println!(
        "mock_ync_response index={} elapsed_ms={} local_time={} operation={} id={}",
        index,
        started_at.elapsed().as_millis(),
        local_time(),
        operation,
        id
    );
    Ok(())
}

fn is_plugin_command_request(request: &[u8]) -> bool {
    request.starts_with(b"POST / HTTP/1.1\r\n")
}

fn request_line(request: &[u8]) -> String {
    let end = request
        .windows(2)
        .position(|window| window == b"\r\n")
        .unwrap_or(request.len());
    String::from_utf8_lossy(&request[..end]).to_string()
}

fn translate_response(id: &str, lang: &str, delay_ms: u64) -> Value {
    sleep_ms(delay_ms);
    json!({
        "operation": "translate",
        "status": "success",
        "id": id,
        "lang": lang,
        "text": format!("mock translated {id}")
    })
}

fn translates_response(param: &Value, id: &str, delay_ms: u64) -> Value {
    sleep_ms(delay_ms);
    let results = param
        .get("lang")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|lang| {
            json!({
                "lang": lang,
                "text": format!("mock translated {id} {lang}")
            })
        })
        .collect::<Vec<_>>();
    json!({
        "operation": "translates",
        "status": "success",
        "id": id,
        "result": results
    })
}

fn speech_response(id: &str, text: &str, talker: &str, delay_ms: u64) -> Value {
    sleep_ms(delay_ms);
    json!({
        "operation": "speech",
        "status": "sended",
        "id": id,
        "text": text,
        "talker": talker
    })
}

fn version_response(id: &str) -> Value {
    json!({
        "operation": "version",
        "status": "success",
        "id": id,
        "version": [
            {
                "System": "mock",
                "Plugin": "mock"
            }
        ]
    })
}

fn voice_list_response(id: &str) -> Value {
    json!({
        "operation": "speech.getvoicelist",
        "status": "success",
        "id": id,
        "voice": ["Microsoft Zira Desktop/SAPI5"]
    })
}

fn speech_stop_response(id: &str) -> Value {
    json!({
        "operation": "speech.stop",
        "status": "sended",
        "id": id
    })
}

fn unknown_operation_response(operation: &str, id: &str) -> Value {
    json!({
        "operation": operation,
        "status": "error",
        "id": id,
        "error": "unknown operation"
    })
}

fn read_http_request(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end = loop {
        let len = stream.read(&mut buffer)?;
        if len == 0 {
            bail!("connection closed before headers completed");
        }
        request.extend_from_slice(&buffer[..len]);
        if let Some(header_end) = find_header_end(&request) {
            break header_end;
        }
        if request.len() > 64 * 1024 {
            bail!("request headers are too large");
        }
    };
    let content_length = content_length(&request[..header_end])?;
    let total_len = header_end + 4 + content_length;
    while request.len() < total_len {
        let len = stream.read(&mut buffer)?;
        if len == 0 {
            bail!("connection closed before body completed");
        }
        request.extend_from_slice(&buffer[..len]);
    }
    Ok(request)
}

fn find_header_end(request: &[u8]) -> Option<usize> {
    request.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &[u8]) -> Result<usize> {
    let headers = String::from_utf8_lossy(headers);
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            return value
                .trim()
                .parse()
                .context("content-length is not a valid number");
        }
    }
    Ok(0)
}

fn request_body(request: &[u8]) -> Result<&[u8]> {
    let header_end = find_header_end(request).context("request has no header terminator")?;
    Ok(&request[header_end + 4..])
}

fn sleep_ms(ms: u64) {
    if ms > 0 {
        thread::sleep(Duration::from_millis(ms));
    }
}

fn local_time() -> String {
    chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S%.3f")
        .to_string()
}
