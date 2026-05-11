#![allow(deprecated)]

use std::{
    io::Write,
    net::{Shutdown, SocketAddr, TcpStream},
    sync::mpsc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use cpal::{
    Device, Sample, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use dasp_sample::ToSample;
use serde_json::json;

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = find_input_device(&args.device_name)?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let config = device.default_input_config()?.config();
    let sample_format = device.default_input_config()?.sample_format();
    println!(
        "capture_device=\"{}\" channels={} sample_rate={} sample_format={sample_format:?}",
        device_name, config.channels, config.sample_rate
    );

    let (sender, receiver) = mpsc::channel::<(Instant, f32)>();
    let stream = build_peak_stream(&device, &config, sample_format, sender)?;
    stream.play()?;

    if args.monitor_ms > 0 {
        monitor_audio(&receiver, args.monitor_ms, args.threshold);
        return Ok(());
    }

    std::thread::sleep(Duration::from_millis(args.pre_roll_ms));
    drain_peaks(&receiver);
    if let Some(neo_text) = &args.neo_text {
        send_neo_text(args.neo_port, neo_text)?;
        println!("neo_text_written");
    }
    if let Some(translate_text) = &args.translate_text {
        send_translate(args.port, &args.target_lang, translate_text)?;
        println!("translate_completed");
    }
    let sent_at = Instant::now();
    if args.wait_response {
        send_speech_and_wait_response(&args, sent_at)?;
    } else {
        send_speech_without_waiting_response(&args)?;
        println!("speech_post_written_ms=0");
    }

    let deadline = sent_at + Duration::from_millis(args.timeout_ms);
    let mut max_peak = 0.0_f32;
    while Instant::now() < deadline {
        if let Ok((captured_at, peak)) = receiver.recv_timeout(Duration::from_millis(50)) {
            max_peak = max_peak.max(peak);
            if peak >= args.threshold {
                println!(
                    "audio_detected_ms={} peak={peak:.6} max_peak={max_peak:.6}",
                    captured_at.duration_since(sent_at).as_millis()
                );
                return Ok(());
            }
        }
    }

    println!(
        "audio_not_detected timeout_ms={} max_peak={max_peak:.6}",
        args.timeout_ms
    );
    Ok(())
}

struct Args {
    port: u16,
    neo_port: u16,
    neo_text: Option<String>,
    translate_text: Option<String>,
    target_lang: String,
    device_name: String,
    text: String,
    talker: String,
    id: String,
    threshold: f32,
    timeout_ms: u64,
    pre_roll_ms: u64,
    monitor_ms: u64,
    wait_response: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            port: 8080,
            neo_port: 15520,
            neo_text: None,
            translate_text: None,
            target_lang: "en_US".to_string(),
            device_name: "CABLE Output".to_string(),
            text: "test".to_string(),
            talker: "Microsoft Zira Desktop/SAPI5".to_string(),
            id: "codex-audio-latency".to_string(),
            threshold: 0.002,
            timeout_ms: 15_000,
            pre_roll_ms: 500,
            monitor_ms: 0,
            wait_response: false,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--port" => parsed.port = next_value(&mut args, "--port")?.parse()?,
                "--neo-port" => parsed.neo_port = next_value(&mut args, "--neo-port")?.parse()?,
                "--neo-text" => parsed.neo_text = Some(next_value(&mut args, "--neo-text")?),
                "--translate-text" => {
                    parsed.translate_text = Some(next_value(&mut args, "--translate-text")?);
                }
                "--target-lang" => parsed.target_lang = next_value(&mut args, "--target-lang")?,
                "--device" => parsed.device_name = next_value(&mut args, "--device")?,
                "--text" => parsed.text = next_value(&mut args, "--text")?,
                "--talker" => parsed.talker = next_value(&mut args, "--talker")?,
                "--id" => parsed.id = next_value(&mut args, "--id")?,
                "--threshold" => {
                    parsed.threshold = next_value(&mut args, "--threshold")?.parse()?;
                }
                "--timeout-ms" => {
                    parsed.timeout_ms = next_value(&mut args, "--timeout-ms")?.parse()?;
                }
                "--pre-roll-ms" => {
                    parsed.pre_roll_ms = next_value(&mut args, "--pre-roll-ms")?.parse()?;
                }
                "--monitor-ms" => {
                    parsed.monitor_ms = next_value(&mut args, "--monitor-ms")?.parse()?;
                }
                "--wait-response" => parsed.wait_response = true,
                "--list-devices" => {
                    list_input_devices();
                    std::process::exit(0);
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

fn monitor_audio(receiver: &mpsc::Receiver<(Instant, f32)>, monitor_ms: u64, threshold: f32) {
    let started_at = Instant::now();
    let deadline = started_at + Duration::from_millis(monitor_ms);
    let mut active = false;
    let mut max_peak = 0.0_f32;
    println!(
        "monitor_start local_time={}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
    );
    while Instant::now() < deadline {
        if let Ok((captured_at, peak)) = receiver.recv_timeout(Duration::from_millis(50)) {
            max_peak = max_peak.max(peak);
            if !active && peak >= threshold {
                active = true;
                println!(
                    "audio_onset elapsed_ms={} local_time={} peak={peak:.6}",
                    captured_at.duration_since(started_at).as_millis(),
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
                );
            } else if active && peak < threshold * 0.5 {
                active = false;
                println!(
                    "audio_below elapsed_ms={} local_time={} peak={peak:.6}",
                    captured_at.duration_since(started_at).as_millis(),
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
                );
            }
        }
    }
    println!(
        "monitor_done local_time={} max_peak={max_peak:.6}",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
    );
}

fn list_input_devices() {
    for host_id in cpal::available_hosts() {
        let Ok(host) = cpal::host_from_id(host_id) else {
            continue;
        };
        let Ok(devices) = host.input_devices() else {
            continue;
        };
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "unknown".to_string());
            let config = device.default_input_config().ok();
            println!("{host_id:?}: {name} {config:?}");
        }
    }
}

fn find_input_device(name_part: &str) -> Result<Device> {
    for host_id in cpal::available_hosts() {
        let host = cpal::host_from_id(host_id)?;
        let Ok(devices) = host.input_devices() else {
            continue;
        };
        for device in devices {
            let name = device.name().unwrap_or_default();
            if name
                .to_ascii_lowercase()
                .contains(&name_part.to_ascii_lowercase())
            {
                return Ok(device);
            }
        }
    }
    bail!("input device containing \"{name_part}\" was not found")
}

fn build_peak_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: mpsc::Sender<(Instant, f32)>,
) -> Result<cpal::Stream> {
    match sample_format {
        SampleFormat::F32 => build_peak_stream_inner::<f32>(device, config, sender),
        SampleFormat::I16 => build_peak_stream_inner::<i16>(device, config, sender),
        SampleFormat::U16 => build_peak_stream_inner::<u16>(device, config, sender),
        SampleFormat::I8 => build_peak_stream_inner::<i8>(device, config, sender),
        SampleFormat::U8 => build_peak_stream_inner::<u8>(device, config, sender),
        SampleFormat::I32 => build_peak_stream_inner::<i32>(device, config, sender),
        SampleFormat::U32 => build_peak_stream_inner::<u32>(device, config, sender),
        SampleFormat::F64 => build_peak_stream_inner::<f64>(device, config, sender),
        _ => bail!("unsupported sample format: {sample_format:?}"),
    }
}

fn build_peak_stream_inner<T>(
    device: &Device,
    config: &StreamConfig,
    sender: mpsc::Sender<(Instant, f32)>,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + ToSample<f32>,
{
    let channels = usize::from(config.channels);
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                let peak = data
                    .chunks(channels.max(1))
                    .flat_map(|frame| frame.iter())
                    .fold(0.0_f32, |acc, sample| acc.max(sample.to_sample().abs()));
                let _ = sender.send((Instant::now(), peak));
            },
            |err| eprintln!("capture_error={err}"),
            None,
        )
        .context("failed to build input stream")
}

fn drain_peaks(receiver: &mpsc::Receiver<(Instant, f32)>) {
    while receiver.try_recv().is_ok() {}
}

fn speech_body(args: &Args) -> String {
    json!({
        "operation": "speech",
        "params": [{
            "id": args.id,
            "text": args.text,
            "talker": args.talker,
            "volume": 1.0
        }]
    })
    .to_string()
}

fn send_speech_without_waiting_response(args: &Args) -> Result<()> {
    let endpoint = format!("127.0.0.1:{}", args.port);
    let address = endpoint.parse::<SocketAddr>()?;
    let body = speech_body(args);
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let headers = format!(
        "POST / HTTP/1.1\r\nHost: {endpoint}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

fn send_speech_and_wait_response(args: &Args, sent_at: Instant) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/", args.port);
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(args.timeout_ms))
        .build()?
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(speech_body(args))
        .send()?;
    let status = response.status();
    let response_body = response.text()?;
    println!(
        "speech_response_ms={} status={} body={}",
        sent_at.elapsed().as_millis(),
        status,
        response_body
    );
    if !status.is_success() {
        bail!("speech returned HTTP {status}");
    }
    Ok(())
}

fn send_translate(port: u16, target_lang: &str, text: &str) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/");
    let body = json!({
        "operation": "translate",
        "params": [{
            "id": "codex-translate-before-speech",
            "lang": target_lang,
            "text": text
        }]
    });
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?
        .post(url)
        .json(&body)
        .send()?;
    let status = response.status();
    let response_body = response.text()?;
    if !status.is_success() {
        bail!("translate returned HTTP {status}: {response_body}");
    }
    Ok(())
}

fn send_neo_text(port: u16, text: &str) -> Result<()> {
    let endpoint = format!("127.0.0.1:{port}");
    let address = endpoint.parse::<SocketAddr>()?;
    let encoded = percent_encode(text);
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let request = format!(
        "GET /api/input?text={encoded} HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

fn percent_encode(text: &str) -> String {
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
