use std::{sync::mpsc, thread, time::Duration};

use serde::Serialize;

use crate::{config::DeveloperConnectionMode, delivery::RecognizedTextOutput};

use super::{DispatchContext, RecognizedTextSink};

pub(crate) static SINK: DeveloperHttpSink = DeveloperHttpSink;

pub(crate) struct DeveloperHttpSink;

#[derive(Debug)]
struct DeveloperHttpRequest {
    url: String,
    body: DeveloperRecognitionEvent,
}

#[derive(Debug, Clone, Serialize)]
struct DeveloperRecognitionEvent {
    version: u8,
    #[serde(rename = "type")]
    event_type: &'static str,
    id: String,
    text: String,
    turn_session_id: u64,
    turn_id: u64,
    revision: u64,
    output_sequence: u64,
    segment_id: u64,
    previous_segment_id: Option<u64>,
    source_asr_model: String,
    source_language: String,
    detected_language: Option<String>,
    recognized_at_ms: u64,
    elapsed_ms: u128,
    audio_duration_ms: Option<u64>,
}

impl RecognizedTextSink for DeveloperHttpSink {
    fn name(&self) -> &'static str {
        "developer_http"
    }

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
        if !ctx.config.streaming_recognition.enabled
            || ctx.config.streaming_recognition.mode != DeveloperConnectionMode::Http
            || !ctx
                .config
                .developer_http_enabled_for_source(output.meta.source().identity.source_id.as_str())
        {
            return;
        }
        let request = DeveloperHttpRequest {
            url: ctx.config.streaming_recognition.http_url.clone(),
            body: build_event(ctx, output),
        };
        if let Err(error) = delivery_queue().send(request) {
            log::warn!("Failed to queue developer HTTP event: {error}");
        }
    }
}

fn build_event(
    ctx: &DispatchContext<'_>,
    output: &RecognizedTextOutput,
) -> DeveloperRecognitionEvent {
    let source = output.meta.source();
    DeveloperRecognitionEvent {
        version: 1,
        event_type: if output.meta.is_final() {
            "turn.final"
        } else {
            "turn.partial"
        },
        id: output.meta.id.clone(),
        text: output.text.clone(),
        turn_session_id: source.turn_session_id,
        turn_id: source.turn_id,
        revision: source.turn_revision,
        output_sequence: source.output_sequence,
        segment_id: source.segment_id,
        previous_segment_id: source.previous_segment_id,
        source_asr_model: serde_json::to_value(output.source_asr_model)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("{:?}", output.source_asr_model)),
        source_language: serde_json::to_value(output.source_language)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("{:?}", output.source_language)),
        detected_language: output.detected_language.clone(),
        recognized_at_ms: ctx.recognized_at_millis,
        elapsed_ms: ctx.elapsed_millis,
        audio_duration_ms: output
            .meta
            .is_final()
            .then(|| audio_duration_millis(ctx.audio_seconds)),
    }
}

fn audio_duration_millis(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    u64::try_from(Duration::from_secs_f64(seconds).as_millis()).unwrap_or(u64::MAX)
}

fn delivery_queue() -> &'static mpsc::Sender<DeveloperHttpRequest> {
    use std::sync::OnceLock;

    static QUEUE: OnceLock<mpsc::Sender<DeveloperHttpRequest>> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("parapper-developer-http-delivery".to_string())
            .spawn(move || run_delivery_queue(receiver))
            .expect("failed to spawn developer HTTP delivery worker");
        sender
    })
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "the worker owns the receiver for its full lifetime"
)]
fn run_delivery_queue(receiver: mpsc::Receiver<DeveloperHttpRequest>) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("failed to build developer HTTP client");
    while let Ok(request) = receiver.recv() {
        let result = client.post(&request.url).json(&request.body).send();
        match result {
            Ok(response) if response.status().is_success() => {}
            Ok(response) => log::warn!(
                "Developer HTTP endpoint returned status {} for {}",
                response.status(),
                request.url
            ),
            Err(error) => log::warn!(
                "Failed to send developer HTTP event to {}: {error}",
                request.url
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use serde_json::json;

    use crate::connect::test_support::{TimedMockHttpServer, json_response};

    use super::{DeveloperHttpRequest, DeveloperRecognitionEvent, run_delivery_queue};

    fn event(id: &str, text: &str) -> DeveloperRecognitionEvent {
        DeveloperRecognitionEvent {
            version: 1,
            event_type: "turn.final",
            id: id.to_owned(),
            text: text.to_owned(),
            turn_session_id: 7,
            turn_id: 3,
            revision: 2,
            output_sequence: 4,
            segment_id: 8,
            previous_segment_id: Some(7),
            source_asr_model: "reazonspeech_k2_v2".to_owned(),
            source_language: "japanese".to_owned(),
            detected_language: None,
            recognized_at_ms: 1_000,
            elapsed_ms: 96,
            audio_duration_ms: Some(1_280),
        }
    }

    #[test]
    fn developer_http_turn_event_preserves_version_identity_ordering_and_final_metadata() {
        let event = event("turn-3", "こんにちは。");

        assert_eq!(
            serde_json::to_value(event).unwrap(),
            json!({
                "version": 1,
                "type": "turn.final",
                "id": "turn-3",
                "text": "こんにちは。",
                "turn_session_id": 7,
                "turn_id": 3,
                "revision": 2,
                "output_sequence": 4,
                "segment_id": 8,
                "previous_segment_id": 7,
                "source_asr_model": "reazonspeech_k2_v2",
                "source_language": "japanese",
                "detected_language": null,
                "recognized_at_ms": 1_000,
                "elapsed_ms": 96,
                "audio_duration_ms": 1_280,
            })
        );
    }

    #[test]
    fn developer_http_worker_posts_each_enabled_profile_event_once_to_the_configured_url() {
        let server = TimedMockHttpServer::start_until_idle(Duration::from_millis(100), |_, _| {
            json_response("{}")
        });
        let url = format!("http://127.0.0.1:{}/events", server.port());
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || run_delivery_queue(receiver));

        for (id, text) in [
            ("profile-a-turn", "Aの発話。"),
            ("profile-b-turn", "Bの発話。"),
        ] {
            sender
                .send(DeveloperHttpRequest {
                    url: url.clone(),
                    body: event(id, text),
                })
                .expect("Developer HTTP request should queue");
        }
        drop(sender);
        worker.join().expect("Developer HTTP worker should stop");

        let requests = [server.recv_request(), server.recv_request()];
        assert_eq!(
            requests.map(|request| {
                let body = request
                    .raw
                    .split("\r\n\r\n")
                    .nth(1)
                    .expect("Developer HTTP request should have a body");
                let value: serde_json::Value =
                    serde_json::from_str(body).expect("request body should be JSON");
                (value["id"].clone(), value["text"].clone())
            }),
            [
                (json!("profile-a-turn"), json!("Aの発話。")),
                (json!("profile-b-turn"), json!("Bの発話。")),
            ]
        );
        assert!(
            server
                .try_recv_request(Duration::from_millis(150))
                .is_none(),
            "each profile event must be posted exactly once"
        );
        server.join();
    }
}
