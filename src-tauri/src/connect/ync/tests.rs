use std::{
    io::ErrorKind,
    net::TcpListener,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use super::{SpeechRequest, YncPluginClient, YncTextInputTransport};
use crate::{
    config::TurnDetector,
    connect::TextTransport,
    connect::test_support::{
        MockHttpServer, TimedMockHttpServer, json_response, request_id_from_plugin_request,
        text_response,
    },
};

// YNC has two distinct HTTP surfaces:
// plugin commands are POSTed to the TransServer root path, while built-in text input uses
// /api/input. The fallback/retry tests below are external contract regressions, not cleanup
// candidates. They prevent accidentally probing alternate command paths or unrelated open ports.

#[test]
fn plugin_client_posts_translate_command() {
    let server = MockHttpServer::start(1, |_request, _index| {
        let body = r#"{"operation":"translate","status":"success","id":"id-1","lang":"ja_JP","text":"Hello."}"#;
        json_response(body)
    });

    let mut client = YncPluginClient::for_command(server.port()).unwrap();
    let response = client.translate("id-1", "en_US", "こんにちは").unwrap();
    let request = server.recv_request();
    server.join();

    assert_plugin_command_request(&request);
    assert!(request.contains(r#""operation":"translate""#));
    assert!(request.contains(r#""lang":"en_US""#));
    assert_eq!(response.text, "Hello.");
}

#[test]
fn plugin_client_does_not_fallback_to_alternate_command_path() {
    let server = MockHttpServer::start(1, |_request, _index| text_response("ok"));

    let mut client = YncPluginClient::for_command(server.port()).unwrap();
    let err = client.translate("id-1", "en_US", "こんにちは").unwrap_err();
    let request = server.recv_request();
    server.join();

    assert_plugin_command_request(&request);
    assert!(
        err.to_string()
            .contains("YNC plugin response is not valid JSON"),
        "unexpected error: {err}"
    );
}

#[test]
fn plugin_client_does_not_retry_command_paths_after_connection_failure() {
    let port = unused_local_port();
    let mut client = YncPluginClient::for_command(port).unwrap();
    let started_at = Instant::now();

    let err = client.translate("id-1", "en_US", "こんにちは").unwrap_err();
    let elapsed = started_at.elapsed();

    assert!(
        elapsed < Duration::from_millis(3_500),
        "connection failure retried command paths: elapsed={elapsed:?}, error={err}"
    );
}

#[test]
fn plugin_client_does_not_fallback_to_another_open_plugin_port() {
    let alternate_listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    alternate_listener.set_nonblocking(true).unwrap();
    let configured_port = unused_local_port();

    let mut client = YncPluginClient::for_command(configured_port).unwrap();
    let err = client.translate("id-1", "en_US", "こんにちは").unwrap_err();

    assert!(
        err.to_string()
            .contains("Failed to send YNC plugin command"),
        "unexpected error: {err}"
    );
    match alternate_listener.accept() {
        Ok((_stream, address)) => {
            panic!("client connected to an alternate open port instead of failing: {address}")
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => {}
        Err(err) => panic!("failed to inspect alternate listener: {err}"),
    }
}

#[test]
fn plugin_port_probe_uses_version_command() {
    let server = MockHttpServer::start(1, |request, _index| {
        assert_plugin_command_request(request);
        assert!(request.contains(r#""operation":"version""#));
        let body = r#"{"operation":"version","status":"success","id":"plugin-port-probe","version":[{"System":"1.959","Plugin":"1.4a"}]}"#;
        json_response(body)
    });

    let client = YncPluginClient::for_command(server.port()).unwrap();
    client.probe_plugin_port(server.port()).unwrap();
    server.join();
}

fn unused_local_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

fn assert_plugin_command_request(request: &str) {
    assert!(
        request.starts_with("POST / HTTP/1.1\r\n"),
        "plugin commands must be sent to TransServer root, got: {request}"
    );
}

#[test]
fn plugin_client_posts_speech_command() {
    let server = MockHttpServer::start(1, |_request, _index| {
        let body = r#"{"operation":"speech","status":"sended","id":"speech-1","text":"Hello."}"#;
        json_response(body)
    });

    let mut client = YncPluginClient::for_command(server.port()).unwrap();
    let response = client
        .speech(SpeechRequest {
            id: "speech-1",
            text: "Hello.",
            talker: "ずんだもん/VOICEVOX",
            volume: 1.0,
        })
        .unwrap();
    let request = server.recv_request();
    server.join();

    assert_plugin_command_request(&request);
    assert!(request.contains(r#""operation":"speech""#));
    assert!(request.contains(r#""talker":"ずんだもん/VOICEVOX""#));
    assert_eq!(response.id, "speech-1");
}

#[test]
fn plugin_client_posts_speech_to_mock_before_waiting_for_slow_response() {
    let response_delay = Duration::from_millis(250);
    let server = TimedMockHttpServer::start(1, move |request, _index| {
        let request_id = request_id_from_plugin_request(request);
        thread::sleep(response_delay);
        let body = format!(
            r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"ok"}}"#
        );
        json_response(&body)
    });
    let port = server.port();
    let started_at = Instant::now();

    let client_handle = thread::spawn(move || {
        let mut client = YncPluginClient::for_speech(port).unwrap();
        client
            .speech(SpeechRequest {
                id: "speech-slow-response",
                text: "Hello.",
                talker: "Microsoft Zira Desktop/SAPI5",
                volume: 1.0,
            })
            .unwrap()
    });

    let received = server.recv_request();
    assert_plugin_command_request(&received.raw);
    assert!(received.raw.contains(r#""operation":"speech""#));
    assert!(
        received
            .raw
            .contains(r#""talker":"Microsoft Zira Desktop/SAPI5""#)
    );
    assert!(
        received.received_at.duration_since(started_at) < Duration::from_millis(80),
        "speech POST did not reach the local mock promptly"
    );

    let response = client_handle.join().unwrap();
    assert_eq!(response.id, "speech-slow-response");
    assert!(
        started_at.elapsed() >= response_delay,
        "client did not wait for the mock speech response"
    );
    server.join();
}

#[test]
fn plugin_client_reads_voice_list() {
    let server = MockHttpServer::start(1, |_request, _index| {
        let body = r#"{"operation":"speech.getvoicelist","status":"success","id":"voice-1","voice":["ずんだもん/VOICEVOX"]}"#;
        json_response(body)
    });

    let mut client = YncPluginClient::for_command(server.port()).unwrap();
    let voices = client.voice_list("voice-1").unwrap();
    let request = server.recv_request();
    server.join();

    assert_plugin_command_request(&request);
    assert!(request.contains(r#""operation":"speech.getvoicelist""#));
    assert_eq!(voices, vec!["ずんだもん/VOICEVOX"]);
}

#[test]
fn text_input_transport_posts_text_to_input_api() {
    let server = MockHttpServer::start(1, |_request, _index| text_response("ok"));

    let mut transport = YncTextInputTransport::localhost(server.port());
    transport.send_text("こんにちは").unwrap();
    let request = server.recv_request();
    server.join();

    assert!(request.starts_with(
        "GET /api/input?text=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF HTTP/1.1\r\n"
    ));
}

#[test]
fn text_input_transport_accepts_keep_alive_response() {
    let server = MockHttpServer::start(1, |_request, _index| {
        std::thread::sleep(std::time::Duration::from_millis(200));
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok".to_string()
    });

    let mut transport = YncTextInputTransport::localhost(server.port());
    transport.send_text("こんにちは").unwrap();
    server.join();
}

#[test]
fn text_input_transport_accepts_empty_response_after_request() {
    let server = MockHttpServer::start(1, |_request, _index| String::new());

    let mut transport = YncTextInputTransport::localhost(server.port());
    transport.send_text("こんにちは").unwrap();
    server.join();
}

#[test]
fn text_input_transport_does_not_wait_for_slow_response() {
    let server = MockHttpServer::start(1, |_request, _index| {
        std::thread::sleep(std::time::Duration::from_millis(300));
        text_response("ok")
    });

    let transport = YncTextInputTransport::localhost(server.port());
    let started_at = Instant::now();
    transport
        .send_text_to_port(server.port(), "こんにちは")
        .unwrap();
    let elapsed = started_at.elapsed();

    assert!(
        elapsed < Duration::from_millis(100),
        "text input waited for the HTTP response: {elapsed:?}"
    );
    server.join();
}

#[test]
fn percent_encode_query_component_encodes_utf8_and_reserved_chars() {
    assert_eq!(
        super::text_input::percent_encode_query_component("a b+c=こんにちは"),
        "a%20b%2Bc%3D%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF"
    );
}

#[test]
fn slow_speech_does_not_block_translation_in_timing_subset() {
    let speech_delay = Duration::from_millis(240);
    let translation_delay = Duration::from_millis(20);
    let max_expected_translation_elapsed = Duration::from_millis(120);

    for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
        let recognition_delay = recognition_delay_for_turn_detector(turn_detector);
        let (port, operation_receiver, server_handle) =
            start_timing_plugin_server_for_requests(vec![
                timing_response(TimingOperation::Speech, speech_delay),
                timing_response(TimingOperation::Translate, translation_delay),
            ]);

        let speech_handle = thread::spawn(move || {
            thread::sleep(recognition_delay);
            let mut speech_client = YncPluginClient::for_speech(port).unwrap();
            speech_client
                .speech(SpeechRequest {
                    id: "speech-slow",
                    text: "Hello.",
                    talker: "ずんだもん/VOICEVOX",
                    volume: 1.0,
                })
                .unwrap();
        });

        assert_eq!(
            operation_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            "speech:speech-slow",
            "turn_detector={turn_detector:?}"
        );

        thread::sleep(recognition_delay);
        let started_at = Instant::now();
        let mut translation_client = YncPluginClient::for_command(port).unwrap();
        let response = translation_client
            .translate("translation-1", "en_US", "こんにちは")
            .unwrap();
        let translation_elapsed = started_at.elapsed();

        assert_eq!(response.text, "Hello.");
        assert_eq!(
            operation_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            "translate:translation-1",
            "turn_detector={turn_detector:?}"
        );
        assert!(
            translation_elapsed < max_expected_translation_elapsed,
            "translation waited behind speech: turn_detector={turn_detector:?}, elapsed={translation_elapsed:?}"
        );

        speech_handle.join().unwrap();
        server_handle.join();
    }
}

#[test]
fn translation_commands_are_not_serialized_in_timing_subset() {
    let command_delay = Duration::from_millis(180);
    let max_expected_elapsed = Duration::from_millis(300);

    for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
        let recognition_delay = recognition_delay_for_turn_detector(turn_detector);
        let (port, operation_receiver, server_handle) =
            start_timing_plugin_server_for_requests(vec![
                timing_response(TimingOperation::Translate, command_delay),
                timing_response(TimingOperation::Translate, command_delay),
            ]);
        let started_at = Instant::now();

        let first = spawn_translate_command(port, "parallel-1", recognition_delay);
        let second = spawn_translate_command(port, "parallel-2", recognition_delay);

        first.join().unwrap();
        second.join().unwrap();
        let elapsed = started_at.elapsed();
        let mut operations = vec![
            operation_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            operation_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
        ];
        operations.sort();

        assert_eq!(
            operations,
            vec![
                "translate:parallel-1".to_string(),
                "translate:parallel-2".to_string()
            ],
            "turn_detector={turn_detector:?}"
        );
        assert!(
            elapsed < max_expected_elapsed,
            "translation commands were serialized: turn_detector={turn_detector:?}, elapsed={elapsed:?}"
        );

        server_handle.join();
    }
}

fn recognition_delay_for_turn_detector(turn_detector: TurnDetector) -> Duration {
    match turn_detector {
        TurnDetector::Simple => Duration::from_millis(10),
        TurnDetector::Namo => Duration::from_millis(15),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimingOperation {
    Translate,
    Speech,
}

impl TimingOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Translate => "translate",
            Self::Speech => "speech",
        }
    }
}

#[derive(Clone, Copy)]
struct TimingResponse {
    operation: TimingOperation,
    delay: Duration,
}

fn timing_response(operation: TimingOperation, delay: Duration) -> TimingResponse {
    TimingResponse { operation, delay }
}

fn spawn_translate_command(
    port: u16,
    id: &'static str,
    recognition_delay: Duration,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        thread::sleep(recognition_delay);
        let mut client = YncPluginClient::for_command(port).unwrap();
        client.translate(id, "en_US", "こんにちは").unwrap();
    })
}

fn start_timing_plugin_server_for_requests(
    responses: Vec<TimingResponse>,
) -> (u16, mpsc::Receiver<String>, MockHttpServer) {
    let (operation_sender, operation_receiver) = mpsc::channel();
    let server = MockHttpServer::start(responses.len(), move |request, index| {
        handle_timing_plugin_request(request, &operation_sender, responses[index])
    });
    (server.port(), operation_receiver, server)
}

fn handle_timing_plugin_request(
    request: &str,
    operation_sender: &mpsc::Sender<String>,
    response: TimingResponse,
) -> String {
    assert_plugin_command_request(request);
    let operation = response.operation.as_str();
    assert!(
        request.contains(&format!(r#""operation":"{operation}""#)),
        "unexpected plugin request: {request}"
    );
    let request_id = request_id_from_plugin_request(request);
    operation_sender
        .send(format!("{operation}:{request_id}"))
        .unwrap();
    thread::sleep(response.delay);
    let body = match response.operation {
        TimingOperation::Translate => format!(
            r#"{{"operation":"translate","status":"success","id":"{request_id}","lang":"ja_JP","text":"Hello."}}"#
        ),
        TimingOperation::Speech => format!(
            r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"Hello."}}"#
        ),
    };
    json_response(&body)
}
