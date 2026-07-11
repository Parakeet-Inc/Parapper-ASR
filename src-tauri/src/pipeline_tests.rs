use std::{sync::mpsc, time::Duration};

use tauri::Listener;

use crate::{
    config::{
        AsrLanguage, AsrModel, LocalTranslationModel, NeoSendTiming, ParapperConfig, SpeechBackend,
        SpeechMapping, SpeechSourceKind, TranslationBackend, TranslationLanguage,
        TranslationMapping,
    },
    connect::test_support::{TimedMockHttpServer, json_response, request_id_from_plugin_request},
    delivery::{
        RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, dispatch_recognized_text,
    },
};

fn source_meta() -> RecognitionSourceMeta {
    RecognitionSourceMeta {
        turn_session_id: 1,
        turn_id: 1,
        turn_revision: 0,
        output_sequence: 1,
        segment_id: 1,
        previous_segment_id: None,
    }
}

fn recognized_output(id: &str, text: &str) -> RecognizedTextOutput {
    RecognizedTextOutput {
        phrase: Vec::new().into(),
        text: text.to_string(),
        source_asr_model: AsrModel::ReazonSpeechK2V2,
        source_language: AsrLanguage::Japanese,
        detected_language: Some("ja".to_string()),
        meta: RecognizedTextMeta::replace_turn(id.to_string(), source_meta(), true),
        elapsed_millis: 0,
    }
}

fn translation_mapping(id: &str, target_lang: &str) -> TranslationMapping {
    TranslationMapping {
        id: id.to_string(),
        source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
        backend: TranslationBackend::Ync,
        local_model: LocalTranslationModel::default(),
        source_lang: TranslationLanguage::Ja,
        target_lang: TranslationLanguage::from_code(target_lang).expect("en/ja translation target"),
    }
}

fn speech_mapping(id: &str, source_kind: SpeechSourceKind) -> SpeechMapping {
    SpeechMapping {
        id: id.to_string(),
        source_kind,
        source_asr_model: None,
        target_lang: None,
        backend: SpeechBackend::Ync,
        talker: "ずんだもん/VOICEVOX".to_string(),
        local_tts_voice: None,
        local_tts_language: None,
        local_tts_speaker_id: None,
        output_device_id: None,
        output_device_host: None,
        output_device_name: None,
        muted: false,
        volume: 0.0,
    }
}

#[test]
#[cfg(not(target_os = "macos"))]
fn recognized_text_pipeline_dispatches_translation_and_speech_sinks() {
    let builder = tauri::Builder::default();
    #[cfg(any(windows, target_os = "linux"))]
    let builder = builder.any_thread();
    let app = builder
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("test app should build");
    let handle = app.handle().clone();
    let (translated_sender, translated_receiver) = mpsc::channel::<String>();
    let _event_id = handle.listen("parapper://translated-text", move |event| {
        translated_sender
            .send(event.payload().to_string())
            .expect("translated event should be recorded");
    });

    let server = TimedMockHttpServer::start_until_idle(
        Duration::from_millis(500),
        move |request, _index| {
            let request_id = request_id_from_plugin_request(request);
            if request.contains(r#""operation":"translate""#) {
                assert!(
                    request.contains(r#""lang":"en""#),
                    "unexpected translation target: {request}"
                );
                assert!(
                    request.contains(r#""text":"翻訳して読み上げます。""#),
                    "unexpected translation source text: {request}"
                );
                let body = format!(
                    r#"{{"operation":"translate","status":"success","id":"{request_id}","lang":"en","text":"translated {request_id}"}}"#
                );
                return json_response(&body);
            }
            assert!(
                request.contains(r#""operation":"speech""#),
                "unexpected mock request: {request}"
            );
            let body = format!(
                r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"ok"}}"#
            );
            json_response(&body)
        },
    );
    let config = pipeline_test_config(server.port());
    let output = recognized_output("turn-pipeline-1", "翻訳して読み上げます。");

    dispatch_recognized_text(&handle, &config, None, &output);

    let mut translated_ids = Vec::new();
    let mut speech_ids = Vec::new();
    for _ in 0..3 {
        let received = server.recv_request();
        let request_id = request_id_from_plugin_request(&received.raw).to_string();
        if received.raw.contains(r#""operation":"translate""#) {
            translated_ids.push(request_id);
        } else {
            speech_ids.push(request_id);
        }
    }
    speech_ids.sort();

    let translated_event = translated_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("translated event should be emitted");

    assert_eq!(translated_ids, vec!["turn-pipeline-1"]);
    assert_eq!(
        speech_ids,
        vec![
            "speech-turn-pipeline-1-speech-recognition",
            "speech-turn-pipeline-1|en-speech-translation",
        ]
    );
    assert!(translated_event.contains(r#""source_recognition_id":"turn-pipeline-1""#));
    assert!(translated_event.contains(r#""target_lang":"en""#));
    assert!(translated_event.contains(r#""translated_text":"translated turn-pipeline-1""#));
    let translated_event: serde_json::Value =
        serde_json::from_str(&translated_event).expect("translated event should be JSON");
    assert_eq!(translated_event["source"]["turn_session_id"], 1);
    assert_eq!(translated_event["source"]["turn_id"], 1);
    assert_eq!(translated_event["source"]["output_sequence"], 1);
    assert!(
        server
            .try_recv_request(Duration::from_millis(500))
            .is_none(),
        "recognized text pipeline must not send extra translation or speech requests"
    );
    server.join();
}

fn pipeline_test_config(port: u16) -> ParapperConfig {
    parapper_config! {
        neo_http_enabled: false,
        translation_enabled: true,
        ync_plugin_port: port,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![translation_mapping("translate-en", "en")],
        speech_mappings: vec![
            SpeechMapping {
                ..speech_mapping("speech-recognition", SpeechSourceKind::Recognition)
            },
            SpeechMapping {
                target_lang: Some("en".to_string()),
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                ..speech_mapping("speech-translation", SpeechSourceKind::Translation)
            },
        ],
        ..ParapperConfig::default()
    }
}
