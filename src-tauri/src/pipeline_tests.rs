use std::{sync::mpsc, time::Duration};

use tauri::Listener;

use crate::{
    config::{
        AsrLanguage, AsrModel, DeliveryRouteSnapshot, LocalTranslationModel, NeoSendTiming,
        ParapperConfig, SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationBackend,
        TranslationLanguage, TranslationMapping,
    },
    connect::test_support::{TimedMockHttpServer, json_response, request_id_from_plugin_request},
    delivery::{
        RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, dispatch_recognized_text,
    },
};

fn source_meta() -> RecognitionSourceMeta {
    RecognitionSourceMeta {
        identity: parapper_stt_engine::SourceIdentitySnapshot::new(
            "channel-1".into(),
            "Speaker 1".to_string(),
            "interface-1".to_string(),
            Some(0),
        ),
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
fn recognized_text_pipeline_dispatches_only_profile_selected_translation_and_speech_sinks() {
    let builder = tauri::Builder::default();
    #[cfg(any(windows, target_os = "linux"))]
    let builder = builder.any_thread();
    let app = builder
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("test app should build");
    let handle = app.handle().clone();
    let (recognized_sender, recognized_receiver) = mpsc::channel::<String>();
    let _recognized_event_id = handle.listen("parapper://recognized-text", move |event| {
        recognized_sender
            .send(event.payload().to_string())
            .expect("recognized event should be recorded");
    });
    let (translated_sender, translated_receiver) = mpsc::channel::<String>();
    let _event_id = handle.listen("parapper://translated-text", move |event| {
        translated_sender
            .send(event.payload().to_string())
            .expect("translated event should be recorded");
    });
    let (speech_sender, speech_receiver) = mpsc::channel::<String>();
    let _speech_event_id = handle.listen("parapper://speech-request", move |event| {
        speech_sender
            .send(event.payload().to_string())
            .expect("speech event should be recorded");
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
    let (config, route) = profile_scoped_pipeline_config(server.port());
    let output = recognized_output("turn-pipeline-1", "翻訳して読み上げます。");

    dispatch_recognized_text(&handle, &config, None, &output, &route);

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

    assert_eq!(translated_ids, vec!["turn-pipeline-1"]);
    assert_eq!(
        speech_ids,
        vec![
            "speech-turn-pipeline-1-speech-recognition",
            "speech-turn-pipeline-1|en-speech-translation",
        ]
    );
    let translated_event = recv_json_event(&translated_receiver, "translated");
    let recognized_event = recv_json_event(&recognized_receiver, "recognized");
    let speech_events = (0..2)
        .map(|_| recv_json_event(&speech_receiver, "speech"))
        .collect::<Vec<_>>();
    assert_eq!(translated_event["source_recognition_id"], "turn-pipeline-1");
    assert_eq!(translated_event["target_lang"], "en");
    assert_eq!(
        translated_event["translated_text"],
        "translated turn-pipeline-1"
    );
    assert_source_metadata(
        [&recognized_event, &translated_event]
            .into_iter()
            .chain(speech_events.iter()),
    );
    assert!(
        server
            .try_recv_request(Duration::from_millis(500))
            .is_none(),
        "recognized text pipeline must not send extra translation or speech requests"
    );
    server.join();
}

fn recv_json_event(receiver: &mpsc::Receiver<String>, event_name: &str) -> serde_json::Value {
    let payload = receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap_or_else(|_| panic!("{event_name} event should be emitted"));
    serde_json::from_str(&payload).unwrap_or_else(|_| panic!("{event_name} event should be JSON"))
}

fn assert_source_metadata<'a>(events: impl IntoIterator<Item = &'a serde_json::Value>) {
    let expected_source = serde_json::json!({
        "identity": {
            "source_id": "channel-1",
            "speaker_label": "Speaker 1",
            "capture_endpoint_id": "interface-1",
            "channel_index": 0,
        },
        "turn_session_id": 1,
        "turn_id": 1,
        "turn_revision": 0,
        "output_sequence": 1,
        "segment_id": 1,
        "previous_segment_id": null,
    });
    for event in events {
        assert_eq!(event["source"], expected_source);
    }
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

fn profile_scoped_pipeline_config(port: u16) -> (ParapperConfig, DeliveryRouteSnapshot) {
    let mut config = pipeline_test_config(port);
    config.translation.mappings.insert(
        0,
        TranslationMapping {
            backend: TranslationBackend::Local,
            ..translation_mapping("translate-excluded", "en")
        },
    );
    config.speech.mappings.push(SpeechMapping {
        ..speech_mapping("speech-excluded", SpeechSourceKind::Recognition)
    });
    let mut route = config.legacy_delivery_route();
    route.translation_mapping_ids = vec!["translate-en".to_owned()];
    route.speech_mapping_ids = vec![
        "speech-recognition".to_owned(),
        "speech-translation".to_owned(),
    ];
    (config, route)
}
