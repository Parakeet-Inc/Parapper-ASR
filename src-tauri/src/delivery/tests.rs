use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use super::{
    QueuedSpeechRequest, RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput,
    SpeechTextSource, TranslationProviderId, build_speech_requests, build_translation_request,
    continuing_turn_text, finalize_turn_text, join_turn_segments, should_send_to_neo,
    spawn_speech_requests, speech_mapping_matches, translate_and_spawn_speech_for_test,
    translation_targets_for_mappings, translation_timing_allows,
};
use crate::config::{
    AsrLanguage, AsrModel, LocalTranslationModel, LocalTtsVoice, NeoSendTiming, ParapperConfig,
    SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationBackend, TranslationLanguage,
    TranslationMapping, TurnDetector,
};
use crate::connect::test_support::{
    MockHttpServer, TimedMockHttpServer, json_response, request_id_from_plugin_request,
};

fn source_meta(source_id: &str, turn_id: u64, output_sequence: u64) -> RecognitionSourceMeta {
    RecognitionSourceMeta {
        turn_session_id: source_hash(source_id),
        turn_id,
        turn_revision: 0,
        output_sequence,
        segment_id: output_sequence,
        previous_segment_id: output_sequence.checked_sub(1),
    }
}

fn turn_meta(id: &str, turn_id: u64, is_final: bool) -> RecognizedTextMeta {
    RecognizedTextMeta::replace_turn(id.to_string(), source_meta(id, turn_id, turn_id), is_final)
}

fn recognized_output(id: &str, turn_id: u64, text: &str, is_final: bool) -> RecognizedTextOutput {
    RecognizedTextOutput {
        phrase: Vec::new().into(),
        text: text.to_string(),
        source_asr_model: AsrModel::ReazonSpeechK2V2,
        source_language: AsrLanguage::Japanese,
        detected_language: None,
        meta: turn_meta(id, turn_id, is_final),
        elapsed_millis: 0,
    }
}

fn translation_mapping(id: &str, target_lang: &str) -> TranslationMapping {
    TranslationMapping {
        id: id.to_string(),
        source_asr_model: None,
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
        talker: "ずんだもん-ノーマル/VOICEVOX".to_string(),
        local_tts_voice: None,
        local_tts_language: None,
        local_tts_speaker_id: None,
        output_device_id: None,
        output_device_host: None,
        output_device_name: None,
        muted: false,
        volume: 1.0,
    }
}

#[derive(Debug, PartialEq)]
struct SpeechRequestSnapshot {
    id: String,
    source_event_id: String,
    source_meta: RecognitionSourceMeta,
    source_kind: SpeechSourceKind,
    target_lang: Option<String>,
    text: String,
    backend: SpeechBackend,
    talker: String,
    local_tts_voice: Option<LocalTtsVoice>,
    local_tts_language: Option<String>,
    local_tts_speaker_id: Option<i32>,
    output_device_host: Option<String>,
    output_device_id: Option<String>,
}

impl From<&QueuedSpeechRequest> for SpeechRequestSnapshot {
    fn from(request: &QueuedSpeechRequest) -> Self {
        Self {
            id: request.id.clone(),
            source_event_id: request.source_event_id.clone(),
            source_meta: request.source_meta.clone(),
            source_kind: request.source_kind,
            target_lang: request.target_lang.clone(),
            text: request.text.clone(),
            backend: request.backend,
            talker: request.talker.clone(),
            local_tts_voice: request.local_tts_voice,
            local_tts_language: request.local_tts_language.clone(),
            local_tts_speaker_id: request.local_tts_speaker_id,
            output_device_host: request.output_device_host.clone(),
            output_device_id: request.output_device_id.clone(),
        }
    }
}

fn speech_request_snapshots(requests: &[QueuedSpeechRequest]) -> Vec<SpeechRequestSnapshot> {
    requests.iter().map(SpeechRequestSnapshot::from).collect()
}

fn source_hash(source_id: &str) -> u64 {
    source_id
        .bytes()
        .fold(1_469_598_103_934_665_603, |hash, byte| {
            hash.wrapping_mul(1_099_511_628_211) ^ u64::from(byte)
        })
}

#[test]
fn turn_display_text_decision_table() {
    enum TextCase<'a> {
        JoinJapanese {
            segments: &'a [&'a str],
            expected: &'a str,
        },
        Continuing {
            input: &'a str,
            expected: &'a str,
        },
        FinalizeJapanese {
            input: &'a str,
            expected: &'a str,
        },
    }

    let cases = [
        (
            "Japanese turn segments preserve sentence punctuation across ASR segment boundaries",
            TextCase::JoinJapanese {
                segments: &["今日は。", "いい天気です"],
                expected: "今日は。いい天気です",
            },
        ),
        (
            "continuing text adds the display continuation marker",
            TextCase::Continuing {
                input: "今日は",
                expected: "今日は...",
            },
        ),
        (
            "continuing text does not duplicate an existing marker",
            TextCase::Continuing {
                input: "今日は...",
                expected: "今日は...",
            },
        ),
        (
            "Japanese final text adds sentence end punctuation",
            TextCase::FinalizeJapanese {
                input: "今日はいい天気です",
                expected: "今日はいい天気です。",
            },
        ),
        (
            "Japanese final text replaces the continuation marker",
            TextCase::FinalizeJapanese {
                input: "今日はいい天気です...",
                expected: "今日はいい天気です。",
            },
        ),
    ];

    for (name, case) in cases {
        let actual = match case {
            TextCase::JoinJapanese { segments, .. } => join_turn_segments(
                &segments
                    .iter()
                    .map(|segment| (*segment).to_string())
                    .collect::<Vec<_>>(),
                AsrLanguage::Japanese,
            ),
            TextCase::Continuing { input, .. } => continuing_turn_text(input),
            TextCase::FinalizeJapanese { input, .. } => {
                finalize_turn_text(input, AsrLanguage::Japanese)
            }
        };
        let expected = match case {
            TextCase::JoinJapanese { expected, .. }
            | TextCase::Continuing { expected, .. }
            | TextCase::FinalizeJapanese { expected, .. } => expected,
        };

        assert_eq!(actual, expected, "case={name}");
    }
}

#[test]
fn neo_enabled_sends_both_interim_and_final_with_fixed_text_flag() {
    let cases = [
        (true, false, true),
        (true, true, true),
        (false, false, false),
        (false, true, false),
    ];

    for (http_enabled, is_final, expected_when_supported) in cases {
        let config = parapper_config! {
            neo_http_enabled: http_enabled,
            // Legacy persisted values must no longer suppress interim events.
            neo_send_timing: NeoSendTiming::Final,
            ..ParapperConfig::default()
        };
        let expected = expected_when_supported && ParapperConfig::neo_http_supported();

        assert_eq!(
            should_send_to_neo(&config, is_final),
            expected,
            "http_enabled={http_enabled}, is_final={is_final}"
        );
    }
}

#[test]
fn translation_timing_decision_table() {
    let cases = [
        (NeoSendTiming::Interim, false, true),
        (NeoSendTiming::Interim, true, true),
        (NeoSendTiming::Final, false, false),
        (NeoSendTiming::Final, true, true),
    ];

    for (timing, is_final, expected) in cases {
        let config = parapper_config! {
            translation_send_timing: timing,
            ..ParapperConfig::default()
        };

        assert_eq!(
            translation_timing_allows(&config, is_final),
            expected,
            "timing={timing:?}, is_final={is_final}"
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn final_only_translation_skips_non_final_turn_results() {
    let config = parapper_config! {
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![translation_mapping("translate-en", "en")],
        ..ParapperConfig::default()
    };
    let output = recognized_output("turn-1", 1, "今日は...", false);

    assert!(build_translation_request(&config, "turn-1", &output).is_none());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn final_only_translation_skips_namo_intermediate_turn_segments() {
    let config = parapper_config! {
        turn_detector: TurnDetector::Namo,
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![TranslationMapping {
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            ..translation_mapping("translate-en", "en")
        }],
        ..ParapperConfig::default()
    };
    let output = RecognizedTextOutput {
        phrase: vec![0.0, 1.0].into(),
        detected_language: Some("ja".to_string()),
        elapsed_millis: 42,
        ..recognized_output("turn-namo-1", 1, "ここまでを翻訳します...", false)
    };

    assert!(build_translation_request(&config, "turn-namo-1", &output).is_none());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn final_only_translation_sends_namo_completed_turn_results() {
    let config = parapper_config! {
        turn_detector: TurnDetector::Namo,
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![TranslationMapping {
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            ..translation_mapping("translate-en", "en")
        }],
        ..ParapperConfig::default()
    };
    let output = RecognizedTextOutput {
        phrase: vec![0.0, 1.0].into(),
        detected_language: Some("ja".to_string()),
        elapsed_millis: 42,
        ..recognized_output("turn-namo-1", 1, "これは中途確定です。", true)
    };

    let request = build_translation_request(&config, "turn-namo-1", &output)
        .expect("Namo completed turn should be translated even with final-only timing");

    assert_eq!(request.source_text(), "これは中途確定です。");
    assert!(request.is_final());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn namo_completed_turn_is_eligible_for_final_translation_and_speech() {
    let config = parapper_config! {
        turn_detector: TurnDetector::Namo,
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![TranslationMapping {
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            ..translation_mapping("translate-en", "en")
        }],
        speech_mappings: vec![SpeechMapping {
            volume: 0.0,
            ..speech_mapping("speech-ja", SpeechSourceKind::Recognition)
        }],
        ..ParapperConfig::default()
    };
    let output = RecognizedTextOutput {
        phrase: vec![0.0, 1.0].into(),
        detected_language: Some("ja".to_string()),
        elapsed_millis: 42,
        ..recognized_output("turn-namo-final", 1, "TDで確定した文です。", true)
    };

    let translation = build_translation_request(&config, "turn-namo-final", &output)
        .expect("Namo completed turn should be translated with final-only timing");
    let speech = build_speech_requests(
        &config,
        "turn-namo-final",
        SpeechTextSource::Recognition,
        output.source_asr_model,
        output.meta.is_final(),
        &output.text,
    );

    assert!(translation.is_final());
    assert_eq!(translation.source_text(), "TDで確定した文です。");
    assert_eq!(
        speech_request_snapshots(&speech),
        vec![SpeechRequestSnapshot {
            id: "speech-turn-namo-final-speech-ja".to_string(),
            source_event_id: "turn-namo-final".to_string(),
            source_meta: source_meta("turn-namo-final", 1, 1),
            source_kind: SpeechSourceKind::Recognition,
            target_lang: None,
            text: "TDで確定した文です。".to_string(),
            backend: SpeechBackend::Ync,
            talker: "ずんだもん-ノーマル/VOICEVOX".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
        }]
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn final_only_translation_builds_only_one_request_for_namo_interim_and_final_pair() {
    let config = parapper_config! {
        turn_detector: TurnDetector::Namo,
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![TranslationMapping {
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            ..translation_mapping("translate-en", "en")
        }],
        ..ParapperConfig::default()
    };
    let outputs = [
        RecognizedTextOutput {
            phrase: vec![0.0, 1.0].into(),
            detected_language: Some("ja".to_string()),
            elapsed_millis: 42,
            ..recognized_output("turn-namo-1", 1, "これは途中です...", false)
        },
        RecognizedTextOutput {
            phrase: vec![0.0, 1.0].into(),
            detected_language: Some("ja".to_string()),
            elapsed_millis: 84,
            ..recognized_output("turn-namo-1", 1, "これは確定です。", true)
        },
    ];

    let requests = outputs
        .iter()
        .filter_map(|output| build_translation_request(&config, "turn-namo-1", output))
        .collect::<Vec<_>>();

    assert_eq!(requests.len(), 1);
    assert!(requests[0].is_final());
    assert_eq!(requests[0].source_text(), "これは確定です。");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn interim_translation_sends_non_final_turn_results_without_continuation_marker() {
    let config = parapper_config! {
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Interim,
        translation_mappings: vec![translation_mapping("translate-en", "en")],
        ..ParapperConfig::default()
    };
    let output = recognized_output("turn-1", 1, "今日は...", false);

    let request = build_translation_request(&config, "turn-1", &output)
        .expect("interim translation should accept non-final turn output");
    assert_eq!(request.source_text(), "今日は");
    assert!(!request.is_final());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn translation_requests_work_across_turn_detector_modes() {
    for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
        let config = parapper_config! {
            turn_detector: turn_detector,
            translation_enabled: true,
            translation_send_timing: NeoSendTiming::Interim,
            translation_mappings: vec![translation_mapping("translate-en", "en")],
            ..ParapperConfig::default()
        };

        for is_final in [false, true] {
            let output = recognized_output(
                "turn-1",
                1,
                if is_final {
                    "今日は。"
                } else {
                    "今日は..."
                },
                is_final,
            );

            let request = build_translation_request(&config, "turn-1", &output)
                .unwrap_or_else(|| panic!("translation request missing for {turn_detector:?}"));

            assert_eq!(
                request.source_text(),
                if is_final {
                    "今日は。"
                } else {
                    "今日は"
                }
            );
            assert_eq!(request.is_final(), is_final);
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn translation_request_survives_when_neo_text_input_is_disabled() {
    let config = parapper_config! {
        neo_http_enabled: false,
        translation_enabled: true,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![translation_mapping("translate-en", "en")],
        ..ParapperConfig::default()
    }
    .normalized();
    let output = recognized_output(
        "turn-translation-without-neo-text",
        1,
        "翻訳対象です。",
        true,
    );

    let request = build_translation_request(&config, "turn-translation-without-neo-text", &output)
        .expect("translation plugin should not depend on NEO text input");

    assert_eq!(request.source_text(), "翻訳対象です。");
    assert!(request.is_final());
}

#[test]
fn translation_mappings_keep_top_priority() {
    let targets = translation_targets_for_mappings(
        &[
            TranslationMapping {
                backend: TranslationBackend::Local,
                local_model: LocalTranslationModel::Lfm2Q4,
                ..translation_mapping("translate-en-primary", "en")
            },
            TranslationMapping {
                backend: TranslationBackend::Ync,
                ..translation_mapping("translate-en-secondary", "en")
            },
            TranslationMapping {
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                ..translation_mapping("translate-en-ja", "ja")
            },
        ],
        AsrModel::ReazonSpeechK2V2,
        AsrLanguage::Japanese,
        None,
    );

    assert_eq!(targets.len(), 1);
    assert_eq!(
        targets[0].provider_id,
        TranslationProviderId::Local(LocalTranslationModel::Lfm2Q4)
    );
    assert_eq!(targets[0].source_lang, TranslationLanguage::Ja);
    assert_eq!(targets[0].target_lang, TranslationLanguage::En);
}

#[test]
fn translation_mapping_source_language_selects_expected_backend_target() {
    let targets = translation_targets_for_mappings(
        &[
            TranslationMapping {
                backend: TranslationBackend::Ync,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
                ..translation_mapping("translate-ja-en", "en")
            },
            TranslationMapping {
                backend: TranslationBackend::Local,
                source_lang: TranslationLanguage::En,
                target_lang: TranslationLanguage::Ja,
                ..translation_mapping("translate-en-ja", "ja")
            },
        ],
        AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        AsrLanguage::Multilingual,
        Some("en"),
    );

    assert_eq!(targets.len(), 1);
    assert_eq!(
        targets[0].provider_id,
        TranslationProviderId::Local(LocalTranslationModel::Lfm2Q4)
    );
    assert_eq!(targets[0].source_lang, TranslationLanguage::En);
    assert_eq!(targets[0].target_lang, TranslationLanguage::Ja);
}

#[test]
fn speech_mapping_matches_recognition_source() {
    let mapping = SpeechMapping {
        talker: "ずんだもん/VOICEVOX".to_string(),
        ..speech_mapping("speech-ja", SpeechSourceKind::Recognition)
    };

    assert!(speech_mapping_matches(
        &mapping,
        SpeechTextSource::Recognition,
        AsrModel::ReazonSpeechK2V2,
    ));
    assert!(!speech_mapping_matches(
        &mapping,
        SpeechTextSource::Translation { target_lang: "en" },
        AsrModel::ReazonSpeechK2V2,
    ));
}

#[test]
fn speech_mapping_matches_recognition_source_asr_model() {
    let mapping = SpeechMapping {
        source_asr_model: Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
        talker: "Microsoft Zira Desktop/SAPI5".to_string(),
        ..speech_mapping("speech-en-asr", SpeechSourceKind::Recognition)
    };

    assert!(speech_mapping_matches(
        &mapping,
        SpeechTextSource::Recognition,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
    ));
    assert!(!speech_mapping_matches(
        &mapping,
        SpeechTextSource::Recognition,
        AsrModel::ReazonSpeechK2V2,
    ));
}

#[test]
fn build_speech_requests_uses_all_matching_mappings() {
    let config = parapper_config! {
        speech_mappings: vec![
            SpeechMapping {
                talker: "First voice".to_string(),
                output_device_id: Some("mapping-output-id".to_string()),
                output_device_host: Some("mapping-host".to_string()),
                output_device_name: Some("Mapping output".to_string()),
                volume: 6.0,
                ..speech_mapping("speech-first", SpeechSourceKind::Recognition)
            },
            SpeechMapping {
                talker: "Second voice".to_string(),
                ..speech_mapping("speech-second", SpeechSourceKind::Recognition)
            },
        ],
        ..ParapperConfig::default()
    };

    let requests = build_speech_requests(
        &config,
        "turn-1",
        SpeechTextSource::Recognition,
        AsrModel::ReazonSpeechK2V2,
        true,
        "こんにちは。",
    );

    assert_eq!(
        speech_request_snapshots(&requests),
        vec![
            SpeechRequestSnapshot {
                id: "speech-turn-1-speech-first".to_string(),
                source_event_id: "turn-1".to_string(),
                source_meta: source_meta("turn-1", 1, 1),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "こんにちは。".to_string(),
                backend: SpeechBackend::Ync,
                talker: "First voice".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: Some("mapping-host".to_string()),
                output_device_id: Some("mapping-output-id".to_string()),
            },
            SpeechRequestSnapshot {
                id: "speech-turn-1-speech-second".to_string(),
                source_event_id: "turn-1".to_string(),
                source_meta: source_meta("turn-1", 1, 1),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "こんにちは。".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Second voice".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
            },
        ]
    );
    assert!((requests[0].volume - 1.995_262_4).abs() < 0.000_1);
}

#[test]
fn build_speech_requests_uses_only_matching_recognition_asr_model_mapping() {
    let config = parapper_config! {
        speech_mappings: vec![
            SpeechMapping {
                source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
                talker: "Japanese ASR voice".to_string(),
                ..speech_mapping("speech-ja-asr", SpeechSourceKind::Recognition)
            },
            SpeechMapping {
                source_asr_model: Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
                talker: "English ASR voice".to_string(),
                ..speech_mapping("speech-en-asr", SpeechSourceKind::Recognition)
            },
            SpeechMapping {
                talker: "Fallback voice".to_string(),
                ..speech_mapping("speech-any-asr", SpeechSourceKind::Recognition)
            },
        ],
        ..ParapperConfig::default()
    };

    let requests = build_speech_requests(
        &config,
        "turn-en",
        SpeechTextSource::Recognition,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        true,
        "Hello.",
    );

    assert_eq!(
        speech_request_snapshots(&requests),
        vec![
            SpeechRequestSnapshot {
                id: "speech-turn-en-speech-en-asr".to_string(),
                source_event_id: "turn-en".to_string(),
                source_meta: source_meta("turn-en", 1, 1),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "Hello.".to_string(),
                backend: SpeechBackend::Ync,
                talker: "English ASR voice".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
            },
            SpeechRequestSnapshot {
                id: "speech-turn-en-speech-any-asr".to_string(),
                source_event_id: "turn-en".to_string(),
                source_meta: source_meta("turn-en", 1, 1),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "Hello.".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Fallback voice".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
            },
        ]
    );
}

#[test]
fn build_speech_requests_skips_muted_mapping() {
    let config = parapper_config! {
        speech_mappings: vec![SpeechMapping {
            talker: "Muted voice".to_string(),
            muted: true,
            ..speech_mapping("speech-muted", SpeechSourceKind::Recognition)
        }],
        ..ParapperConfig::default()
    };

    let requests = build_speech_requests(
        &config,
        "turn-muted",
        SpeechTextSource::Recognition,
        AsrModel::ReazonSpeechK2V2,
        true,
        "ミュート中です。",
    );

    assert!(requests.is_empty());
}

#[test]
fn build_speech_requests_sends_once_for_namo_interim_and_final_pair_by_source_kind() {
    struct SpeechOnceCase<'a> {
        name: &'a str,
        mapping: SpeechMapping,
        source_event_id: &'a str,
        source: SpeechTextSource<'a>,
        interim_text: &'a str,
        final_text: &'a str,
        expected_id: &'a str,
        expected_kind: SpeechSourceKind,
        expected_target_lang: Option<&'a str>,
        expected_talker: &'a str,
    }

    let cases = [
        SpeechOnceCase {
            name: "recognition",
            mapping: speech_mapping("speech-ja", SpeechSourceKind::Recognition),
            source_event_id: "turn-namo-1",
            source: SpeechTextSource::Recognition,
            interim_text: "途中読み上げです...",
            final_text: "確定読み上げです。",
            expected_id: "speech-turn-namo-1-speech-ja",
            expected_kind: SpeechSourceKind::Recognition,
            expected_target_lang: None,
            expected_talker: "ずんだもん-ノーマル/VOICEVOX",
        },
        SpeechOnceCase {
            name: "translation",
            mapping: SpeechMapping {
                target_lang: Some("en".to_string()),
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                ..speech_mapping("speech-en", SpeechSourceKind::Translation)
            },
            source_event_id: "turn-namo-1|en",
            source: SpeechTextSource::Translation { target_lang: "en" },
            interim_text: "This is interim.",
            final_text: "This is final.",
            expected_id: "speech-turn-namo-1|en-speech-en",
            expected_kind: SpeechSourceKind::Translation,
            expected_target_lang: Some("en"),
            expected_talker: "Microsoft Zira Desktop/SAPI5",
        },
    ];

    for case in cases {
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            speech_mappings: vec![case.mapping],
            ..ParapperConfig::default()
        };

        let requests = [
            build_speech_requests(
                &config,
                case.source_event_id,
                case.source,
                AsrModel::ReazonSpeechK2V2,
                false,
                case.interim_text,
            ),
            build_speech_requests(
                &config,
                case.source_event_id,
                case.source,
                AsrModel::ReazonSpeechK2V2,
                true,
                case.final_text,
            ),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        assert_eq!(
            speech_request_snapshots(&requests),
            vec![SpeechRequestSnapshot {
                id: case.expected_id.to_string(),
                source_event_id: case.source_event_id.to_string(),
                source_meta: source_meta(case.source_event_id, 1, 1),
                source_kind: case.expected_kind,
                target_lang: case.expected_target_lang.map(ToString::to_string),
                text: case.final_text.to_string(),
                backend: SpeechBackend::Ync,
                talker: case.expected_talker.to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
            }],
            "case={}",
            case.name
        );
    }
}

#[test]
fn build_speech_requests_skips_namo_intermediate_segments_by_source_kind() {
    struct NamoIntermediateSkipCase<'a> {
        name: &'a str,
        mapping: SpeechMapping,
        source_event_id: &'a str,
        source: SpeechTextSource<'a>,
        text: &'a str,
    }

    let cases = [
        NamoIntermediateSkipCase {
            name: "recognition",
            mapping: speech_mapping("speech-namo", SpeechSourceKind::Recognition),
            source_event_id: "turn-namo-1",
            source: SpeechTextSource::Recognition,
            text: "ここまでを読み上げます...",
        },
        NamoIntermediateSkipCase {
            name: "translation",
            mapping: SpeechMapping {
                target_lang: Some("en".to_string()),
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                ..speech_mapping("speech-en", SpeechSourceKind::Translation)
            },
            source_event_id: "turn-namo-1|en",
            source: SpeechTextSource::Translation { target_lang: "en" },
            text: "I will read this segment.",
        },
    ];

    for case in cases {
        let config = parapper_config! {
            turn_detector: TurnDetector::Namo,
            speech_mappings: vec![case.mapping],
            ..ParapperConfig::default()
        };

        let requests = build_speech_requests(
            &config,
            case.source_event_id,
            case.source,
            AsrModel::ReazonSpeechK2V2,
            false,
            case.text,
        );

        assert!(requests.is_empty(), "case={}", case.name);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn recognition_speech_is_sent_once_for_interim_and_final_pair() {
    let server = MockHttpServer::start_until_idle(
        Duration::from_millis(500),
        move |request, _index| {
            assert!(
                request.contains(r#""operation":"speech""#),
                "unexpected mock request: {request}"
            );
            assert!(
                request.contains(r#""id":"speech-turn-namo-1-speech-ja""#),
                "unexpected speech id: {request}"
            );
            assert!(
                request.contains("確定読み上げです。"),
                "unexpected speech text: {request}"
            );
            json_response(
                r#"{"operation":"speech","status":"sended","id":"speech-turn-namo-1-speech-ja","text":"ok"}"#,
            )
        },
    );
    let config = parapper_config! {
        turn_detector: TurnDetector::Namo,
        ync_plugin_port: server.port(),
        speech_mappings: vec![speech_mapping("speech-ja", SpeechSourceKind::Recognition)],
        ..ParapperConfig::default()
    };
    let requests = [
        build_speech_requests(
            &config,
            "turn-namo-1",
            SpeechTextSource::Recognition,
            AsrModel::ReazonSpeechK2V2,
            false,
            "途中読み上げです...",
        ),
        build_speech_requests(
            &config,
            "turn-namo-1",
            SpeechTextSource::Recognition,
            AsrModel::ReazonSpeechK2V2,
            true,
            "確定読み上げです。",
        ),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    spawn_speech_requests(None, requests);

    let received = server.recv_request();
    assert_eq!(
        request_id_from_plugin_request(&received),
        "speech-turn-namo-1-speech-ja"
    );
    assert!(
        server
            .try_recv_request(Duration::from_millis(500))
            .is_none(),
        "interim/final pair must not send a second recognition speech request"
    );
    server.join();
}

#[test]
fn speech_mapping_matches_translation_target() {
    let mapping = SpeechMapping {
        target_lang: Some("en".to_string()),
        talker: "ずんだもん/VOICEVOX".to_string(),
        ..speech_mapping("speech-en", SpeechSourceKind::Translation)
    };

    assert!(speech_mapping_matches(
        &mapping,
        SpeechTextSource::Translation { target_lang: "en" },
        AsrModel::ReazonSpeechK2V2,
    ));
    assert!(!speech_mapping_matches(
        &mapping,
        SpeechTextSource::Translation { target_lang: "ja" },
        AsrModel::ReazonSpeechK2V2,
    ));
}

#[test]
fn build_speech_requests_uses_translated_text_for_translation_mapping() {
    let config = parapper_config! {
        speech_mappings: vec![SpeechMapping {
            target_lang: Some("en".to_string()),
            talker: "Microsoft Zira Desktop/SAPI5".to_string(),
            ..speech_mapping("speech-en", SpeechSourceKind::Translation)
        }],
        ..ParapperConfig::default()
    };

    let requests = build_speech_requests(
        &config,
        "turn-1|en",
        SpeechTextSource::Translation { target_lang: "en" },
        AsrModel::ReazonSpeechK2V2,
        true,
        "Hello.",
    );

    assert_eq!(
        speech_request_snapshots(&requests),
        vec![SpeechRequestSnapshot {
            id: "speech-turn-1|en-speech-en".to_string(),
            source_event_id: "turn-1|en".to_string(),
            source_meta: source_meta("turn-1|en", 1, 1),
            source_kind: SpeechSourceKind::Translation,
            target_lang: Some("en".to_string()),
            text: "Hello.".to_string(),
            backend: SpeechBackend::Ync,
            talker: "Microsoft Zira Desktop/SAPI5".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
        }]
    );
}

#[test]
fn build_speech_requests_keeps_supertonic_language_and_speaker() {
    let config = parapper_config! {
        speech_mappings: vec![SpeechMapping {
            target_lang: Some("es_ES".to_string()),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
            local_tts_language: Some("es".to_string()),
            local_tts_speaker_id: Some(3),
            ..speech_mapping("speech-supertonic", SpeechSourceKind::Translation)
        }],
        ..ParapperConfig::default()
    };

    let requests = build_speech_requests(
        &config,
        "turn-1|es_ES",
        SpeechTextSource::Translation {
            target_lang: "es_ES",
        },
        AsrModel::ReazonSpeechK2V2,
        true,
        "Hola.",
    );

    assert_eq!(
        speech_request_snapshots(&requests),
        vec![SpeechRequestSnapshot {
            id: "speech-turn-1|es_ES-speech-supertonic".to_string(),
            source_event_id: "turn-1|es_ES".to_string(),
            source_meta: source_meta("turn-1|es_ES", 1, 1),
            source_kind: SpeechSourceKind::Translation,
            target_lang: Some("es_ES".to_string()),
            text: "Hola.".to_string(),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
            local_tts_language: Some("es".to_string()),
            local_tts_speaker_id: Some(3),
            output_device_host: None,
            output_device_id: None,
        }]
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn speech_requests_are_sent_in_queue_order() {
    let (id_sender, id_receiver) = mpsc::channel::<String>();
    let server = MockHttpServer::start(2, move |request, index| {
        handle_speech_test_request(request, &id_sender, index == 0)
    });
    let port = server.port();
    spawn_speech_requests(
        None,
        vec![QueuedSpeechRequest {
            port,
            id: "speech-order-1".to_string(),
            source_event_id: "speech-order-1".to_string(),
            source_meta: source_meta("speech-order-1", 1, 1),
            source_kind: SpeechSourceKind::Recognition,
            target_lang: None,
            text: "first".to_string(),
            backend: SpeechBackend::Ync,
            talker: "ずんだもん/VOICEVOX".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }],
    );
    spawn_speech_requests(
        None,
        vec![QueuedSpeechRequest {
            port,
            id: "speech-order-2".to_string(),
            source_event_id: "speech-order-2".to_string(),
            source_meta: source_meta("speech-order-2", 2, 2),
            source_kind: SpeechSourceKind::Recognition,
            target_lang: None,
            text: "second".to_string(),
            backend: SpeechBackend::Ync,
            talker: "ずんだもん/VOICEVOX".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }],
    );

    let ids = vec![
        id_receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
        id_receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
    ];
    assert_eq!(ids, vec!["speech-order-1", "speech-order-2"]);
    server.join();
}

#[cfg(not(target_os = "macos"))]
#[test]
#[expect(clippy::too_many_lines)]
fn consecutive_speech_requests_are_received_by_mock_in_queue_order() {
    let response_delay = Duration::from_millis(20);
    let server = TimedMockHttpServer::start(4, move |request, _index| {
        let request_id = request_id_from_plugin_request(request);
        std::thread::sleep(response_delay);
        let body = format!(
            r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"ok"}}"#
        );
        json_response(&body)
    });
    let port = server.port();
    let started_at = Instant::now();

    spawn_speech_requests(
        None,
        vec![
            QueuedSpeechRequest {
                port,
                id: "speech-consecutive-1".to_string(),
                source_event_id: "speech-consecutive-1".to_string(),
                source_meta: source_meta("speech-consecutive-1", 1, 1),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "first".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
                volume: 1.0,
            },
            QueuedSpeechRequest {
                port,
                id: "speech-consecutive-2".to_string(),
                source_event_id: "speech-consecutive-2".to_string(),
                source_meta: source_meta("speech-consecutive-2", 2, 2),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "second".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
                volume: 1.0,
            },
            QueuedSpeechRequest {
                port,
                id: "speech-consecutive-3".to_string(),
                source_event_id: "speech-consecutive-3".to_string(),
                source_meta: source_meta("speech-consecutive-3", 3, 3),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "third".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
                volume: 1.0,
            },
            QueuedSpeechRequest {
                port,
                id: "speech-consecutive-4".to_string(),
                source_event_id: "speech-consecutive-4".to_string(),
                source_meta: source_meta("speech-consecutive-4", 4, 4),
                source_kind: SpeechSourceKind::Recognition,
                target_lang: None,
                text: "fourth".to_string(),
                backend: SpeechBackend::Ync,
                talker: "Microsoft Zira Desktop/SAPI5".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_host: None,
                output_device_id: None,
                volume: 1.0,
            },
        ],
    );

    let mut received_ids = Vec::new();
    for _ in 0..4 {
        let received = server.recv_request();
        let elapsed = received.received_at.duration_since(started_at);
        let request_id = request_id_from_plugin_request(&received.raw).to_string();
        println!(
            "mock_received_speech id={} elapsed_ms={}",
            request_id,
            elapsed.as_millis()
        );
        assert!(
            received.raw.contains(r#""operation":"speech""#),
            "mock received a non-speech request: {}",
            received.raw
        );
        assert!(
            received
                .raw
                .contains(r#""talker":"Microsoft Zira Desktop/SAPI5""#),
            "mock received an unexpected talker: {}",
            received.raw
        );
        received_ids.push(request_id);
    }

    assert_eq!(
        received_ids,
        vec![
            "speech-consecutive-1",
            "speech-consecutive-2",
            "speech-consecutive-3",
            "speech-consecutive-4",
        ]
    );
    server.join();
}

#[cfg(not(target_os = "macos"))]
#[test]
fn consecutive_translation_results_send_speech_to_mock_in_queue_order() {
    let speech_response_delay = Duration::from_millis(20);
    let max_expected_receive_elapsed = Duration::from_secs(1);
    let server = start_translation_speech_mock(speech_response_delay);
    let config = translation_speech_config(server.port());
    let started_at = Instant::now();

    send_mock_translation_turns(&config);
    let (mut translated_ids, mut speech_ids) =
        collect_translation_speech_mock_requests(&server, started_at, max_expected_receive_elapsed);

    translated_ids.sort();
    speech_ids.sort();
    assert_eq!(
        translated_ids,
        vec![
            "turn-translation-1",
            "turn-translation-2",
            "turn-translation-3",
            "turn-translation-4",
        ]
    );
    assert_eq!(
        speech_ids,
        vec![
            "speech-turn-translation-1|en-speech-en",
            "speech-turn-translation-2|en-speech-en",
            "speech-turn-translation-3|en-speech-en",
            "speech-turn-translation-4|en-speech-en",
        ]
    );
    server.join();
}

#[cfg(not(target_os = "macos"))]
#[test]
fn final_only_translation_and_translation_speech_are_sent_once_for_interim_and_final_pair() {
    let server = TimedMockHttpServer::start_until_idle(
        Duration::from_millis(500),
        move |request, _index| {
            let request_id = request_id_from_plugin_request(request);
            if request.contains(r#""operation":"translate""#) {
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
    let config = translation_speech_config(server.port());
    let outputs = [
        recognized_output("turn-translation-once", 1, "翻訳途中です...", false),
        recognized_output("turn-translation-once", 1, "翻訳確定です。", true),
    ];

    for output in &outputs {
        if let Some(request) = build_translation_request(&config, "turn-translation-once", output) {
            let translations = translate_and_spawn_speech_for_test(&request)
                .expect("mock translation should succeed");
            assert_eq!(
                translations,
                vec![(
                    "en".to_string(),
                    "translated turn-translation-once".to_string()
                )]
            );
        }
    }

    let mut translated_ids = Vec::new();
    let mut speech_ids = Vec::new();
    for _ in 0..2 {
        let received = server.recv_request();
        let request_id = request_id_from_plugin_request(&received.raw).to_string();
        if received.raw.contains(r#""operation":"translate""#) {
            translated_ids.push(request_id);
        } else {
            assert!(received.raw.contains(r#""operation":"speech""#));
            speech_ids.push(request_id);
        }
    }

    assert_eq!(translated_ids, vec!["turn-translation-once"]);
    assert_eq!(
        speech_ids,
        vec!["speech-turn-translation-once|en-speech-en"]
    );
    assert!(
        server
            .try_recv_request(Duration::from_millis(500))
            .is_none(),
        "interim/final pair must not send extra translation or speech requests"
    );
    server.join();
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_final_translation_result_does_not_send_speech_to_mock() {
    let server = TimedMockHttpServer::start_until_idle(
        Duration::from_millis(500),
        move |request, _index| {
            let request_id = request_id_from_plugin_request(request);
            if request.contains(r#""operation":"translate""#) {
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
    let config = parapper_config! {
        translation_send_timing: NeoSendTiming::Interim,
        ..translation_speech_config(server.port())
    };
    let recognized_text_id = "turn-translation-interim";
    let output = RecognizedTextOutput {
        phrase: vec![0.0, 1.0].into(),
        detected_language: Some("ja".to_string()),
        ..recognized_output(recognized_text_id, 1, "翻訳途中です...", false)
    };
    let request = build_translation_request(&config, recognized_text_id, &output)
        .expect("Namo-style interim translation request should be built");

    let translations =
        translate_and_spawn_speech_for_test(&request).expect("mock translation should succeed");

    assert_eq!(
        translations,
        vec![("en".to_string(), format!("translated {recognized_text_id}"))]
    );
    let received = server.recv_request();
    assert!(received.raw.contains(r#""operation":"translate""#));
    assert!(
        server
            .try_recv_request(Duration::from_millis(500))
            .is_none(),
        "non-final translation result must not trigger speech"
    );
    server.join();
}

fn start_translation_speech_mock(speech_response_delay: Duration) -> TimedMockHttpServer {
    TimedMockHttpServer::start(8, move |request, _index| {
        let request_id = request_id_from_plugin_request(request);
        if request.contains(r#""operation":"translate""#) {
            let body = format!(
                r#"{{"operation":"translate","status":"success","id":"{request_id}","lang":"en","text":"translated {request_id}"}}"#
            );
            return json_response(&body);
        }
        assert!(
            request.contains(r#""operation":"speech""#),
            "unexpected mock request: {request}"
        );
        std::thread::sleep(speech_response_delay);
        let body = format!(
            r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"ok"}}"#
        );
        json_response(&body)
    })
}

fn translation_speech_config(port: u16) -> ParapperConfig {
    parapper_config! {
        translation_enabled: true,
        ync_plugin_port: port,
        translation_send_timing: NeoSendTiming::Final,
        translation_mappings: vec![TranslationMapping {
            source_asr_model: Some(AsrModel::ReazonSpeechK2V2),
            ..translation_mapping("translate-en", "en")
        }],
        speech_mappings: vec![SpeechMapping {
            target_lang: Some("en".to_string()),
            talker: "Microsoft Zira Desktop/SAPI5".to_string(),
            ..speech_mapping("speech-en", SpeechSourceKind::Translation)
        }],
        ..ParapperConfig::default()
    }
}

fn send_mock_translation_turns(config: &ParapperConfig) {
    for turn_index in 1..=4 {
        let recognized_text_id = format!("turn-translation-{turn_index}");
        let text = format!("翻訳テスト{turn_index}。");
        let output = recognized_output(&recognized_text_id, turn_index, &text, true);
        let request = build_translation_request(config, &recognized_text_id, &output)
            .expect("final translation request should be built");
        let translations =
            translate_and_spawn_speech_for_test(&request).expect("mock translation should succeed");
        assert_eq!(
            translations,
            vec![("en".to_string(), format!("translated {recognized_text_id}"))]
        );
    }
}

fn collect_translation_speech_mock_requests(
    server: &TimedMockHttpServer,
    started_at: Instant,
    max_expected_receive_elapsed: Duration,
) -> (Vec<String>, Vec<String>) {
    let mut translated_ids = Vec::new();
    let mut speech_ids = Vec::new();
    for _ in 0..8 {
        let received = server.recv_request();
        let elapsed = received.received_at.duration_since(started_at);
        let request_id = request_id_from_plugin_request(&received.raw).to_string();
        if received.raw.contains(r#""operation":"translate""#) {
            println!(
                "mock_received_translate id={} elapsed_ms={}",
                request_id,
                elapsed.as_millis()
            );
            translated_ids.push(request_id);
        } else {
            assert_translation_speech_request(
                &received.raw,
                &request_id,
                elapsed,
                max_expected_receive_elapsed,
            );
            speech_ids.push(request_id);
        }
    }
    (translated_ids, speech_ids)
}

fn assert_translation_speech_request(
    request: &str,
    request_id: &str,
    elapsed: Duration,
    max_expected_receive_elapsed: Duration,
) {
    assert!(
        request.contains(r#""operation":"speech""#),
        "mock received unexpected request: {request}"
    );
    println!(
        "mock_received_translation_speech id={} elapsed_ms={}",
        request_id,
        elapsed.as_millis()
    );
    assert!(
        elapsed < max_expected_receive_elapsed,
        "translation speech POST did not reach the mock promptly: elapsed={elapsed:?}, request={request}"
    );
    assert!(
        request.contains(r#""talker":"Microsoft Zira Desktop/SAPI5""#),
        "mock received an unexpected talker: {request}"
    );
}

fn handle_speech_test_request(
    request: &str,
    id_sender: &mpsc::Sender<String>,
    delay_response: bool,
) -> String {
    let request_id = request_id_from_plugin_request(request).to_string();
    id_sender.send(request_id.clone()).unwrap();
    if delay_response {
        std::thread::sleep(Duration::from_millis(180));
    }
    let body =
        format!(r#"{{"operation":"speech","status":"sended","id":"{request_id}","text":"ok"}}"#);
    json_response(&body)
}
