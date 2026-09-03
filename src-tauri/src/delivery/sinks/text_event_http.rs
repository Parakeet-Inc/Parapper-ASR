use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::{
    config::HttpArtifactKind,
    delivery::router::{TextArtifact, TextEvent},
};

/// Sends the versioned downstream HTTP contract. This deliberately has no
/// endpoint, port, or profile fallback.
pub(crate) fn send_text_event_v1(
    client: &reqwest::blocking::Client,
    url: &str,
    event: &TextEvent,
) -> Result<()> {
    let body = TextEventV1::from_event(event);
    let response = client
        .post(url)
        .json(&body)
        .send()
        .with_context(|| format!("sending text_event_v1 to {url}"))?;
    if !response.status().is_success() {
        bail!(
            "text_event_v1 endpoint {url} returned {}",
            response.status()
        );
    }
    Ok(())
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct TextEventV1 {
    version: &'static str,
    delivery_profile_id: String,
    source: TextEventSourceV1,
    turn: TextEventTurnV1,
    artifact: TextEventArtifactV1,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct TextEventSourceV1 {
    source_id: String,
    speaker_label: String,
    capture_endpoint_id: String,
    channel_index: Option<u16>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct TextEventTurnV1 {
    turn_session_id: u64,
    turn_id: u64,
    revision: u64,
    output_sequence: u64,
    segment_id: u64,
    previous_segment_id: Option<u64>,
    source_asr_model: String,
    source_language: String,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct TextEventArtifactV1 {
    id: String,
    #[serde(rename = "kind")]
    artifact_kind: HttpArtifactKind,
    text: String,
    target_language: Option<String>,
    detected_language: Option<String>,
    is_final: bool,
    update_mode: crate::recognition::events::RecognizedTextUpdateMode,
    elapsed_millis: u128,
    source_recognition_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<crate::recognition::events::TranslationTextStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl TextEventV1 {
    fn from_event(event: &TextEvent) -> Self {
        let source = &event.source;
        let artifact = match &event.artifact {
            TextArtifact::Recognition {
                id,
                text,
                detected_language,
                is_final,
                update_mode,
                elapsed_millis,
            } => TextEventArtifactV1 {
                id: id.clone(),
                artifact_kind: HttpArtifactKind::Recognition,
                text: text.clone(),
                target_language: None,
                detected_language: detected_language.clone(),
                is_final: *is_final,
                update_mode: *update_mode,
                elapsed_millis: *elapsed_millis,
                source_recognition_id: None,
                status: None,
                error: None,
            },
            TextArtifact::Translation {
                id,
                source_recognition_id,
                text,
                target_language,
                is_final,
                update_mode,
                elapsed_millis,
                status,
                error,
            } => TextEventArtifactV1 {
                id: id.clone(),
                artifact_kind: HttpArtifactKind::Translation,
                text: text.clone(),
                target_language: Some(target_language.clone()),
                detected_language: None,
                is_final: *is_final,
                update_mode: *update_mode,
                elapsed_millis: *elapsed_millis,
                source_recognition_id: Some(source_recognition_id.clone()),
                status: Some(*status),
                error: error.clone(),
            },
        };
        Self {
            version: "text_event_v1",
            delivery_profile_id: event.route.profile_id.clone(),
            source: TextEventSourceV1 {
                source_id: source.identity.source_id.as_str().to_owned(),
                speaker_label: source.identity.speaker_label.clone(),
                capture_endpoint_id: source.identity.capture_endpoint_id.clone(),
                channel_index: source.identity.channel_index,
            },
            turn: TextEventTurnV1 {
                turn_session_id: source.turn_session_id,
                turn_id: source.turn_id,
                revision: source.turn_revision,
                output_sequence: source.output_sequence,
                segment_id: source.segment_id,
                previous_segment_id: source.previous_segment_id,
                source_asr_model: serde_json::to_value(event.source_asr_model)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| format!("{:?}", event.source_asr_model)),
                source_language: serde_json::to_value(event.source_language)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| format!("{:?}", event.source_language)),
            },
            artifact,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        config::{AsrLanguage, AsrModel, DeliveryRouteSnapshot, HttpPayloadFormat, NeoSendTiming},
        recognition::events::{RecognitionSourceMeta, TranslationTextStatus},
    };

    #[test]
    fn translation_payload_keeps_success_and_failure_status_explicit() {
        let mut event = TextEvent {
            source: RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                    "source-a".into(),
                    "Speaker A".to_owned(),
                    "capture-a".to_owned(),
                    Some(0),
                ),
                turn_session_id: 8,
                turn_id: 9,
                turn_revision: 2,
                output_sequence: 10,
                segment_id: 11,
                previous_segment_id: Some(7),
            },
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            route: DeliveryRouteSnapshot {
                profile_id: "source-profile".to_owned(),
                gui_enabled: false,
                translation_mapping_ids: Vec::new(),
                speech_mapping_ids: Vec::new(),
                http_profiles: vec![crate::config::HttpDeliveryProfileConfig {
                    id: "http".to_owned(),
                    url: "http://127.0.0.1:1/".to_owned(),
                    payload_format: HttpPayloadFormat::TextEventV1,
                    artifact_kinds: vec![HttpArtifactKind::Translation],
                    send_timing: NeoSendTiming::Final,
                }],
                neo_text_enabled: false,
            },
            artifact: TextArtifact::Translation {
                id: "translation-1".to_owned(),
                source_recognition_id: "recognition-1".to_owned(),
                text: String::new(),
                target_language: "en".to_owned(),
                is_final: true,
                update_mode: crate::recognition::events::RecognizedTextUpdateMode::Replace,
                elapsed_millis: 12,
                status: TranslationTextStatus::Failure,
                error: Some("provider unavailable".to_owned()),
            },
        };

        assert_eq!(
            serde_json::to_value(TextEventV1::from_event(&event)).unwrap(),
            json!({
                "version": "text_event_v1",
                "delivery_profile_id": "source-profile",
                "source": {
                    "source_id": "source-a",
                    "speaker_label": "Speaker A",
                    "capture_endpoint_id": "capture-a",
                    "channel_index": 0,
                },
                "turn": {
                    "turn_session_id": 8,
                    "turn_id": 9,
                    "revision": 2,
                    "output_sequence": 10,
                    "segment_id": 11,
                    "previous_segment_id": 7,
                    "source_asr_model": "reazonspeech_k2_v2",
                    "source_language": "japanese",
                },
                "artifact": {
                    "id": "translation-1",
                    "kind": "translation",
                    "text": "",
                    "target_language": "en",
                    "detected_language": null,
                    "is_final": true,
                    "update_mode": "replace",
                    "elapsed_millis": 12,
                    "source_recognition_id": "recognition-1",
                    "status": "failure",
                    "error": "provider unavailable",
                }
            })
        );

        let TextArtifact::Translation { status, error, .. } = &mut event.artifact else {
            unreachable!("translation event fixture")
        };
        *status = TranslationTextStatus::Success;
        *error = None;
        assert_eq!(
            serde_json::to_value(TextEventV1::from_event(&event)).unwrap()["artifact"],
            json!({
                "id": "translation-1",
                "kind": "translation",
                "text": "",
                "target_language": "en",
                "detected_language": null,
                "is_final": true,
                "update_mode": "replace",
                "elapsed_millis": 12,
                "source_recognition_id": "recognition-1",
                "status": "success",
            })
        );
    }
}
