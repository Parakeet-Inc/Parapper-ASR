use crate::{
    config::{AsrModel, LocalTtsVoice, ParapperConfig, SpeechBackend, SpeechSourceKind},
    delivery::{
        RecognitionSourceMeta, RecognizedTextOutput,
        common::{
            SpeechTextSource, speech_mapping_matches, speech_timing_allows,
            text_format::trim_continuation_marker,
        },
    },
};

const MIN_SPEECH_VOLUME_DB: f32 = -20.0;
const MAX_SPEECH_VOLUME_DB: f32 = 20.0;

#[derive(Clone)]
pub(crate) struct QueuedSpeechRequest {
    pub(crate) port: u16,
    pub(crate) id: String,
    pub(crate) source_event_id: String,
    pub(crate) source_meta: RecognitionSourceMeta,
    pub(crate) source_kind: SpeechSourceKind,
    pub(crate) target_lang: Option<String>,
    pub(crate) text: String,
    pub(crate) backend: SpeechBackend,
    pub(crate) talker: String,
    pub(crate) local_tts_voice: Option<LocalTtsVoice>,
    pub(crate) local_tts_language: Option<String>,
    pub(crate) local_tts_speaker_id: Option<i32>,
    pub(crate) output_device_host: Option<String>,
    pub(crate) output_device_id: Option<String>,
    pub(crate) volume: f32,
}

pub(crate) fn build_speech_requests_with_source_meta(
    config: &ParapperConfig,
    source_event_id: &str,
    source_meta: &RecognitionSourceMeta,
    source: SpeechTextSource<'_>,
    source_asr_model: AsrModel,
    is_final: bool,
    text: &str,
) -> Vec<QueuedSpeechRequest> {
    build_speech_requests_inner(
        config,
        source_event_id,
        source_meta,
        source,
        source_asr_model,
        is_final,
        text,
    )
}

fn build_speech_requests_inner(
    config: &ParapperConfig,
    source_event_id: &str,
    source_meta: &RecognitionSourceMeta,
    source: SpeechTextSource<'_>,
    source_asr_model: AsrModel,
    is_final: bool,
    text: &str,
) -> Vec<QueuedSpeechRequest> {
    if !speech_timing_allows(is_final) {
        return Vec::new();
    }
    let text = trim_continuation_marker(text.trim());
    if text.is_empty() {
        return Vec::new();
    }
    config
        .speech
        .mappings
        .iter()
        .filter(|mapping| speech_mapping_matches(mapping, source, source_asr_model))
        .map(|mapping| QueuedSpeechRequest {
            port: config.translation.ync_plugin_port,
            id: format!("speech-{source_event_id}-{}", mapping.id),
            source_event_id: source_event_id.to_string(),
            source_meta: source_meta.clone(),
            source_kind: speech_source_kind(source),
            target_lang: speech_source_target_lang(source),
            text: text.to_string(),
            backend: mapping.backend,
            talker: mapping.talker.clone(),
            local_tts_voice: mapping.local_tts_voice,
            local_tts_language: mapping.local_tts_language.clone(),
            local_tts_speaker_id: mapping.local_tts_speaker_id,
            output_device_host: mapping.output_device_host.clone(),
            output_device_id: mapping.output_device_id.clone(),
            volume: speech_volume_db_to_gain(mapping.volume),
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn build_speech_requests(
    config: &ParapperConfig,
    source_event_id: &str,
    source: SpeechTextSource<'_>,
    source_asr_model: AsrModel,
    is_final: bool,
    text: &str,
) -> Vec<QueuedSpeechRequest> {
    let source_meta = test_source_meta(source_event_id, 1);
    build_speech_requests_inner(
        config,
        source_event_id,
        &source_meta,
        source,
        source_asr_model,
        is_final,
        text,
    )
}

#[cfg(test)]
fn test_source_meta(source_event_id: &str, output_sequence: u64) -> RecognitionSourceMeta {
    RecognitionSourceMeta {
        turn_session_id: test_source_hash(source_event_id),
        turn_id: output_sequence,
        turn_revision: 0,
        output_sequence,
        segment_id: output_sequence,
        previous_segment_id: output_sequence.checked_sub(1),
    }
}

#[cfg(test)]
fn test_source_hash(source_event_id: &str) -> u64 {
    source_event_id
        .bytes()
        .fold(1_469_598_103_934_665_603, |hash, byte| {
            hash.wrapping_mul(1_099_511_628_211) ^ u64::from(byte)
        })
}

pub(crate) fn speech_requests_for_recognized_text(
    config: &ParapperConfig,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) -> Vec<QueuedSpeechRequest> {
    let text = trim_continuation_marker(output.text.trim()).to_string();
    build_speech_requests_with_source_meta(
        config,
        recognized_text_id,
        output.meta.source(),
        SpeechTextSource::Recognition,
        output.source_asr_model,
        output.meta.is_final,
        &text,
    )
}

fn speech_volume_db_to_gain(volume_db: f32) -> f32 {
    let volume_db = if volume_db.is_finite() {
        volume_db.clamp(MIN_SPEECH_VOLUME_DB, MAX_SPEECH_VOLUME_DB)
    } else {
        0.0
    };
    10.0_f32.powf(volume_db / 20.0)
}

fn speech_source_kind(source: SpeechTextSource<'_>) -> SpeechSourceKind {
    match source {
        SpeechTextSource::Recognition => SpeechSourceKind::Recognition,
        SpeechTextSource::Translation { .. } => SpeechSourceKind::Translation,
    }
}

fn speech_source_target_lang(source: SpeechTextSource<'_>) -> Option<String> {
    match source {
        SpeechTextSource::Recognition => None,
        SpeechTextSource::Translation { target_lang } => Some(target_lang.to_string()),
    }
}
