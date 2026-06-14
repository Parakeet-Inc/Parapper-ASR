use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter};

use crate::{
    config::AsrModel,
    delivery::common::SpeechTextSource,
    recognition::control::events::{
        RecognitionSourceMeta, RecognizedTextUpdateMode, TranslationTextEvent,
        TranslationTextStatus,
    },
    synthesis::{build_speech_requests_with_source_meta, spawn_speech_requests},
};

use super::request::{TranslationRequest, translation_event_id};

pub(super) struct TranslationResult {
    id: String,
    source_recognition_id: String,
    source: RecognitionSourceMeta,
    source_asr_model: AsrModel,
    source_text: String,
    source_detected_language: Option<String>,
    target_lang: String,
    translated_text: String,
    is_final: bool,
    update_mode: RecognizedTextUpdateMode,
    elapsed_millis: u128,
}

pub(super) fn spawn_translation_speech_if_needed(
    handle: Option<&AppHandle>,
    request: &TranslationRequest,
    target_lang: &str,
    translated_text: &str,
) {
    if !request.is_final {
        log::info!(
            "Skipping translation speech for non-final source_id={} target={}",
            request.source_recognition_id,
            target_lang
        );
        return;
    }
    log::info!(
        "Translation speech queue source_id={} target={} text_chars={}",
        request.source_recognition_id,
        target_lang,
        translated_text.chars().count()
    );
    let requests = build_speech_requests_with_source_meta(
        &request.config,
        &translation_event_id(&request.source_recognition_id, target_lang),
        &request.source_meta,
        SpeechTextSource::Translation { target_lang },
        request.source_asr_model,
        request.is_final,
        translated_text,
    );
    spawn_speech_requests(handle, requests);
}

pub(super) fn emit_translation_text_event(
    handle: &AppHandle,
    result: TranslationResult,
    status: TranslationTextStatus,
    error: Option<String>,
) {
    let translated_at_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default();
    let _ = handle.emit(
        "parapper://translated-text",
        TranslationTextEvent {
            id: result.id,
            source_recognition_id: result.source_recognition_id,
            source: result.source,
            source_asr_model: result.source_asr_model,
            source_text: result.source_text,
            source_detected_language: result.source_detected_language,
            target_lang: result.target_lang,
            translated_text: result.translated_text,
            is_final: result.is_final,
            update_mode: result.update_mode,
            translated_at_millis,
            elapsed_millis: result.elapsed_millis,
            status,
            error,
        },
    );
}

pub(super) fn translation_result(
    request: &TranslationRequest,
    target_lang: String,
    translated_text: String,
    elapsed_millis: u128,
) -> TranslationResult {
    TranslationResult {
        id: translation_event_id(&request.source_recognition_id, &target_lang),
        source_recognition_id: request.source_recognition_id.clone(),
        source: request.source_meta.clone(),
        source_asr_model: request.source_asr_model,
        source_text: request.source_text.clone(),
        source_detected_language: request.source_detected_language.clone(),
        target_lang,
        translated_text,
        is_final: request.is_final,
        update_mode: request.update_mode,
        elapsed_millis,
    }
}
