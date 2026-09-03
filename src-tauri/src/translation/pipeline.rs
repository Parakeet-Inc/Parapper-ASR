use std::{
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use tauri::{AppHandle, Emitter};

use crate::{
    config::{AsrModel, DeliveryRouteSnapshot, ParapperConfig, SpeechSourceKind},
    delivery::{
        RecognizedTextOutput,
        common::SpeechTextSource,
        router::{TextArtifact, TextEvent, enqueue_text_event},
    },
    processing::ProcessingContext,
    recognition::events::{
        RecognitionSourceMeta, RecognizedTextUpdateMode, TranslationTextEvent,
        TranslationTextStatus,
    },
    synthesis::{build_speech_requests_with_source_meta, spawn_speech_requests},
};

use super::{
    provider::{TranslationProviderRegistry, TranslationTask},
    queue::{TranslationQueueState, push_translation_request},
    request::{TranslationRequest, build_translation_request_with_route, translation_event_id},
};

pub(crate) fn spawn_translation_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
    delivery_route: &DeliveryRouteSnapshot,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    let Some(request) =
        build_translation_request_with_route(config, delivery_route, recognized_text_id, output)
    else {
        return;
    };
    TranslationManager::global().submit(handle.clone(), request);
}

pub(crate) fn submit_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    delivery_route: &DeliveryRouteSnapshot,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    spawn_translation_if_needed(handle, config, delivery_route, recognized_text_id, output);
}

struct TranslationManager {
    state: Mutex<TranslationQueueState>,
    ready: Condvar,
}

static TRANSLATION_MANAGER: OnceLock<Arc<TranslationManager>> = OnceLock::new();

impl TranslationManager {
    fn global() -> Arc<Self> {
        Arc::clone(TRANSLATION_MANAGER.get_or_init(|| Arc::new(Self::new())))
    }

    fn new() -> Self {
        Self {
            state: Mutex::new(TranslationQueueState::new()),
            ready: Condvar::new(),
        }
    }

    fn submit(self: &Arc<Self>, handle: AppHandle, request: TranslationRequest) {
        {
            let mut state = self.state.lock().expect("translation queue lock poisoned");
            push_translation_request(&mut state, handle, request);
            self.start_worker_if_needed(&mut state);
        }
        self.ready.notify_one();
    }

    fn start_worker_if_needed(self: &Arc<Self>, state: &mut TranslationQueueState) {
        if state.worker_started {
            return;
        }
        state.worker_started = true;
        let manager = Arc::clone(self);
        if let Err(err) = thread::Builder::new()
            .name("parapper-translation".to_string())
            .spawn(move || manager.run_worker())
        {
            state.worker_started = false;
            log::warn!("Failed to spawn translation worker: {err}");
        }
    }

    fn run_worker(self: Arc<Self>) {
        loop {
            let item = {
                let mut state = self.state.lock().expect("translation queue lock poisoned");
                while state.is_empty() {
                    state = self
                        .ready
                        .wait(state)
                        .expect("translation queue lock poisoned");
                }
                state.pop_next().expect("translation request")
            };
            run_translation_request(&item.handle, &item.request);
        }
    }
}

fn run_translation_request(handle: &AppHandle, request: &TranslationRequest) {
    log::info!(
        "Translation request start source_id={} final={} targets={}",
        request.source_recognition_id,
        request.is_final,
        request.target_lang_codes().join(",")
    );
    let started_at = Instant::now();
    let result = translate_request(Some(handle), request);
    let elapsed_millis = started_at.elapsed().as_millis();
    match result {
        Ok(translations) => {
            log::info!(
                "Translation request success source_id={} elapsed_ms={} count={}",
                request.source_recognition_id,
                elapsed_millis,
                translations.len()
            );
            for (target_lang, translated_text) in translations {
                log::info!(
                    "Translation text ready source_id={} target={} text_chars={}",
                    request.source_recognition_id,
                    target_lang,
                    translated_text.chars().count()
                );
                spawn_translation_speech_if_needed(
                    Some(handle),
                    request,
                    &target_lang,
                    &translated_text,
                );
                let result =
                    translation_result(request, target_lang, translated_text, elapsed_millis);
                enqueue_translation_text_event(
                    request,
                    &result,
                    TranslationTextStatus::Success,
                    None,
                );
                if request.delivery_route.gui_enabled {
                    emit_translation_text_event(
                        handle,
                        result,
                        TranslationTextStatus::Success,
                        None,
                    );
                }
            }
        }
        Err(err) => {
            log::warn!(
                "Translation failed for {} after {} ms: {err}",
                request.source_recognition_id,
                elapsed_millis
            );
            for target_lang in request.target_lang_codes() {
                let result = translation_result(
                    request,
                    target_lang.to_string(),
                    String::new(),
                    elapsed_millis,
                );
                enqueue_translation_text_event(
                    request,
                    &result,
                    TranslationTextStatus::Failure,
                    Some(err.to_string()),
                );
                if request.delivery_route.gui_enabled {
                    emit_translation_text_event(
                        handle,
                        result,
                        TranslationTextStatus::Failure,
                        Some(err.to_string()),
                    );
                }
            }
        }
    }
}

fn enqueue_translation_text_event(
    request: &TranslationRequest,
    result: &TranslationEventPayload,
    status: TranslationTextStatus,
    error: Option<String>,
) {
    let event = TextEvent {
        source: result.source.clone(),
        source_asr_model: result.source_asr_model,
        source_language: request.source_language,
        route: request.delivery_route.clone(),
        artifact: TextArtifact::Translation {
            id: result.id.clone(),
            source_recognition_id: result.source_recognition_id.clone(),
            text: result.translated_text.clone(),
            target_language: result.target_lang.clone(),
            is_final: result.is_final,
            update_mode: result.update_mode,
            elapsed_millis: result.elapsed_millis,
            status,
            error,
        },
    };
    for failure in enqueue_text_event(&event) {
        log::warn!("translated text delivery route rejected event: {failure:?}");
    }
}

fn translate_request(
    handle: Option<&AppHandle>,
    request: &TranslationRequest,
) -> anyhow::Result<Vec<(String, String)>> {
    let registry = TranslationProviderRegistry::for_request(
        handle,
        request.config.translation.ync_plugin_port,
        request.targets.iter().map(|target| target.provider_id),
    );
    request
        .targets
        .iter()
        .try_fold(Vec::new(), |mut translations, target| {
            let task = TranslationTask {
                id: request.source_recognition_id.clone(),
                context: ProcessingContext::from_source(
                    &request.source_meta,
                    SpeechSourceKind::Recognition,
                    request.source_detected_language.clone(),
                ),
                source_lang: target.source_lang,
                target_lang: target.target_lang,
                text: request.source_text.clone(),
                is_final: request.is_final,
            };
            if let Some(result) = registry.translate(target.provider_id, &task)? {
                debug_assert_eq!(result.task_id, task.id);
                debug_assert_eq!(result.context, task.context);
                translations.push((result.target_lang.as_code().to_string(), result.text));
            }
            Ok(translations)
        })
}

struct TranslationEventPayload {
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

fn spawn_translation_speech_if_needed(
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

fn emit_translation_text_event(
    handle: &AppHandle,
    result: TranslationEventPayload,
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

fn translation_result(
    request: &TranslationRequest,
    target_lang: String,
    translated_text: String,
    elapsed_millis: u128,
) -> TranslationEventPayload {
    TranslationEventPayload {
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

#[cfg(test)]
pub(crate) fn translate_and_spawn_speech_for_test(
    request: &TranslationRequest,
) -> anyhow::Result<Vec<(String, String)>> {
    let translations = translate_request(None, request)?;
    for (target_lang, translated_text) in &translations {
        spawn_translation_speech_if_needed(None, request, target_lang, translated_text);
    }
    Ok(translations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{AsrModel, LocalTranslationModel, ParapperConfig, TranslationLanguage},
        delivery::{
            RecognitionSourceMeta,
            common::{TranslationProviderId, TranslationTarget},
        },
    };

    #[test]
    fn local_translation_interim_request_does_not_require_engine_or_app_handle() {
        let request = TranslationRequest {
            config: ParapperConfig::default(),
            delivery_route: ParapperConfig::default().legacy_delivery_route(),
            source_recognition_id: "recognition-interim".to_string(),
            source_meta: RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
                turn_session_id: 1,
                turn_id: 1,
                turn_revision: 0,
                output_sequence: 1,
                segment_id: 1,
                previous_segment_id: None,
            },
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: crate::config::AsrLanguage::Japanese,
            source_text: "こんにちは".to_string(),
            source_detected_language: Some("ja".to_string()),
            targets: vec![TranslationTarget {
                provider_id: TranslationProviderId::Local(LocalTranslationModel::default()),
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            is_final: false,
            update_mode: RecognizedTextUpdateMode::Replace,
        };

        let translations = translate_request(None, &request)
            .expect("interim local translation should be skipped before engine loading");

        assert_eq!(translations, Vec::<(String, String)>::new());
    }
}
