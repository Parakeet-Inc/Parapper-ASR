use std::{
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::Instant,
};

use tauri::AppHandle;

use crate::{config::ParapperConfig, delivery::RecognizedTextOutput};

use super::{
    clients::translate_text,
    event::{emit_translation_text_event, spawn_translation_speech_if_needed, translation_result},
    queue::{TranslationQueueState, push_translation_request},
    request::{TranslationRequest, build_translation_request},
};
use crate::recognition::control::events::TranslationTextStatus;

pub(crate) fn spawn_translation_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    let Some(request) = build_translation_request(config, recognized_text_id, output) else {
        return;
    };
    TranslationManager::global().submit(handle.clone(), request);
}

pub(crate) fn submit_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    spawn_translation_if_needed(handle, config, recognized_text_id, output);
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
                while state.queue.is_empty() {
                    state = self
                        .ready
                        .wait(state)
                        .expect("translation queue lock poisoned");
                }
                state.queue.pop_front().expect("translation request")
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
    let result = translate_text(Some(handle), request);
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
                emit_translation_text_event(handle, result, TranslationTextStatus::Success, None);
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

#[cfg(test)]
pub(crate) fn translate_and_spawn_speech_for_test(
    request: &TranslationRequest,
) -> anyhow::Result<Vec<(String, String)>> {
    let translations = translate_text(None, request)?;
    for (target_lang, translated_text) in &translations {
        spawn_translation_speech_if_needed(None, request, target_lang, translated_text);
    }
    Ok(translations)
}
