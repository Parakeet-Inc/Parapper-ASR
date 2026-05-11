use std::{
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::Instant,
};

use tauri::{AppHandle, Emitter};

use crate::{
    config::{ParapperConfig, SpeechBackend},
    delivery::RecognizedTextOutput,
    recognition::events::{SpeechRequestEvent, SpeechRequestStatus},
};

use super::{
    clients::send_ync_speech_request,
    local::enqueue_local_tts_request,
    queue::{TtsQueueState, push_tts_requests},
    request::{QueuedSpeechRequest, speech_requests_for_recognized_text},
};

pub(crate) use super::request::build_speech_requests_with_source_meta;

struct TtsManager {
    state: Mutex<TtsQueueState>,
    ready: Condvar,
}

static TTS_MANAGER: OnceLock<Arc<TtsManager>> = OnceLock::new();

impl TtsManager {
    fn global() -> Arc<Self> {
        Arc::clone(TTS_MANAGER.get_or_init(|| Arc::new(Self::new())))
    }

    fn new() -> Self {
        Self {
            state: Mutex::new(TtsQueueState::new()),
            ready: Condvar::new(),
        }
    }

    fn submit_many(
        self: &Arc<Self>,
        handle: Option<&AppHandle>,
        requests: Vec<QueuedSpeechRequest>,
    ) {
        {
            let mut state = self.state.lock().expect("TTS queue lock poisoned");
            push_tts_requests(&mut state, handle, requests);
            self.start_worker_if_needed(&mut state);
        }
        self.ready.notify_one();
    }

    fn start_worker_if_needed(self: &Arc<Self>, state: &mut TtsQueueState) {
        if state.worker_started {
            return;
        }
        state.worker_started = true;
        let manager = Arc::clone(self);
        if let Err(err) = thread::Builder::new()
            .name("parapper-tts".to_string())
            .spawn(move || manager.run_worker())
        {
            state.worker_started = false;
            log::warn!("Failed to spawn TTS worker: {err}");
        }
    }

    fn run_worker(self: Arc<Self>) {
        loop {
            let item = {
                let mut state = self.state.lock().expect("TTS queue lock poisoned");
                while state.queue.is_empty() {
                    state = self.ready.wait(state).expect("TTS queue lock poisoned");
                }
                state.queue.pop_front().expect("TTS request")
            };
            process_speech_request(item.handle.as_ref(), item.request);
        }
    }
}

pub(crate) fn submit_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    let requests = speech_requests_for_recognized_text(config, recognized_text_id, output);
    spawn_speech_requests(Some(handle), requests);
}

pub(crate) fn spawn_speech_requests(
    handle: Option<&AppHandle>,
    requests: Vec<QueuedSpeechRequest>,
) {
    if requests.is_empty() {
        return;
    }
    if !ParapperConfig::neo_http_supported() {
        log::warn!("Skipping speech: translation/speech plugin HTTP is unsupported");
        return;
    }
    TtsManager::global().submit_many(handle, requests);
}

fn process_speech_request(handle: Option<&AppHandle>, request: QueuedSpeechRequest) {
    if request.backend == SpeechBackend::LocalTts {
        enqueue_local_tts_request(handle, request);
        return;
    }
    log::info!(
        "Speech request enqueue id={} talker={} text_chars={}",
        request.id,
        request.talker,
        request.text.chars().count()
    );
    let started_at = Instant::now();
    match send_ync_speech_request(&request, started_at) {
        Ok(elapsed_millis) => emit_speech_request_event(
            handle,
            &request,
            elapsed_millis,
            SpeechRequestStatus::Accepted,
            None,
        ),
        Err(err) => {
            let elapsed_millis = started_at.elapsed().as_millis();
            log::warn!("Speech request failed for {}: {err}", request.id);
            emit_speech_request_event(
                handle,
                &request,
                elapsed_millis,
                SpeechRequestStatus::Failure,
                Some(err.to_string()),
            );
        }
    }
}

pub(super) fn emit_speech_request_event(
    handle: Option<&AppHandle>,
    request: &QueuedSpeechRequest,
    elapsed_millis: u128,
    status: SpeechRequestStatus,
    error: Option<String>,
) {
    let Some(handle) = handle else {
        return;
    };
    let _ = handle.emit(
        "parapper://speech-request",
        SpeechRequestEvent {
            id: request.id.clone(),
            source_event_id: request.source_event_id.clone(),
            source_kind: request.source_kind,
            target_lang: request.target_lang.clone(),
            elapsed_millis,
            status,
            error,
        },
    );
}
