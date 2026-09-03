use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::Instant,
};

use anyhow::{Context, Result};
use tauri::{AppHandle, Emitter};

use crate::{
    config::{DeliveryRouteSnapshot, ParapperConfig, SpeechBackend},
    connect::{SpeechRequest, YncPluginClient},
    delivery::RecognizedTextOutput,
    processing::ProcessingContext,
    recognition::events::{SpeechRequestEvent, SpeechRequestStatus},
};

use super::{
    local::enqueue_local_tts_request,
    queue::{TtsQueueState, push_tts_requests},
    request::{QueuedSpeechRequest, speech_requests_for_recognized_text},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SpeechOutputProviderId {
    Ync,
    Local,
}

#[derive(Debug, Clone)]
struct SpeechTask {
    id: String,
    context: ProcessingContext,
    text: String,
    language: Option<String>,
    volume: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechOutcome {
    Accepted { elapsed_millis: u128 },
    Deferred,
}

trait SpeechOutputProvider: Send + Sync {
    fn submit(
        &self,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome>;
}

struct SpeechOutputProviderRegistry {
    providers: HashMap<SpeechOutputProviderId, Arc<dyn SpeechOutputProvider>>,
}

impl SpeechOutputProviderRegistry {
    fn standard() -> Self {
        let mut providers: HashMap<SpeechOutputProviderId, Arc<dyn SpeechOutputProvider>> =
            HashMap::new();
        providers.insert(
            SpeechOutputProviderId::Local,
            Arc::new(InProcessSpeechOutputProvider),
        );
        providers.insert(
            SpeechOutputProviderId::Ync,
            Arc::new(YncSpeechOutputProvider),
        );
        Self { providers }
    }

    #[cfg(test)]
    fn empty() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    fn submit(
        &self,
        provider_id: SpeechOutputProviderId,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        self.providers
            .get(&provider_id)
            .with_context(|| format!("speech output provider is not registered: {provider_id:?}"))?
            .submit(handle, task, request)
    }
}

struct InProcessSpeechOutputProvider;

impl SpeechOutputProvider for InProcessSpeechOutputProvider {
    fn submit(
        &self,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        debug_assert_eq!(task.id, request.id);
        debug_assert_eq!(task.context.source_kind, request.source_kind);
        debug_assert_eq!(task.text, request.text);
        debug_assert!((task.volume - request.volume).abs() < f32::EPSILON);
        enqueue_local_tts_request(handle, request);
        Ok(SpeechOutcome::Deferred)
    }
}

struct YncSpeechOutputProvider;

impl SpeechOutputProvider for YncSpeechOutputProvider {
    fn submit(
        &self,
        _handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        if !ParapperConfig::neo_http_supported() {
            anyhow::bail!("translation/speech plugin HTTP is unsupported");
        }
        debug_assert_eq!(task.id, request.id);
        debug_assert_eq!(task.context.source_kind, request.source_kind);
        debug_assert_eq!(task.text, request.text);
        debug_assert_eq!(task.language, request.target_lang);
        debug_assert!((task.volume - request.volume).abs() < f32::EPSILON);
        let started_at = Instant::now();
        let elapsed_millis = send_ync_speech_request(&request, started_at)?;
        Ok(SpeechOutcome::Accepted { elapsed_millis })
    }
}

fn send_ync_speech_request(
    request: &QueuedSpeechRequest,
    started_at: Instant,
) -> anyhow::Result<u128> {
    log::info!(
        "Speech request send start id={} port={} talker={} text_chars={}",
        request.id,
        request.port,
        request.talker,
        request.text.chars().count()
    );
    let mut client = YncPluginClient::for_speech(request.port)?;
    let response = client.speech(SpeechRequest {
        id: &request.id,
        text: &request.text,
        talker: &request.talker,
        volume: request.volume,
    })?;
    if response.id != request.id {
        log::warn!(
            "YNC speech response id differs: request={}, response={}",
            request.id,
            response.id
        );
    }
    log::info!(
        "Speech request accepted id={} response_id={} elapsed_ms={}",
        request.id,
        response.id,
        started_at.elapsed().as_millis()
    );
    Ok(started_at.elapsed().as_millis())
}

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
                state.queue.pop().expect("TTS request")
            };
            process_speech_request(item.handle.as_ref(), item.request);
        }
    }
}

pub(crate) fn submit_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    delivery_route: &DeliveryRouteSnapshot,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) {
    let requests =
        speech_requests_for_recognized_text(config, delivery_route, recognized_text_id, output);
    spawn_speech_requests(Some(handle), requests);
}

pub(crate) fn spawn_speech_requests(
    handle: Option<&AppHandle>,
    requests: Vec<QueuedSpeechRequest>,
) {
    if requests.is_empty() {
        return;
    }
    TtsManager::global().submit_many(handle, requests);
}

fn process_speech_request(handle: Option<&AppHandle>, request: QueuedSpeechRequest) {
    let provider_id = SpeechOutputProviderId::from(request.backend);
    let task = SpeechTask {
        id: request.id.clone(),
        context: ProcessingContext::from_source(
            &request.source_meta,
            request.source_kind,
            request.target_lang.clone(),
        ),
        text: request.text.clone(),
        language: request
            .local_tts_language
            .clone()
            .or_else(|| request.target_lang.clone()),
        volume: request.volume,
    };
    log::info!(
        "Speech request dispatch id={} provider={provider_id:?} text_chars={}",
        request.id,
        request.text.chars().count()
    );
    let started_at = Instant::now();
    let event_request = request.clone();
    match SpeechOutputProviderRegistry::standard().submit(provider_id, handle, &task, request) {
        Ok(SpeechOutcome::Accepted { elapsed_millis }) => emit_speech_request_event(
            handle,
            &event_request,
            elapsed_millis,
            SpeechRequestStatus::Accepted,
            None,
        ),
        Ok(SpeechOutcome::Deferred) => {}
        Err(err) => {
            let elapsed_millis = started_at.elapsed().as_millis();
            log::warn!("Speech request failed for {}: {err}", event_request.id);
            emit_speech_request_event(
                handle,
                &event_request,
                elapsed_millis,
                SpeechRequestStatus::Failure,
                Some(err.to_string()),
            );
        }
    }
}

impl From<SpeechBackend> for SpeechOutputProviderId {
    fn from(value: SpeechBackend) -> Self {
        match value {
            SpeechBackend::Ync => Self::Ync,
            SpeechBackend::LocalTts => Self::Local,
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
            source: request.source_meta.clone(),
            source_kind: request.source_kind,
            target_lang: request.target_lang.clone(),
            elapsed_millis,
            status,
            error,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpeechSourceKind;

    fn task() -> SpeechTask {
        SpeechTask {
            id: "speech-1".to_string(),
            context: ProcessingContext {
                turn_session_id: 1,
                turn_id: 2,
                turn_revision: 0,
                segment_id: 3,
                source_kind: SpeechSourceKind::Recognition,
                source_language: Some("ja".to_string()),
            },
            text: "hello".to_string(),
            language: Some("en".to_string()),
            volume: 1.0,
        }
    }

    fn request() -> QueuedSpeechRequest {
        QueuedSpeechRequest {
            port: 8080,
            id: "speech-1".to_string(),
            source_event_id: "recognition-1".to_string(),
            source_meta: crate::delivery::RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
                turn_session_id: 1,
                turn_id: 2,
                turn_revision: 0,
                output_sequence: 1,
                segment_id: 3,
                previous_segment_id: None,
            },
            source_kind: SpeechSourceKind::Recognition,
            target_lang: Some("en".to_string()),
            text: "hello".to_string(),
            backend: SpeechBackend::Ync,
            talker: "voice".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }
    }

    #[test]
    fn unknown_speech_provider_is_an_error_without_fallback() {
        let error = SpeechOutputProviderRegistry::empty()
            .submit(SpeechOutputProviderId::Ync, None, &task(), request())
            .expect_err("an unregistered provider must not fall back");

        assert!(error.to_string().contains("not registered"));
    }
}
