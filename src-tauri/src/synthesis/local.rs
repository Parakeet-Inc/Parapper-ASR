use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::Instant,
};

use parapper_models::tts::{LocalTtsEngine, SynthesizedTtsAudio};
use tauri::AppHandle;

use crate::{
    config::{LocalTtsVoice, ParapperConfig, SpeechBackend},
    model::local_tts_model_dir,
    playback::{PlaybackEvent, PlaybackManager, PlaybackRequest},
    recognition::events::SpeechRequestStatus,
    synthesis::{
        dispatch::emit_speech_request_event,
        queue::{SourceRoundRobinQueue, tts_job_is_stale},
        request::QueuedSpeechRequest,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LocalTtsQueueKey {
    voice: Option<LocalTtsVoice>,
}

fn local_tts_queue_key(request: &QueuedSpeechRequest) -> LocalTtsQueueKey {
    LocalTtsQueueKey {
        voice: request.local_tts_voice,
    }
}

struct GeneratedLocalTtsItem {
    handle: Option<AppHandle>,
    request: QueuedSpeechRequest,
    audio: SynthesizedTtsAudio,
}

struct TtsArtifact {
    request_id: String,
    samples: Vec<f32>,
    sample_rate: i32,
    volume: f32,
    output_device_host: Option<String>,
    output_device_id: Option<String>,
}

impl TtsArtifact {
    fn into_playback_request(
        self,
        on_finished: Box<dyn FnOnce(PlaybackEvent) + Send>,
    ) -> PlaybackRequest {
        PlaybackRequest::new(
            self.request_id,
            self.samples,
            self.sample_rate,
            self.volume,
            self.output_device_host,
            self.output_device_id,
            on_finished,
        )
    }
}

fn ensure_local_tts_engine(
    engine: &mut Option<LocalTtsEngine>,
    handle: &AppHandle,
    queue_key: LocalTtsQueueKey,
) -> anyhow::Result<()> {
    if engine.is_some() {
        return Ok(());
    }
    let voice = queue_key
        .voice
        .ok_or_else(|| anyhow::anyhow!("Local TTS queue has no voice"))?;
    let model_dir = local_tts_model_dir(handle, voice)?;
    *engine = Some(LocalTtsEngine::load(&model_dir, voice, 2)?);
    Ok(())
}

fn synthesize_cached_local_tts_request(
    engine: &mut Option<LocalTtsEngine>,
    queue_key: LocalTtsQueueKey,
    handle: Option<&AppHandle>,
    request: &QueuedSpeechRequest,
    started_at: Instant,
) -> anyhow::Result<SynthesizedTtsAudio> {
    let handle = handle.ok_or_else(|| anyhow::anyhow!("AppHandle is required for local TTS"))?;
    let voice = request
        .local_tts_voice
        .ok_or_else(|| anyhow::anyhow!("Sherpa ONNX TTS voice is not configured"))?;
    if queue_key.voice != Some(voice) {
        anyhow::bail!(
            "Local TTS queue voice mismatch: queue={:?}, request={}",
            queue_key,
            voice.dir_name()
        );
    }
    ensure_local_tts_engine(engine, handle, queue_key)?;
    log::info!(
        "Local TTS synth start id={} voice={} text_chars={}",
        request.id,
        voice.dir_name(),
        request.text.chars().count()
    );
    let audio = engine
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Local TTS engine is not initialized"))?
        .synthesize(
            &request.text,
            request.local_tts_speaker_id,
            request.local_tts_language.as_deref(),
        )?;
    log::info!(
        "Local TTS synth finished id={} voice={} elapsed_ms={}",
        request.id,
        voice.dir_name(),
        started_at.elapsed().as_millis()
    );
    Ok(audio)
}

fn submit_generated_local_tts_for_playback(item: GeneratedLocalTtsItem) {
    log::info!(
        "Local TTS playback queue id={} text_chars={}",
        item.request.id,
        item.request.text.chars().count()
    );
    let artifact = TtsArtifact {
        request_id: item.request.id.clone(),
        samples: item.audio.samples,
        sample_rate: item.audio.sample_rate,
        volume: item.request.volume,
        output_device_host: item.request.output_device_host.clone(),
        output_device_id: item.request.output_device_id.clone(),
    };
    PlaybackManager::global().submit(artifact.into_playback_request(Box::new(move |event| {
        match event {
            PlaybackEvent::Finished {
                request_id,
                elapsed_millis,
            } => {
                log::info!(
                    "Local TTS playback finished id={request_id} elapsed_ms={elapsed_millis}"
                );
                emit_speech_request_event(
                    item.handle.as_ref(),
                    &item.request,
                    elapsed_millis,
                    SpeechRequestStatus::Accepted,
                    None,
                );
            }
            PlaybackEvent::Failed {
                request_id,
                elapsed_millis,
                error,
            } => {
                log::warn!("Local TTS playback failed for {request_id}: {error}");
                emit_speech_request_event(
                    item.handle.as_ref(),
                    &item.request,
                    elapsed_millis,
                    SpeechRequestStatus::Failure,
                    Some(error),
                );
            }
        }
    })));
}

static LOCAL_TTS_QUEUES: OnceLock<LocalTtsQueueRegistry> = OnceLock::new();

pub(crate) fn prewarm_local_tts_engines(handle: &AppHandle, config: &ParapperConfig) {
    for voice in config
        .speech
        .mappings
        .iter()
        .filter(|mapping| mapping.backend == SpeechBackend::LocalTts)
        .filter_map(|mapping| mapping.local_tts_voice)
    {
        let queue = LOCAL_TTS_QUEUES
            .get_or_init(LocalTtsQueueRegistry::new)
            .queue_for(LocalTtsQueueKey { voice: Some(voice) });
        queue.prewarm(handle.clone());
    }
}

pub(super) fn enqueue_local_tts_request(handle: Option<&AppHandle>, request: QueuedSpeechRequest) {
    let queue_key = local_tts_queue_key(&request);
    let queue = LOCAL_TTS_QUEUES
        .get_or_init(LocalTtsQueueRegistry::new)
        .queue_for(queue_key);
    queue.enqueue(handle.cloned(), request);
}

struct LocalTtsQueueRegistry {
    queues: Mutex<HashMap<LocalTtsQueueKey, Arc<LocalTtsQueue>>>,
}

struct LocalTtsQueue {
    queue_key: LocalTtsQueueKey,
    state: Mutex<LocalTtsQueueState>,
    ready: Condvar,
    playback_state: Mutex<LocalTtsPlaybackState>,
    playback_ready: Condvar,
}

struct LocalTtsQueueState {
    queue: SourceRoundRobinQueue<LocalTtsQueueItem>,
    worker_started: bool,
    prewarm_handle: Option<AppHandle>,
}

struct LocalTtsQueueItem {
    handle: Option<AppHandle>,
    request: QueuedSpeechRequest,
}

struct LocalTtsPlaybackState {
    queue: VecDeque<GeneratedLocalTtsItem>,
    worker_started: bool,
}

impl LocalTtsQueueRegistry {
    fn new() -> Self {
        Self {
            queues: Mutex::new(HashMap::new()),
        }
    }

    fn queue_for(&self, queue_key: LocalTtsQueueKey) -> Arc<LocalTtsQueue> {
        let mut queues = self
            .queues
            .lock()
            .expect("local TTS registry lock poisoned");
        Arc::clone(
            queues
                .entry(queue_key)
                .or_insert_with(|| Arc::new(LocalTtsQueue::new(queue_key))),
        )
    }
}

impl LocalTtsQueue {
    fn new(queue_key: LocalTtsQueueKey) -> Self {
        Self {
            queue_key,
            state: Mutex::new(LocalTtsQueueState {
                queue: SourceRoundRobinQueue::new(),
                worker_started: false,
                prewarm_handle: None,
            }),
            ready: Condvar::new(),
            playback_state: Mutex::new(LocalTtsPlaybackState {
                queue: VecDeque::new(),
                worker_started: false,
            }),
            playback_ready: Condvar::new(),
        }
    }

    fn prewarm(self: &Arc<Self>, handle: AppHandle) {
        {
            let mut state = self.state.lock().expect("local TTS queue lock poisoned");
            if state.prewarm_handle.is_none() {
                state.prewarm_handle = Some(handle);
            }
            self.start_worker_if_needed(&mut state);
        }
        self.ready.notify_one();
    }

    fn enqueue(self: &Arc<Self>, handle: Option<AppHandle>, request: QueuedSpeechRequest) {
        let request_id = request.id.clone();
        {
            let mut state = self.state.lock().expect("local TTS queue lock poisoned");
            enqueue_local_tts_item(&mut state, handle, request);
            self.start_worker_if_needed(&mut state);
        }
        log::info!(
            "Local TTS request queued id={} queue={:?}",
            request_id,
            self.queue_key
        );
        self.ready.notify_one();
    }

    fn start_worker_if_needed(self: &Arc<Self>, state: &mut LocalTtsQueueState) {
        self.start_playback_worker_if_needed();
        if state.worker_started {
            return;
        }
        state.worker_started = true;
        let queue = Arc::clone(self);
        if let Err(err) = thread::Builder::new()
            .name("parapper-local-tts".to_string())
            .spawn(move || queue.run_worker())
        {
            state.worker_started = false;
            log::warn!("Failed to spawn local TTS worker: {err}");
        }
    }

    fn start_playback_worker_if_needed(self: &Arc<Self>) {
        let mut state = self
            .playback_state
            .lock()
            .expect("local TTS playback queue lock poisoned");
        if state.worker_started {
            return;
        }
        state.worker_started = true;
        let queue = Arc::clone(self);
        if let Err(err) = thread::Builder::new()
            .name("parapper-local-tts-playback".to_string())
            .spawn(move || queue.run_playback_worker())
        {
            state.worker_started = false;
            log::warn!("Failed to spawn local TTS playback worker: {err}");
        }
    }

    fn run_worker(self: Arc<Self>) {
        let mut engine = None;
        loop {
            let (prewarm_handle, item) = self.wait_for_next_item();
            if let Some(handle) = prewarm_handle
                && let Err(err) = ensure_local_tts_engine(&mut engine, &handle, self.queue_key)
            {
                log::warn!(
                    "Failed to prewarm local TTS engine queue={:?}: {err}",
                    self.queue_key
                );
            }
            let Some(item) = item else {
                continue;
            };
            self.synthesize_item(&mut engine, item);
        }
    }

    fn wait_for_next_item(&self) -> (Option<AppHandle>, Option<LocalTtsQueueItem>) {
        let mut state = self.state.lock().expect("local TTS queue lock poisoned");
        while state.queue.is_empty() && state.prewarm_handle.is_none() {
            state = self
                .ready
                .wait(state)
                .expect("local TTS queue lock poisoned");
        }
        let prewarm_handle = state.prewarm_handle.take();
        let item = state.queue.pop();
        (prewarm_handle, item)
    }

    fn synthesize_item(
        self: &Arc<Self>,
        engine: &mut Option<LocalTtsEngine>,
        item: LocalTtsQueueItem,
    ) {
        let started_at = Instant::now();
        match synthesize_cached_local_tts_request(
            engine,
            self.queue_key,
            item.handle.as_ref(),
            &item.request,
            started_at,
        ) {
            Ok(audio) => self.enqueue_generated_audio(GeneratedLocalTtsItem {
                handle: item.handle,
                request: item.request,
                audio,
            }),
            Err(err) => {
                let elapsed_millis = started_at.elapsed().as_millis();
                log::warn!("Local TTS request failed for {}: {err}", item.request.id);
                emit_speech_request_event(
                    item.handle.as_ref(),
                    &item.request,
                    elapsed_millis,
                    SpeechRequestStatus::Failure,
                    Some(err.to_string()),
                );
            }
        }
    }

    fn enqueue_generated_audio(self: &Arc<Self>, item: GeneratedLocalTtsItem) {
        {
            let mut state = self
                .playback_state
                .lock()
                .expect("local TTS playback queue lock poisoned");
            state.queue.push_back(item);
        }
        self.start_playback_worker_if_needed();
        self.playback_ready.notify_one();
    }

    fn run_playback_worker(self: Arc<Self>) {
        loop {
            let item = {
                let mut state = self
                    .playback_state
                    .lock()
                    .expect("local TTS playback queue lock poisoned");
                while state.queue.is_empty() {
                    state = self
                        .playback_ready
                        .wait(state)
                        .expect("local TTS playback queue lock poisoned");
                }
                state.queue.pop_front().expect("generated TTS item")
            };
            submit_generated_local_tts_for_playback(item);
        }
    }
}

fn enqueue_local_tts_item(
    state: &mut LocalTtsQueueState,
    handle: Option<AppHandle>,
    request: QueuedSpeechRequest,
) {
    state
        .queue
        .retain(|queued| !tts_job_is_stale(&queued.request, &request));
    let source = request.source_meta.source_session_key();
    state
        .queue
        .push(source, LocalTtsQueueItem { handle, request });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{LocalTtsVoice, SpeechBackend, SpeechSourceKind},
        delivery::RecognitionSourceMeta,
    };

    fn local_tts_request_with_source(
        source_event_id: &str,
        source_meta: RecognitionSourceMeta,
    ) -> QueuedSpeechRequest {
        QueuedSpeechRequest {
            port: 0,
            id: format!("speech-{source_event_id}"),
            source_event_id: source_event_id.to_string(),
            source_meta,
            source_kind: SpeechSourceKind::Translation,
            target_lang: Some("en_US".to_string()),
            text: "test".to_string(),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }
    }

    fn local_tts_item_with_source(
        source_event_id: &str,
        source_meta: RecognitionSourceMeta,
    ) -> LocalTtsQueueItem {
        LocalTtsQueueItem {
            handle: None,
            request: local_tts_request_with_source(source_event_id, source_meta),
        }
    }

    fn source_meta(
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
            turn_session_id,
            turn_id,
            turn_revision: 0,
            output_sequence,
            segment_id: output_sequence,
            previous_segment_id: output_sequence.checked_sub(1),
        }
    }

    fn local_tts_request_with_voice(voice: LocalTtsVoice) -> QueuedSpeechRequest {
        QueuedSpeechRequest {
            port: 0,
            id: "speech-test".to_string(),
            source_event_id: "turn-1-1-0".to_string(),
            source_meta: source_meta(1, 1, 1),
            source_kind: SpeechSourceKind::Recognition,
            target_lang: None,
            text: "test".to_string(),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(voice),
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }
    }

    #[test]
    fn local_tts_queue_key_is_split_by_voice_model() {
        let supertonic2 = local_tts_queue_key(&local_tts_request_with_voice(
            LocalTtsVoice::Supertonic2Onnx,
        ));
        let supertonic3 = local_tts_queue_key(&local_tts_request_with_voice(
            LocalTtsVoice::Supertonic3Onnx,
        ));

        assert_ne!(supertonic2, supertonic3);
    }

    #[test]
    fn fp32_and_quantized_supertonic3_use_separate_engine_queues() {
        let fp32 = local_tts_queue_key(&local_tts_request_with_voice(
            LocalTtsVoice::Supertonic3Onnx,
        ));
        let quantized = local_tts_queue_key(&local_tts_request_with_voice(
            LocalTtsVoice::Supertonic3OnnxQuantized,
        ));

        assert_ne!(fp32, quantized);
    }

    #[test]
    fn local_tts_same_voice_queue_round_robins_ready_sources() {
        let mut state = LocalTtsQueueState {
            queue: SourceRoundRobinQueue::new(),
            worker_started: false,
            prewarm_handle: None,
        };
        for (id, source, sequence) in [
            ("A1", "source-a", 1),
            ("A2", "source-a", 2),
            ("B1", "source-b", 1),
            ("B2", "source-b", 2),
        ] {
            let meta = RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                    source.into(),
                    source.to_owned(),
                    "interface-1".to_owned(),
                    None,
                ),
                turn_session_id: if source == "source-a" { 10 } else { 20 },
                turn_id: sequence,
                turn_revision: 0,
                output_sequence: sequence,
                segment_id: sequence,
                previous_segment_id: sequence.checked_sub(1),
            };
            enqueue_local_tts_item(&mut state, None, local_tts_request_with_source(id, meta));
        }

        let released = (0..4)
            .map(|_| {
                state
                    .queue
                    .pop()
                    .expect("local TTS item")
                    .request
                    .source_event_id
            })
            .collect::<Vec<_>>();

        assert_eq!(released, vec!["A1", "B1", "A2", "B2"]);
    }

    #[test]
    fn local_tts_keeps_previous_recognition_session_before_restart_sequence_reset() {
        let mut state = LocalTtsQueueState {
            queue: SourceRoundRobinQueue::new(),
            worker_started: false,
            prewarm_handle: None,
        };
        enqueue_local_tts_item(
            &mut state,
            None,
            local_tts_request_with_source("turn-2-1-0|en_US", source_meta(2, 1, 1)),
        );
        enqueue_local_tts_item(
            &mut state,
            None,
            local_tts_request_with_source("turn-1-5-0|en_US", source_meta(1, 5, 5)),
        );

        let released = (0..2)
            .map(|_| {
                state
                    .queue
                    .pop()
                    .expect("local TTS item")
                    .request
                    .source_event_id
            })
            .collect::<Vec<_>>();

        assert_eq!(
            released,
            vec!["turn-1-5-0|en_US", "turn-2-1-0|en_US"],
            "local TTS must not play a restarted recognition session ahead of queued older audio"
        );
    }

    #[test]
    fn local_tts_queue_keeps_enqueue_order_for_one_source_session() {
        let mut queue = SourceRoundRobinQueue::new();
        for id in ["A1", "A2", "A3"] {
            let item = local_tts_item_with_source(id, source_meta(1, 1, 1));
            queue.push(item.request.source_meta.source_session_key(), item);
        }

        let released = (0..3)
            .map(|_| queue.pop().expect("local TTS item").request.source_event_id)
            .collect::<Vec<_>>();

        assert_eq!(released, vec!["A1", "A2", "A3"]);
    }

    #[test]
    fn local_tts_same_voice_queue_replaces_stale_pending_request_only_for_that_source_turn() {
        let mut state = LocalTtsQueueState {
            queue: SourceRoundRobinQueue::new(),
            worker_started: false,
            prewarm_handle: None,
        };
        let old = QueuedSpeechRequest {
            id: "old".to_owned(),
            ..local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx)
        };
        let new = QueuedSpeechRequest {
            id: "new".to_owned(),
            ..old.clone()
        };

        enqueue_local_tts_item(&mut state, None, old);
        enqueue_local_tts_item(&mut state, None, new);

        assert_eq!(
            state.queue.pop().expect("latest local request").request.id,
            "new"
        );
        assert!(state.queue.is_empty());
    }
}
