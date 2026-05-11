use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, VecDeque},
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
    time::Instant,
};

use tauri::AppHandle;

use crate::{
    config::{ParapperConfig, SpeechBackend},
    recognition::events::SpeechRequestStatus,
    synthesis::{
        manager::emit_speech_request_event,
        queue::{SpeechOrderKey, speech_order_key_for_request},
        request::QueuedSpeechRequest,
    },
};

use super::{
    audio::GeneratedLocalTtsItem,
    engine::{LocalTtsEngine, ensure_local_tts_engine, synthesize_cached_local_tts_request},
    key::{LocalTtsQueueKey, local_tts_queue_key},
    playback::submit_generated_local_tts_for_playback,
};

static LOCAL_TTS_QUEUES: OnceLock<LocalTtsQueueRegistry> = OnceLock::new();

pub(crate) fn prewarm_local_tts_engines(handle: &AppHandle, config: &ParapperConfig) {
    for voice in config
        .speech_mappings
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

pub(in crate::synthesis) fn enqueue_local_tts_request(
    handle: Option<&AppHandle>,
    request: QueuedSpeechRequest,
) {
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
    heap: BinaryHeap<LocalTtsQueueItem>,
    next_sequence: u64,
    worker_started: bool,
    prewarm_handle: Option<AppHandle>,
}

struct LocalTtsQueueItem {
    key: SpeechOrderKey,
    sequence: u64,
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
                heap: BinaryHeap::new(),
                next_sequence: 0,
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
        let key = speech_order_key_for_request(&request);
        {
            let mut state = self.state.lock().expect("local TTS queue lock poisoned");
            let sequence = state.next_sequence;
            state.next_sequence = state.next_sequence.saturating_add(1);
            state.heap.push(LocalTtsQueueItem {
                key,
                sequence,
                handle,
                request,
            });
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
            if let Some(handle) = prewarm_handle {
                if let Err(err) = ensure_local_tts_engine(&mut engine, &handle, self.queue_key) {
                    log::warn!(
                        "Failed to prewarm local TTS engine queue={:?}: {err}",
                        self.queue_key
                    );
                }
            }
            let Some(item) = item else {
                continue;
            };
            self.synthesize_item(&mut engine, item);
        }
    }

    fn wait_for_next_item(&self) -> (Option<AppHandle>, Option<LocalTtsQueueItem>) {
        let mut state = self.state.lock().expect("local TTS queue lock poisoned");
        while state.heap.is_empty() && state.prewarm_handle.is_none() {
            state = self
                .ready
                .wait(state)
                .expect("local TTS queue lock poisoned");
        }
        let prewarm_handle = state.prewarm_handle.take();
        let item = state.heap.pop();
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

impl Ord for LocalTtsQueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for LocalTtsQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for LocalTtsQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.sequence == other.sequence
    }
}

impl Eq for LocalTtsQueueItem {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{LocalTtsVoice, SpeechBackend, SpeechSourceKind},
        delivery::RecognitionSourceMeta,
    };

    fn local_tts_item(
        source_event_id: &str,
        output_sequence: u64,
        sequence: u64,
    ) -> LocalTtsQueueItem {
        let request = QueuedSpeechRequest {
            port: 0,
            id: format!("speech-{source_event_id}"),
            source_event_id: source_event_id.to_string(),
            source_meta: source_meta(output_sequence),
            source_kind: SpeechSourceKind::Translation,
            target_lang: Some("en_US".to_string()),
            text: "test".to_string(),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(LocalTtsVoice::Kristin),
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        };
        LocalTtsQueueItem {
            key: speech_order_key_for_request(&request),
            sequence,
            handle: None,
            request,
        }
    }

    fn source_meta(output_sequence: u64) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            turn_session_id: 1,
            turn_id: output_sequence,
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
            source_meta: source_meta(1),
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
        let kristin = local_tts_queue_key(&local_tts_request_with_voice(LocalTtsVoice::Kristin));
        let supertonic = local_tts_queue_key(&local_tts_request_with_voice(
            LocalTtsVoice::Supertonic2Onnx,
        ));

        assert_ne!(kristin, supertonic);
    }

    #[test]
    fn local_tts_heap_releases_requests_by_source_sequence_order() {
        let mut heap = BinaryHeap::new();
        heap.push(local_tts_item("turn-1-10-0|en_US", 10, 0));
        heap.push(local_tts_item("turn-1-2-0|en_US", 2, 1));
        heap.push(local_tts_item("turn-1-2-1|en_US", 3, 2));

        let released = (0..3)
            .map(|_| heap.pop().expect("heap item").request.source_event_id)
            .collect::<Vec<_>>();

        assert_eq!(
            released,
            vec!["turn-1-2-0|en_US", "turn-1-2-1|en_US", "turn-1-10-0|en_US",]
        );
    }

    #[test]
    fn local_tts_heap_keeps_enqueue_order_for_same_source_id() {
        let mut heap = BinaryHeap::new();
        heap.push(local_tts_item("turn-1-1-0|en_US", 1, 2));
        heap.push(local_tts_item("turn-1-1-0|en_US", 1, 0));
        heap.push(local_tts_item("turn-1-1-0|en_US", 1, 1));

        let released = (0..3)
            .map(|_| heap.pop().expect("heap item").sequence)
            .collect::<Vec<_>>();

        assert_eq!(released, vec![0, 1, 2]);
    }
}
