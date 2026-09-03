use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex, OnceLock},
    thread,
};

use crate::{
    config::{
        AsrLanguage, AsrModel, DeliveryRouteSnapshot, HttpArtifactKind, HttpDeliveryProfileConfig,
        NeoSendTiming,
    },
    delivery::{RecognizedTextOutput, sinks::text_event_http},
    recognition::events::RecognizedTextUpdateMode,
};

/// Immutable downstream envelope. A recognition source resolves `route` once
/// at startup; later events cannot be redirected by a live config lookup.
#[derive(Debug, Clone)]
pub(crate) struct TextEvent {
    pub(crate) source: crate::recognition::events::RecognitionSourceMeta,
    pub(crate) source_asr_model: AsrModel,
    pub(crate) source_language: AsrLanguage,
    pub(crate) route: DeliveryRouteSnapshot,
    pub(crate) artifact: TextArtifact,
}

#[derive(Debug, Clone)]
pub(crate) enum TextArtifact {
    Recognition {
        id: String,
        text: String,
        detected_language: Option<String>,
        is_final: bool,
        update_mode: RecognizedTextUpdateMode,
        elapsed_millis: u128,
    },
    Translation {
        id: String,
        source_recognition_id: String,
        text: String,
        target_language: String,
        is_final: bool,
        update_mode: RecognizedTextUpdateMode,
        elapsed_millis: u128,
        status: crate::recognition::events::TranslationTextStatus,
        error: Option<String>,
    },
}

impl TextEvent {
    #[must_use]
    pub(crate) fn recognition(output: &RecognizedTextOutput, route: DeliveryRouteSnapshot) -> Self {
        Self {
            source: output.meta.source().clone(),
            source_asr_model: output.source_asr_model,
            source_language: output.source_language,
            route,
            artifact: TextArtifact::Recognition {
                id: output.meta.id.clone(),
                text: output.text.clone(),
                detected_language: output.detected_language.clone(),
                is_final: output.meta.is_final(),
                update_mode: output.meta.update_mode(),
                elapsed_millis: output.elapsed_millis,
            },
        }
    }

    #[must_use]
    pub(crate) fn artifact_kind(&self) -> HttpArtifactKind {
        match &self.artifact {
            TextArtifact::Recognition { .. } => HttpArtifactKind::Recognition,
            TextArtifact::Translation { .. } => HttpArtifactKind::Translation,
        }
    }

    #[must_use]
    pub(crate) fn is_final(&self) -> bool {
        match &self.artifact {
            TextArtifact::Recognition { is_final, .. }
            | TextArtifact::Translation { is_final, .. } => *is_final,
        }
    }

    #[must_use]
    pub(crate) fn source_session_key(&self) -> parapper_stt_engine::SourceSessionKey {
        self.source.source_session_key()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeliveryRouteFailure {
    QueueFull {
        profile_id: String,
        source: parapper_stt_engine::SourceSessionKey,
    },
    WorkerUnavailable {
        profile_id: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HttpDeliveryMetrics {
    pub(crate) enqueued: u64,
    pub(crate) queue_full: u64,
    pub(crate) sent: u64,
    pub(crate) failed: u64,
}

pub(crate) struct TextEventDeliveryRouter {
    capacity_per_profile: usize,
    profiles: Mutex<HashMap<HttpProfileWorkerKey, Arc<HttpProfileWorker>>>,
}

/// The definition, rather than the profile ID, identifies a live worker. A
/// source-start snapshot can therefore drain through its original URL after a
/// later config save reuses the same profile ID for a different endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HttpProfileWorkerKey {
    id: String,
    url: String,
    payload_format: String,
    artifact_kinds: Vec<HttpArtifactKind>,
    send_timing: String,
}

impl From<&HttpDeliveryProfileConfig> for HttpProfileWorkerKey {
    fn from(profile: &HttpDeliveryProfileConfig) -> Self {
        Self {
            id: profile.id.clone(),
            url: profile.url.clone(),
            payload_format: format!("{:?}", profile.payload_format),
            artifact_kinds: profile.artifact_kinds.clone(),
            send_timing: format!("{:?}", profile.send_timing),
        }
    }
}

impl TextEventDeliveryRouter {
    #[must_use]
    pub(crate) fn new(capacity_per_profile: usize) -> Self {
        Self {
            capacity_per_profile: capacity_per_profile.max(1),
            profiles: Mutex::new(HashMap::new()),
        }
    }

    /// Enqueues matching HTTP consumers without doing network I/O. Each
    /// profile has an independent bounded queue and worker; a failure cannot
    /// trigger a fallback to another profile.
    #[must_use]
    pub(crate) fn try_enqueue(&self, event: &TextEvent) -> Vec<DeliveryRouteFailure> {
        let mut failures = Vec::new();
        for profile in event.route.http_profiles.clone() {
            if !profile_matches(&profile, event) {
                continue;
            }
            let worker_key = HttpProfileWorkerKey::from(&profile);
            let worker = {
                let mut workers = self
                    .profiles
                    .lock()
                    .expect("HTTP profile workers lock poisoned");
                if let Some(worker) = workers.get(&worker_key) {
                    Arc::clone(worker)
                } else {
                    let worker = match HttpProfileWorker::spawn(
                        profile.clone(),
                        self.capacity_per_profile,
                    ) {
                        Ok(worker) => worker,
                        Err(message) => {
                            failures.push(DeliveryRouteFailure::WorkerUnavailable {
                                profile_id: profile.id.clone(),
                                message,
                            });
                            continue;
                        }
                    };
                    workers.insert(worker_key, Arc::clone(&worker));
                    worker
                }
            };
            if worker.try_enqueue(event.clone()).is_err() {
                failures.push(DeliveryRouteFailure::QueueFull {
                    profile_id: profile.id,
                    source: event.source_session_key(),
                });
            }
        }
        failures
    }

    #[cfg(test)]
    pub(crate) fn metrics(&self, profile_id: &str) -> Option<HttpDeliveryMetrics> {
        self.profiles
            .lock()
            .expect("HTTP profile workers lock poisoned")
            .values()
            .find(|worker| worker.shared.profile.id == profile_id)
            .map(|worker| worker.metrics())
    }

    #[cfg(test)]
    fn wait_for_profile_in_flight(
        &self,
        profile: &HttpDeliveryProfileConfig,
        source: &parapper_stt_engine::SourceSessionKey,
    ) {
        let worker = self
            .profiles
            .lock()
            .expect("HTTP profile workers lock poisoned")
            .get(&HttpProfileWorkerKey::from(profile))
            .cloned()
            .expect("matching HTTP profile worker");
        worker.wait_for_in_flight(source);
    }

    #[cfg(test)]
    fn wait_for_profile_completed(&self, profile: &HttpDeliveryProfileConfig, expected_total: u64) {
        let worker = self
            .profiles
            .lock()
            .expect("HTTP profile workers lock poisoned")
            .get(&HttpProfileWorkerKey::from(profile))
            .cloned()
            .expect("matching HTTP profile worker");
        worker.wait_for_completed(expected_total);
    }

    #[cfg(test)]
    fn wait_for_profile_workers(
        &self,
        profile: &HttpDeliveryProfileConfig,
        expected_workers: usize,
    ) {
        let worker = self
            .profiles
            .lock()
            .expect("HTTP profile workers lock poisoned")
            .get(&HttpProfileWorkerKey::from(profile))
            .cloned()
            .expect("matching HTTP profile worker");
        worker.wait_for_active_workers(expected_workers);
    }
}

pub(crate) fn enqueue_text_event(event: &TextEvent) -> Vec<DeliveryRouteFailure> {
    static ROUTER: OnceLock<TextEventDeliveryRouter> = OnceLock::new();
    ROUTER
        .get_or_init(|| TextEventDeliveryRouter::new(64))
        .try_enqueue(event)
}

fn profile_matches(profile: &HttpDeliveryProfileConfig, event: &TextEvent) -> bool {
    matches!(
        profile.payload_format,
        crate::config::HttpPayloadFormat::TextEventV1
    ) && profile.artifact_kinds.contains(&event.artifact_kind())
        && (matches!(profile.send_timing, NeoSendTiming::Interim) || event.is_final())
}

struct HttpProfileWorker {
    shared: Arc<HttpProfileWorkerShared>,
    joins: Mutex<Vec<thread::JoinHandle<()>>>,
}

struct HttpProfileWorkerShared {
    profile: HttpDeliveryProfileConfig,
    capacity: usize,
    client: reqwest::blocking::Client,
    state: Mutex<HttpQueueState>,
    ready: Condvar,
    state_changed: Condvar,
    startup: Mutex<WorkerStartup>,
    startup_changed: Condvar,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerStartup {
    Pending,
    Started,
    Cancelled,
}

#[derive(Default)]
struct HttpQueueState {
    by_source: HashMap<parapper_stt_engine::SourceSessionKey, VecDeque<TextEvent>>,
    ready_sources: VecDeque<parapper_stt_engine::SourceSessionKey>,
    ready_set: HashSet<parapper_stt_engine::SourceSessionKey>,
    in_flight: HashSet<parapper_stt_engine::SourceSessionKey>,
    reserved: usize,
    metrics: HttpDeliveryMetrics,
    stopped: bool,
    active_workers: usize,
}

impl HttpProfileWorker {
    fn spawn(profile: HttpDeliveryProfileConfig, capacity: usize) -> Result<Arc<Self>, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .map_err(|error| format!("building HTTP client: {error}"))?;
        let shared = Arc::new(HttpProfileWorkerShared {
            profile,
            capacity,
            client,
            state: Mutex::new(HttpQueueState::default()),
            ready: Condvar::new(),
            state_changed: Condvar::new(),
            startup: Mutex::new(WorkerStartup::Pending),
            startup_changed: Condvar::new(),
        });
        let mut joins: Vec<thread::JoinHandle<()>> = Vec::with_capacity(2);
        for worker_index in 0..2 {
            let running = Arc::clone(&shared);
            let join = match spawn_http_profile_thread(
                thread::Builder::new().name(format!(
                    "parapper-text-event-http-{}-{worker_index}",
                    shared.profile.id
                )),
                move || running.run(),
            ) {
                Ok(join) => join,
                Err(error) => {
                    shared.cancel_startup();
                    for join in joins {
                        let _ = join.join();
                    }
                    return Err(format!("spawning HTTP profile worker: {error}"));
                }
            };
            joins.push(join);
        }
        shared.start();
        Ok(Arc::new(Self {
            shared,
            joins: Mutex::new(joins),
        }))
    }

    fn try_enqueue(&self, event: TextEvent) -> Result<(), ()> {
        self.shared.try_enqueue(event)
    }

    #[cfg(test)]
    fn metrics(&self) -> HttpDeliveryMetrics {
        self.shared.metrics()
    }

    #[cfg(test)]
    fn wait_for_in_flight(&self, source: &parapper_stt_engine::SourceSessionKey) {
        self.shared.wait_for_in_flight(source);
    }

    #[cfg(test)]
    fn wait_for_completed(&self, expected_total: u64) {
        self.shared.wait_for_completed(expected_total);
    }

    #[cfg(test)]
    fn wait_for_active_workers(&self, expected_workers: usize) {
        self.shared.wait_for_active_workers(expected_workers);
    }
}

impl Drop for HttpProfileWorker {
    fn drop(&mut self) {
        self.shared.shutdown();
        let joins = self
            .joins
            .get_mut()
            .expect("HTTP profile joins lock poisoned");
        for join in joins.drain(..) {
            let _ = join.join();
        }
    }
}

impl HttpProfileWorkerShared {
    fn start(&self) {
        *self
            .startup
            .lock()
            .expect("HTTP profile startup lock poisoned") = WorkerStartup::Started;
        self.startup_changed.notify_all();
    }

    fn cancel_startup(&self) {
        *self
            .startup
            .lock()
            .expect("HTTP profile startup lock poisoned") = WorkerStartup::Cancelled;
        self.startup_changed.notify_all();
    }

    fn await_start(&self) -> bool {
        let mut startup = self
            .startup
            .lock()
            .expect("HTTP profile startup lock poisoned");
        while *startup == WorkerStartup::Pending {
            startup = self
                .startup_changed
                .wait(startup)
                .expect("HTTP profile startup lock poisoned");
        }
        *startup == WorkerStartup::Started
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        state.stopped = true;
        self.ready.notify_all();
        self.state_changed.notify_all();
    }

    fn try_enqueue(&self, event: TextEvent) -> Result<(), ()> {
        let source = event.source_session_key();
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        if state.stopped || state.reserved >= self.capacity {
            state.metrics.queue_full = state.metrics.queue_full.saturating_add(1);
            return Err(());
        }
        let queue = state.by_source.entry(source.clone()).or_default();
        queue.push_back(event);
        state.reserved += 1;
        state.metrics.enqueued = state.metrics.enqueued.saturating_add(1);
        if !state.in_flight.contains(&source) && state.ready_set.insert(source.clone()) {
            state.ready_sources.push_back(source);
            self.ready.notify_one();
        }
        Ok(())
    }

    fn run(self: Arc<Self>) {
        if !self.await_start() {
            return;
        }
        self.mark_worker_started();
        let _active_worker = ActiveWorkerGuard { worker: &self };
        #[cfg(test)]
        let _live_worker = LiveHttpProfileWorkerGuard::enter();
        loop {
            let Some((source, event)) = self.take_next() else {
                return;
            };
            let result =
                text_event_http::send_text_event_v1(&self.client, &self.profile.url, &event);
            let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
            state.in_flight.remove(&source);
            state.reserved = state.reserved.saturating_sub(1);
            match result {
                Ok(()) => state.metrics.sent = state.metrics.sent.saturating_add(1),
                Err(error) => {
                    state.metrics.failed = state.metrics.failed.saturating_add(1);
                    log::warn!(
                        "text_event_v1 profile {} failed for source {}: {error}",
                        self.profile.id,
                        source.source_id
                    );
                }
            }
            self.state_changed.notify_all();
            if state
                .by_source
                .get(&source)
                .is_some_and(|queue| !queue.is_empty())
                && state.ready_set.insert(source.clone())
            {
                state.ready_sources.push_back(source);
                self.ready.notify_one();
            }
        }
    }

    fn mark_worker_started(&self) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        state.active_workers += 1;
        self.state_changed.notify_all();
    }

    fn mark_worker_stopped(&self) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        state.active_workers = state.active_workers.saturating_sub(1);
        self.state_changed.notify_all();
    }

    fn take_next(&self) -> Option<(parapper_stt_engine::SourceSessionKey, TextEvent)> {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        loop {
            if state.stopped {
                return None;
            }
            while let Some(source) = state.ready_sources.pop_front() {
                state.ready_set.remove(&source);
                if state.in_flight.contains(&source) {
                    continue;
                }
                let Some(queue) = state.by_source.get_mut(&source) else {
                    continue;
                };
                let Some(event) = queue.pop_front() else {
                    continue;
                };
                if queue.is_empty() {
                    state.by_source.remove(&source);
                }
                state.in_flight.insert(source.clone());
                self.state_changed.notify_all();
                return Some((source, event));
            }
            state = self
                .ready
                .wait(state)
                .expect("HTTP profile queue lock poisoned");
        }
    }

    #[cfg(test)]
    fn metrics(&self) -> HttpDeliveryMetrics {
        self.state
            .lock()
            .expect("HTTP profile queue lock poisoned")
            .metrics
    }

    #[cfg(test)]
    fn wait_for_in_flight(&self, source: &parapper_stt_engine::SourceSessionKey) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        while !state.in_flight.contains(source) {
            let (next, timeout) = self
                .state_changed
                .wait_timeout(state, std::time::Duration::from_secs(1))
                .expect("HTTP profile queue lock poisoned");
            assert!(
                !timeout.timed_out(),
                "HTTP profile worker never started source {source:?}"
            );
            state = next;
        }
    }

    #[cfg(test)]
    fn wait_for_completed(&self, expected_total: u64) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        while state.metrics.sent + state.metrics.failed < expected_total {
            let completed = state.metrics.sent + state.metrics.failed;
            let (next, timeout) = self
                .state_changed
                .wait_timeout(state, std::time::Duration::from_secs(1))
                .expect("HTTP profile queue lock poisoned");
            assert!(
                !timeout.timed_out(),
                "HTTP profile worker completed only {completed} of {expected_total} events",
            );
            state = next;
        }
    }

    #[cfg(test)]
    fn wait_for_active_workers(&self, expected_workers: usize) {
        let mut state = self.state.lock().expect("HTTP profile queue lock poisoned");
        while state.active_workers < expected_workers {
            let active = state.active_workers;
            let (next, timeout) = self
                .state_changed
                .wait_timeout(state, std::time::Duration::from_secs(1))
                .expect("HTTP profile queue lock poisoned");
            assert!(
                !timeout.timed_out(),
                "HTTP profile worker started only {active} of {expected_workers} threads"
            );
            state = next;
        }
    }
}

struct ActiveWorkerGuard<'a> {
    worker: &'a HttpProfileWorkerShared,
}

impl Drop for ActiveWorkerGuard<'_> {
    fn drop(&mut self) {
        self.worker.mark_worker_stopped();
    }
}

fn spawn_http_profile_thread(
    builder: thread::Builder,
    run: impl FnOnce() + Send + 'static,
) -> std::io::Result<thread::JoinHandle<()>> {
    #[cfg(test)]
    if FAIL_HTTP_PROFILE_THREAD_SPAWN_AFTER.with(|after| {
        let Some(remaining) = after.get() else {
            return false;
        };
        if remaining == 0 {
            after.set(None);
            true
        } else {
            after.set(Some(remaining - 1));
            false
        }
    }) {
        return Err(std::io::Error::other(
            "injected HTTP profile worker spawn failure",
        ));
    }
    builder.spawn(run)
}

#[cfg(test)]
thread_local! {
    static FAIL_HTTP_PROFILE_THREAD_SPAWN_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
static LIVE_HTTP_PROFILE_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static HTTP_PROFILE_WORKER_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
struct LiveHttpProfileWorkerGuard;

#[cfg(test)]
impl LiveHttpProfileWorkerGuard {
    fn enter() -> Self {
        LIVE_HTTP_PROFILE_WORKERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self
    }

    fn count() -> usize {
        LIVE_HTTP_PROFILE_WORKERS.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
impl Drop for LiveHttpProfileWorkerGuard {
    fn drop(&mut self) {
        LIVE_HTTP_PROFILE_WORKERS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
fn fail_http_profile_thread_spawn_after(successful_spawns: usize) {
    FAIL_HTTP_PROFILE_THREAD_SPAWN_AFTER.with(|after| after.set(Some(successful_spawns)));
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, mpsc};

    use serde_json::{Value, json};

    use super::*;
    use crate::{
        config::{HttpPayloadFormat, NeoSendTiming},
        connect::test_support::{MockHttpServer, json_response},
        recognition::events::RecognitionSourceMeta,
    };

    fn worker_test_guard() -> std::sync::MutexGuard<'static, ()> {
        HTTP_PROFILE_WORKER_TEST_LOCK
            .lock()
            .expect("HTTP profile worker test lock poisoned")
    }

    #[test]
    fn http_profile_preserves_source_fifo_and_ready_source_round_robin_without_extra_requests() {
        let _test_guard = worker_test_guard();
        let (release_a, a_released) = mpsc::channel();
        let a_released = Arc::new(Mutex::new(a_released));
        let (release_b, b_released) = mpsc::channel();
        let b_released = Arc::new(Mutex::new(b_released));
        let server = MockHttpServer::start(4, move |_request, index| {
            if index == 0 {
                a_released.lock().unwrap().recv().unwrap();
            } else if index == 1 {
                b_released.lock().unwrap().recv().unwrap();
            }
            json_response("{}")
        });
        let router = TextEventDeliveryRouter::new(8);
        let profile = http_profile("events", server.port());

        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-a",
                    1,
                    "A1",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&profile, &source_session("source-a"));
        let first = parse_http_json(&server.recv_request());
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-a",
                    2,
                    "A2",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-b",
                    1,
                    "B1",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-b",
                    2,
                    "B2",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        let second = parse_http_json(&server.recv_request());
        assert_eq!(second["artifact"]["text"], "B1");
        release_a.send(()).unwrap();
        let third = parse_http_json(&server.recv_request());
        assert_eq!(third["artifact"]["text"], "A2");
        release_b.send(()).unwrap();
        let fourth = parse_http_json(&server.recv_request());
        let received = vec![first, second, third, fourth];
        router.wait_for_profile_completed(&profile, 4);
        server.join();

        assert_eq!(
            received
                .iter()
                .map(|body| body["artifact"]["text"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["A1", "B1", "A2", "B2"]
        );
        assert_eq!(
            received,
            vec![
                expected_recognition_payload("source-a", 1, "A1"),
                expected_recognition_payload("source-b", 1, "B1"),
                expected_recognition_payload("source-a", 2, "A2"),
                expected_recognition_payload("source-b", 2, "B2"),
            ]
        );
        assert_eq!(router.metrics("events").unwrap().sent, 4);
    }

    #[test]
    fn same_profile_id_with_a_new_url_keeps_each_source_snapshot_on_its_original_endpoint() {
        let _test_guard = worker_test_guard();
        let old = MockHttpServer::start(1, |_request, _| json_response("{}"));
        let new = MockHttpServer::start(1, |_request, _| json_response("{}"));
        let router = TextEventDeliveryRouter::new(4);
        let old_profile = http_profile("reused-id", old.port());
        let new_profile = http_profile("reused-id", new.port());

        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-old",
                    1,
                    "old",
                    vec![old_profile.clone()]
                ))
                .is_empty()
        );
        router.wait_for_profile_completed(&old_profile, 1);
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-new",
                    1,
                    "new",
                    vec![new_profile.clone()]
                ))
                .is_empty()
        );
        router.wait_for_profile_completed(&new_profile, 1);

        assert_eq!(
            parse_http_json(&old.recv_request())["artifact"]["text"],
            "old"
        );
        assert_eq!(
            parse_http_json(&new.recv_request())["artifact"]["text"],
            "new"
        );
        old.join();
        new.join();
    }

    #[test]
    fn full_http_profile_queue_returns_structured_failure_without_fallback_or_extra_request() {
        let _test_guard = worker_test_guard();
        let (release_first, first_released) = mpsc::channel();
        let first_released = Arc::new(Mutex::new(first_released));
        let server = MockHttpServer::start(1, move |_request, _| {
            first_released.lock().unwrap().recv().unwrap();
            json_response("{}")
        });
        let router = TextEventDeliveryRouter::new(1);
        let profile = http_profile("bounded", server.port());

        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-a",
                    1,
                    "A1",
                    vec![profile.clone()]
                ))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&profile, &source_session("source-a"));
        let _first = server.recv_request();
        assert_eq!(
            router.try_enqueue(&recognition_event(
                "source-b",
                1,
                "B1",
                vec![profile.clone()],
            )),
            vec![DeliveryRouteFailure::QueueFull {
                profile_id: "bounded".to_owned(),
                source: parapper_stt_engine::SourceSessionKey::new(7, "source-b".into()),
            }]
        );
        release_first.send(()).unwrap();
        router.wait_for_profile_completed(&profile, 1);
        server.join();
        assert_eq!(router.metrics("bounded").unwrap().queue_full, 1);
        assert_eq!(router.metrics("bounded").unwrap().sent, 1);
    }

    #[test]
    fn worker_spawn_failure_is_a_structured_ingress_failure_without_fallback() {
        let _test_guard = worker_test_guard();
        let live_before = LiveHttpProfileWorkerGuard::count();
        let router = TextEventDeliveryRouter::new(1);
        fail_http_profile_thread_spawn_after(0);
        let failures = router.try_enqueue(&recognition_event(
            "source-a",
            1,
            "A1",
            vec![http_profile("unavailable", 1)],
        ));

        assert_eq!(
            failures,
            vec![DeliveryRouteFailure::WorkerUnavailable {
                profile_id: "unavailable".to_owned(),
                message: "spawning HTTP profile worker: injected HTTP profile worker spawn failure"
                    .to_owned(),
            }]
        );
        assert_eq!(LiveHttpProfileWorkerGuard::count(), live_before);
    }

    #[test]
    fn second_worker_spawn_failure_leaves_no_live_worker() {
        let _test_guard = worker_test_guard();
        let live_before = LiveHttpProfileWorkerGuard::count();
        let router = TextEventDeliveryRouter::new(1);
        fail_http_profile_thread_spawn_after(1);

        let failures = router.try_enqueue(&recognition_event(
            "source-a",
            1,
            "A1",
            vec![http_profile("second-spawn-failure", 1)],
        ));
        assert!(matches!(
            failures.as_slice(),
            [DeliveryRouteFailure::WorkerUnavailable { profile_id, .. }]
                if profile_id == "second-spawn-failure"
        ));
        assert_eq!(LiveHttpProfileWorkerGuard::count(), live_before);
    }

    #[test]
    fn dropping_router_stops_and_joins_its_profile_workers() {
        let _test_guard = worker_test_guard();
        let live_before = LiveHttpProfileWorkerGuard::count();
        let profile = http_profile("drop", 1);
        let router = TextEventDeliveryRouter::new(1);
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-a",
                    1,
                    "A1",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&profile, &source_session("source-a"));
        router.wait_for_profile_workers(&profile, 2);
        assert_eq!(LiveHttpProfileWorkerGuard::count(), live_before + 2);
        drop(router);
        assert_eq!(LiveHttpProfileWorkerGuard::count(), live_before);
    }

    #[test]
    fn slow_or_failing_http_profile_does_not_block_another_profile() {
        let _test_guard = worker_test_guard();
        let (release_slow, slow_released) = mpsc::channel();
        let slow_released = Arc::new(Mutex::new(slow_released));
        let slow = MockHttpServer::start(1, move |_request, _| {
            slow_released.lock().unwrap().recv().unwrap();
            json_response("{}")
        });
        let failing = MockHttpServer::start(1, |_request, _| {
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_owned()
        });
        let fast = MockHttpServer::start(1, |_request, _| json_response("{}"));
        let router = TextEventDeliveryRouter::new(4);
        let slow_profile = http_profile("slow", slow.port());
        let failing_profile = http_profile("failing", failing.port());
        let fast_profile = http_profile("fast", fast.port());
        let profiles = vec![
            slow_profile.clone(),
            failing_profile.clone(),
            fast_profile.clone(),
        ];

        assert!(
            router
                .try_enqueue(&recognition_event("source-a", 1, "A1", profiles))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&slow_profile, &source_session("source-a"));
        let _slow = slow.recv_request();
        router.wait_for_profile_completed(&fast_profile, 1);
        router.wait_for_profile_completed(&failing_profile, 1);
        let fast_request = fast.recv_request();
        let failed_request = failing.recv_request();
        assert_eq!(parse_http_json(&fast_request)["artifact"]["text"], "A1");
        assert_eq!(parse_http_json(&failed_request)["artifact"]["text"], "A1");
        release_slow.send(()).unwrap();
        router.wait_for_profile_completed(&slow_profile, 1);
        slow.join();
        failing.join();
        fast.join();
        assert_eq!(router.metrics("fast").unwrap().sent, 1);
        assert_eq!(router.metrics("failing").unwrap().failed, 1);
    }

    #[test]
    fn slow_source_a_does_not_block_source_b_in_the_same_http_profile() {
        let _test_guard = worker_test_guard();
        let (release_a, a_released) = mpsc::channel();
        let a_released = Arc::new(Mutex::new(a_released));
        let server = MockHttpServer::start(2, move |_request, index| {
            if index == 0 {
                a_released.lock().unwrap().recv().unwrap();
            }
            json_response("{}")
        });
        let router = TextEventDeliveryRouter::new(4);
        let profile = http_profile("shared", server.port());

        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-a",
                    1,
                    "A1",
                    vec![profile.clone()]
                ))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&profile, &source_session("source-a"));
        let first = parse_http_json(&server.recv_request());
        assert_eq!(first["artifact"]["text"], "A1");
        assert!(
            router
                .try_enqueue(&recognition_event(
                    "source-b",
                    1,
                    "B1",
                    vec![profile.clone()],
                ))
                .is_empty()
        );
        router.wait_for_profile_in_flight(&profile, &source_session("source-b"));
        let second = parse_http_json(&server.recv_request());
        assert_eq!(second["artifact"]["text"], "B1");
        release_a.send(()).unwrap();
        router.wait_for_profile_completed(&profile, 2);
        server.join();
    }

    fn recognition_event(
        source_id: &str,
        output_sequence: u64,
        text: &str,
        profiles: Vec<HttpDeliveryProfileConfig>,
    ) -> TextEvent {
        TextEvent {
            source: RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                    source_id.into(),
                    format!("Speaker {source_id}"),
                    "capture-1".to_owned(),
                    Some(1),
                ),
                turn_session_id: 7,
                turn_id: output_sequence,
                turn_revision: 3,
                output_sequence,
                segment_id: output_sequence + 10,
                previous_segment_id: output_sequence.checked_sub(1).map(|id| id + 10),
            },
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            route: DeliveryRouteSnapshot {
                profile_id: "profile-a".to_owned(),
                gui_enabled: false,
                translation_mapping_ids: Vec::new(),
                speech_mapping_ids: Vec::new(),
                http_profiles: profiles,
                neo_text_enabled: false,
            },
            artifact: TextArtifact::Recognition {
                id: format!("event-{source_id}-{output_sequence}"),
                text: text.to_owned(),
                detected_language: Some("ja".to_owned()),
                is_final: true,
                update_mode: RecognizedTextUpdateMode::Replace,
                elapsed_millis: 42,
            },
        }
    }

    fn http_profile(id: &str, port: u16) -> HttpDeliveryProfileConfig {
        HttpDeliveryProfileConfig {
            id: id.to_owned(),
            url: format!("http://127.0.0.1:{port}/events"),
            payload_format: HttpPayloadFormat::TextEventV1,
            artifact_kinds: vec![HttpArtifactKind::Recognition],
            send_timing: NeoSendTiming::Final,
        }
    }

    fn source_session(source_id: &str) -> parapper_stt_engine::SourceSessionKey {
        parapper_stt_engine::SourceSessionKey::new(7, source_id.into())
    }

    fn parse_http_json(raw: &str) -> Value {
        let body = raw.split("\r\n\r\n").nth(1).expect("HTTP body");
        serde_json::from_str(body).expect("text_event_v1 JSON body")
    }

    fn expected_recognition_payload(source_id: &str, output_sequence: u64, text: &str) -> Value {
        json!({
            "version": "text_event_v1",
            "delivery_profile_id": "profile-a",
            "source": {
                "source_id": source_id,
                "speaker_label": format!("Speaker {source_id}"),
                "capture_endpoint_id": "capture-1",
                "channel_index": 1,
            },
            "turn": {
                "turn_session_id": 7,
                "turn_id": output_sequence,
                "revision": 3,
                "output_sequence": output_sequence,
                "segment_id": output_sequence + 10,
                "previous_segment_id": output_sequence.checked_sub(1).map(|id| id + 10),
                "source_asr_model": "reazonspeech_k2_v2",
                "source_language": "japanese",
            },
            "artifact": {
                "id": format!("event-{source_id}-{output_sequence}"),
                "kind": "recognition",
                "text": text,
                "target_language": null,
                "detected_language": "ja",
                "is_final": true,
                "update_mode": "replace",
                "elapsed_millis": 42,
                "source_recognition_id": null,
            }
        })
    }
}
