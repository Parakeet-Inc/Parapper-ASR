use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use parapper_stt_engine::{AsrExecutionRuntime, SourceId, SourceSessionKey};
use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    recognition::{
        events::{MissingModelKind, emit_missing_model_event},
        model_factory::load_asr_models,
    },
};
use parapper_stt_engine::transcription::task::{AsrRequest, AsrResult, AsrResultStatus};

const ASR_WORKER_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
pub(crate) use parapper_stt_engine::ports::AsrRequestRunner;

pub(crate) type AsrWorkerStartupResult = Result<(), Vec<String>>;

#[derive(Debug)]
pub(crate) struct AsrWorkerStartupReport {
    pub(crate) source_id: SourceId,
    pub(crate) result: AsrWorkerStartupResult,
}

#[derive(Clone)]
pub(crate) struct AsrWorkerStartupSender {
    source_id: SourceId,
    sender: Sender<AsrWorkerStartupReport>,
}

impl AsrWorkerStartupSender {
    pub(crate) fn new(source_id: SourceId, sender: Sender<AsrWorkerStartupReport>) -> Self {
        Self { source_id, sender }
    }

    pub(crate) fn send(
        &self,
        result: AsrWorkerStartupResult,
    ) -> Result<(), std::sync::mpsc::SendError<AsrWorkerStartupReport>> {
        self.sender.send(AsrWorkerStartupReport {
            source_id: self.source_id.clone(),
            result,
        })
    }
}

pub(crate) fn emit_asr_warning(handle: &AppHandle, error: &anyhow::Error) {
    emit_parapper_error(
        handle,
        ParapperErrorType::Asr,
        ErrorSeverity::Warning,
        Some(error.to_string()),
    );
}

#[cfg(test)]
pub(crate) struct NoopAsrRequestRunner;

#[cfg(test)]
impl parapper_stt_engine::ports::AsrRequestRunner for NoopAsrRequestRunner {
    fn submit(&mut self, _request: AsrRequest) -> bool {
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        None
    }
}

pub(crate) struct EngineAsrRequestRunner {
    request_sender: Option<Sender<AsrWorkerCommand>>,
    result_receiver: Receiver<AsrResult>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
    routed_result: bool,
    result_sender: Option<Sender<AsrResult>>,
    normalize_state: Option<Arc<Mutex<Option<bool>>>>,
}

enum AsrWorkerCommand {
    Request(Box<AsrRequest>),
    RequestFor(Box<AsrRequest>, Sender<AsrResult>),
    SetNormalizeInputAudio(bool),
    ResetStreamingSessions,
    ResetStreamingSessionsForSource(SourceSessionKey),
}

struct PendingAsrRequest {
    request: Box<AsrRequest>,
    destination: Option<Sender<AsrResult>>,
}

/// Fair dispatch for the shared runtime. Each model keeps source-local FIFO
/// queues; models are drained in insertion order and each model rotates ready
/// sources. Thus A1,A2,B1,B2 is dispatched A1,B1,A2,B2 without allowing one
/// source or model to monopolize the runtime.
#[derive(Default)]
struct PendingAsrRequests {
    queues: HashMap<(parapper_stt_engine::AsrModel, SourceSessionKey), VecDeque<PendingAsrRequest>>,
    ready_models: VecDeque<parapper_stt_engine::AsrModel>,
    ready_sources: HashMap<parapper_stt_engine::AsrModel, VecDeque<SourceSessionKey>>,
}

impl PendingAsrRequests {
    fn push(&mut self, request: Box<AsrRequest>, destination: Option<Sender<AsrResult>>) {
        let model = request.route.model;
        let source = request.target.source_session.clone();
        let key = (model, source.clone());
        let queue = self.queues.entry(key).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(PendingAsrRequest {
            request,
            destination,
        });
        if was_empty {
            let sources = self.ready_sources.entry(model).or_default();
            if sources.is_empty() {
                self.ready_models.push_back(model);
            }
            sources.push_back(source);
        }
    }

    fn pop(&mut self) -> Option<PendingAsrRequest> {
        let model = self.ready_models.pop_front()?;
        let source = self.ready_sources.get_mut(&model)?.pop_front()?;
        let key = (model, source.clone());
        let request = self.queues.get_mut(&key)?.pop_front()?;
        let has_more = self.queues.get(&key).is_some_and(|queue| !queue.is_empty());
        if has_more {
            self.ready_sources.get_mut(&model)?.push_back(source);
        } else {
            self.queues.remove(&key);
        }
        if self
            .ready_sources
            .get(&model)
            .is_none_or(VecDeque::is_empty)
        {
            self.ready_sources.remove(&model);
        } else {
            self.ready_models.push_back(model);
        }
        Some(request)
    }

    fn remove_source(&mut self, source: &SourceSessionKey) {
        let models = self
            .queues
            .keys()
            .filter(|(_, queued_source)| queued_source == source)
            .map(|(model, _)| *model)
            .collect::<Vec<_>>();
        for model in models {
            self.queues.remove(&(model, source.clone()));
            if let Some(sources) = self.ready_sources.get_mut(&model) {
                sources.retain(|queued_source| queued_source != source);
            }
            if self
                .ready_sources
                .get(&model)
                .is_none_or(VecDeque::is_empty)
            {
                self.ready_sources.remove(&model);
                self.ready_models
                    .retain(|ready_model| *ready_model != model);
            }
        }
    }

    fn clear(&mut self) {
        self.queues.clear();
        self.ready_models.clear();
        self.ready_sources.clear();
    }
}

/// A single model registry/runtime shared by all explicit capture lanes.
/// Handles only own their request and result channels; dropping a handle does
/// not stop the pool or reset another source's stream.
pub(crate) struct AsrRuntimePool {
    command_sender: Option<Sender<AsrWorkerCommand>>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub(crate) struct AsrRuntimePoolHandle {
    command_sender: Sender<AsrWorkerCommand>,
    normalize_state: Arc<Mutex<Option<bool>>>,
}

impl EngineAsrRequestRunner {
    pub(crate) fn new(
        handle: AppHandle,
        config: &ParapperConfig,
        startup_sender: Option<AsrWorkerStartupSender>,
    ) -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let startup_sender_for_spawn_error = startup_sender.clone();
        let join_handle = match thread::Builder::new()
            .name("parapper-next-asr-runner".to_string())
            .spawn(move || {
                run_engine_asr_request_worker(
                    &handle,
                    &worker_config,
                    &request_receiver,
                    &result_sender,
                    &worker_stop,
                    startup_sender,
                );
            }) {
            Ok(join_handle) => Some(join_handle),
            Err(err) => {
                let reason = format!("Failed to spawn ASR request worker: {err}");
                log::warn!("{reason}");
                if let Some(sender) = startup_sender_for_spawn_error {
                    let _ = sender.send(Err(vec![reason]));
                }
                None
            }
        };

        Self {
            request_sender: Some(request_sender),
            result_receiver,
            stop_requested,
            join_handle,
            routed_result: false,
            result_sender: None,
            normalize_state: None,
        }
    }

    pub(crate) fn from_pool(pool: &AsrRuntimePoolHandle) -> Self {
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        Self {
            request_sender: Some(pool.command_sender.clone()),
            result_receiver,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            routed_result: true,
            result_sender: Some(result_sender),
            normalize_state: Some(pool.normalize_state.clone()),
        }
    }
}

impl AsrRuntimePool {
    /// Loads the union of models once, then exposes a cheap source-lane handle.
    /// The returned pool is ready before this method returns, which lets the
    /// capture startup transaction keep `capture.play()` as its final step.
    pub(crate) fn start(
        handle: AppHandle,
        config: &ParapperConfig,
    ) -> Result<(Self, AsrRuntimePoolHandle), Vec<String>> {
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let (result_sender, _result_receiver) = std::sync::mpsc::channel();
        let (startup_sender, startup_receiver) = std::sync::mpsc::channel();
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = stop_requested.clone();
        let worker_config = config.clone();
        let join_handle = thread::Builder::new()
            .name("parapper-asr-runtime-pool".to_owned())
            .spawn(move || {
                run_engine_asr_request_worker_with_pool_startup(
                    &handle,
                    &worker_config,
                    &request_receiver,
                    &result_sender,
                    &worker_stop,
                    None,
                    Some(startup_sender),
                );
            })
            .map_err(|error| vec![format!("Failed to spawn ASR runtime pool: {error}")])?;
        let startup = startup_receiver.recv().unwrap_or_else(|_| {
            Err(vec![
                "ASR runtime pool stopped during model preload".to_owned(),
            ])
        });
        if let Err(errors) = startup {
            stop_requested.store(true, Ordering::Release);
            drop(request_sender);
            let _ = join_handle.join();
            return Err(errors);
        }
        let normalize_state = Arc::new(Mutex::new(Some(config.asr.normalize_input_audio)));
        let pool = Self {
            command_sender: Some(request_sender.clone()),
            stop_requested,
            join_handle: Some(join_handle),
        };
        Ok((
            pool,
            AsrRuntimePoolHandle {
                command_sender: request_sender,
                normalize_state,
            },
        ))
    }

    pub(crate) fn shutdown(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        self.command_sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            let started_at = Instant::now();
            while !join_handle.is_finished() && started_at.elapsed() < ASR_WORKER_JOIN_TIMEOUT {
                thread::sleep(Duration::from_millis(1));
            }
            if join_handle.is_finished() {
                let _ = join_handle.join();
            } else {
                log::warn!("Timed out waiting for ASR runtime pool shutdown; detaching worker");
            }
        }
    }
}

impl Drop for AsrRuntimePool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl parapper_stt_engine::ports::AsrRequestRunner for EngineAsrRequestRunner {
    fn set_normalize_input_audio(&mut self, enabled: bool) {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!(
                "Failed to update ASR input normalization because the request sender is closed"
            );
            return;
        };
        if let Some(state) = &self.normalize_state {
            let mut state = state.lock().expect("ASR normalize state lock");
            if *state == Some(enabled) {
                return;
            }
            *state = Some(enabled);
        }
        if let Err(err) = sender.send(AsrWorkerCommand::SetNormalizeInputAudio(enabled)) {
            log::warn!("Failed to update ASR input normalization: {err}");
        }
    }

    fn reset_streaming_sessions(&mut self) {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!(
                "Failed to reset ASR streaming sessions because the request sender is closed"
            );
            return;
        };
        if let Err(err) = sender.send(AsrWorkerCommand::ResetStreamingSessions) {
            log::warn!(
                "Failed to submit ASR streaming session reset to next runtime runner: {err}"
            );
        }
    }

    fn reset_streaming_sessions_for_source(&mut self, source: &SourceSessionKey) {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!(
                "Failed to reset source-scoped ASR streaming sessions because the request sender is closed"
            );
            return;
        };
        if let Err(err) = sender.send(AsrWorkerCommand::ResetStreamingSessionsForSource(
            source.clone(),
        )) {
            log::warn!("Failed to submit source-scoped ASR streaming reset: {err}");
        }
    }

    fn submit(&mut self, request: AsrRequest) -> bool {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!("Failed to submit ASR request because the request sender is closed");
            return false;
        };
        let command = if self.routed_result {
            AsrWorkerCommand::RequestFor(
                Box::new(request),
                self.result_sender
                    .as_ref()
                    .expect("pooled runner result sender")
                    .clone(),
            )
        } else {
            AsrWorkerCommand::Request(Box::new(request))
        };
        if let Err(err) = sender.send(command) {
            log::warn!("Failed to submit ASR request to next runtime runner: {err}");
            return false;
        }
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.result_receiver.try_recv().ok()
    }

    fn shutdown(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        self.request_sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            let started_at = Instant::now();
            while !join_handle.is_finished() && started_at.elapsed() < ASR_WORKER_JOIN_TIMEOUT {
                thread::sleep(Duration::from_millis(1));
            }
            if join_handle.is_finished() {
                if let Err(err) = join_handle.join() {
                    log::warn!("RecognitionSession ASR runner thread panicked: {err:?}");
                }
            } else {
                log::warn!(
                    "Timed out waiting for recognition ASR runner shutdown; detaching worker"
                );
            }
        }
    }
}

impl Drop for EngineAsrRequestRunner {
    fn drop(&mut self) {
        parapper_stt_engine::ports::AsrRequestRunner::shutdown(self);
    }
}

fn run_engine_asr_request_worker(
    handle: &AppHandle,
    initial_config: &ParapperConfig,
    request_receiver: &Receiver<AsrWorkerCommand>,
    result_sender: &Sender<AsrResult>,
    stop_requested: &Arc<AtomicBool>,
    startup_sender: Option<AsrWorkerStartupSender>,
) {
    run_engine_asr_request_worker_with_pool_startup(
        handle,
        initial_config,
        request_receiver,
        result_sender,
        stop_requested,
        startup_sender,
        None,
    );
}

fn run_engine_asr_request_worker_with_pool_startup(
    handle: &AppHandle,
    initial_config: &ParapperConfig,
    request_receiver: &Receiver<AsrWorkerCommand>,
    result_sender: &Sender<AsrResult>,
    stop_requested: &Arc<AtomicBool>,
    startup_sender: Option<AsrWorkerStartupSender>,
    pool_startup_sender: Option<Sender<AsrWorkerStartupResult>>,
) {
    let mut current_config = initial_config.clone();
    let (models, startup_errors) = load_asr_models(handle, &current_config);
    let mut asr = AsrExecutionRuntime::new(models);
    for reason in &startup_errors {
        log::warn!("{reason}");
        emit_missing_model_event(handle, MissingModelKind::Asr, reason.clone());
    }
    let startup_result = if startup_errors.is_empty() {
        Ok(())
    } else {
        Err(startup_errors.clone())
    };
    if let Some(sender) = startup_sender {
        let _ = sender.send(startup_result.clone());
    }
    if let Some(sender) = pool_startup_sender {
        let _ = sender.send(startup_result);
    }

    let mut pending = PendingAsrRequests::default();
    while !stop_requested.load(Ordering::Acquire) {
        // Drain all currently available commands before selecting one request.
        // Control messages remain ordered and are applied immediately; request
        // messages enter the source/model fair scheduler below.
        let first_command = if pending.ready_models.is_empty() {
            match request_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(command) => Some(command),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            None
        };
        if let Some(command) = first_command {
            apply_asr_worker_command(command, &mut pending, &mut current_config, &mut asr);
        }
        while let Ok(command) = request_receiver.try_recv() {
            apply_asr_worker_command(command, &mut pending, &mut current_config, &mut asr);
        }
        let Some(PendingAsrRequest {
            request,
            destination,
        }) = pending.pop()
        else {
            continue;
        };
        let result = run_engine_asr_request(handle, &current_config, &mut asr, request.as_ref());
        match destination {
            Some(destination) => {
                if destination.send(result).is_err() {
                    log::debug!("ASR lane receiver closed before result delivery");
                }
            }
            None if result_sender.send(result).is_err() => break,
            None => {}
        }
    }
    asr.reset_streaming_sessions();
}

fn apply_asr_worker_command(
    command: AsrWorkerCommand,
    pending: &mut PendingAsrRequests,
    current_config: &mut ParapperConfig,
    asr: &mut AsrExecutionRuntime,
) {
    match command {
        AsrWorkerCommand::Request(request) => pending.push(request, None),
        AsrWorkerCommand::RequestFor(request, destination) => {
            pending.push(request, Some(destination));
        }
        AsrWorkerCommand::SetNormalizeInputAudio(enabled) => {
            current_config.asr.normalize_input_audio = enabled;
        }
        AsrWorkerCommand::ResetStreamingSessions => {
            pending.clear();
            asr.reset_streaming_sessions();
        }
        AsrWorkerCommand::ResetStreamingSessionsForSource(source) => {
            pending.remove_source(&source);
            asr.reset_streaming_sessions_for_source(&source);
        }
    }
}

pub(crate) fn run_engine_asr_request(
    handle: &AppHandle,
    config: &ParapperConfig,
    asr: &mut AsrExecutionRuntime,
    request: &AsrRequest,
) -> AsrResult {
    let request_id = request.request_id;
    let kind = request.kind;
    let target = request.target.clone();
    let route = request.route;
    let completed_at_frame = request.created_at_frame;
    let started_at = Instant::now();
    let status = match asr.execute(
        &crate::recognition::config::stt_engine_config(config).asr,
        request,
    ) {
        Ok(transcript) => AsrResultStatus::Ok(transcript),
        Err(err) => {
            emit_asr_warning(handle, &err);
            AsrResultStatus::Failed(err.to_string())
        }
    };

    AsrResult {
        request_id,
        kind,
        target,
        route,
        status,
        completed_at_frame,
        elapsed_millis: started_at.elapsed().as_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWave {
        sample_rate: i32,
        samples: Vec<f32>,
    }

    fn read_test_wave(path: &std::path::Path) -> TestWave {
        let mut reader = hound::WavReader::open(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let spec = reader.spec();
        assert_eq!(
            spec.channels,
            1,
            "test WAV must be mono: {}",
            path.display()
        );
        let samples = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.expect("float WAV sample should decode"))
                .collect(),
            hound::SampleFormat::Int => {
                assert_eq!(
                    spec.bits_per_sample,
                    16,
                    "test WAV must use PCM16 or float samples: {}",
                    path.display()
                );
                reader
                    .samples::<i16>()
                    .map(|sample| {
                        f32::from(sample.expect("PCM16 WAV sample should decode")) / 32_768.0
                    })
                    .collect()
            }
        };
        TestWave {
            sample_rate: i32::try_from(spec.sample_rate).expect("WAV sample rate fits in i32"),
            samples,
        }
    }
    use parapper_stt_engine::SegmentCloseReason;
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use crate::{config::AsrModel, recognition::events::MissingModelEvent};
    use parapper_models::vad::VadResult;
    use parapper_stt_engine::{
        SegmentId, TurnId,
        transcription::{
            route::RecognitionRoute,
            task::{
                AsrRequestId, AsrTarget, AsrTaskKind, AudioRange, GlobalSampleIndex, TurnRevision,
                VadFrameIndex,
            },
        },
    };

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_reports_initial_preload_failure_without_waiting_for_request() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("worker-startup-signal");
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (_request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let (startup_report_sender, startup_receiver) = mpsc::channel();
        let startup_sender =
            AsrWorkerStartupSender::new(SourceId::legacy_single_source(), startup_report_sender);
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                Some(startup_sender),
            );
        });

        let startup_report = startup_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker should report initial ASR preload before any request is submitted");
        assert_eq!(
            startup_report.source_id,
            SourceId::legacy_single_source(),
            "startup readiness must retain the source selected by the recognition worker"
        );

        stop_requested.store(true, Ordering::Release);
        worker.join().expect("worker should exit cleanly");
        assert!(
            matches!(startup_report.result, Err(ref errors) if errors.iter().any(|error| error.contains("Failed to preload"))),
            "missing ASR models must be reported through startup readiness, got {:?}",
            startup_report.result
        );
        assert!(
            result_receiver.try_recv().is_err(),
            "startup preload failure must not require or synthesize an ASR request result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_processes_request_and_returns_failed_result_when_engine_missing() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("worker-processes-request");
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                None,
            );
        });
        let request = test_asr_request(7);

        request_sender
            .send(AsrWorkerCommand::Request(Box::new(request.clone())))
            .expect("worker request channel should accept a request");
        let result = result_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker should send a result for the submitted request");
        stop_requested.store(true, Ordering::Release);
        drop(request_sender);
        worker.join().expect("worker should exit cleanly");

        assert_eq!(result.request_id, request.request_id);
        assert_eq!(result.kind, request.kind);
        assert_eq!(result.target, request.target);
        assert_eq!(result.route, request.route);
        assert!(
            matches!(result.status, AsrResultStatus::Failed(ref reason) if reason.contains("was not preloaded")),
            "a missing model must surface as a failed ASR result, got {:?}",
            result.status
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_exits_after_request_channel_disconnects() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("worker-disconnect");
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        drop(request_sender);

        run_engine_asr_request_worker(
            &handle,
            &config,
            &request_receiver,
            &result_sender,
            &stop_requested,
            None,
        );

        assert!(
            result_receiver.try_recv().is_err(),
            "disconnecting the request channel without a request must not produce a result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_observes_stop_request_after_timeout_tick() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("worker-stop");
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (_request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                None,
            );
        });

        thread::sleep(Duration::from_millis(150));
        stop_requested.store(true, Ordering::Release);
        worker
            .join()
            .expect("worker should stop after a timeout tick");

        assert!(
            result_receiver.try_recv().is_err(),
            "a stop request without submitted ASR work must not produce a result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_runner_shutdown_sets_stop_and_takes_worker_handles() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("runner-shutdown");
        let mut runner = EngineAsrRequestRunner::new(handle, &config, None);

        runner.shutdown();
        runner.shutdown();

        assert!(runner.stop_requested.load(Ordering::Acquire));
        assert!(runner.request_sender.is_none());
        assert!(runner.join_handle.is_none());
    }

    #[test]
    fn engine_asr_request_runner_submit_sends_request_over_channel() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            routed_result: false,
            result_sender: None,
            normalize_state: None,
        };
        let request = test_asr_request(9);

        runner.submit(request.clone());

        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("submit should forward the ASR request to the worker channel");
        let AsrWorkerCommand::Request(submitted) = submitted else {
            panic!("submit must send an ASR request command");
        };
        assert_eq!(submitted.request_id, request.request_id);
        assert_eq!(submitted.target, request.target);
    }

    #[test]
    fn normalization_update_is_ordered_before_the_following_asr_request() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            routed_result: false,
            result_sender: None,
            normalize_state: None,
        };
        let request = test_asr_request(10);

        runner.set_normalize_input_audio(false);
        runner.submit(request.clone());

        assert!(matches!(
            request_receiver.recv_timeout(Duration::from_secs(1)),
            Ok(AsrWorkerCommand::SetNormalizeInputAudio(false))
        ));
        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("the ASR request must follow its normalization update");
        let AsrWorkerCommand::Request(submitted) = submitted else {
            panic!("normalization must be followed by the ASR request command");
        };
        assert_eq!(submitted.request_id, request.request_id);
    }

    #[test]
    fn engine_asr_request_runner_reset_streaming_sessions_sends_reset_command() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            routed_result: false,
            result_sender: None,
            normalize_state: None,
        };

        runner.reset_streaming_sessions();

        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("reset should forward the streaming reset command to the worker channel");
        assert!(matches!(
            submitted,
            AsrWorkerCommand::ResetStreamingSessions
        ));
    }

    #[test]
    fn engine_asr_request_runner_source_reset_forwards_only_the_source_key() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
            routed_result: false,
            result_sender: None,
            normalize_state: None,
        };
        let source = SourceSessionKey::new(42, SourceId::from("source-a"));

        runner.reset_streaming_sessions_for_source(&source);

        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("source reset should reach the worker command channel");
        assert!(matches!(
            submitted,
            AsrWorkerCommand::ResetStreamingSessionsForSource(ref actual) if actual == &source
        ));
        assert!(
            request_receiver
                .recv_timeout(Duration::from_millis(20))
                .is_err(),
            "source reset must not enqueue an additional pool-wide reset"
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "diagnostic: requires downloaded Nemotron 3.5 and Parakeet TDT CTC JA models"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "the ignored benchmark keeps its setup and bounded sample-count ratios together"
    )]
    fn measure_cpu4_rtf_current_worker_nemotron_streaming_delta_vs_parakeet_tdt_ctc_ja() {
        use std::time::{Duration, Instant};

        const NEMOTRON_CHUNK_SAMPLES: usize = crate::audio::ASR_SAMPLE_RATE as usize * 160 / 1_000;

        fn models_root_for_diagnostic() -> std::path::PathBuf {
            std::env::var_os("PARAPPER_MODELS_ROOT").map_or_else(
                || {
                    std::path::PathBuf::from(
                        std::env::var_os("APPDATA")
                            .expect("APPDATA or PARAPPER_MODELS_ROOT is required"),
                    )
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                },
                std::path::PathBuf::from,
            )
        }

        fn diagnostic_config(model: AsrModel, models_root: &std::path::Path) -> ParapperConfig {
            let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
            parapper_config! {
                model_dir: Some(model_dir.to_string_lossy().into_owned()),
                asr_language: model.language(),
                asr_model: model,
                asr_num_threads: 4,
                asr_normalize_input_audio: false,
                ..ParapperConfig::default()
            }
            .normalized()
        }

        fn vad_results_for_sample_len(sample_len: usize) -> Vec<VadResult> {
            let frames = sample_len.div_ceil(NEMOTRON_CHUNK_SAMPLES).max(1);
            vec![
                VadResult {
                    probability: 0.9,
                    is_speech: true,
                };
                frames
            ]
        }

        fn nemotron_interim_request(
            request_id: u64,
            chunk: &[f32],
            source_audio: &[f32],
        ) -> AsrRequest {
            let end_sample = source_audio.len();
            AsrRequest {
                request_id: AsrRequestId(request_id),
                kind: AsrTaskKind::InterimDisplay,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(end_sample as u64)),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
                detected_language: None,
                audio: chunk.to_vec(),
                vad_results: vec![VadResult {
                    probability: 0.9,
                    is_speech: true,
                }],
                source_audio: source_audio.to_vec(),
                source_vad_results: vad_results_for_sample_len(end_sample),
                close_reason: Some(SegmentCloseReason::InterimChunkReached),
                created_at_frame: VadFrameIndex(request_id),
            }
        }

        fn parakeet_completion_request(samples: &[f32]) -> AsrRequest {
            AsrRequest {
                request_id: AsrRequestId(1),
                kind: AsrTaskKind::CompletionCheck,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(
                        GlobalSampleIndex(0),
                        GlobalSampleIndex(samples.len() as u64),
                    ),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8),
                detected_language: None,
                audio: samples.to_vec(),
                vad_results: vad_results_for_sample_len(samples.len()),
                source_audio: samples.to_vec(),
                source_vad_results: vad_results_for_sample_len(samples.len()),
                close_reason: Some(SegmentCloseReason::EndSilenceReached),
                created_at_frame: VadFrameIndex(1),
            }
        }

        fn measure_nemotron_worker_delta_batch(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            samples: &[f32],
            audio_sec: f64,
            label: &str,
            batch_samples: usize,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 0..=repeats {
                let (models, preload_errors) = load_asr_models(handle, config);
                let mut asr = AsrExecutionRuntime::new(models);
                assert!(
                    preload_errors.is_empty(),
                    "Nemotron preload should succeed: {preload_errors:?}"
                );
                let started_at = Instant::now();
                let mut request_id = 1_u64;
                let mut start = 0;
                while start < samples.len() {
                    let end = (start + batch_samples).min(samples.len());
                    let chunk = &samples[start..end];
                    let result = run_engine_asr_request(
                        handle,
                        config,
                        &mut asr,
                        &nemotron_interim_request(request_id, chunk, &samples[..end]),
                    );
                    let AsrResultStatus::Ok(transcript) = result.status else {
                        panic!("Nemotron delta request failed: {:?}", result.status);
                    };
                    last_text = transcript.text;
                    request_id += 1;
                    start = end;
                }
                let elapsed = started_at.elapsed();
                if iteration == 0 {
                    println!(
                        "nemotron_worker_delta_cpu4_{label} warmup requests={} text={last_text:?}",
                        request_id - 1
                    );
                    continue;
                }
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "nemotron_worker_delta_cpu4_{label} iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!("nemotron_worker_delta_cpu4_{label} avg_rtf={avg_rtf:.3} repeats={repeats}");
            (avg_rtf, last_text)
        }

        fn measure_parakeet_worker_full(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            samples: &[f32],
            audio_sec: f64,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let (models, preload_errors) = load_asr_models(handle, config);
            let mut asr = AsrExecutionRuntime::new(models);
            assert!(
                preload_errors.is_empty(),
                "Parakeet preload should succeed: {preload_errors:?}"
            );
            let warmup = run_engine_asr_request(
                handle,
                config,
                &mut asr,
                &parakeet_completion_request(samples),
            );
            let AsrResultStatus::Ok(transcript) = warmup.status else {
                panic!("Parakeet warmup failed: {:?}", warmup.status);
            };
            println!(
                "parakeet_tdt_ctc_ja_worker_full_cpu4 warmup text: {:?}",
                transcript.text
            );

            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 1..=repeats {
                let started_at = Instant::now();
                let result = run_engine_asr_request(
                    handle,
                    config,
                    &mut asr,
                    &parakeet_completion_request(samples),
                );
                let elapsed = started_at.elapsed();
                let AsrResultStatus::Ok(transcript) = result.status else {
                    panic!("Parakeet request failed: {:?}", result.status);
                };
                last_text = transcript.text;
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "parakeet_tdt_ctc_ja_worker_full_cpu4 iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!("parakeet_tdt_ctc_ja_worker_full_cpu4 avg_rtf={avg_rtf:.3} repeats={repeats}");
            (avg_rtf, last_text)
        }

        let handle = tauri_test_handle();
        let models_root = models_root_for_diagnostic();
        let nemotron_dir = models_root.join(crate::model::catalog::asr_model_dir_name(
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        ));
        let wav_path = std::env::var_os("PARAPPER_ASR_RTF_WAV").map_or_else(
            || nemotron_dir.join("test_wavs").join("ja.wav"),
            std::path::PathBuf::from,
        );
        let wave = read_test_wave(&wav_path);
        assert_eq!(
            wave.sample_rate,
            i32::try_from(crate::audio::ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32")
        );
        let audio_sec = wave.samples.len() as f64 / f64::from(crate::audio::ASR_SAMPLE_RATE);
        println!(
            "current worker RTF input: {} samples={} audio_sec={audio_sec:.3}",
            wav_path.display(),
            wave.samples.len(),
        );

        let nemotron_config =
            diagnostic_config(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8, &models_root);
        let parakeet_config =
            diagnostic_config(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, &models_root);
        let nemotron_measurements = [
            ("160ms", NEMOTRON_CHUNK_SAMPLES),
            ("320ms", NEMOTRON_CHUNK_SAMPLES * 2),
            ("640ms", NEMOTRON_CHUNK_SAMPLES * 4),
            ("1280ms", NEMOTRON_CHUNK_SAMPLES * 8),
            ("all", wave.samples.len()),
        ]
        .into_iter()
        .map(|(label, batch_samples)| {
            let (rtf, text) = measure_nemotron_worker_delta_batch(
                &handle,
                &nemotron_config,
                &wave.samples,
                audio_sec,
                label,
                batch_samples,
            );
            (label, rtf, text)
        })
        .collect::<Vec<_>>();
        let (parakeet_rtf, parakeet_text) =
            measure_parakeet_worker_full(&handle, &parakeet_config, &wave.samples, audio_sec);

        for (label, nemotron_rtf, _) in &nemotron_measurements {
            println!(
                "current worker RTF comparison cpu4: nemotron_delta_{label}={nemotron_rtf:.3} parakeet_tdt_ctc_ja={parakeet_rtf:.3} ratio={:.3}",
                nemotron_rtf / parakeet_rtf
            );
        }
        for (label, _, nemotron_text) in &nemotron_measurements {
            assert!(
                !nemotron_text.trim().is_empty(),
                "Nemotron worker delta {label} should produce non-empty text"
            );
        }
        assert!(
            !parakeet_text.trim().is_empty(),
            "Parakeet worker full should produce non-empty text"
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "diagnostic: requires downloaded Nemotron English streaming model"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "the ignored benchmark keeps its setup and bounded sample-count ratios together"
    )]
    fn measure_cpu4_rtf_current_worker_nemotron_en_streaming_batches() {
        use std::time::{Duration, Instant};

        const NEMOTRON_CHUNK_SAMPLES: usize = crate::audio::ASR_SAMPLE_RATE as usize * 160 / 1_000;

        fn models_root_for_diagnostic() -> std::path::PathBuf {
            std::env::var_os("PARAPPER_MODELS_ROOT").map_or_else(
                || {
                    std::path::PathBuf::from(
                        std::env::var_os("APPDATA")
                            .expect("APPDATA or PARAPPER_MODELS_ROOT is required"),
                    )
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                },
                std::path::PathBuf::from,
            )
        }

        fn diagnostic_config(model: AsrModel, models_root: &std::path::Path) -> ParapperConfig {
            let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
            parapper_config! {
                model_dir: Some(model_dir.to_string_lossy().into_owned()),
                asr_language: model.language(),
                asr_model: model,
                asr_num_threads: 4,
                asr_normalize_input_audio: false,
                ..ParapperConfig::default()
            }
            .normalized()
        }

        fn vad_results_for_sample_len(sample_len: usize) -> Vec<VadResult> {
            let frames = sample_len.div_ceil(NEMOTRON_CHUNK_SAMPLES).max(1);
            vec![
                VadResult {
                    probability: 0.9,
                    is_speech: true,
                };
                frames
            ]
        }

        fn nemotron_interim_request(
            model: AsrModel,
            request_id: u64,
            chunk: &[f32],
            source_audio: &[f32],
        ) -> AsrRequest {
            let end_sample = source_audio.len();
            AsrRequest {
                request_id: AsrRequestId(request_id),
                kind: AsrTaskKind::InterimDisplay,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(end_sample as u64)),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(model),
                detected_language: None,
                audio: chunk.to_vec(),
                vad_results: vec![VadResult {
                    probability: 0.9,
                    is_speech: true,
                }],
                source_audio: source_audio.to_vec(),
                source_vad_results: vad_results_for_sample_len(end_sample),
                close_reason: Some(SegmentCloseReason::InterimChunkReached),
                created_at_frame: VadFrameIndex(request_id),
            }
        }

        fn measure_nemotron_worker_delta_batch(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            model: AsrModel,
            samples: &[f32],
            audio_sec: f64,
            label: &str,
            batch_samples: usize,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 0..=repeats {
                let (models, preload_errors) = load_asr_models(handle, config);
                let mut asr = AsrExecutionRuntime::new(models);
                assert!(
                    preload_errors.is_empty(),
                    "Nemotron preload should succeed: {preload_errors:?}"
                );
                let started_at = Instant::now();
                let mut request_id = 1_u64;
                let mut start = 0;
                while start < samples.len() {
                    let end = (start + batch_samples).min(samples.len());
                    let result = run_engine_asr_request(
                        handle,
                        config,
                        &mut asr,
                        &nemotron_interim_request(
                            model,
                            request_id,
                            &samples[start..end],
                            &samples[..end],
                        ),
                    );
                    let AsrResultStatus::Ok(transcript) = result.status else {
                        panic!("Nemotron {label} request failed: {:?}", result.status);
                    };
                    last_text = transcript.text;
                    request_id += 1;
                    start = end;
                }
                let elapsed = started_at.elapsed();
                if iteration == 0 {
                    println!(
                        "nemotron_en_worker_delta_cpu4_{label} warmup requests={} text={last_text:?}",
                        request_id - 1
                    );
                    continue;
                }
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "nemotron_en_worker_delta_cpu4_{label} iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!(
                "nemotron_en_worker_delta_cpu4_{label} avg_rtf={avg_rtf:.3} repeats={repeats}"
            );
            (avg_rtf, last_text)
        }

        let model = AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8;
        let handle = tauri_test_handle();
        let models_root = models_root_for_diagnostic();
        let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
        let wav_path = std::env::var_os("PARAPPER_NEMOTRON_EN_RTF_WAV")
            .or_else(|| std::env::var_os("PARAPPER_ASR_RTF_WAV"))
            .map_or_else(
                || model_dir.join("test_wavs").join("0.wav"),
                std::path::PathBuf::from,
            );
        let wave = read_test_wave(&wav_path);
        assert_eq!(
            wave.sample_rate,
            i32::try_from(crate::audio::ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32")
        );
        let audio_sec = wave.samples.len() as f64 / f64::from(crate::audio::ASR_SAMPLE_RATE);
        println!(
            "current worker Nemotron EN RTF input: {} samples={} audio_sec={audio_sec:.3}",
            wav_path.display(),
            wave.samples.len(),
        );

        let config = diagnostic_config(model, &models_root);
        let measurements = [
            ("160ms", NEMOTRON_CHUNK_SAMPLES),
            ("320ms", NEMOTRON_CHUNK_SAMPLES * 2),
            ("640ms", NEMOTRON_CHUNK_SAMPLES * 4),
            ("1280ms", NEMOTRON_CHUNK_SAMPLES * 8),
            ("all", wave.samples.len()),
        ]
        .into_iter()
        .map(|(label, batch_samples)| {
            let (rtf, text) = measure_nemotron_worker_delta_batch(
                &handle,
                &config,
                model,
                &wave.samples,
                audio_sec,
                label,
                batch_samples,
            );
            (label, rtf, text)
        })
        .collect::<Vec<_>>();

        for (label, rtf, text) in &measurements {
            println!("current worker Nemotron EN RTF cpu4: delta_{label}={rtf:.3} text={text:?}");
            assert!(
                !text.trim().is_empty(),
                "Nemotron EN worker delta {label} should produce non-empty text"
            );
        }
    }

    fn test_asr_request(request_id: u64) -> AsrRequest {
        AsrRequest {
            request_id: AsrRequestId(request_id),
            kind: AsrTaskKind::CompletionCheck,
            target: AsrTarget::new(
                TurnId(1),
                TurnRevision(0),
                AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(4)),
                Some(SegmentId(1)),
                Some(SegmentId(1)),
            ),
            route: RecognitionRoute::from_model(ParapperConfig::default().asr.model),
            detected_language: None,
            audio: vec![0.0, 0.25, -0.25, 0.5],
            vad_results: vec![VadResult {
                probability: 0.9,
                is_speech: true,
            }],
            source_audio: vec![0.0, 0.25, -0.25, 0.5],
            source_vad_results: vec![VadResult {
                probability: 0.9,
                is_speech: true,
            }],
            close_reason: Some(SegmentCloseReason::EndSilenceReached),
            created_at_frame: VadFrameIndex(1),
        }
    }

    fn config_with_missing_model_dir(test_name: &str) -> ParapperConfig {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let missing_dir = std::env::temp_dir().join(format!(
            "parapper-missing-asr-model-{test_name}-{}-{unique}",
            std::process::id()
        ));
        parapper_config! {
            model_dir: Some(missing_dir.to_string_lossy().into_owned()),
            ..ParapperConfig::default()
        }
    }

    #[test]
    fn pooled_dispatch_rotates_sources_while_preserving_each_source_fifo() {
        let source_a = SourceSessionKey::new(1, SourceId::from("source-a"));
        let source_b = SourceSessionKey::new(1, SourceId::from("source-b"));
        let model_one = AsrModel::ReazonSpeechK2V2;
        let mut pending = PendingAsrRequests::default();
        for (id, source) in [
            (1, source_a.clone()),
            (2, source_a.clone()),
            (3, source_b.clone()),
            (4, source_b.clone()),
        ] {
            let mut request = test_asr_request(id);
            request.route = RecognitionRoute::from_model(model_one);
            request.target.set_source_session(source);
            pending.push(Box::new(request), None);
        }

        let dispatched = (0..4)
            .map(|_| pending.pop().expect("queued request"))
            .map(|request| request.request.request_id.0)
            .collect::<Vec<_>>();
        assert_eq!(dispatched, vec![1, 3, 2, 4]);
    }

    #[test]
    fn pooled_dispatch_rotates_ready_models_instead_of_draining_one_model_first() {
        let source_a = SourceSessionKey::new(1, SourceId::from("source-a"));
        let source_b = SourceSessionKey::new(1, SourceId::from("source-b"));
        let model_one = AsrModel::ReazonSpeechK2V2;
        let model_two = AsrModel::NemoParakeetTdt0_6BV2Int8;
        let mut pending = PendingAsrRequests::default();
        for (id, source, model) in [
            (1, source_a.clone(), model_one),
            (2, source_a.clone(), model_one),
            (3, source_b.clone(), model_two),
            (4, source_b.clone(), model_two),
        ] {
            let mut request = test_asr_request(id);
            request.route = RecognitionRoute::from_model(model);
            request.target.set_source_session(source);
            pending.push(Box::new(request), None);
        }

        let dispatched = (0..4)
            .map(|_| pending.pop().expect("queued request"))
            .map(|request| request.request.request_id.0)
            .collect::<Vec<_>>();
        assert_eq!(dispatched, vec![1, 3, 2, 4]);
    }

    #[test]
    fn pooled_runner_routes_reverse_completion_to_the_originating_source_handle() {
        let (request_sender, request_receiver) = mpsc::channel();
        let pool = AsrRuntimePoolHandle {
            command_sender: request_sender,
            normalize_state: Arc::new(Mutex::new(None)),
        };
        let mut runner_a = EngineAsrRequestRunner::from_pool(&pool);
        let mut runner_b = EngineAsrRequestRunner::from_pool(&pool);
        let request_a = test_asr_request(41);
        let mut request_b = test_asr_request(42);
        request_b
            .target
            .set_source_session(SourceSessionKey::new(3, SourceId::from("source-b")));

        assert!(runner_a.submit(request_a.clone()));
        assert!(runner_b.submit(request_b.clone()));
        let AsrWorkerCommand::RequestFor(request_a, destination_a) =
            request_receiver.recv().expect("request A")
        else {
            panic!("pooled runner must use routed request commands");
        };
        let AsrWorkerCommand::RequestFor(request_b, destination_b) =
            request_receiver.recv().expect("request B")
        else {
            panic!("pooled runner must use routed request commands");
        };
        let result_b = AsrResult {
            request_id: request_b.request_id,
            kind: request_b.kind,
            target: request_b.target.clone(),
            route: request_b.route,
            status: AsrResultStatus::Failed("b".to_owned()),
            completed_at_frame: request_b.created_at_frame,
            elapsed_millis: 2,
        };
        let result_a = AsrResult {
            request_id: request_a.request_id,
            kind: request_a.kind,
            target: request_a.target.clone(),
            route: request_a.route,
            status: AsrResultStatus::Failed("a".to_owned()),
            completed_at_frame: request_a.created_at_frame,
            elapsed_millis: 1,
        };
        destination_b.send(result_b.clone()).expect("B receiver");
        destination_a.send(result_a.clone()).expect("A receiver");
        assert_eq!(runner_a.try_recv_result(), Some(result_a));
        assert_eq!(runner_b.try_recv_result(), Some(result_b));
    }

    #[test]
    fn pooled_source_reset_discards_only_queued_requests_for_that_source() {
        let source_a = SourceSessionKey::new(8, SourceId::from("source-a"));
        let source_b = SourceSessionKey::new(8, SourceId::from("source-b"));
        let mut pending = PendingAsrRequests::default();
        for (id, source) in [(51, source_a.clone()), (52, source_b.clone())] {
            let mut request = test_asr_request(id);
            request.target.set_source_session(source);
            pending.push(Box::new(request), None);
        }
        pending.remove_source(&source_a);
        assert_eq!(
            pending
                .pop()
                .expect("other source remains queued")
                .request
                .request_id
                .0,
            52
        );
        assert!(pending.pop().is_none());
    }

    #[cfg(not(target_os = "macos"))]
    fn tauri_test_handle() -> tauri::AppHandle {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("test app should build");
        app.handle().clone()
    }
}
