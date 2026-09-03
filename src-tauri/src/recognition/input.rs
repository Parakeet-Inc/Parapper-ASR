use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU8, AtomicU64, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use parapper_models::vad::{OnnxRuntimeSileroVadEngine, VadEngine, VadResult};
use tauri::{AppHandle, Emitter};

use crate::{
    audio::{
        AudioInputProcessor, ExplicitAudioLaneStartup, InputChunk, ProcessedAudioChunk,
        RunningAudioInput, SourceQueueOverrun,
    },
    config::{CaptureEndpointConfig, ParapperConfig, RecognitionSourceConfig, SttProfileConfig},
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    model::vad_model_path,
    recognition::events::{VadState, VadStateEvent},
};
use parapper_stt_engine::{SourceId, SourceIdentitySnapshot};

use super::{
    AsrRuntimePool, AsrRuntimePoolHandle, AsrWorkerStartupReport, AsrWorkerStartupSender,
    DeliveryTurnOutputSink, RecognitionDriver, RecognitionDriverHandle, RecognitionShutdownResult,
    TurnOutputSink,
    config::stt_runtime_parameters,
    input_source::{
        InputDisconnectPolicy, InputSourceConfig, InputSourceLifetime, RunningInputSource,
    },
};

pub struct RunningRecognitionInput {
    stop_control: Arc<RecognitionStopControl>,
    source_lifetime: Option<InputSourceLifetime>,
    /// One prepared CPAL capture per distinct physical endpoint. Explicit
    /// profile mode keeps all captures alive until its shared scheduler drains.
    captures: Vec<RunningAudioInput>,
    worker_joins: Vec<Box<dyn RecognitionWorkerJoinHandle>>,
    /// Pools are grouped by an exactly matching ASR execution configuration;
    /// profiles with different decoder/hotword settings must never share one.
    asr_pools: Vec<AsrRuntimePool>,
}

pub(crate) use parapper_stt_server::RecognitionStreamEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum RecognitionStopMode {
    Running = 0,
    Graceful = 1,
    Cancel = 2,
}

#[derive(Default)]
struct RecognitionStopControl {
    mode: AtomicU8,
}

impl RecognitionStopControl {
    fn request(&self, mode: RecognitionStopMode) {
        self.mode.fetch_max(mode as u8, Ordering::AcqRel);
    }

    fn mode(&self) -> RecognitionStopMode {
        match self.mode.load(Ordering::Acquire) {
            0 => RecognitionStopMode::Running,
            1 => RecognitionStopMode::Graceful,
            _ => RecognitionStopMode::Cancel,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecognitionInputMode {
    Legacy,
    Explicit,
    Profile,
}

#[derive(Debug, PartialEq, Eq)]
struct ProfileCaptureGroup {
    endpoint_id: String,
    device_host: Option<String>,
    device_id: Option<String>,
    profile_ids: Vec<String>,
}

type ProfileAsrPools = (Vec<AsrRuntimePool>, Vec<(String, AsrRuntimePoolHandle)>);

fn recognition_input_mode(config: &ParapperConfig) -> RecognitionInputMode {
    if !config.stt_profiles.is_empty() {
        RecognitionInputMode::Profile
    } else if config.input.capture_endpoint.is_some() {
        RecognitionInputMode::Explicit
    } else {
        RecognitionInputMode::Legacy
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkerStartupCollectionError {
    ChannelClosed {
        source_ids: Vec<SourceId>,
    },
    UnexpectedSource {
        source_id: SourceId,
    },
    Asr {
        source_id: SourceId,
        errors: Vec<String>,
    },
}

impl std::fmt::Display for WorkerStartupCollectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChannelClosed { source_ids } => write!(
                formatter,
                "recognition sources [{}] closed before reporting ASR startup readiness",
                source_ids
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::UnexpectedSource { source_id } => {
                write!(
                    formatter,
                    "unexpected or duplicate ASR startup report from source {source_id}"
                )
            }
            Self::Asr { source_id, errors } => write!(
                formatter,
                "recognition source {source_id} failed to preload required ASR models: {}",
                errors.join("; ")
            ),
        }
    }
}

impl std::error::Error for WorkerStartupCollectionError {}

fn collect_worker_startup_results(
    expected_source_ids: &[SourceId],
    receiver: &Receiver<AsrWorkerStartupReport>,
) -> std::result::Result<(), WorkerStartupCollectionError> {
    let mut pending = expected_source_ids.iter().cloned().collect::<HashSet<_>>();
    while !pending.is_empty() {
        match receiver.recv() {
            Ok(report) if !pending.remove(&report.source_id) => {
                return Err(WorkerStartupCollectionError::UnexpectedSource {
                    source_id: report.source_id,
                });
            }
            Ok(AsrWorkerStartupReport {
                source_id: _,
                result: Ok(()),
            }) => {}
            Ok(AsrWorkerStartupReport {
                source_id,
                result: Err(errors),
            }) => {
                return Err(WorkerStartupCollectionError::Asr { source_id, errors });
            }
            Err(_) => {
                return Err(WorkerStartupCollectionError::ChannelClosed {
                    source_ids: expected_source_ids
                        .iter()
                        .filter(|source_id| pending.contains(*source_id))
                        .cloned()
                        .collect(),
                });
            }
        }
    }
    Ok(())
}

trait RecognitionWorkerJoinHandle: Send {
    fn source_id(&self) -> &SourceId;
    fn join(self: Box<Self>) -> RecognitionShutdownResult;
}

struct ThreadRecognitionWorkerJoinHandle {
    source_id: SourceId,
    join_handle: JoinHandle<RecognitionShutdownResult>,
}

impl RecognitionWorkerJoinHandle for ThreadRecognitionWorkerJoinHandle {
    fn source_id(&self) -> &SourceId {
        &self.source_id
    }

    fn join(self: Box<Self>) -> RecognitionShutdownResult {
        match self.join_handle.join() {
            Ok(result) => result,
            Err(error) => {
                log::warn!(
                    "Recognition input worker for source {} panicked: {error:?}",
                    self.source_id
                );
                RecognitionShutdownResult::Cancelled
            }
        }
    }
}

fn join_all_workers(
    workers: Vec<Box<dyn RecognitionWorkerJoinHandle>>,
) -> RecognitionShutdownResult {
    if workers.is_empty() {
        return RecognitionShutdownResult::Cancelled;
    }

    let mut aggregate = RecognitionShutdownResult::Completed;
    for worker in workers {
        let source_id = worker.source_id().clone();
        let result = worker.join();
        log::debug!("Recognition input worker for source {source_id} stopped: {result:?}");
        aggregate = aggregate_shutdown_result(aggregate, result);
    }
    aggregate
}

fn aggregate_shutdown_result(
    current: RecognitionShutdownResult,
    next: RecognitionShutdownResult,
) -> RecognitionShutdownResult {
    use RecognitionShutdownResult::{Cancelled, Completed, TimedOut};
    match (current, next) {
        (TimedOut, _) | (_, TimedOut) => TimedOut,
        (Cancelled, _) | (_, Cancelled) => Cancelled,
        (Completed, Completed) => Completed,
    }
}

#[derive(Debug)]
pub enum RecognitionStartError {
    AudioInput(anyhow::Error),
    Asr(anyhow::Error),
    Busy,
}

impl std::fmt::Display for RecognitionStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioInput(err) | Self::Asr(err) => std::fmt::Display::fmt(err, f),
            Self::Busy => write!(f, "another recognition session is active"),
        }
    }
}

impl std::error::Error for RecognitionStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AudioInput(err) | Self::Asr(err) => err.source(),
            Self::Busy => None,
        }
    }
}

pub(crate) struct RuntimeConfigState {
    config: RwLock<Arc<ParapperConfig>>,
    revision: AtomicU64,
}

impl RuntimeConfigState {
    pub(crate) fn new(config: ParapperConfig) -> Self {
        Self {
            config: RwLock::new(Arc::new(config)),
            revision: AtomicU64::new(0),
        }
    }

    pub(crate) fn replace(&self, config: ParapperConfig) {
        if let Ok(mut current) = self.config.write() {
            if **current == config {
                return;
            }
            *current = Arc::new(config);
            // Publish the revision while the write lock still owns the matching Arc. A
            // consumer that observes this revision either blocks on this writer or reads a
            // later complete pair; it cannot pair this revision with the previous config.
            self.revision.fetch_add(1, Ordering::Release);
        }
    }

    pub(crate) fn snapshot(&self) -> Result<ParapperConfig> {
        self.config
            .read()
            .map(|config| (**config).clone())
            .map_err(|_| anyhow!("runtime config lock is poisoned"))
    }

    fn take_updated_config(&self, cursor: &mut RuntimeConfigCursor) -> Option<RuntimeConfigUpdate> {
        if self.revision.load(Ordering::Acquire) == cursor.revision {
            return None;
        }

        // Writers update the Arc and publish its revision under the same write lock. Reading
        // both while holding the corresponding read lock keeps the snapshot pair coherent,
        // including when several intermediate revisions are coalesced.
        let config = self.config.read().ok()?;
        let revision = self.revision.load(Ordering::Acquire);
        if revision == cursor.revision {
            return None;
        }
        let config = config.clone();
        cursor.revision = revision;
        Some(RuntimeConfigUpdate { revision, config })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct RuntimeConfigCursor {
    revision: u64,
}

#[derive(Debug, Clone)]
struct RuntimeConfigUpdate {
    revision: u64,
    config: Arc<ParapperConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeConfigDirty {
    bits: u8,
}

impl RuntimeConfigDirty {
    const AUDIO: u8 = 1 << 0;
    const VAD: u8 = 1 << 1;
    const DRIVER: u8 = 1 << 2;

    fn between(current: &ParapperConfig, next: &ParapperConfig) -> Self {
        let mut bits = 0;
        if current.input.volume_db.to_bits() != next.input.volume_db.to_bits()
            || current.input.muted != next.input.muted
        {
            bits |= Self::AUDIO;
        }
        if current.segmentation.vad_threshold.to_bits() != next.segmentation.vad_threshold.to_bits()
        {
            bits |= Self::VAD;
        }
        if driver_config_changed(current, next) {
            bits |= Self::DRIVER;
        }
        Self { bits }
    }

    fn is_empty(self) -> bool {
        self.bits == 0
    }

    fn vad(self) -> bool {
        self.bits & Self::VAD != 0
    }

    fn driver(self) -> bool {
        self.bits & Self::DRIVER != 0
    }
}

fn driver_config_changed(current: &ParapperConfig, next: &ParapperConfig) -> bool {
    stt_runtime_parameters(current) != stt_runtime_parameters(next)
}

/// Resolves the configuration owned by a scheduler source.  In profile mode,
/// the shared state intentionally contains every profile, but each lane must
/// only observe the one selected at startup by its source identity.
fn resolve_runtime_config_for_source(
    config: &ParapperConfig,
    source_id: &SourceId,
    profile_mode: bool,
) -> Result<ParapperConfig> {
    if profile_mode {
        config.config_for_stt_profile(&source_id.to_string())
    } else {
        Ok(config.clone())
    }
}

impl RunningRecognitionInput {
    pub fn start(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: Arc<RuntimeConfigState>,
    ) -> Result<Self, RecognitionStartError> {
        config
            .validate()
            .context("recognition input config is invalid")
            .map_err(RecognitionStartError::AudioInput)?;
        match recognition_input_mode(config) {
            RecognitionInputMode::Profile => {
                return Self::start_profiles(&handle, config, &runtime_config);
            }
            RecognitionInputMode::Explicit => {
                return Self::start_explicit(&handle, config, &runtime_config);
            }
            RecognitionInputMode::Legacy => {}
        }

        let source = InputSourceConfig::from_config(config)
            .start(config)
            .map_err(RecognitionStartError::AudioInput)?;
        let output_sink = Box::new(DeliveryTurnOutputSink::new(handle.clone(), config));
        Self::start_with_source_and_sink(handle, config, runtime_config, source, output_sink, None)
    }

    pub(crate) fn start_with_source_and_sink(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: Arc<RuntimeConfigState>,
        source: RunningInputSource,
        output_sink: Box<dyn TurnOutputSink>,
        activity_sender: Option<Sender<RecognitionStreamEvent>>,
    ) -> Result<Self, RecognitionStartError> {
        config
            .validate()
            .context("recognition input config is invalid")
            .map_err(RecognitionStartError::AudioInput)?;
        let source = source.into_parts();
        let source_lifetime = source.lifetime;
        let receiver = source.receiver;
        let source_sample_rate = source.source_sample_rate;
        let disconnect_policy = source.disconnect_policy;
        let startup =
            build_recognition_startup(&handle, config, source_sample_rate, activity_sender, None)
                .map_err(RecognitionStartError::AudioInput)?;
        let stop_control = Arc::new(RecognitionStopControl::default());
        let source_identity = SourceIdentitySnapshot::legacy_single_source();
        let source_id = source_identity.source_id.clone();
        let (startup_report_sender, startup_report_receiver) =
            std::sync::mpsc::channel::<AsrWorkerStartupReport>();
        let asr_startup_sender =
            AsrWorkerStartupSender::new(source_id.clone(), startup_report_sender);
        let RecognitionStartup {
            audio_processor,
            vad_stage,
        } = startup;
        let worker_startup = RecognitionWorkerStartup {
            config: config.clone(),
            source_identity,
            receiver,
            audio_processor,
            vad_stage,
            asr_startup_sender,
            output_sink,
            queue_overrun: None,
            asr_pool: None,
            profile_mode: false,
        };
        let worker_join = spawn_recognition_worker(
            handle,
            config.clone(),
            runtime_config,
            worker_startup,
            stop_control.clone(),
            disconnect_policy,
        )
        .with_context(|| format!("failed to spawn recognition source {source_id}"))
        .map_err(RecognitionStartError::AudioInput)?;

        let mut running = Self {
            stop_control,
            source_lifetime: Some(source_lifetime),
            captures: Vec::new(),
            worker_joins: vec![worker_join],
            asr_pools: Vec::new(),
        };
        let readiness = collect_worker_startup_results(&[source_id], &startup_report_receiver);
        if let Err(error) = readiness {
            let _ = running.stop_inner(RecognitionStopMode::Cancel);
            return Err(RecognitionStartError::Asr(anyhow!(error)));
        }
        Ok(running)
    }

    fn start_explicit(
        handle: &AppHandle,
        config: &ParapperConfig,
        runtime_config: &Arc<RuntimeConfigState>,
    ) -> Result<Self, RecognitionStartError> {
        let prepared = RunningAudioInput::prepare_explicit(config)
            .map_err(RecognitionStartError::AudioInput)?;
        let mut capture = prepared.input;
        let stop_control = Arc::new(RecognitionStopControl::default());
        let (asr_pool, asr_pool_handle) = match AsrRuntimePool::start(handle.clone(), config) {
            Ok(pool) => pool,
            Err(errors) => {
                capture.cancel();
                return Err(RecognitionStartError::Asr(anyhow!(errors.join("; "))));
            }
        };
        let (startup_report_sender, startup_report_receiver) =
            std::sync::mpsc::channel::<AsrWorkerStartupReport>();
        let (worker_specs, expected_source_ids) = build_explicit_worker_specs(
            handle,
            config,
            prepared.lanes,
            &asr_pool_handle,
            &startup_report_sender,
        )
        .inspect_err(|_| {
            capture.cancel();
        })?;
        drop(startup_report_sender);

        let mut running = Self {
            stop_control,
            source_lifetime: None,
            captures: vec![capture],
            worker_joins: Vec::with_capacity(1),
            asr_pools: vec![asr_pool],
        };
        match spawn_recognition_scheduler(
            handle.clone(),
            config.clone(),
            runtime_config.clone(),
            worker_specs,
            running.stop_control.clone(),
        ) {
            Ok(worker_join) => running.worker_joins.push(worker_join),
            Err(error) => {
                let _ = running.stop_inner(RecognitionStopMode::Cancel);
                return Err(RecognitionStartError::AudioInput(anyhow!(
                    "failed to spawn explicit recognition scheduler: {error}"
                )));
            }
        }

        if let Err(error) =
            collect_worker_startup_results(&expected_source_ids, &startup_report_receiver)
        {
            let _ = running.stop_inner(RecognitionStopMode::Cancel);
            return Err(RecognitionStartError::Asr(anyhow!(error)));
        }

        if let Err(error) = running.play_all_captures() {
            let source_ids = config
                .input
                .recognition_sources
                .iter()
                .map(|source| source.source_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = running.stop_inner(RecognitionStopMode::Cancel);
            return Err(RecognitionStartError::AudioInput(anyhow!(
                "failed to play explicit capture for recognition sources [{source_ids}]: {error}"
            )));
        }

        Ok(running)
    }

    fn start_profiles(
        handle: &AppHandle,
        config: &ParapperConfig,
        runtime_config: &Arc<RuntimeConfigState>,
    ) -> Result<Self, RecognitionStartError> {
        let stop_control = Arc::new(RecognitionStopControl::default());
        let (mut captures, lanes) = prepare_profile_captures(config)?;

        let (mut asr_pools, pool_handles) =
            start_profile_asr_pools(handle, config).inspect_err(|_| {
                for capture in &mut captures {
                    capture.cancel();
                }
            })?;
        let (startup_report_sender, startup_report_receiver) =
            std::sync::mpsc::channel::<AsrWorkerStartupReport>();
        let (worker_specs, expected_source_ids) = build_profile_worker_specs(
            handle,
            config,
            lanes,
            &pool_handles,
            &startup_report_sender,
        )
        .inspect_err(|_| {
            for capture in &mut captures {
                capture.cancel();
            }
            for pool in &mut asr_pools {
                pool.shutdown();
            }
        })?;
        drop(startup_report_sender);

        let mut running = Self {
            stop_control,
            source_lifetime: None,
            captures,
            worker_joins: Vec::with_capacity(1),
            asr_pools,
        };
        match spawn_recognition_scheduler(
            handle.clone(),
            config.clone(),
            runtime_config.clone(),
            worker_specs,
            running.stop_control.clone(),
        ) {
            Ok(worker_join) => running.worker_joins.push(worker_join),
            Err(error) => {
                let _ = running.stop_inner(RecognitionStopMode::Cancel);
                return Err(RecognitionStartError::AudioInput(anyhow!(
                    "failed to spawn STT profile recognition scheduler: {error}"
                )));
            }
        }

        if let Err(error) =
            collect_worker_startup_results(&expected_source_ids, &startup_report_receiver)
        {
            let _ = running.stop_inner(RecognitionStopMode::Cancel);
            return Err(RecognitionStartError::Asr(anyhow!(error)));
        }

        if let Err(error) = running.play_all_captures() {
            let source_ids = expected_source_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let _ = running.stop_inner(RecognitionStopMode::Cancel);
            return Err(RecognitionStartError::AudioInput(anyhow!(
                "failed to play STT profile captures for [{source_ids}]: {error}"
            )));
        }

        Ok(running)
    }

    pub fn stop(mut self) -> RecognitionShutdownResult {
        self.stop_inner(RecognitionStopMode::Graceful)
    }

    pub(crate) fn cancel(mut self) -> RecognitionShutdownResult {
        self.stop_inner(RecognitionStopMode::Cancel)
    }

    fn stop_inner(&mut self, mode: RecognitionStopMode) -> RecognitionShutdownResult {
        self.stop_control.request(mode);
        // Stop the producer before waiting for a graceful worker drain. This closes
        // desktop input channels; network senders are closed by their session owner.
        self.source_lifetime.take();
        for mut capture in std::mem::take(&mut self.captures) {
            if mode == RecognitionStopMode::Cancel {
                capture.cancel();
            }
            drop(capture);
        }
        let result = join_all_workers(std::mem::take(&mut self.worker_joins));
        // Lane runners only release their handles during the join above. The
        // shared model/runtime must outlive every lane and is joined last.
        for mut pool in std::mem::take(&mut self.asr_pools) {
            pool.shutdown();
        }
        result
    }

    fn play_all_captures(&self) -> Result<()> {
        for capture in &self.captures {
            capture.play()?;
        }
        Ok(())
    }
}

impl Drop for RunningRecognitionInput {
    fn drop(&mut self) {
        let _ = self.stop_inner(RecognitionStopMode::Graceful);
    }
}

fn spawn_recognition_worker(
    handle: AppHandle,
    config: ParapperConfig,
    runtime_config: Arc<RuntimeConfigState>,
    startup: RecognitionWorkerStartup,
    stop_control: Arc<RecognitionStopControl>,
    disconnect_policy: InputDisconnectPolicy,
) -> std::io::Result<Box<dyn RecognitionWorkerJoinHandle>> {
    let source_id = startup.source_identity.source_id.clone();
    let worker_source_id = source_id.clone();
    let join_handle = thread::Builder::new()
        .name("parapper-recognition-input".to_owned())
        .spawn(move || {
            run_recognition_input_worker(
                &handle,
                &config,
                runtime_config,
                startup,
                &stop_control,
                disconnect_policy,
            )
        })?;
    Ok(Box::new(ThreadRecognitionWorkerJoinHandle {
        source_id: worker_source_id,
        join_handle,
    }))
}

/// Starts the single shared scheduler used by explicit multi-source capture.
/// Legacy and WebSocket input intentionally keep `spawn_recognition_worker` so
/// their existing one-source lifecycle and wire contracts are unchanged.
fn spawn_recognition_scheduler(
    handle: AppHandle,
    config: ParapperConfig,
    runtime_config: Arc<RuntimeConfigState>,
    startups: Vec<RecognitionWorkerStartup>,
    stop_control: Arc<RecognitionStopControl>,
) -> std::io::Result<Box<dyn RecognitionWorkerJoinHandle>> {
    let join_handle = thread::Builder::new()
        .name("parapper-recognition-scheduler".to_owned())
        .spawn(move || {
            let mut scheduler =
                RecognitionSourceSchedulerLoop::new(&handle, &config, &runtime_config, startups);
            drive_recognition_input_loop(
                &mut scheduler,
                &stop_control,
                InputDisconnectPolicy::Cancel,
            )
        })?;
    Ok(Box::new(ThreadRecognitionWorkerJoinHandle {
        source_id: SourceId::from("explicit-source-scheduler"),
        join_handle,
    }))
}

fn run_recognition_input_worker(
    handle: &AppHandle,
    config: &ParapperConfig,
    runtime_config: Arc<RuntimeConfigState>,
    startup: RecognitionWorkerStartup,
    stop_control: &RecognitionStopControl,
    disconnect_policy: InputDisconnectPolicy,
) -> RecognitionShutdownResult {
    let mut outer_loop = RecognitionOuterLoop::new(handle, config, runtime_config, startup);
    drive_recognition_input_loop(&mut outer_loop, stop_control, disconnect_policy)
}

trait RecognitionInputLoop {
    fn step(&mut self) -> RecognitionLoopStep;
    fn stop(&mut self) -> RecognitionShutdownResult;
    fn cancel(&mut self);
}

fn drive_recognition_input_loop(
    outer_loop: &mut impl RecognitionInputLoop,
    stop_control: &RecognitionStopControl,
    disconnect_policy: InputDisconnectPolicy,
) -> RecognitionShutdownResult {
    loop {
        if stop_control.mode() == RecognitionStopMode::Cancel {
            outer_loop.cancel();
            return RecognitionShutdownResult::Cancelled;
        }
        match outer_loop.step() {
            RecognitionLoopStep::Progressed | RecognitionLoopStep::Idle => {}
            RecognitionLoopStep::InputDisconnected => {
                if stop_control.mode() == RecognitionStopMode::Running
                    && disconnect_policy == InputDisconnectPolicy::Cancel
                {
                    outer_loop.cancel();
                    return RecognitionShutdownResult::Cancelled;
                }
                return outer_loop.stop();
            }
        }
    }
}

struct RecognitionOuterLoop<'a> {
    handle: &'a AppHandle,
    source_id: SourceId,
    queue_overrun: Option<Arc<SourceQueueOverrun>>,
    observed_overrun_epoch: u64,
    runtime_config: Arc<RuntimeConfigState>,
    runtime_config_cursor: RuntimeConfigCursor,
    applied_config: Arc<ParapperConfig>,
    // A profile may be removed while its shared capture endpoint remains
    // active for other sources.  Keep this lane fail-closed without stopping
    // those unrelated sources.
    profile_mode: bool,
    profile_disabled: bool,
    receiver: Receiver<InputChunk>,
    pending_input: PendingInputChunks,
    pending_vad_frames: PendingVadFrames,
    audio_processor: AudioInputProcessor,
    vad_stage: Option<RecognitionVadStage>,
    driver: Option<Box<dyn RecognitionDriverHandle>>,
}

/// The explicit capture path owns exactly one scheduling thread.  Each source
/// keeps its own processor, VAD, driver, clock and FIFO; only the choice of the
/// next ready source is shared.
struct RecognitionSourceSchedulerLoop<S> {
    sources: Vec<S>,
    ready: ReadySourceScheduler,
    input_closed: Vec<bool>,
    stopped: Vec<bool>,
    idle_wait: Duration,
}

/// Production source-runtime seam for the shared scheduler.  Test sources use
/// this same trait and `RecognitionSourceSchedulerLoop::step_inner`; they do
/// not duplicate the round-robin or lifecycle decisions.
trait ScheduledRecognitionSource {
    fn source_id(&self) -> &SourceId;
    fn take_queue_overrun(&mut self) -> Option<SourceDiscontinuity>;
    fn collect_available_input(&mut self) -> RecognitionLoopStep;
    fn has_pending_input(&self) -> bool;
    fn process_one_pending_input(&mut self) -> bool;
    fn step_driver_only(&mut self);
    fn flush_input_for_source_stop(&mut self);
    fn finish_source_drain_if_idle(&mut self) -> bool;
    fn shutdown_source(&mut self) -> RecognitionShutdownResult;
    fn cancel_source(&mut self);
}

struct RecognitionStartup {
    audio_processor: AudioInputProcessor,
    vad_stage: RecognitionVadStage,
}

struct RecognitionWorkerStartup {
    config: ParapperConfig,
    source_identity: SourceIdentitySnapshot,
    receiver: Receiver<InputChunk>,
    audio_processor: AudioInputProcessor,
    vad_stage: RecognitionVadStage,
    asr_startup_sender: AsrWorkerStartupSender,
    output_sink: Box<dyn TurnOutputSink>,
    queue_overrun: Option<Arc<SourceQueueOverrun>>,
    asr_pool: Option<AsrRuntimePoolHandle>,
    profile_mode: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecognitionLoopStep {
    Progressed,
    Idle,
    InputDisconnected,
}

/// A gap in one source's audio timeline.  It is intentionally source-scoped:
/// a discontinuity never authorizes clearing another source's VAD, Segment,
/// Turn or ASR state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceDiscontinuity {
    QueueOverrun { epoch: u64, dropped_samples: u64 },
}

#[derive(Default)]
struct PendingInputChunks {
    chunks: VecDeque<InputChunk>,
}

#[derive(Default)]
struct PendingVadFrames {
    frames: VecDeque<ProcessedAudioChunk>,
}

/// The shared deterministic scheduling policy used by explicit source runtimes.
/// A source can occur in the ready queue at most once until it has received its
/// fixed one-chunk quota and is requeued at the tail.
#[derive(Default)]
struct ReadySourceScheduler {
    ready: VecDeque<usize>,
}

impl ReadySourceScheduler {
    fn mark_ready(&mut self, source_index: usize) {
        if !self.ready.contains(&source_index) {
            self.ready.push_back(source_index);
        }
    }

    fn next_ready(&mut self) -> Option<usize> {
        self.ready.pop_front()
    }

    fn requeue_after_quota(&mut self, source_index: usize, still_ready: bool) {
        if still_ready {
            self.mark_ready(source_index);
        }
    }
}

struct VadFrame {
    asr_samples: Vec<f32>,
    result: VadResult,
}

struct RecognitionVadStage {
    handle: AppHandle,
    vad: Box<dyn VadEngine>,
    activity_sender: Option<Sender<RecognitionStreamEvent>>,
    was_speech: bool,
}

impl PendingInputChunks {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    fn pop_front(&mut self) -> Option<InputChunk> {
        self.chunks.pop_front()
    }

    fn collect_from(
        &mut self,
        receiver: &Receiver<InputChunk>,
        wait_timeout: Duration,
    ) -> RecognitionLoopStep {
        if self.chunks.is_empty() {
            match receiver.recv_timeout(wait_timeout) {
                Ok(chunk) => self.chunks.push_back(chunk),
                Err(RecvTimeoutError::Timeout) => return RecognitionLoopStep::Idle,
                Err(RecvTimeoutError::Disconnected) => {
                    return RecognitionLoopStep::InputDisconnected;
                }
            }
        }

        loop {
            match receiver.try_recv() {
                Ok(chunk) => self.chunks.push_back(chunk),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if self.chunks.is_empty() {
                        return RecognitionLoopStep::InputDisconnected;
                    }
                    break;
                }
            }
        }

        RecognitionLoopStep::Progressed
    }

    fn collect_available_from(&mut self, receiver: &Receiver<InputChunk>) -> RecognitionLoopStep {
        loop {
            match receiver.try_recv() {
                Ok(chunk) => self.chunks.push_back(chunk),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return if self.chunks.is_empty() {
                        RecognitionLoopStep::InputDisconnected
                    } else {
                        RecognitionLoopStep::Progressed
                    };
                }
            }
        }
        if self.chunks.is_empty() {
            RecognitionLoopStep::Idle
        } else {
            RecognitionLoopStep::Progressed
        }
    }
}

impl PendingVadFrames {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.frames.len()
    }

    fn push(&mut self, samples: ProcessedAudioChunk) {
        self.frames.push_back(samples);
    }

    fn pop_front(&mut self) -> Option<ProcessedAudioChunk> {
        self.frames.pop_front()
    }
}

impl RecognitionVadStage {
    fn new(
        handle: AppHandle,
        config: &ParapperConfig,
        activity_sender: Option<Sender<RecognitionStreamEvent>>,
    ) -> Result<Self> {
        let vad_path = vad_model_path(&handle)?;
        let vad = OnnxRuntimeSileroVadEngine::new(&vad_path, config.segmentation.vad_threshold)?;
        Ok(Self {
            handle,
            vad: Box::new(vad),
            activity_sender,
            was_speech: false,
        })
    }

    fn update_config(&mut self, config: &ParapperConfig) {
        self.vad.set_threshold(config.segmentation.vad_threshold);
    }

    fn process(&mut self, samples: ProcessedAudioChunk) -> Result<VadFrame> {
        let result = self.vad.process(samples.vad_samples())?;
        let state = if result.is_speech {
            VadState::Speech
        } else {
            VadState::Silence
        };
        if result.is_speech
            && !self.was_speech
            && let Some(sender) = &self.activity_sender
        {
            let _ = sender.send(RecognitionStreamEvent::SpeechStarted);
        }
        self.was_speech = result.is_speech;
        let _ = self.handle.emit(
            "parapper://vad-state",
            VadStateEvent {
                state,
                probability: result.probability,
            },
        );

        Ok(VadFrame {
            asr_samples: samples.into_asr_samples(),
            result,
        })
    }
}

fn build_recognition_startup(
    handle: &AppHandle,
    config: &ParapperConfig,
    source_sample_rate: u32,
    activity_sender: Option<Sender<RecognitionStreamEvent>>,
    source_id: Option<SourceId>,
) -> Result<RecognitionStartup> {
    let audio_processor = AudioInputProcessor::initialize_for_source(
        handle.clone(),
        config,
        source_sample_rate,
        source_id,
    )?;
    let vad_stage = match RecognitionVadStage::new(handle.clone(), config, activity_sender) {
        Ok(stage) => stage,
        Err(err) => {
            emit_parapper_error(
                handle,
                ParapperErrorType::Vad,
                ErrorSeverity::Fatal,
                Some(err.to_string()),
            );
            return Err(err);
        }
    };
    Ok(RecognitionStartup {
        audio_processor,
        vad_stage,
    })
}

fn build_explicit_worker_specs(
    handle: &AppHandle,
    config: &ParapperConfig,
    lanes: Vec<ExplicitAudioLaneStartup>,
    asr_pool: &AsrRuntimePoolHandle,
    startup_report_sender: &Sender<AsrWorkerStartupReport>,
) -> Result<(Vec<RecognitionWorkerStartup>, Vec<SourceId>), RecognitionStartError> {
    let mut worker_specs = Vec::with_capacity(lanes.len());
    let mut expected_source_ids = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let source_id = lane.identity.source_id.clone();
        let startup = build_recognition_startup(
            handle,
            config,
            lane.source_sample_rate,
            None,
            Some(source_id.clone()),
        )
        .map_err(|error| {
            RecognitionStartError::AudioInput(anyhow!(
                "recognition source {source_id} failed to initialize: {error}"
            ))
        })?;
        let source_config = config
            .with_asr_route_for_source(source_id.as_str())
            .map_err(RecognitionStartError::Asr)?;
        let RecognitionStartup {
            audio_processor,
            vad_stage,
        } = startup;
        expected_source_ids.push(source_id.clone());
        let output_sink =
            DeliveryTurnOutputSink::new_for_source(handle.clone(), config, source_id.as_str())
                .map_err(RecognitionStartError::Asr)?;
        worker_specs.push(RecognitionWorkerStartup {
            source_identity: lane.identity,
            config: source_config,
            receiver: lane.receiver,
            audio_processor,
            vad_stage,
            asr_startup_sender: AsrWorkerStartupSender::new(
                source_id,
                startup_report_sender.clone(),
            ),
            output_sink: Box::new(output_sink),
            queue_overrun: Some(lane.queue_overrun),
            asr_pool: Some(asr_pool.clone()),
            profile_mode: false,
        });
    }
    Ok((worker_specs, expected_source_ids))
}

fn prepare_profile_captures(
    config: &ParapperConfig,
) -> Result<(Vec<RunningAudioInput>, Vec<ExplicitAudioLaneStartup>), RecognitionStartError> {
    let mut captures = Vec::new();
    let mut lanes = Vec::new();
    for ProfileCaptureGroup {
        endpoint_id,
        device_host,
        device_id,
        profile_ids,
    } in profile_capture_groups(config)
    {
        let sources = profile_ids
            .iter()
            .map(|profile_id| {
                config
                    .resolved_stt_profile(profile_id)
                    .map(|profile| profile_recognition_source(profile, &endpoint_id))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(RecognitionStartError::AudioInput)?;
        let prepared = match (device_host, device_id) {
            (Some(device_host), Some(device_id)) => {
                let endpoint = CaptureEndpointConfig {
                    id: endpoint_id,
                    device_host,
                    device_id,
                    device_name: None,
                };
                RunningAudioInput::prepare_profile_endpoint(&endpoint, &sources)
            }
            (None, None) => {
                let default_profile_config = config
                    .config_for_stt_profile(&profile_ids[0])
                    .map_err(RecognitionStartError::AudioInput)?;
                RunningAudioInput::prepare_default_profile_endpoint(
                    &default_profile_config,
                    &endpoint_id,
                    &sources,
                )
            }
            _ => Err(anyhow!(
                "STT profile capture endpoint must configure both device host and device id"
            )),
        }
        .map_err(RecognitionStartError::AudioInput)?;
        captures.push(prepared.input);
        lanes.extend(prepared.lanes);
    }
    Ok((captures, lanes))
}

fn profile_capture_groups(config: &ParapperConfig) -> Vec<ProfileCaptureGroup> {
    let mut groups: Vec<(Option<String>, Option<String>, Vec<String>)> = Vec::new();
    for profile in config.stt_profiles.iter().filter(|profile| profile.enabled) {
        let key = (
            profile.input.device_host.clone(),
            profile.input.device_id.clone(),
        );
        if let Some((_, _, profile_ids)) = groups
            .iter_mut()
            .find(|(host, id, _)| *host == key.0 && *id == key.1)
        {
            profile_ids.push(profile.id.clone());
        } else {
            groups.push((key.0, key.1, vec![profile.id.clone()]));
        }
    }
    groups
        .into_iter()
        .map(|(host, id, profile_ids)| ProfileCaptureGroup {
            endpoint_id: profile_capture_endpoint_id(host.as_deref(), id.as_deref()),
            device_host: host,
            device_id: id,
            profile_ids,
        })
        .collect()
}

fn profile_capture_endpoint_id(device_host: Option<&str>, device_id: Option<&str>) -> String {
    match (device_host, device_id) {
        (Some(host), Some(id)) => format!("stt-device:{}:{host}:{}:{id}", host.len(), id.len()),
        (None, None) => "stt-default-capture".to_owned(),
        _ => "stt-invalid-capture".to_owned(),
    }
}

fn profile_recognition_source(
    profile: &SttProfileConfig,
    capture_endpoint_id: &str,
) -> RecognitionSourceConfig {
    RecognitionSourceConfig {
        source_id: profile.id.clone(),
        speaker_label: profile.name.clone(),
        capture_endpoint_id: capture_endpoint_id.to_owned(),
        channel_index: profile.input.channel_index,
        delivery_profile_id: profile.delivery_profile_id.clone(),
        asr_route_policy: None,
    }
}

fn start_profile_asr_pools(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> std::result::Result<ProfileAsrPools, RecognitionStartError> {
    let mut pools = Vec::new();
    let mut keyed_handles: Vec<(String, AsrRuntimePoolHandle)> = Vec::new();
    let mut profile_handles = Vec::with_capacity(config.stt_profiles.len());
    for profile in config.stt_profiles.iter().filter(|profile| profile.enabled) {
        let profile_config = config
            .config_for_stt_profile(&profile.id)
            .map_err(RecognitionStartError::Asr)?;
        let key = serde_json::to_string(&profile_config.asr).map_err(|error| {
            RecognitionStartError::Asr(anyhow!(
                "failed to serialize ASR configuration for STT profile {:?}: {error}",
                profile.id
            ))
        })?;
        let handle = if let Some((_, handle)) = keyed_handles
            .iter()
            .find(|(existing_key, _)| *existing_key == key)
        {
            handle.clone()
        } else {
            let mut pool_config = profile_config;
            // The profile's own ASR config is the complete pool key. The
            // top-level profile collection is only a startup plan and must not
            // make this pool preload unrelated profile models.
            pool_config.stt_profiles.clear();
            match AsrRuntimePool::start(handle.clone(), &pool_config) {
                Ok((pool, pool_handle)) => {
                    pools.push(pool);
                    keyed_handles.push((key.clone(), pool_handle.clone()));
                    pool_handle
                }
                Err(errors) => {
                    for pool in &mut pools {
                        pool.shutdown();
                    }
                    return Err(RecognitionStartError::Asr(anyhow!(errors.join("; "))));
                }
            }
        };
        profile_handles.push((profile.id.clone(), handle));
    }
    Ok((pools, profile_handles))
}

fn build_profile_worker_specs(
    handle: &AppHandle,
    config: &ParapperConfig,
    lanes: Vec<ExplicitAudioLaneStartup>,
    pool_handles: &[(String, AsrRuntimePoolHandle)],
    startup_report_sender: &Sender<AsrWorkerStartupReport>,
) -> Result<(Vec<RecognitionWorkerStartup>, Vec<SourceId>), RecognitionStartError> {
    let mut worker_specs = Vec::with_capacity(lanes.len());
    let mut expected_source_ids = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let source_id = lane.identity.source_id.clone();
        let source_config = config
            .config_for_stt_profile(source_id.as_str())
            .map_err(RecognitionStartError::AudioInput)?;
        let startup = build_recognition_startup(
            handle,
            &source_config,
            lane.source_sample_rate,
            None,
            Some(source_id.clone()),
        )
        .map_err(|error| {
            RecognitionStartError::AudioInput(anyhow!(
                "STT profile {source_id} failed to initialize: {error}"
            ))
        })?;
        let pool = pool_handles
            .iter()
            .find(|(profile_id, _)| profile_id == source_id.as_str())
            .map(|(_, pool)| pool)
            .with_context(|| format!("STT profile {source_id:?} has no ASR runtime pool"))
            .map_err(RecognitionStartError::Asr)?;
        let output_sink =
            DeliveryTurnOutputSink::new_for_stt_profile(handle.clone(), config, source_id.as_str())
                .map_err(RecognitionStartError::Asr)?;
        let RecognitionStartup {
            audio_processor,
            vad_stage,
        } = startup;
        expected_source_ids.push(source_id.clone());
        worker_specs.push(RecognitionWorkerStartup {
            source_identity: lane.identity,
            config: source_config,
            receiver: lane.receiver,
            audio_processor,
            vad_stage,
            asr_startup_sender: AsrWorkerStartupSender::new(
                source_id,
                startup_report_sender.clone(),
            ),
            output_sink: Box::new(output_sink),
            queue_overrun: Some(lane.queue_overrun),
            asr_pool: Some(pool.clone()),
            profile_mode: true,
        });
    }
    Ok((worker_specs, expected_source_ids))
}

impl<'a> RecognitionOuterLoop<'a> {
    fn new(
        handle: &'a AppHandle,
        _config: &'a ParapperConfig,
        runtime_config: Arc<RuntimeConfigState>,
        startup: RecognitionWorkerStartup,
    ) -> Self {
        let RecognitionWorkerStartup {
            config: source_config,
            source_identity,
            receiver,
            audio_processor,
            vad_stage,
            asr_startup_sender,
            output_sink,
            queue_overrun,
            asr_pool,
            profile_mode,
        } = startup;
        let source_id = source_identity.source_id.clone();
        let pooled = asr_pool.is_some();
        let driver = if let Some(pool) = asr_pool {
            RecognitionDriver::new_for_production_with_pool_and_output_sink_and_source_identity(
                handle,
                &source_config,
                &pool,
                source_identity.clone(),
                output_sink,
            )
        } else if source_identity == SourceIdentitySnapshot::legacy_single_source() {
            RecognitionDriver::new_for_production_with_output_sink(
                handle,
                &source_config,
                Some(asr_startup_sender.clone()),
                output_sink,
            )
        } else {
            RecognitionDriver::new_for_production_with_output_sink_and_source_identity(
                handle,
                &source_config,
                Some(asr_startup_sender.clone()),
                source_identity,
                output_sink,
            )
        };
        if pooled {
            // Pool preload completed before the scheduler was spawned; this
            // report marks this lane's scheduler construction as ready.
            let _ = asr_startup_sender.send(Ok(()));
        }
        Self {
            handle,
            source_id,
            observed_overrun_epoch: queue_overrun
                .as_ref()
                .map_or(0, |overrun| overrun.snapshot().0),
            queue_overrun,
            runtime_config,
            runtime_config_cursor: RuntimeConfigCursor::default(),
            applied_config: Arc::new(source_config),
            profile_mode,
            profile_disabled: false,
            receiver,
            pending_input: PendingInputChunks::default(),
            pending_vad_frames: PendingVadFrames::default(),
            audio_processor,
            vad_stage: Some(vad_stage),
            driver: Some(Box::new(driver)),
        }
    }

    fn step_inner(&mut self) -> RecognitionLoopStep {
        self.apply_runtime_config_update();
        let current_config = self.applied_config.clone();
        let input_status = self.collect_input(&current_config);
        if matches!(input_status, RecognitionLoopStep::InputDisconnected) {
            return RecognitionLoopStep::InputDisconnected;
        }

        let audio_progressed = self.process_pending_input(&current_config);
        let vad_progressed = self.process_pending_vad_frames();
        if let Some(driver) = self.driver.as_mut() {
            driver.step();
        }

        if audio_progressed || vad_progressed {
            RecognitionLoopStep::Progressed
        } else {
            input_status
        }
    }

    fn collect_available_input(&mut self) -> RecognitionLoopStep {
        self.pending_input.collect_available_from(&self.receiver)
    }

    /// Services one input chunk and every VAD frame produced from that chunk.
    /// It intentionally does not drain `pending_input`: the source scheduler is
    /// responsible for returning a still-ready source to the round-robin tail.
    fn process_one_pending_input(&mut self) -> bool {
        let Some(chunk) = self.pending_input.pop_front() else {
            return false;
        };
        if self.profile_disabled {
            drop(chunk);
            return true;
        }
        let current_config = self.applied_config.clone();
        let vad_enabled = self.vad_stage.is_some();
        let pending_vad_frames = &mut self.pending_vad_frames;
        self.audio_processor
            .process(&chunk, &current_config, |samples| {
                if vad_enabled {
                    pending_vad_frames.push(samples);
                }
            });
        // A native device callback may be smaller than one fixed 32 ms VAD
        // frame. Consuming it still releases capture-queue capacity and must
        // keep the shared scheduler hot; otherwise each partial callback is
        // followed by an artificial idle sleep and multiple live devices
        // inevitably fall behind their clocks.
        let _ = self.process_pending_vad_frames();
        true
    }

    fn step_driver_only(&mut self) {
        if let Some(driver) = self.driver.as_mut() {
            // This only advances ASR completions/pending work. Runtime ticks and
            // source audio clocks advance exclusively in `push_vad_frame`.
            driver.step();
        }
    }

    fn flush_input_for_source_stop(&mut self) {
        if let Some(driver) = self.driver.as_mut() {
            driver.flush_input();
        }
    }

    fn finish_source_drain_if_idle(&mut self) -> bool {
        let Some(driver) = self.driver.as_mut() else {
            return true;
        };
        if driver.has_pending_work() {
            return false;
        }
        driver.finalize_open_turn_after_drain();
        !driver.has_pending_work()
    }

    fn take_queue_overrun(&mut self) -> Option<SourceDiscontinuity> {
        let overrun = self.queue_overrun.as_ref()?;
        let (epoch, dropped_samples) = overrun.snapshot();
        if epoch == self.observed_overrun_epoch {
            return None;
        }
        self.observed_overrun_epoch = epoch;
        Some(SourceDiscontinuity::QueueOverrun {
            epoch,
            dropped_samples,
        })
    }

    fn stop_inner(&mut self) -> RecognitionShutdownResult {
        if let Some(mut driver) = self.driver.take() {
            return driver.shutdown();
        }
        RecognitionShutdownResult::Cancelled
    }

    fn cancel_inner(&mut self) {
        if let Some(mut driver) = self.driver.take() {
            driver.cancel();
        }
    }

    fn apply_runtime_config_update(&mut self) {
        let Some(update) = self
            .runtime_config
            .take_updated_config(&mut self.runtime_config_cursor)
        else {
            return;
        };
        debug_assert_eq!(update.revision, self.runtime_config_cursor.revision);
        // A source removed from the active profile plan cannot safely rebuild
        // its VAD/driver in place. It remains closed until the next global
        // recognition start, while still advancing this cursor on later edits.
        if self.profile_disabled {
            return;
        }
        let resolved_config = match resolve_runtime_config_for_source(
            &update.config,
            &self.source_id,
            self.profile_mode,
        ) {
            Ok(config) => config,
            Err(error) => {
                log::error!(
                    "stt profile source {} is no longer resolvable after runtime config update: {error}",
                    self.source_id
                );
                self.profile_disabled = true;
                self.pending_input = PendingInputChunks::default();
                self.pending_vad_frames = PendingVadFrames::default();
                self.vad_stage = None;
                self.cancel_inner();
                return;
            }
        };
        let dirty = RuntimeConfigDirty::between(&self.applied_config, &resolved_config);
        self.applied_config = Arc::new(resolved_config);
        if dirty.is_empty() {
            return;
        }
        if dirty.driver()
            && let Some(driver) = self.driver.as_mut()
        {
            driver.update_runtime_parameters(stt_runtime_parameters(&self.applied_config));
        }
        if dirty.vad()
            && let Some(vad_stage) = self.vad_stage.as_mut()
        {
            vad_stage.update_config(&self.applied_config);
        }
    }

    fn collect_input(&mut self, current_config: &ParapperConfig) -> RecognitionLoopStep {
        self.pending_input.collect_from(
            &self.receiver,
            recognition_input_wait_timeout(current_config),
        )
    }

    fn process_pending_input(&mut self, current_config: &ParapperConfig) -> bool {
        let mut progressed = false;
        while let Some(chunk) = self.pending_input.pop_front() {
            if self.profile_disabled {
                drop(chunk);
                progressed = true;
                continue;
            }
            let vad_enabled = self.vad_stage.is_some();
            let pending_vad_frames = &mut self.pending_vad_frames;
            self.audio_processor
                .process(&chunk, current_config, |samples| {
                    progressed = true;
                    if vad_enabled {
                        pending_vad_frames.push(samples);
                    }
                });
        }
        progressed
    }

    fn process_pending_vad_frames(&mut self) -> bool {
        let mut progressed = false;
        while let Some(samples) = self.pending_vad_frames.pop_front() {
            let Some(vad_stage) = self.vad_stage.as_mut() else {
                continue;
            };
            match vad_stage.process(samples) {
                Ok(frame) => {
                    progressed = true;
                    if let Some(driver) = self.driver.as_mut() {
                        driver.push_vad_frame(&frame.asr_samples, frame.result);
                    }
                }
                Err(err) => {
                    emit_parapper_error(
                        self.handle,
                        ParapperErrorType::Vad,
                        ErrorSeverity::Warning,
                        Some(err.to_string()),
                    );
                }
            }
        }
        progressed
    }
}

impl ScheduledRecognitionSource for RecognitionOuterLoop<'_> {
    fn source_id(&self) -> &SourceId {
        &self.source_id
    }

    fn take_queue_overrun(&mut self) -> Option<SourceDiscontinuity> {
        Self::take_queue_overrun(self)
    }

    fn collect_available_input(&mut self) -> RecognitionLoopStep {
        Self::collect_available_input(self)
    }

    fn has_pending_input(&self) -> bool {
        !self.pending_input.is_empty()
    }

    fn process_one_pending_input(&mut self) -> bool {
        Self::process_one_pending_input(self)
    }

    fn step_driver_only(&mut self) {
        Self::step_driver_only(self);
    }

    fn flush_input_for_source_stop(&mut self) {
        Self::flush_input_for_source_stop(self);
    }

    fn finish_source_drain_if_idle(&mut self) -> bool {
        Self::finish_source_drain_if_idle(self)
    }

    fn shutdown_source(&mut self) -> RecognitionShutdownResult {
        Self::stop_inner(self)
    }

    fn cancel_source(&mut self) {
        Self::cancel_inner(self);
    }
}

impl<'a> RecognitionSourceSchedulerLoop<RecognitionOuterLoop<'a>> {
    fn new(
        handle: &'a AppHandle,
        config: &'a ParapperConfig,
        runtime_config: &Arc<RuntimeConfigState>,
        startups: Vec<RecognitionWorkerStartup>,
    ) -> Self {
        let idle_wait = recognition_input_wait_timeout(config);
        RecognitionSourceSchedulerLoop::from_sources(
            startups
                .into_iter()
                .map(|startup| {
                    RecognitionOuterLoop::new(handle, config, runtime_config.clone(), startup)
                })
                .collect(),
            idle_wait,
        )
    }
}

impl<S: ScheduledRecognitionSource> RecognitionSourceSchedulerLoop<S> {
    fn from_sources(sources: Vec<S>, idle_wait: Duration) -> Self {
        let source_count = sources.len();
        Self {
            sources,
            ready: ReadySourceScheduler::default(),
            input_closed: vec![false; source_count],
            stopped: vec![false; source_count],
            idle_wait,
        }
    }

    fn all_sources_stopped(&self) -> bool {
        self.stopped.iter().all(|stopped| *stopped)
    }

    fn collect_ready_sources(&mut self) {
        for index in 0..self.sources.len() {
            if self.stopped[index] || self.input_closed[index] {
                continue;
            }
            let source = &mut self.sources[index];
            if let Some(discontinuity) = source.take_queue_overrun() {
                log::warn!(
                    "recognition source {} discontinuity={discontinuity:?}; continuing this source",
                    source.source_id(),
                );
            }
            if matches!(
                source.collect_available_input(),
                RecognitionLoopStep::InputDisconnected
            ) && !source.has_pending_input()
            {
                self.input_closed[index] = true;
                source.flush_input_for_source_stop();
                log::info!(
                    "recognition source {} input closed; flushing and draining only this source",
                    source.source_id()
                );
                continue;
            }
            if source.has_pending_input() {
                self.ready.mark_ready(index);
            }
        }
    }

    fn step_inner(&mut self) -> RecognitionLoopStep {
        self.step_inner_with_sleeper(thread::sleep)
    }

    /// Executes one shared scheduling turn. The injected sleeper is a narrow
    /// test seam: a real scheduler sleeps only when it consumed no source
    /// input, so tests can assert that partial native-rate packets never add
    /// artificial latency before they form a VAD frame.
    fn step_inner_with_sleeper(&mut self, sleep: impl FnOnce(Duration)) -> RecognitionLoopStep {
        self.collect_ready_sources();
        let mut progressed = false;
        if let Some(index) = self.ready.next_ready() {
            let source = &mut self.sources[index];
            progressed = source.process_one_pending_input();
            self.ready.requeue_after_quota(
                index,
                source.has_pending_input() && !self.input_closed[index] && !self.stopped[index],
            );
        }

        // Results and pending finalization remain source-local.  Stepping an
        // idle source does not advance its audio clock because no VAD frame is
        // pushed here.
        for (index, source) in self.sources.iter_mut().enumerate() {
            if !self.stopped[index] {
                source.step_driver_only();
            }
        }

        for index in 0..self.sources.len() {
            if self.input_closed[index]
                && !self.stopped[index]
                && self.sources[index].finish_source_drain_if_idle()
            {
                let result = self.sources[index].shutdown_source();
                log::debug!(
                    "recognition source {} drained after input close: {result:?}",
                    self.sources[index].source_id()
                );
                self.stopped[index] = true;
            }
        }

        if self.all_sources_stopped() {
            RecognitionLoopStep::InputDisconnected
        } else if progressed {
            RecognitionLoopStep::Progressed
        } else {
            sleep(self.idle_wait);
            RecognitionLoopStep::Idle
        }
    }

    fn stop_inner(&mut self) -> RecognitionShutdownResult {
        self.sources.iter_mut().enumerate().fold(
            RecognitionShutdownResult::Completed,
            |result, (index, source)| {
                if self.stopped[index] {
                    result
                } else {
                    aggregate_shutdown_result(result, source.shutdown_source())
                }
            },
        )
    }

    fn cancel_inner(&mut self) {
        for (index, source) in self.sources.iter_mut().enumerate() {
            if !self.stopped[index] {
                source.cancel_source();
            }
        }
    }
}

impl<S: ScheduledRecognitionSource> RecognitionInputLoop for RecognitionSourceSchedulerLoop<S> {
    fn step(&mut self) -> RecognitionLoopStep {
        self.step_inner()
    }

    fn stop(&mut self) -> RecognitionShutdownResult {
        self.stop_inner()
    }

    fn cancel(&mut self) {
        self.cancel_inner();
    }
}

impl RecognitionInputLoop for RecognitionOuterLoop<'_> {
    fn step(&mut self) -> RecognitionLoopStep {
        self.step_inner()
    }

    fn stop(&mut self) -> RecognitionShutdownResult {
        self.stop_inner()
    }

    fn cancel(&mut self) {
        self.cancel_inner();
    }
}

fn recognition_input_wait_timeout(config: &ParapperConfig) -> Duration {
    let half_vad_interval_ms = u64::from(config.segmentation.vad_interval_ms.max(1)).div_ceil(2);
    Duration::from_millis(half_vad_interval_ms.max(1))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use super::{
        PendingInputChunks, PendingVadFrames, ProfileCaptureGroup, RecognitionInputLoop,
        RecognitionInputMode, RecognitionLoopStep, RecognitionOuterLoop, RecognitionShutdownResult,
        RecognitionSourceSchedulerLoop, RecognitionStopControl, RecognitionStopMode,
        RecognitionVadStage, RuntimeConfigCursor, RuntimeConfigDirty, RuntimeConfigState,
        ScheduledRecognitionSource, SourceDiscontinuity, WorkerStartupCollectionError,
        build_recognition_startup, collect_worker_startup_results, drive_recognition_input_loop,
        join_all_workers, profile_capture_endpoint_id, profile_capture_groups,
        recognition_input_mode, recognition_input_wait_timeout, resolve_runtime_config_for_source,
    };
    use crate::{
        audio::{ASR_SAMPLE_RATE, AudioInputProcessor, InputChunk, ProcessedAudioChunk},
        config::{
            AsrConfig, CaptureEndpointConfig, NoiseCancellationConfig, ParapperConfig,
            RecognitionSourceConfig, SegmentationConfig, SttProfileConfig, SttProfileDisplayColor,
            SttProfileInputConfig, TurnConfig,
        },
        recognition::AsrWorkerStartupReport,
        recognition::input_source::InputDisconnectPolicy,
    };
    use parapper_models::vad::{VadEngine, VadResult};
    use parapper_stt_engine::SourceId;

    struct RecordingVad {
        observed: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    fn stt_profile(
        id: &str,
        name: &str,
        device_host: Option<&str>,
        device_id: Option<&str>,
        channel_index: u16,
    ) -> SttProfileConfig {
        SttProfileConfig {
            id: id.to_owned(),
            name: name.to_owned(),
            enabled: true,
            neo_http_enabled: true,
            developer_http_enabled: true,
            display_color: SttProfileDisplayColor::Green,
            input: SttProfileInputConfig {
                device_host: device_host.map(str::to_owned),
                device_id: device_id.map(str::to_owned),
                device_name: None,
                channel_index,
                volume_percent: 100,
                muted: false,
            },
            noise_cancellation: NoiseCancellationConfig::default(),
            segmentation: SegmentationConfig::default(),
            turn: TurnConfig::default(),
            asr: AsrConfig::default(),
            delivery_profile_id: None,
        }
    }

    impl VadEngine for RecordingVad {
        fn process(&mut self, samples: &[f32]) -> anyhow::Result<VadResult> {
            self.observed.lock().unwrap().push(samples.to_vec());
            Ok(VadResult {
                probability: 0.9,
                is_speech: true,
            })
        }
    }

    struct ScriptedInputLoop {
        steps: VecDeque<RecognitionLoopStep>,
        observed_steps: usize,
        stopped: bool,
        cancelled: bool,
    }

    struct FakeWorkerJoin {
        source_id: SourceId,
        result: RecognitionShutdownResult,
        joined: Arc<Mutex<Vec<SourceId>>>,
    }

    impl super::RecognitionWorkerJoinHandle for FakeWorkerJoin {
        fn source_id(&self) -> &SourceId {
            &self.source_id
        }

        fn join(self: Box<Self>) -> RecognitionShutdownResult {
            self.joined.lock().unwrap().push(self.source_id.clone());
            self.result
        }
    }

    struct FakeStopWorkerJoin {
        source_id: SourceId,
        stop_control: Arc<RecognitionStopControl>,
        observed: Arc<Mutex<Vec<(SourceId, RecognitionStopMode)>>>,
    }

    struct FakeScheduledSource {
        source_id: SourceId,
        input: VecDeque<f32>,
        input_closed: bool,
        queue_overrun: Option<SourceDiscontinuity>,
        events: Arc<Mutex<Vec<String>>>,
    }

    impl FakeScheduledSource {
        fn new(
            source_id: &str,
            input: impl IntoIterator<Item = f32>,
            input_closed: bool,
            queue_overrun: Option<SourceDiscontinuity>,
            events: Arc<Mutex<Vec<String>>>,
        ) -> Self {
            Self {
                source_id: SourceId::from(source_id),
                input: input.into_iter().collect(),
                input_closed,
                queue_overrun,
                events,
            }
        }

        fn record(&self, event: &str) {
            self.events
                .lock()
                .unwrap()
                .push(format!("{}:{event}", self.source_id));
        }
    }

    impl ScheduledRecognitionSource for FakeScheduledSource {
        fn source_id(&self) -> &SourceId {
            &self.source_id
        }

        fn take_queue_overrun(&mut self) -> Option<SourceDiscontinuity> {
            self.queue_overrun.take()
        }

        fn collect_available_input(&mut self) -> RecognitionLoopStep {
            self.record(if self.input_closed {
                "input_closed"
            } else {
                "collect"
            });
            if self.input_closed && self.input.is_empty() {
                RecognitionLoopStep::InputDisconnected
            } else if self.input.is_empty() {
                RecognitionLoopStep::Idle
            } else {
                RecognitionLoopStep::Progressed
            }
        }

        fn has_pending_input(&self) -> bool {
            !self.input.is_empty()
        }

        fn process_one_pending_input(&mut self) -> bool {
            let Some(sample) = self.input.pop_front() else {
                return false;
            };
            self.record(&format!("process:{sample}"));
            true
        }

        fn step_driver_only(&mut self) {
            self.record("pending_drain");
        }

        fn flush_input_for_source_stop(&mut self) {
            self.record("flush");
        }

        fn finish_source_drain_if_idle(&mut self) -> bool {
            self.record("finalize");
            true
        }

        fn shutdown_source(&mut self) -> RecognitionShutdownResult {
            self.record("shutdown");
            RecognitionShutdownResult::Completed
        }

        fn cancel_source(&mut self) {
            self.record("cancel");
        }
    }

    fn queued_callback_packet(
        sender: &mpsc::Sender<InputChunk>,
        queued_samples: &Arc<AtomicUsize>,
        marker: f32,
    ) {
        // A 48 kHz device callback delivering 512 mono samples contains
        // 10.67 ms of audio. Keep the production queue permit on the chunk so
        // this test observes whether the shared scheduler actually releases
        // capture backlog as it consumes each callback packet.
        let samples = vec![marker; 512];
        queued_samples.fetch_add(samples.len(), Ordering::AcqRel);
        sender
            .send(InputChunk::with_queue_permit(
                samples,
                Arc::clone(queued_samples),
            ))
            .expect("test capture lane should remain connected");
    }

    fn profile_outer_loop_with_48khz_input<'a>(
        handle: &'a tauri::AppHandle,
        source_id: &str,
        receiver: mpsc::Receiver<InputChunk>,
        runtime_config: Arc<RuntimeConfigState>,
        observed_vad_frames: Arc<Mutex<Vec<Vec<f32>>>>,
    ) -> RecognitionOuterLoop<'a> {
        let config = ParapperConfig::default();
        let source_id = SourceId::from(source_id);
        RecognitionOuterLoop {
            handle,
            source_id: source_id.clone(),
            queue_overrun: None,
            observed_overrun_epoch: 0,
            runtime_config,
            runtime_config_cursor: RuntimeConfigCursor::default(),
            applied_config: Arc::new(config.clone()),
            profile_mode: true,
            profile_disabled: false,
            receiver,
            pending_input: PendingInputChunks::default(),
            pending_vad_frames: PendingVadFrames::default(),
            audio_processor: AudioInputProcessor::initialize_for_source(
                handle.clone(),
                &config,
                48_000,
                Some(source_id),
            )
            .expect("48 kHz test processor should initialize"),
            vad_stage: Some(RecognitionVadStage {
                handle: handle.clone(),
                vad: Box::new(RecordingVad {
                    observed: observed_vad_frames,
                }),
                activity_sender: None,
                was_speech: false,
            }),
            driver: None,
        }
    }

    impl super::RecognitionWorkerJoinHandle for FakeStopWorkerJoin {
        fn source_id(&self) -> &SourceId {
            &self.source_id
        }

        fn join(self: Box<Self>) -> RecognitionShutdownResult {
            self.observed
                .lock()
                .unwrap()
                .push((self.source_id.clone(), self.stop_control.mode()));
            RecognitionShutdownResult::Cancelled
        }
    }

    #[test]
    fn worker_startup_result_collector_identifies_the_failing_source() {
        let (sender, receiver) = mpsc::channel();
        sender
            .send(AsrWorkerStartupReport {
                source_id: SourceId::from("speaker-b"),
                result: Err(vec!["model unavailable".to_owned()]),
            })
            .unwrap();

        assert_eq!(
            collect_worker_startup_results(
                &[SourceId::from("speaker-a"), SourceId::from("speaker-b"),],
                &receiver,
            ),
            Err(WorkerStartupCollectionError::Asr {
                source_id: SourceId::from("speaker-b"),
                errors: vec!["model unavailable".to_owned()],
            })
        );
    }

    #[test]
    fn worker_join_collector_joins_every_lane_and_uses_timeout_cancel_complete_priority() {
        // This seam bypasses production thread::JoinHandle only; it exercises the same
        // production collector and aggregation used after real recognition workers exit.
        let joined = Arc::new(Mutex::new(Vec::new()));
        let workers: Vec<Box<dyn super::RecognitionWorkerJoinHandle>> = vec![
            Box::new(FakeWorkerJoin {
                source_id: SourceId::from("completed"),
                result: RecognitionShutdownResult::Completed,
                joined: joined.clone(),
            }),
            Box::new(FakeWorkerJoin {
                source_id: SourceId::from("timed-out"),
                result: RecognitionShutdownResult::TimedOut,
                joined: joined.clone(),
            }),
            Box::new(FakeWorkerJoin {
                source_id: SourceId::from("cancelled"),
                result: RecognitionShutdownResult::Cancelled,
                joined: joined.clone(),
            }),
        ];

        assert_eq!(
            join_all_workers(workers),
            RecognitionShutdownResult::TimedOut
        );
        assert_eq!(
            *joined.lock().unwrap(),
            vec![
                SourceId::from("completed"),
                SourceId::from("timed-out"),
                SourceId::from("cancelled"),
            ],
            "a higher-priority result must not short-circuit the remaining joins"
        );
    }

    #[test]
    fn running_input_cancel_requests_every_lane_and_joins_them_all() {
        let stop_control = Arc::new(RecognitionStopControl::default());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let worker_joins: Vec<Box<dyn super::RecognitionWorkerJoinHandle>> =
            ["speaker-a", "speaker-b", "speaker-c"]
                .into_iter()
                .map(|source_id| {
                    Box::new(FakeStopWorkerJoin {
                        source_id: SourceId::from(source_id),
                        stop_control: stop_control.clone(),
                        observed: observed.clone(),
                    }) as Box<dyn super::RecognitionWorkerJoinHandle>
                })
                .collect();
        let mut running = super::RunningRecognitionInput {
            stop_control,
            source_lifetime: None,
            captures: Vec::new(),
            worker_joins,
            asr_pools: Vec::new(),
        };

        assert_eq!(
            running.stop_inner(RecognitionStopMode::Cancel),
            RecognitionShutdownResult::Cancelled
        );
        assert_eq!(
            *observed.lock().unwrap(),
            vec![
                (SourceId::from("speaker-a"), RecognitionStopMode::Cancel),
                (SourceId::from("speaker-b"), RecognitionStopMode::Cancel),
                (SourceId::from("speaker-c"), RecognitionStopMode::Cancel),
            ],
            "stop_inner must publish Cancel before collecting every worker"
        );
    }

    #[test]
    fn recognition_input_mode_selects_explicit_only_for_a_configured_capture_endpoint() {
        let legacy = ParapperConfig::default();
        let mut explicit = ParapperConfig::default();
        explicit.input.capture_endpoint = Some(CaptureEndpointConfig {
            id: "interface-1".to_owned(),
            device_host: "WASAPI".to_owned(),
            device_id: "device-1".to_owned(),
            device_name: Some("Interface 1".to_owned()),
        });
        explicit.input.recognition_sources = vec![RecognitionSourceConfig {
            source_id: "speaker-a".to_owned(),
            speaker_label: "Speaker A".to_owned(),
            capture_endpoint_id: "interface-1".to_owned(),
            channel_index: 0,
            asr_route_policy: None,
            delivery_profile_id: None,
        }];

        assert_eq!(
            recognition_input_mode(&legacy),
            RecognitionInputMode::Legacy
        );
        assert_eq!(
            recognition_input_mode(&explicit),
            RecognitionInputMode::Explicit
        );
    }

    impl RecognitionInputLoop for ScriptedInputLoop {
        fn step(&mut self) -> RecognitionLoopStep {
            self.observed_steps += 1;
            self.steps.pop_front().expect("scripted step")
        }

        fn stop(&mut self) -> RecognitionShutdownResult {
            self.stopped = true;
            RecognitionShutdownResult::Completed
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    #[test]
    fn graceful_stop_processes_accepted_input_until_the_source_disconnects() {
        let control = RecognitionStopControl::default();
        control.request(RecognitionStopMode::Graceful);
        let mut input_loop = ScriptedInputLoop {
            steps: VecDeque::from([
                RecognitionLoopStep::Progressed,
                RecognitionLoopStep::Progressed,
                RecognitionLoopStep::InputDisconnected,
            ]),
            observed_steps: 0,
            stopped: false,
            cancelled: false,
        };

        drive_recognition_input_loop(&mut input_loop, &control, InputDisconnectPolicy::Cancel);

        assert_eq!(input_loop.observed_steps, 3);
        assert!(input_loop.stopped);
        assert!(!input_loop.cancelled);
    }

    #[test]
    fn unexpected_network_disconnect_cancels_incomplete_work() {
        let control = RecognitionStopControl::default();
        let mut input_loop = ScriptedInputLoop {
            steps: VecDeque::from([RecognitionLoopStep::InputDisconnected]),
            observed_steps: 0,
            stopped: false,
            cancelled: false,
        };

        drive_recognition_input_loop(&mut input_loop, &control, InputDisconnectPolicy::Cancel);

        assert!(input_loop.cancelled);
        assert!(!input_loop.stopped);
    }

    #[test]
    fn unexpected_desktop_disconnect_flushes_the_last_processed_segment() {
        let control = RecognitionStopControl::default();
        let mut input_loop = ScriptedInputLoop {
            steps: VecDeque::from([RecognitionLoopStep::InputDisconnected]),
            observed_steps: 0,
            stopped: false,
            cancelled: false,
        };

        drive_recognition_input_loop(&mut input_loop, &control, InputDisconnectPolicy::Graceful);

        assert!(input_loop.stopped);
        assert!(!input_loop.cancelled);
    }

    fn chunk(value: f32) -> InputChunk {
        InputChunk::new(vec![value])
    }

    #[test]
    fn outer_loop_input_collects_all_available_chunks_without_skipping() {
        let (sender, receiver) = mpsc::channel();
        for value in [0.0_f32, 1.0, 2.0, 3.0] {
            sender.send(chunk(value)).unwrap();
        }
        drop(sender);
        let mut pending = PendingInputChunks::default();

        let status = pending.collect_from(&receiver, Duration::from_millis(1));

        assert_eq!(status, RecognitionLoopStep::Progressed);
        assert_eq!(pending.len(), 4);
        let samples = std::iter::from_fn(|| pending.pop_front())
            .map(|chunk| chunk.samples[0].to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            samples,
            vec![
                0.0_f32.to_bits(),
                1.0_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
            ]
        );
    }

    #[test]
    fn production_source_scheduler_services_one_chunk_per_source_before_returning_to_a_backlog() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = RecognitionSourceSchedulerLoop::from_sources(
            vec![
                FakeScheduledSource::new(
                    "speaker-a",
                    [10.0, 11.0, 12.0],
                    false,
                    None,
                    events.clone(),
                ),
                FakeScheduledSource::new("speaker-b", [20.0], false, None, events.clone()),
            ],
            Duration::ZERO,
        );

        for _ in 0..4 {
            assert_eq!(scheduler.step_inner(), RecognitionLoopStep::Progressed);
        }

        let processed = events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.contains(":process:"))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            processed,
            vec![
                "speaker-a:process:10".to_owned(),
                "speaker-b:process:20".to_owned(),
                "speaker-a:process:11".to_owned(),
                "speaker-a:process:12".to_owned(),
            ],
            "the production scheduler seam must requeue a still-ready source only after its one-chunk quota"
        );
    }

    #[test]
    fn two_48khz_capture_lanes_consume_each_512_sample_packet_without_sleeping_before_vad_frame() {
        let handle = crate::recognition::tests::tauri_test_handle();
        let config = ParapperConfig::default();
        let runtime_config = Arc::new(RuntimeConfigState::new(config));
        let (sender_a, receiver_a) = mpsc::channel();
        let (sender_b, receiver_b) = mpsc::channel();
        let queued_a = Arc::new(AtomicUsize::new(0));
        let queued_b = Arc::new(AtomicUsize::new(0));
        let observed_a = Arc::new(Mutex::new(Vec::new()));
        let observed_b = Arc::new(Mutex::new(Vec::new()));

        for marker in [1.0_f32, 2.0, 3.0] {
            queued_callback_packet(&sender_a, &queued_a, marker);
        }
        for marker in [11.0_f32, 12.0, 13.0] {
            queued_callback_packet(&sender_b, &queued_b, marker);
        }
        drop(sender_a);
        drop(sender_b);

        let mut scheduler = RecognitionSourceSchedulerLoop::from_sources(
            vec![
                profile_outer_loop_with_48khz_input(
                    &handle,
                    "motu-in",
                    receiver_a,
                    runtime_config.clone(),
                    observed_a.clone(),
                ),
                profile_outer_loop_with_48khz_input(
                    &handle,
                    "motu-out",
                    receiver_b,
                    runtime_config,
                    observed_b.clone(),
                ),
            ],
            Duration::from_millis(16),
        );
        let expected_queued_samples = [
            (1_024, 1_536),
            (1_024, 1_024),
            (512, 1_024),
            (512, 512),
            (0, 512),
            (0, 0),
        ];
        let mut requested_sleeps = Vec::new();

        for (step, expected) in expected_queued_samples.into_iter().enumerate() {
            let result =
                scheduler.step_inner_with_sleeper(|duration| requested_sleeps.push(duration));

            assert_eq!(
                result,
                RecognitionLoopStep::Progressed,
                "step {} must make capture progress even when a 32 ms VAD frame is not ready",
                step + 1
            );
            assert_eq!(
                (
                    queued_a.load(Ordering::Acquire),
                    queued_b.load(Ordering::Acquire)
                ),
                expected,
                "the shared scheduler must release one actual capture packet per source in A/B FIFO round-robin order"
            );
        }

        assert!(
            requested_sleeps.is_empty(),
            "a ready source that consumed a native callback packet must not enter the scheduler idle sleep before its 32 ms VAD frame exists"
        );
        assert_eq!(
            observed_a.lock().unwrap().len(),
            1,
            "three 512-sample callbacks must produce exactly one 1536-sample VAD frame for source A"
        );
        assert_eq!(
            observed_b.lock().unwrap().len(),
            1,
            "three 512-sample callbacks must produce exactly one 1536-sample VAD frame for source B"
        );
        assert!(
            scheduler
                .sources
                .iter()
                .all(|source| source.pending_input.is_empty()),
            "draining both native callback bursts must release queue permits instead of approaching the two-second source overrun budget"
        );
    }

    #[test]
    fn production_source_scheduler_queue_overrun_warns_and_keeps_both_sources_fifo_round_robin() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = RecognitionSourceSchedulerLoop::from_sources(
            vec![
                FakeScheduledSource::new(
                    "speaker-a",
                    [10.0, 11.0],
                    false,
                    Some(SourceDiscontinuity::QueueOverrun {
                        epoch: 3,
                        dropped_samples: 512,
                    }),
                    events.clone(),
                ),
                FakeScheduledSource::new("speaker-b", [20.0, 21.0], false, None, events.clone()),
            ],
            Duration::ZERO,
        );

        for _ in 0..4 {
            assert_eq!(scheduler.step_inner(), RecognitionLoopStep::Progressed);
        }

        assert_eq!(scheduler.stopped, vec![false, false]);
        let events = events.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![
                "speaker-a:collect".to_owned(),
                "speaker-b:collect".to_owned(),
                "speaker-a:process:10".to_owned(),
                "speaker-a:pending_drain".to_owned(),
                "speaker-b:pending_drain".to_owned(),
                "speaker-a:collect".to_owned(),
                "speaker-b:collect".to_owned(),
                "speaker-b:process:20".to_owned(),
                "speaker-a:pending_drain".to_owned(),
                "speaker-b:pending_drain".to_owned(),
                "speaker-a:collect".to_owned(),
                "speaker-b:collect".to_owned(),
                "speaker-a:process:11".to_owned(),
                "speaker-a:pending_drain".to_owned(),
                "speaker-b:pending_drain".to_owned(),
                "speaker-a:collect".to_owned(),
                "speaker-b:collect".to_owned(),
                "speaker-b:process:21".to_owned(),
                "speaker-a:pending_drain".to_owned(),
                "speaker-b:pending_drain".to_owned(),
            ],
            "QueueOverrun is a source-local warning: the same scheduling turn must still collect/process A, and following turns must retain A/B FIFO round-robin service"
        );
        assert!(
            events.iter().all(|event| !event.ends_with(":cancel")),
            "a QueueOverrun warning alone must not permanently stop either source"
        );
    }

    #[test]
    fn production_source_scheduler_input_close_flushes_drains_finalizes_and_stops_only_that_source()
    {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = RecognitionSourceSchedulerLoop::from_sources(
            vec![
                FakeScheduledSource::new("speaker-a", [], true, None, events.clone()),
                FakeScheduledSource::new("speaker-b", [20.0], false, None, events.clone()),
            ],
            Duration::ZERO,
        );

        assert_eq!(scheduler.step_inner(), RecognitionLoopStep::Progressed);

        assert_eq!(scheduler.stopped, vec![true, false]);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                "speaker-a:input_closed".to_owned(),
                "speaker-a:flush".to_owned(),
                "speaker-b:collect".to_owned(),
                "speaker-b:process:20".to_owned(),
                "speaker-a:pending_drain".to_owned(),
                "speaker-b:pending_drain".to_owned(),
                "speaker-a:finalize".to_owned(),
                "speaker-a:shutdown".to_owned(),
            ],
            "A input close must follow flush -> pending drain -> finalize -> shutdown while B remains scheduled"
        );
    }

    #[test]
    fn outer_loop_input_idles_when_no_source_chunk_is_available() {
        let (_sender, receiver) = mpsc::channel();
        let mut pending = PendingInputChunks::default();
        let started_at = Instant::now();

        let status = pending.collect_from(&receiver, Duration::from_millis(16));

        assert_eq!(status, RecognitionLoopStep::Idle);
        assert!(pending.is_empty());
        assert!(
            started_at.elapsed() < Duration::from_millis(100),
            "idle wait should stay bounded to the configured short sleep"
        );
    }

    #[test]
    fn recognition_input_wait_timeout_uses_half_vad_interval() {
        let config = parapper_config! {
            vad_interval_ms: 32,
            ..ParapperConfig::default()
        };

        assert_eq!(
            recognition_input_wait_timeout(&config),
            Duration::from_millis(16)
        );
    }

    #[test]
    fn stt_profile_capture_endpoint_identity_is_stable_when_profiles_are_reordered() {
        let first = stt_profile(
            "speaker-a",
            "Speaker A",
            Some("wasapi"),
            Some("usb-interface"),
            0,
        );
        let second = stt_profile(
            "speaker-b",
            "Speaker B",
            Some("wasapi"),
            Some("usb-interface"),
            1,
        );
        let third = stt_profile("speaker-c", "Speaker C", Some("wasapi"), Some("webcam"), 0);
        let original = ParapperConfig {
            stt_profiles: vec![first.clone(), second.clone(), third.clone()],
            ..ParapperConfig::default()
        };
        let mut reordered = original.clone();
        reordered.stt_profiles = vec![third, second, first];

        let endpoint_for = |config: &ParapperConfig, profile_id: &str| {
            profile_capture_groups(config)
                .into_iter()
                .find_map(|group| {
                    group
                        .profile_ids
                        .iter()
                        .any(|id| id == profile_id)
                        .then_some(group.endpoint_id)
                })
                .expect("configured profile must belong to exactly one capture endpoint")
        };

        for profile_id in ["speaker-a", "speaker-b", "speaker-c"] {
            assert_eq!(
                endpoint_for(&original, profile_id),
                endpoint_for(&reordered, profile_id),
                "profile reorder must not change its capture endpoint identity"
            );
        }
        assert_eq!(
            endpoint_for(&original, "speaker-a"),
            endpoint_for(&original, "speaker-b"),
            "channels on the same device must share one capture endpoint"
        );
    }

    #[test]
    fn disabled_stt_profiles_do_not_create_capture_sources() {
        let enabled = stt_profile("enabled", "Enabled", Some("wasapi"), Some("interface"), 0);
        let mut disabled =
            stt_profile("disabled", "Disabled", Some("wasapi"), Some("interface"), 1);
        disabled.enabled = false;
        let config = ParapperConfig {
            stt_profiles: vec![enabled, disabled],
            ..ParapperConfig::default()
        };

        assert_eq!(
            profile_capture_groups(&config),
            vec![ProfileCaptureGroup {
                endpoint_id: profile_capture_endpoint_id(Some("wasapi"), Some("interface")),
                device_host: Some("wasapi".to_owned()),
                device_id: Some("interface".to_owned()),
                profile_ids: vec!["enabled".to_owned()],
            }]
        );
    }

    #[test]
    fn profile_runtime_update_resolves_only_the_source_owned_profile() {
        let profile_a = stt_profile("speaker-a", "Speaker A", None, None, 0);
        let profile_b = stt_profile("speaker-b", "Speaker B", None, None, 1);
        let initial = ParapperConfig {
            stt_profiles: vec![profile_a, profile_b],
            ..ParapperConfig::default()
        };
        let applied_a =
            resolve_runtime_config_for_source(&initial, &SourceId::from("speaker-a"), true)
                .expect("source A profile must resolve");
        let applied_b =
            resolve_runtime_config_for_source(&initial, &SourceId::from("speaker-b"), true)
                .expect("source B profile must resolve");

        let mut updated = initial.clone();
        let profile_b = updated
            .stt_profiles
            .iter_mut()
            .find(|profile| profile.id == "speaker-b")
            .expect("profile B must exist");
        profile_b.input.volume_percent = 20;
        profile_b.input.muted = true;
        profile_b.segmentation.vad_threshold = 0.42;

        let next_a =
            resolve_runtime_config_for_source(&updated, &SourceId::from("speaker-a"), true)
                .expect("source A must still resolve after B changes");
        let next_b =
            resolve_runtime_config_for_source(&updated, &SourceId::from("speaker-b"), true)
                .expect("source B must resolve its updated profile");

        assert_eq!(
            RuntimeConfigDirty::between(&applied_a, &next_a),
            RuntimeConfigDirty { bits: 0 },
            "changing B must not reconfigure A's VAD, turn, or input gain"
        );
        assert_eq!(
            RuntimeConfigDirty::between(&applied_b, &next_b),
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::AUDIO | RuntimeConfigDirty::VAD,
            },
            "B mute and VAD changes must be applied only to B"
        );
        assert!(next_b.input.muted);
        updated
            .stt_profiles
            .retain(|profile| profile.id != "speaker-b");
        assert!(
            resolve_runtime_config_for_source(&updated, &SourceId::from("speaker-b"), true)
                .is_err(),
            "a removed profile must fail closed instead of inheriting another source's config"
        );
    }

    #[test]
    fn cancel_control_overrides_graceful_stop_without_entering_the_audio_queue() {
        let control = RecognitionStopControl::default();

        control.request(RecognitionStopMode::Graceful);
        control.request(RecognitionStopMode::Cancel);

        assert_eq!(control.mode(), RecognitionStopMode::Cancel);
    }

    #[test]
    fn runtime_config_state_broadcasts_one_revision_once_to_each_cursor() {
        let initial = ParapperConfig::default();
        let state = super::RuntimeConfigState::new(initial.clone());
        let mut source_a = RuntimeConfigCursor::default();
        let mut source_b = RuntimeConfigCursor::default();

        assert!(
            state.take_updated_config(&mut source_a).is_none(),
            "initial config is already applied by startup and must not fan out on every loop step"
        );

        let updated = parapper_config! {
            vad_threshold: 0.42,
            ..ParapperConfig::default()
        };
        state.replace(updated.clone());

        let first_observation = state
            .take_updated_config(&mut source_a)
            .expect("source A must observe the frontend update");
        let independent_observation = state
            .take_updated_config(&mut source_b)
            .expect("source B must independently observe the same frontend update");
        assert_eq!(first_observation.revision, independent_observation.revision);
        assert!(Arc::ptr_eq(
            &first_observation.config,
            &independent_observation.config
        ));
        assert_f32_close(
            first_observation.config.segmentation.vad_threshold,
            updated.segmentation.vad_threshold,
        );
        assert_eq!(
            RuntimeConfigDirty::between(&initial, &first_observation.config),
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::VAD
            }
        );
        assert!(
            state.take_updated_config(&mut source_a).is_none(),
            "the same cursor must not receive one revision twice"
        );
    }

    #[test]
    fn runtime_config_state_coalesces_revisions_to_the_latest_snapshot() {
        let initial = ParapperConfig::default();
        let state = super::RuntimeConfigState::new(initial.clone());
        let mut cursor = RuntimeConfigCursor::default();
        let intermediate = parapper_config! {
            input_volume_db: 6.0,
            ..initial.clone()
        };
        let latest = parapper_config! {
            vad_threshold: 0.42,
            ..initial.clone()
        };

        state.replace(intermediate);
        state.replace(latest.clone());

        let update = state
            .take_updated_config(&mut cursor)
            .expect("coalesced cursor must receive the latest config");
        assert_eq!(update.revision, 2);
        assert_eq!(update.config.as_ref(), &latest);
        assert_eq!(
            RuntimeConfigDirty::between(&initial, &update.config),
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::VAD
            },
            "the source-owned runtime must compare its applied config to the latest snapshot, not OR intermediate changes"
        );
    }

    #[test]
    fn runtime_config_state_never_pairs_a_revision_with_another_revisions_snapshot() {
        const REPLACEMENTS: u64 = 64;
        const BASE_PORT: u16 = 15_000;

        let initial = ParapperConfig::default();
        let state = Arc::new(super::RuntimeConfigState::new(initial.clone()));
        let writer_state = state.clone();
        let writer = std::thread::spawn(move || {
            for revision in 1..=REPLACEMENTS {
                let mut config = ParapperConfig::default();
                config.neo.http_port = BASE_PORT
                    + u16::try_from(revision).expect("test revision must fit the HTTP port");
                writer_state.replace(config);
            }
        });
        let mut cursor = RuntimeConfigCursor::default();
        while cursor.revision < REPLACEMENTS {
            if let Some(update) = state.take_updated_config(&mut cursor) {
                assert_eq!(
                    update.config.neo.http_port,
                    BASE_PORT
                        + u16::try_from(update.revision)
                            .expect("published test revision must fit the HTTP port"),
                    "a consumer must observe the config Arc and revision from one publication"
                );
            } else {
                std::thread::yield_now();
            }
        }
        writer.join().expect("config writer must finish");
    }

    #[test]
    fn runtime_config_state_equal_replace_does_not_advance_revision() {
        let initial = ParapperConfig::default();
        let state = super::RuntimeConfigState::new(initial.clone());
        let mut cursor = RuntimeConfigCursor::default();
        let first = parapper_config! {
            vad_threshold: 0.42,
            ..initial.clone()
        };
        state.replace(first.clone());
        let first_update = state
            .take_updated_config(&mut cursor)
            .expect("first changed config must advance the revision");

        state.replace(first.clone());
        assert!(
            state.take_updated_config(&mut cursor).is_none(),
            "an equal replacement must not publish another revision"
        );

        let second = parapper_config! {
            vad_threshold: 0.55,
            ..initial
        };
        state.replace(second);
        let second_update = state
            .take_updated_config(&mut cursor)
            .expect("the next changed config must still be published");
        assert_eq!(second_update.revision, first_update.revision + 1);
    }

    #[test]
    fn runtime_config_state_marks_input_volume_update_without_driver_or_vad_dirty() {
        let initial = ParapperConfig::default();
        let state = super::RuntimeConfigState::new(initial.clone());
        let mut cursor = RuntimeConfigCursor::default();
        let updated = parapper_config! {
            input_volume_db: 6.0,
            ..ParapperConfig::default()
        };

        state.replace(updated.clone());

        let update = state
            .take_updated_config(&mut cursor)
            .expect("input volume update should be visible to the audio processor");
        assert_f32_close(update.config.input.volume_db, updated.input.volume_db);
        assert_eq!(
            RuntimeConfigDirty::between(&initial, &update.config),
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::AUDIO
            },
            "audio-only changes must not fan out to recognition driver or VAD stage"
        );
    }

    #[test]
    fn delivery_only_config_update_replaces_snapshot_without_waking_stt_runtime() {
        let initial = ParapperConfig::default();
        let state = super::RuntimeConfigState::new(initial.clone());
        let mut cursor = RuntimeConfigCursor::default();
        let mut updated = initial.clone();
        updated.neo.http_port = 15521;

        state.replace(updated.clone());

        assert_eq!(
            state.snapshot().unwrap().neo.http_port,
            15521,
            "delivery config must remain visible through the shared snapshot"
        );
        let update = state
            .take_updated_config(&mut cursor)
            .expect("every consumer cursor must advance to the delivery-only revision");
        assert!(
            RuntimeConfigDirty::between(&initial, &update.config).is_empty(),
            "delivery-only changes must not wake the STT driver, VAD, or audio processor"
        );
    }

    #[test]
    fn outer_loop_input_reports_disconnect_after_buffered_chunks_are_consumed() {
        let (sender, receiver) = mpsc::channel();
        sender.send(chunk(1.0)).unwrap();
        drop(sender);
        let mut pending = PendingInputChunks::default();

        assert_eq!(
            pending.collect_from(&receiver, Duration::from_millis(1)),
            RecognitionLoopStep::Progressed
        );
        assert!(pending.pop_front().is_some());
        assert_eq!(
            pending.collect_from(&receiver, Duration::from_millis(1)),
            RecognitionLoopStep::InputDisconnected
        );
    }

    #[test]
    fn outer_loop_vad_queue_preserves_processed_audio_fifo_order() {
        let mut pending = PendingVadFrames::default();

        pending.push(ProcessedAudioChunk::shared(vec![1.0]));
        pending.push(ProcessedAudioChunk::shared(vec![2.0]));
        pending.push(ProcessedAudioChunk::shared(vec![3.0]));

        assert_eq!(pending.len(), 3);
        let samples = std::iter::from_fn(|| pending.pop_front())
            .map(|samples| samples.into_asr_samples()[0].to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            samples,
            vec![1.0_f32.to_bits(), 2.0_f32.to_bits(), 3.0_f32.to_bits()]
        );
    }

    #[test]
    fn vad_stage_decides_on_enhanced_audio_but_forwards_the_aligned_raw_frame_to_asr() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut stage = RecognitionVadStage {
            handle: crate::recognition::tests::tauri_test_handle(),
            vad: Box::new(RecordingVad {
                observed: observed.clone(),
            }),
            activity_sender: None,
            was_speech: false,
        };

        let frame = stage
            .process(ProcessedAudioChunk::split(
                vec![101.0, 102.0],
                vec![1.0, 2.0],
            ))
            .unwrap();

        assert_eq!(*observed.lock().unwrap(), vec![vec![101.0, 102.0]]);
        assert_eq!(frame.asr_samples, vec![1.0, 2.0]);
        assert_eq!(
            frame.result,
            VadResult {
                probability: 0.9,
                is_speech: true,
            }
        );
    }

    #[test]
    fn recognition_startup_fails_when_vad_model_is_missing() {
        let handle = crate::recognition::tests::tauri_test_handle();
        let config = parapper_config! {
            model_dir: Some(missing_model_dir("vad-init-failure")),
            ..ParapperConfig::default()
        };

        let err = build_recognition_startup(&handle, &config, ASR_SAMPLE_RATE, None, None)
            .err()
            .expect("missing VAD model should fail recognition startup");

        assert!(
            err.to_string().contains("VAD model not found"),
            "unexpected VAD init error: {err}"
        );
    }

    fn missing_model_dir(test_name: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "parapper-missing-recognition-input-model-{test_name}-{}-{unique}",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "actual={actual}, expected={expected}"
        );
    }
}
