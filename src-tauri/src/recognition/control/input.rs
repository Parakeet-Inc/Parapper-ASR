use std::{
    collections::VecDeque,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use tauri::{AppHandle, Emitter};

use crate::{
    audio::{AudioInputProcessor, InputChunk, RunningAudioInput},
    config::ParapperConfig,
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    model::vad_model_path,
    recognition::{
        control::events::{VadState, VadStateEvent},
        segmentation::vad::engine::{OnnxRuntimeSileroVadEngine, VadEngine, VadResult},
    },
};

use super::{
    AsrWorkerStartupResult, AsrWorkerStartupSender, RecognitionDriver, RecognitionDriverHandle,
};

pub struct RunningRecognitionInput {
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
pub enum RecognitionStartError {
    AudioInput(anyhow::Error),
    Asr(anyhow::Error),
}

impl RecognitionStartError {
    pub fn is_asr(&self) -> bool {
        matches!(self, Self::Asr(_))
    }
}

impl std::fmt::Display for RecognitionStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioInput(err) | Self::Asr(err) => std::fmt::Display::fmt(err, f),
        }
    }
}

impl std::error::Error for RecognitionStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AudioInput(err) | Self::Asr(err) => err.source(),
        }
    }
}

pub(crate) struct RuntimeConfigState {
    config: RwLock<Arc<ParapperConfig>>,
    dirty_bits: AtomicU8,
}

impl RuntimeConfigState {
    pub(crate) fn new(config: ParapperConfig) -> Self {
        Self {
            config: RwLock::new(Arc::new(config)),
            dirty_bits: AtomicU8::new(0),
        }
    }

    pub(crate) fn replace(&self, config: ParapperConfig) {
        if let Ok(mut current) = self.config.write() {
            let dirty = RuntimeConfigDirty::between(&current, &config);
            if dirty.is_empty() {
                return;
            }
            *current = Arc::new(config);
            self.dirty_bits.fetch_or(dirty.bits, Ordering::Release);
        }
    }

    pub(crate) fn snapshot(&self) -> Result<ParapperConfig> {
        self.config
            .read()
            .map(|config| (**config).clone())
            .map_err(|_| anyhow!("runtime config lock is poisoned"))
    }

    fn take_updated_config(&self) -> Option<RuntimeConfigUpdate> {
        let bits = self.dirty_bits.swap(0, Ordering::AcqRel);
        if bits == 0 {
            return None;
        }
        if let Ok(config) = self.config.read() {
            Some(RuntimeConfigUpdate {
                config: config.clone(),
                dirty: RuntimeConfigDirty { bits },
            })
        } else {
            self.dirty_bits.fetch_or(bits, Ordering::Release);
            None
        }
    }
}

#[derive(Debug, Clone)]
struct RuntimeConfigUpdate {
    config: Arc<ParapperConfig>,
    dirty: RuntimeConfigDirty,
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
        if current.input != next.input || current.noise_cancellation != next.noise_cancellation {
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
    current.neo != next.neo
        || current.asr != next.asr
        || current.translation != next.translation
        || current.speech != next.speech
        || current.models != next.models
        || current.segmentation != next.segmentation
        || current.turn != next.turn
        || current.debug != next.debug
}

impl RunningRecognitionInput {
    pub fn start(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: Arc<RuntimeConfigState>,
    ) -> Result<Self, RecognitionStartError> {
        let audio_startup =
            RunningAudioInput::start(config).map_err(RecognitionStartError::AudioInput)?;
        let audio_input = audio_startup.input;
        let receiver = audio_startup.receiver;
        let source_sample_rate = audio_startup.source_sample_rate;
        let startup = build_recognition_startup(&handle, config, source_sample_rate)
            .map_err(RecognitionStartError::AudioInput)?;
        let (asr_startup_sender, asr_startup_receiver) =
            std::sync::mpsc::channel::<AsrWorkerStartupResult>();

        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = stop_requested.clone();
        let recognition_config = config.clone();
        let join_handle = thread::Builder::new()
            .name("parapper-recognition-input".to_string())
            .spawn(move || {
                let audio_input = audio_input;
                let RecognitionStartup {
                    audio_processor,
                    vad_stage,
                } = startup;
                let worker_startup = RecognitionWorkerStartup {
                    receiver,
                    audio_processor,
                    vad_stage,
                    asr_startup_sender,
                };
                run_recognition_input_worker(
                    &handle,
                    &recognition_config,
                    runtime_config,
                    worker_startup,
                    &worker_stop,
                );
                drop(audio_input);
            })
            .context("Failed to spawn recognition input worker")
            .map_err(RecognitionStartError::AudioInput)?;

        let mut running = Self {
            stop_requested,
            join_handle: Some(join_handle),
        };
        match asr_startup_receiver
            .recv()
            .context("ASR worker closed before reporting startup readiness")
            .map_err(RecognitionStartError::Asr)?
        {
            Ok(()) => Ok(running),
            Err(errors) => {
                running.stop_inner();
                Err(RecognitionStartError::Asr(anyhow!(
                    "ASR worker failed to preload required models: {}",
                    errors.join("; ")
                )))
            }
        }
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        if let Some(join_handle) = self.join_handle.take()
            && let Err(err) = join_handle.join()
        {
            log::warn!("Recognition input worker thread panicked: {err:?}");
        }
    }
}

impl Drop for RunningRecognitionInput {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

fn run_recognition_input_worker(
    handle: &AppHandle,
    config: &ParapperConfig,
    runtime_config: Arc<RuntimeConfigState>,
    startup: RecognitionWorkerStartup,
    stop_requested: &AtomicBool,
) {
    let RecognitionWorkerStartup {
        receiver,
        audio_processor,
        vad_stage,
        asr_startup_sender,
    } = startup;
    let mut outer_loop = RecognitionOuterLoop::new(
        handle,
        config,
        runtime_config,
        receiver,
        audio_processor,
        vad_stage,
        asr_startup_sender,
    );

    while !stop_requested.load(Ordering::Acquire) {
        match outer_loop.step() {
            RecognitionLoopStep::Progressed | RecognitionLoopStep::Idle => {}
            RecognitionLoopStep::InputDisconnected => break,
        }
    }

    outer_loop.stop();
}

struct RecognitionOuterLoop<'a> {
    handle: &'a AppHandle,
    runtime_config: Arc<RuntimeConfigState>,
    applied_config: Arc<ParapperConfig>,
    receiver: Receiver<InputChunk>,
    pending_input: PendingInputChunks,
    pending_vad_frames: PendingVadFrames,
    audio_processor: AudioInputProcessor,
    vad_stage: Option<RecognitionVadStage>,
    driver: Option<Box<dyn RecognitionDriverHandle>>,
}

struct RecognitionStartup {
    audio_processor: AudioInputProcessor,
    vad_stage: RecognitionVadStage,
}

struct RecognitionWorkerStartup {
    receiver: Receiver<InputChunk>,
    audio_processor: AudioInputProcessor,
    vad_stage: RecognitionVadStage,
    asr_startup_sender: AsrWorkerStartupSender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecognitionLoopStep {
    Progressed,
    Idle,
    InputDisconnected,
}

#[derive(Default)]
struct PendingInputChunks {
    chunks: VecDeque<InputChunk>,
}

#[derive(Default)]
struct PendingVadFrames {
    frames: VecDeque<Vec<f32>>,
}

struct VadFrame {
    samples: Vec<f32>,
    result: VadResult,
}

struct RecognitionVadStage {
    handle: AppHandle,
    vad: Box<dyn VadEngine>,
}

impl PendingInputChunks {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    #[cfg(test)]
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
}

impl PendingVadFrames {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.frames.len()
    }

    fn push(&mut self, samples: Vec<f32>) {
        self.frames.push_back(samples);
    }

    fn pop_front(&mut self) -> Option<Vec<f32>> {
        self.frames.pop_front()
    }
}

impl RecognitionVadStage {
    fn new(handle: AppHandle, config: &ParapperConfig) -> Result<Self> {
        let vad_path = vad_model_path(&handle)?;
        let vad = OnnxRuntimeSileroVadEngine::new(&vad_path, config.segmentation.vad_threshold)?;
        Ok(Self {
            handle,
            vad: Box::new(vad),
        })
    }

    fn update_config(&mut self, config: &ParapperConfig) {
        self.vad.set_threshold(config.segmentation.vad_threshold);
    }

    fn process(&mut self, samples: Vec<f32>) -> Result<VadFrame> {
        let result = self.vad.process(&samples)?;
        let state = if result.is_speech {
            VadState::Speech
        } else {
            VadState::Silence
        };
        let _ = self.handle.emit(
            "parapper://vad-state",
            VadStateEvent {
                state,
                probability: result.probability,
            },
        );

        Ok(VadFrame { samples, result })
    }
}

fn build_recognition_startup(
    handle: &AppHandle,
    config: &ParapperConfig,
    source_sample_rate: u32,
) -> Result<RecognitionStartup> {
    let audio_processor =
        AudioInputProcessor::initialize(handle.clone(), config, source_sample_rate)?;
    let vad_stage = match RecognitionVadStage::new(handle.clone(), config) {
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

impl<'a> RecognitionOuterLoop<'a> {
    fn new(
        handle: &'a AppHandle,
        config: &'a ParapperConfig,
        runtime_config: Arc<RuntimeConfigState>,
        receiver: Receiver<InputChunk>,
        audio_processor: AudioInputProcessor,
        vad_stage: RecognitionVadStage,
        asr_startup_sender: AsrWorkerStartupSender,
    ) -> Self {
        Self {
            handle,
            runtime_config,
            applied_config: Arc::new(config.clone()),
            receiver,
            pending_input: PendingInputChunks::default(),
            pending_vad_frames: PendingVadFrames::default(),
            audio_processor,
            vad_stage: Some(vad_stage),
            driver: Some(Box::new(RecognitionDriver::new_for_production(
                handle,
                config,
                Some(asr_startup_sender),
            ))),
        }
    }

    fn step(&mut self) -> RecognitionLoopStep {
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

    fn stop(mut self) {
        if let Some(mut driver) = self.driver.take() {
            driver.shutdown();
        }
    }

    fn apply_runtime_config_update(&mut self) {
        let Some(update) = self.runtime_config.take_updated_config() else {
            return;
        };
        self.applied_config = update.config.clone();
        if update.dirty.driver()
            && let Some(driver) = self.driver.as_mut()
        {
            driver.update_config(&update.config);
        }
        if update.dirty.vad()
            && let Some(vad_stage) = self.vad_stage.as_mut()
        {
            vad_stage.update_config(&update.config);
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
                        driver.push_vad_frame(&frame.samples, frame.result);
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

fn recognition_input_wait_timeout(config: &ParapperConfig) -> Duration {
    let half_vad_interval_ms = u64::from(config.segmentation.vad_interval_ms.max(1)).div_ceil(2);
    Duration::from_millis(half_vad_interval_ms.max(1))
}

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use super::{
        PendingInputChunks, PendingVadFrames, RecognitionLoopStep, RuntimeConfigDirty,
        build_recognition_startup, recognition_input_wait_timeout,
    };
    use crate::{
        audio::{ASR_SAMPLE_RATE, InputChunk},
        config::ParapperConfig,
    };

    fn chunk(value: f32) -> InputChunk {
        InputChunk {
            samples: vec![value],
        }
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
    fn runtime_config_state_reports_update_only_after_frontend_replaces_config() {
        let state = super::RuntimeConfigState::new(ParapperConfig::default());

        assert!(
            state.take_updated_config().is_none(),
            "initial config is already applied by startup and must not fan out on every loop step"
        );

        let updated = parapper_config! {
            vad_threshold: 0.42,
            ..ParapperConfig::default()
        };
        state.replace(updated.clone());

        let update = state
            .take_updated_config()
            .expect("frontend config replacement should mark one runtime update");
        let applied = update.config;
        assert_f32_close(
            applied.segmentation.vad_threshold,
            updated.segmentation.vad_threshold,
        );
        assert_eq!(
            update.dirty,
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::VAD | RuntimeConfigDirty::DRIVER
            }
        );
        assert!(
            state.take_updated_config().is_none(),
            "a single frontend update must be consumed once, not replayed every step"
        );
    }

    #[test]
    fn runtime_config_state_marks_input_volume_update_without_driver_or_vad_dirty() {
        let state = super::RuntimeConfigState::new(ParapperConfig::default());
        let updated = parapper_config! {
            input_volume_db: 6.0,
            ..ParapperConfig::default()
        };

        state.replace(updated.clone());

        let update = state
            .take_updated_config()
            .expect("input volume update should be visible to the audio processor");
        assert_f32_close(update.config.input.volume_db, updated.input.volume_db);
        assert_eq!(
            update.dirty,
            RuntimeConfigDirty {
                bits: RuntimeConfigDirty::AUDIO
            },
            "audio-only changes must not fan out to recognition driver or VAD stage"
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

        pending.push(vec![1.0]);
        pending.push(vec![2.0]);
        pending.push(vec![3.0]);

        assert_eq!(pending.len(), 3);
        let samples = std::iter::from_fn(|| pending.pop_front())
            .map(|samples| samples[0].to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            samples,
            vec![1.0_f32.to_bits(), 2.0_f32.to_bits(), 3.0_f32.to_bits()]
        );
    }

    #[test]
    fn recognition_startup_fails_when_vad_model_is_missing() {
        let handle = crate::recognition::control::tests::tauri_test_handle();
        let config = parapper_config! {
            model_dir: Some(missing_model_dir("vad-init-failure")),
            ..ParapperConfig::default()
        };

        let err = build_recognition_startup(&handle, &config, ASR_SAMPLE_RATE)
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
