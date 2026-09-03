use std::{
    collections::VecDeque,
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result, anyhow};
use cpal::{Stream, traits::StreamTrait};
use parapper_models::nc::{NoiseCancellationEngine, UlUnasNoiseCancellationEngine};
use parapper_stt_engine::{SourceId, SourceIdentitySnapshot};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::{
    config::{
        CaptureEndpointConfig, NoiseCancellationTarget, ParapperConfig, RecognitionSourceConfig,
    },
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    model::noise_cancellation_model_dir,
};

use super::{
    device::{InputDeviceSelection, selected_input_device, selected_input_device_strict},
    resampler::{MonoFastFixedInResampler, validated_vad_interval_ms},
    stream::{
        CaptureSequence, ChannelDemuxError, InputChunk, InterleavedPcmChunk, build_input_stream,
        build_interleaved_input_stream, demux_selected_channels, peak_level,
    },
};

pub const ASR_SAMPLE_RATE: u32 = 16_000;
const INPUT_LEVEL_EMIT_CHUNKS: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunkEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    pub source_sample_rate: u32,
    pub sample_rate: u32,
    pub frames: usize,
    pub level: f32,
    pub pre_gain_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputLevelEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    pub pre_gain_level: f32,
    pub post_gain_level: f32,
}

pub struct RunningAudioInput {
    stream: Option<Stream>,
    demux_worker: Option<JoinHandle<std::result::Result<(), DemuxRunError>>>,
    demux_control: Option<Arc<DemuxControl>>,
}

pub(crate) struct RunningAudioInputStartup {
    pub(crate) input: RunningAudioInput,
    pub(crate) receiver: Receiver<InputChunk>,
    pub(crate) source_sample_rate: u32,
}

pub(crate) struct ExplicitAudioLaneStartup {
    pub(crate) identity: SourceIdentitySnapshot,
    pub(crate) receiver: Receiver<InputChunk>,
    pub(crate) queue_overrun: Arc<SourceQueueOverrun>,
    /// Native sample rate of the capture endpoint that produced this lane.
    /// Every lane is independently resampled before it joins the shared 16 kHz
    /// scheduler, so endpoints may use different device clocks.
    pub(crate) source_sample_rate: u32,
}

pub(crate) struct PreparedExplicitAudioInput {
    pub(crate) input: RunningAudioInput,
    pub(crate) lanes: Vec<ExplicitAudioLaneStartup>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExplicitChannelLayoutError {
    ZeroChannelCount {
        endpoint_id: String,
    },
    ChannelIndexOutOfRange {
        endpoint_id: String,
        source_id: String,
        channel_index: u16,
        channel_count: u16,
    },
}

impl fmt::Display for ExplicitChannelLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroChannelCount { endpoint_id } => {
                write!(
                    formatter,
                    "explicit capture endpoint {endpoint_id} has zero channels"
                )
            }
            Self::ChannelIndexOutOfRange {
                endpoint_id,
                source_id,
                channel_index,
                channel_count,
            } => write!(
                formatter,
                "recognition source {source_id} selects channel {channel_index}, outside explicit capture endpoint {endpoint_id} with {channel_count} channels"
            ),
        }
    }
}

impl Error for ExplicitChannelLayoutError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DemuxRunError {
    Structural {
        endpoint_id: String,
        capture_sequence: CaptureSequence,
        source: ChannelDemuxError,
    },
    LaneDisconnected {
        endpoint_id: String,
        source_id: String,
        capture_sequence: CaptureSequence,
    },
}

impl fmt::Display for DemuxRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural {
                endpoint_id,
                capture_sequence,
                source,
            } => write!(
                formatter,
                "explicit capture demux failed for endpoint {endpoint_id} at sequence {}: {source}",
                capture_sequence.0
            ),
            Self::LaneDisconnected {
                endpoint_id,
                source_id,
                capture_sequence,
            } => write!(
                formatter,
                "explicit capture lane {source_id} disconnected from endpoint {endpoint_id} at sequence {}",
                capture_sequence.0
            ),
        }
    }
}

impl Error for DemuxRunError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Structural { source, .. } => Some(source),
            Self::LaneDisconnected { .. } => None,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct DemuxControl {
    cancelled: AtomicBool,
}

impl DemuxControl {
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub(crate) struct DemuxLane {
    identity: SourceIdentitySnapshot,
    channel_index: u16,
    sender: BoundedDemuxLaneSender,
}

/// Coalesced producer-side discontinuity state.  The callback/demux thread
/// records overflow atomically; it never waits for scheduler control traffic.
#[derive(Debug, Default)]
pub(crate) struct SourceQueueOverrun {
    epoch: AtomicU64,
    dropped_samples: AtomicU64,
}

impl SourceQueueOverrun {
    pub(crate) fn snapshot(&self) -> (u64, u64) {
        (
            self.epoch.load(Ordering::Acquire),
            self.dropped_samples.load(Ordering::Acquire),
        )
    }

    fn record(&self, samples: usize) {
        self.dropped_samples
            .fetch_add(u64::try_from(samples).unwrap_or(u64::MAX), Ordering::AcqRel);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedDemuxSendError {
    Overrun,
    Disconnected,
}

struct BoundedDemuxLaneSender {
    sender: Sender<InputChunk>,
    queued_samples: Arc<AtomicUsize>,
    max_queued_samples: usize,
    overrun: Arc<SourceQueueOverrun>,
}

impl BoundedDemuxLaneSender {
    fn try_send(
        &self,
        samples: Vec<f32>,
        capture_sequence: CaptureSequence,
    ) -> Result<(), BoundedDemuxSendError> {
        let sample_count = samples.len();
        if self
            .queued_samples
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued
                    .checked_add(sample_count)
                    .filter(|next| *next <= self.max_queued_samples)
            })
            .is_err()
        {
            self.overrun.record(sample_count);
            return Err(BoundedDemuxSendError::Overrun);
        }
        let chunk = InputChunk::with_capture_sequence_and_queue_permit(
            samples,
            capture_sequence,
            self.queued_samples.clone(),
        );
        self.sender
            .send(chunk)
            .map_err(|_| BoundedDemuxSendError::Disconnected)
    }
}

// Match the existing WebSocket input budget: it is long enough for normal
// scheduler jitter while bounding retained audio to two seconds per source.
const EXPLICIT_SOURCE_QUEUE_BUDGET_MS: usize = 2_000;

impl RunningAudioInput {
    pub(crate) fn start(config: &ParapperConfig) -> Result<RunningAudioInputStartup> {
        let selection = selected_input_device(config)?;
        let source_sample_rate = selection.stream_config.sample_rate;
        // Intentionally unbounded: recognition quality depends on preserving the
        // exact audio stream, and the CPAL callback must not block or drop chunks.
        // The worker drains this queue before each VAD step.
        let (sender, receiver) = channel();
        let stream = build_input_stream(
            &selection.device,
            &selection.stream_config,
            selection.sample_format,
            sender,
        )?;
        stream.play().context("Failed to start input stream")?;

        Ok(RunningAudioInputStartup {
            input: Self {
                stream: Some(stream),
                demux_worker: None,
                demux_control: None,
            },
            receiver,
            source_sample_rate,
        })
    }

    pub(crate) fn prepare_explicit(config: &ParapperConfig) -> Result<PreparedExplicitAudioInput> {
        let endpoint = config
            .input
            .capture_endpoint
            .as_ref()
            .context("explicit input startup requires capture_endpoint")?;
        if config.input.recognition_sources.is_empty() {
            anyhow::bail!("explicit input startup requires recognition_sources");
        }

        Self::prepare_profile_endpoint(endpoint, &config.input.recognition_sources)
    }

    /// Prepares one physical capture endpoint without starting playback.
    ///
    /// Profile-mode startup calls this once per distinct `(device_host,
    /// device_id)` group, then merges the returned source lanes into one shared
    /// recognition scheduler. Keeping this operation endpoint-scoped preserves
    /// one CPAL stream and one demux worker for profiles that select different
    /// channels of the same audio interface.
    pub(crate) fn prepare_profile_endpoint(
        endpoint: &CaptureEndpointConfig,
        sources: &[RecognitionSourceConfig],
    ) -> Result<PreparedExplicitAudioInput> {
        if sources.is_empty() {
            anyhow::bail!("capture endpoint requires at least one recognition source");
        }

        let selection = selected_input_device_strict(&endpoint.device_host, &endpoint.device_id)?;
        Self::prepare_selected_profile_endpoint(&endpoint.id, &selection, sources)
    }

    /// Like [`Self::prepare_profile_endpoint`], but resolves the operating
    /// system default input device. Profile schema permits this only for a
    /// single profile, so it cannot accidentally make a multi-device session
    /// depend on an ambiguous default device.
    pub(crate) fn prepare_default_profile_endpoint(
        config: &ParapperConfig,
        endpoint_id: &str,
        sources: &[RecognitionSourceConfig],
    ) -> Result<PreparedExplicitAudioInput> {
        if sources.is_empty() {
            anyhow::bail!("default capture endpoint requires at least one recognition source");
        }
        let selection = selected_input_device(config)?;
        Self::prepare_selected_profile_endpoint(endpoint_id, &selection, sources)
    }

    fn prepare_selected_profile_endpoint(
        endpoint_id: &str,
        selection: &InputDeviceSelection,
        sources: &[RecognitionSourceConfig],
    ) -> Result<PreparedExplicitAudioInput> {
        validate_explicit_channel_layout(endpoint_id, sources, selection.stream_config.channels)?;

        let source_sample_rate = selection.stream_config.sample_rate;
        let (demux_lanes, lanes) = build_explicit_lanes(sources, source_sample_rate);
        let (central_sender, central_receiver) = channel();
        let stream = build_interleaved_input_stream(
            &selection.device,
            &selection.stream_config,
            selection.sample_format,
            central_sender,
        )?;
        let demux_control = Arc::new(DemuxControl::default());
        let worker_control = Arc::clone(&demux_control);
        let worker_endpoint_id = endpoint_id.to_owned();
        let demux_worker = thread::Builder::new()
            .name("parapper-audio-demux".to_owned())
            .spawn(move || {
                let result = run_demux(
                    &worker_endpoint_id,
                    central_receiver,
                    demux_lanes,
                    &worker_control,
                );
                if let Err(error) = &result {
                    log::error!("{error}");
                }
                result
            })
            .context("Failed to spawn explicit input demux worker")?;

        Ok(PreparedExplicitAudioInput {
            input: Self {
                stream: Some(stream),
                demux_worker: Some(demux_worker),
                demux_control: Some(demux_control),
            },
            lanes,
        })
    }

    pub(crate) fn play(&self) -> Result<()> {
        self.stream
            .as_ref()
            .context("audio input stream is no longer available")?
            .play()
            .context("Failed to start explicit input stream")
    }

    pub(crate) fn cancel(&mut self) {
        if let Some(control) = &self.demux_control {
            control.cancel();
        }
        self.shutdown_demux();
    }

    fn shutdown_demux(&mut self) {
        // The callback owns the only central sender. Dropping the stream closes it,
        // allowing a graceful worker to drain every chunk already captured.
        drop(self.stream.take());
        if let Some(worker) = self.demux_worker.take() {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => log::error!("{error}"),
                Err(_) => log::error!("Explicit audio input demux worker panicked"),
            }
        }
        self.demux_control = None;
    }
}

impl Drop for RunningAudioInput {
    fn drop(&mut self) {
        self.shutdown_demux();
    }
}

pub(crate) fn validate_explicit_channel_layout(
    endpoint_id: &str,
    sources: &[RecognitionSourceConfig],
    channel_count: u16,
) -> std::result::Result<(), ExplicitChannelLayoutError> {
    if channel_count == 0 {
        return Err(ExplicitChannelLayoutError::ZeroChannelCount {
            endpoint_id: endpoint_id.to_owned(),
        });
    }
    if let Some(source) = sources
        .iter()
        .find(|source| source.channel_index >= channel_count)
    {
        return Err(ExplicitChannelLayoutError::ChannelIndexOutOfRange {
            endpoint_id: endpoint_id.to_owned(),
            source_id: source.source_id.clone(),
            channel_index: source.channel_index,
            channel_count,
        });
    }
    Ok(())
}

fn build_explicit_lanes(
    sources: &[RecognitionSourceConfig],
    source_sample_rate: u32,
) -> (Vec<DemuxLane>, Vec<ExplicitAudioLaneStartup>) {
    sources
        .iter()
        .map(|source| {
            let identity = SourceIdentitySnapshot::new(
                SourceId::from(source.source_id.clone()),
                source.speaker_label.clone(),
                source.capture_endpoint_id.clone(),
                Some(source.channel_index),
            );
            let (sender, receiver) = channel();
            let queue_overrun = Arc::new(SourceQueueOverrun::default());
            let queued_samples = Arc::new(AtomicUsize::new(0));
            let max_queued_samples = usize::try_from(source_sample_rate)
                .unwrap_or(usize::MAX)
                .saturating_mul(EXPLICIT_SOURCE_QUEUE_BUDGET_MS)
                / 1_000;
            (
                DemuxLane {
                    identity: identity.clone(),
                    channel_index: source.channel_index,
                    sender: BoundedDemuxLaneSender {
                        sender,
                        queued_samples,
                        max_queued_samples,
                        overrun: queue_overrun.clone(),
                    },
                },
                ExplicitAudioLaneStartup {
                    identity,
                    receiver,
                    queue_overrun,
                    source_sample_rate,
                },
            )
        })
        .unzip()
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "the worker owns all lane senders so every receiver closes together on return"
)]
pub(crate) fn run_demux(
    endpoint_id: &str,
    receiver: Receiver<InterleavedPcmChunk>,
    lanes: Vec<DemuxLane>,
    control: &DemuxControl,
) -> std::result::Result<(), DemuxRunError> {
    let selected_channels = lanes
        .iter()
        .map(|lane| lane.channel_index)
        .collect::<Vec<_>>();

    for chunk in receiver {
        if control.is_cancelled() {
            return Ok(());
        }
        let capture_sequence = chunk.capture_sequence;
        let mono_chunks =
            demux_selected_channels(&chunk, &selected_channels).map_err(|source| {
                DemuxRunError::Structural {
                    endpoint_id: endpoint_id.to_owned(),
                    capture_sequence,
                    source,
                }
            })?;

        for (lane, mono_chunk) in lanes.iter().zip(mono_chunks) {
            if control.is_cancelled() {
                return Ok(());
            }
            match lane
                .sender
                .try_send(mono_chunk.samples, mono_chunk.capture_sequence)
            {
                Ok(()) | Err(BoundedDemuxSendError::Overrun) => {}
                Err(BoundedDemuxSendError::Disconnected) => {
                    return Err(DemuxRunError::LaneDisconnected {
                        endpoint_id: endpoint_id.to_owned(),
                        source_id: lane.identity.source_id.to_string(),
                        capture_sequence,
                    });
                }
            }
        }
    }
    Ok(())
}

pub(crate) struct AudioInputProcessor {
    handle: AppHandle,
    resampler: MonoFastFixedInResampler,
    resampled_chunks: Vec<Vec<f32>>,
    noise_cancellation: Option<NoiseCancellationStage>,
    input_level_emitter: InputLevelEmitter,
    source_sample_rate: u32,
    source_id: Option<String>,
}

pub(crate) struct ProcessedAudioChunk {
    vad_samples: Vec<f32>,
    asr_samples: Option<Vec<f32>>,
}

impl ProcessedAudioChunk {
    pub(crate) fn shared(samples: Vec<f32>) -> Self {
        Self {
            vad_samples: samples,
            asr_samples: None,
        }
    }

    pub(crate) fn split(vad_samples: Vec<f32>, asr_samples: Vec<f32>) -> Self {
        debug_assert_eq!(vad_samples.len(), asr_samples.len());
        Self {
            vad_samples,
            asr_samples: Some(asr_samples),
        }
    }

    pub(crate) fn vad_samples(&self) -> &[f32] {
        &self.vad_samples
    }

    pub(crate) fn into_asr_samples(self) -> Vec<f32> {
        self.asr_samples.unwrap_or(self.vad_samples)
    }
}

struct NoiseCancellationStage {
    engine: Box<dyn NoiseCancellationEngine>,
    target: NoiseCancellationTarget,
    delayed_raw_samples: VecDeque<f32>,
}

impl NoiseCancellationStage {
    fn new(engine: Box<dyn NoiseCancellationEngine>, target: NoiseCancellationTarget) -> Self {
        let output_delay_samples = engine.output_delay_samples();
        Self {
            engine,
            target,
            delayed_raw_samples: VecDeque::from(vec![0.0; output_delay_samples]),
        }
    }

    fn process(&mut self, raw_samples: &[f32]) -> Result<ProcessedAudioChunk> {
        if self.target == NoiseCancellationTarget::VadOnly {
            self.delayed_raw_samples.extend(raw_samples.iter().copied());
        }
        let enhanced_samples = self.engine.process(raw_samples)?;
        if self.target == NoiseCancellationTarget::VadAndAsr {
            return Ok(ProcessedAudioChunk::shared(enhanced_samples));
        }
        if enhanced_samples.len() > self.delayed_raw_samples.len() {
            return Err(anyhow!(
                "noise cancellation produced {} samples with only {} aligned raw samples available",
                enhanced_samples.len(),
                self.delayed_raw_samples.len()
            ));
        }
        let aligned_raw_samples = self
            .delayed_raw_samples
            .drain(..enhanced_samples.len())
            .collect();
        Ok(ProcessedAudioChunk::split(
            enhanced_samples,
            aligned_raw_samples,
        ))
    }
}

impl AudioInputProcessor {
    #[cfg(test)]
    pub(crate) fn initialize(
        handle: AppHandle,
        config: &ParapperConfig,
        source_sample_rate: u32,
    ) -> Result<Self> {
        Self::initialize_for_source(handle, config, source_sample_rate, None)
    }

    pub(crate) fn initialize_for_source(
        handle: AppHandle,
        config: &ParapperConfig,
        source_sample_rate: u32,
        source_id: Option<SourceId>,
    ) -> Result<Self> {
        let vad_interval_ms = validated_vad_interval_ms(config.segmentation.vad_interval_ms);
        let resampler = match MonoFastFixedInResampler::new(
            source_sample_rate,
            ASR_SAMPLE_RATE,
            vad_interval_ms,
        ) {
            Ok(resampler) => resampler,
            Err(err) => {
                emit_parapper_error(
                    &handle,
                    ParapperErrorType::Resampler,
                    ErrorSeverity::Fatal,
                    Some(err.to_string()),
                );
                return Err(err);
            }
        };
        let noise_cancellation = match create_noise_cancellation_engine(&handle, config) {
            Ok(noise_cancellation) => noise_cancellation,
            Err(err) => {
                emit_parapper_error(
                    &handle,
                    ParapperErrorType::AudioInput,
                    ErrorSeverity::Fatal,
                    Some(err.to_string()),
                );
                return Err(err);
            }
        };
        let noise_cancellation = noise_cancellation
            .map(|engine| NoiseCancellationStage::new(engine, config.noise_cancellation.target));
        Ok(Self {
            handle,
            resampler,
            resampled_chunks: Vec::with_capacity(1),
            noise_cancellation,
            input_level_emitter: InputLevelEmitter::default(),
            source_sample_rate,
            source_id: source_id.map(|source_id| source_id.to_string()),
        })
    }

    pub(crate) fn process(
        &mut self,
        chunk: &InputChunk,
        config: &ParapperConfig,
        mut on_processed_chunk: impl FnMut(ProcessedAudioChunk),
    ) {
        let input_gain = configured_input_gain(config);
        let Ok(()) = self
            .resampler
            .push_into(&chunk.samples, &mut self.resampled_chunks)
        else {
            emit_parapper_error(
                &self.handle,
                ParapperErrorType::Resampler,
                ErrorSeverity::Warning,
                Some("Failed to resample input audio".to_string()),
            );
            return;
        };
        let mut resampled_chunks = std::mem::take(&mut self.resampled_chunks);
        for mut samples in resampled_chunks.drain(..) {
            let pre_gain_level = peak_level(&samples);
            apply_input_gain(&mut samples, input_gain);
            let processed = if let Some(noise_cancellation) = self.noise_cancellation.as_mut() {
                match noise_cancellation.process(&samples) {
                    Ok(processed) => processed,
                    Err(err) => {
                        emit_parapper_error(
                            &self.handle,
                            ParapperErrorType::AudioInput,
                            ErrorSeverity::Warning,
                            Some(err.to_string()),
                        );
                        continue;
                    }
                }
            } else {
                ProcessedAudioChunk::shared(samples)
            };
            let post_gain_level = peak_level(processed.vad_samples());
            self.input_level_emitter.push_for_source(
                &self.handle,
                pre_gain_level,
                post_gain_level,
                self.source_id.as_deref(),
            );
            let event = AudioChunkEvent {
                source_id: self.source_id.clone(),
                source_sample_rate: self.source_sample_rate,
                sample_rate: ASR_SAMPLE_RATE,
                frames: processed.vad_samples().len(),
                level: post_gain_level,
                pre_gain_level,
            };
            let _ = self.handle.emit("parapper://audio-chunk", event);
            on_processed_chunk(processed);
        }
        self.resampled_chunks = resampled_chunks;
    }
}

fn create_noise_cancellation_engine(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<Option<Box<dyn NoiseCancellationEngine>>> {
    if !config.noise_cancellation.enabled {
        return Ok(None);
    }

    let model_dir = noise_cancellation_model_dir(handle, config.noise_cancellation.model)?;
    Ok(Some(Box::new(UlUnasNoiseCancellationEngine::new(
        &model_dir,
    )?)))
}

fn input_volume_db_to_gain(volume_db: f32) -> f32 {
    let volume_db = if volume_db.is_finite() {
        volume_db.clamp(-30.0, 30.0)
    } else {
        0.0
    };
    10.0_f32.powf(volume_db / 20.0)
}

fn configured_input_gain(config: &ParapperConfig) -> f32 {
    // Mute is intentionally distinct from an ordinary 0% gain. It still
    // advances this source through VAD with silent frames, allowing any active
    // segment to close deterministically while every other source remains
    // scheduled normally.
    if config.input.muted {
        0.0
    } else {
        input_volume_db_to_gain(config.input.volume_db)
    }
}

fn apply_input_gain(samples: &mut [f32], gain: f32) {
    let gain = if gain.is_finite() { gain } else { 1.0 };
    for sample in samples {
        *sample *= gain;
    }
}

#[derive(Default)]
struct InputLevelEmitter {
    chunks_since_emit: u32,
    pre_gain_peak_level: f32,
    post_gain_peak_level: f32,
}

impl InputLevelEmitter {
    #[cfg(test)]
    fn push(&mut self, handle: &AppHandle, pre_gain_level: f32, post_gain_level: f32) {
        self.push_for_source(handle, pre_gain_level, post_gain_level, None);
    }

    fn push_for_source(
        &mut self,
        handle: &AppHandle,
        pre_gain_level: f32,
        post_gain_level: f32,
        source_id: Option<&str>,
    ) {
        self.chunks_since_emit += 1;
        if pre_gain_level.is_finite() {
            self.pre_gain_peak_level = self.pre_gain_peak_level.max(pre_gain_level);
        }
        if post_gain_level.is_finite() {
            self.post_gain_peak_level = self.post_gain_peak_level.max(post_gain_level);
        }

        if self.chunks_since_emit >= INPUT_LEVEL_EMIT_CHUNKS {
            let _ = handle.emit(
                "parapper://input-level",
                InputLevelEvent {
                    source_id: source_id.map(str::to_owned),
                    pre_gain_level: self.pre_gain_peak_level,
                    post_gain_level: self.post_gain_peak_level,
                },
            );
            self.chunks_since_emit = 0;
            self.pre_gain_peak_level = 0.0;
            self.post_gain_peak_level = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicUsize, mpsc},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use tauri::Listener;

    use super::{
        ASR_SAMPLE_RATE, AudioInputProcessor, BoundedDemuxLaneSender, DemuxControl, DemuxLane,
        DemuxRunError, ExplicitChannelLayoutError, InputLevelEmitter, InputLevelEvent,
        NoiseCancellationStage, SourceQueueOverrun, apply_input_gain, build_explicit_lanes,
        configured_input_gain, input_volume_db_to_gain, run_demux,
        validate_explicit_channel_layout,
    };
    use crate::audio::stream::{CaptureSequence, InterleavedPcmChunk, peak_level};
    use crate::config::{
        NoiseCancellationModel, NoiseCancellationTarget, ParapperConfig, RecognitionSourceConfig,
    };
    use parapper_models::nc::NoiseCancellationEngine;
    use parapper_stt_engine::{SourceId, SourceIdentitySnapshot};

    struct DelayedMarkerNoiseCancellation;

    #[test]
    fn explicit_channel_validation_rejects_out_of_range_before_stream_or_worker_creation() {
        let sources = vec![RecognitionSourceConfig {
            source_id: "speaker-b".to_owned(),
            speaker_label: "Speaker B".to_owned(),
            capture_endpoint_id: "interface-1".to_owned(),
            channel_index: 2,
            asr_route_policy: None,
            delivery_profile_id: None,
        }];

        assert_eq!(
            validate_explicit_channel_layout("interface-1", &sources, 2),
            Err(ExplicitChannelLayoutError::ChannelIndexOutOfRange {
                endpoint_id: "interface-1".to_owned(),
                source_id: "speaker-b".to_owned(),
                channel_index: 2,
                channel_count: 2,
            })
        );
    }

    #[test]
    fn explicit_lane_startup_preserves_config_order_and_identity_snapshot() {
        let sources = vec![
            RecognitionSourceConfig {
                source_id: "speaker-b".to_owned(),
                speaker_label: "Speaker B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: None,
                delivery_profile_id: None,
            },
            RecognitionSourceConfig {
                source_id: "speaker-a".to_owned(),
                speaker_label: "Speaker A".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 0,
                asr_route_policy: None,
                delivery_profile_id: None,
            },
        ];

        let (demux_lanes, startup_lanes) = build_explicit_lanes(&sources, ASR_SAMPLE_RATE);

        assert_eq!(
            startup_lanes
                .iter()
                .map(|lane| lane.identity.clone())
                .collect::<Vec<_>>(),
            vec![
                SourceIdentitySnapshot::new(
                    SourceId::from("speaker-b"),
                    "Speaker B".to_owned(),
                    "interface-1".to_owned(),
                    Some(1),
                ),
                SourceIdentitySnapshot::new(
                    SourceId::from("speaker-a"),
                    "Speaker A".to_owned(),
                    "interface-1".to_owned(),
                    Some(0),
                ),
            ]
        );
        assert_eq!(
            demux_lanes
                .iter()
                .map(|lane| lane.channel_index)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert_eq!(
            startup_lanes
                .iter()
                .map(|lane| lane.source_sample_rate)
                .collect::<Vec<_>>(),
            vec![ASR_SAMPLE_RATE, ASR_SAMPLE_RATE],
            "every lane must retain the native rate of its endpoint so a later multi-device scheduler can resample each source independently"
        );
    }

    #[test]
    fn demux_worker_preserves_capture_sequence_and_sample_order_for_every_lane() {
        let (central_sender, central_receiver) = mpsc::channel();
        central_sender
            .send(InterleavedPcmChunk::new(
                CaptureSequence(41),
                2,
                vec![1.0, 10.0, 2.0, 20.0],
            ))
            .unwrap();
        central_sender
            .send(InterleavedPcmChunk::new(
                CaptureSequence(42),
                2,
                vec![3.0, 30.0],
            ))
            .unwrap();
        drop(central_sender);
        let (speaker_b, first_lane_receiver) = demux_lane("speaker-b", "Speaker B", 1);
        let (speaker_a, second_lane_receiver) = demux_lane("speaker-a", "Speaker A", 0);
        let control = Arc::new(DemuxControl::default());

        run_demux(
            "interface-1",
            central_receiver,
            vec![speaker_b, speaker_a],
            &control,
        )
        .expect("valid central chunks must drain to every configured lane");

        assert_eq!(
            captured_lane_chunks(first_lane_receiver),
            vec![
                (CaptureSequence(41), vec![10.0, 20.0]),
                (CaptureSequence(42), vec![30.0]),
            ]
        );
        assert_eq!(
            captured_lane_chunks(second_lane_receiver),
            vec![
                (CaptureSequence(41), vec![1.0, 2.0]),
                (CaptureSequence(42), vec![3.0]),
            ]
        );
    }

    #[test]
    fn explicit_lane_queue_overrun_records_discontinuities_without_aborting_capture() {
        let sources = vec![
            RecognitionSourceConfig {
                source_id: "speaker-a".to_owned(),
                speaker_label: "Speaker A".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 0,
                asr_route_policy: None,
                delivery_profile_id: None,
            },
            RecognitionSourceConfig {
                source_id: "speaker-b".to_owned(),
                speaker_label: "Speaker B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: None,
                delivery_profile_id: None,
            },
        ];
        // At 1 Hz the documented two-second budget admits two one-sample
        // chunks. A third capture payload overruns both full lanes, and its
        // discontinuity is observable rather than silently accepted.
        let (lanes, startups) = build_explicit_lanes(&sources, 1);
        let (central_sender, central_receiver) = mpsc::channel();
        for (sequence, samples) in [
            (1, vec![1.0, 10.0]),
            (2, vec![2.0, 20.0]),
            (3, vec![3.0, 30.0]),
        ] {
            central_sender
                .send(InterleavedPcmChunk::new(
                    CaptureSequence(sequence),
                    2,
                    samples,
                ))
                .unwrap();
        }
        drop(central_sender);

        run_demux(
            "interface-1",
            central_receiver,
            lanes,
            &DemuxControl::default(),
        )
        .expect("one source queue overrun must not terminate the capture endpoint");

        let delivered = startups
            .into_iter()
            .map(|lane| {
                let queue_state = lane.queue_overrun.snapshot();
                let samples = captured_lane_chunks(lane.receiver);
                (samples, queue_state)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            delivered,
            vec![
                (
                    vec![
                        (CaptureSequence(1), vec![1.0]),
                        (CaptureSequence(2), vec![2.0]),
                    ],
                    (1, 1),
                ),
                (
                    vec![
                        (CaptureSequence(1), vec![10.0]),
                        (CaptureSequence(2), vec![20.0]),
                    ],
                    (1, 1),
                ),
            ],
            "each full source queue records its own loss and retains FIFO audio accepted before the discontinuity"
        );
    }

    #[test]
    fn cancelled_demux_discards_central_backlog_without_lane_delivery() {
        let (central_sender, central_receiver) = mpsc::channel();
        central_sender
            .send(InterleavedPcmChunk::new(
                CaptureSequence(41),
                2,
                vec![1.0, 10.0],
            ))
            .unwrap();
        drop(central_sender);
        let (lane, lane_receiver) = demux_lane("speaker-a", "Speaker A", 0);
        let control = Arc::new(DemuxControl::default());
        control.cancel();

        run_demux("interface-1", central_receiver, vec![lane], &control)
            .expect("cancelled demux must drain and close cleanly");

        assert_eq!(captured_lane_chunks(lane_receiver), Vec::new());
    }

    #[test]
    fn disconnected_source_sender_stops_demux_and_closes_every_lane() {
        let (central_sender, central_receiver) = mpsc::channel();
        central_sender
            .send(InterleavedPcmChunk::new(
                CaptureSequence(41),
                2,
                vec![1.0, 10.0],
            ))
            .unwrap();
        drop(central_sender);
        let (disconnected, disconnected_receiver) = demux_lane("speaker-a", "Speaker A", 0);
        let (other, other_receiver) = demux_lane("speaker-b", "Speaker B", 1);
        drop(disconnected_receiver);

        assert_eq!(
            run_demux(
                "interface-1",
                central_receiver,
                vec![disconnected, other],
                &DemuxControl::default(),
            ),
            Err(DemuxRunError::LaneDisconnected {
                endpoint_id: "interface-1".to_owned(),
                source_id: "speaker-a".to_owned(),
                capture_sequence: CaptureSequence(41),
            })
        );
        assert!(
            other_receiver.recv().is_err(),
            "one disconnected source must terminate demux and close every remaining sender"
        );
    }

    fn demux_lane(
        source_id: &str,
        speaker_label: &str,
        channel_index: u16,
    ) -> (DemuxLane, mpsc::Receiver<crate::audio::InputChunk>) {
        let (sender, receiver) = mpsc::channel();
        (
            DemuxLane {
                identity: SourceIdentitySnapshot::new(
                    SourceId::from(source_id),
                    speaker_label.to_owned(),
                    "interface-1".to_owned(),
                    Some(channel_index),
                ),
                channel_index,
                sender: BoundedDemuxLaneSender {
                    sender,
                    queued_samples: Arc::new(AtomicUsize::new(0)),
                    max_queued_samples: ASR_SAMPLE_RATE as usize * 2,
                    overrun: Arc::new(SourceQueueOverrun::default()),
                },
            },
            receiver,
        )
    }

    fn captured_lane_chunks(
        receiver: mpsc::Receiver<crate::audio::InputChunk>,
    ) -> Vec<(CaptureSequence, Vec<f32>)> {
        receiver
            .into_iter()
            .map(|chunk| {
                (
                    chunk
                        .capture_sequence
                        .expect("explicit lane chunk must retain capture sequence"),
                    chunk.samples,
                )
            })
            .collect()
    }

    impl NoiseCancellationEngine for DelayedMarkerNoiseCancellation {
        fn process(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
            Ok(samples.iter().map(|sample| sample + 100.0).collect())
        }

        fn output_delay_samples(&self) -> usize {
            2
        }
    }

    #[test]
    fn input_volume_db_to_gain_uses_decibel_scale() {
        assert!((input_volume_db_to_gain(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((input_volume_db_to_gain(20.0) - 10.0).abs() < 0.0001);
        assert!((input_volume_db_to_gain(-20.0) - 0.1).abs() < 0.0001);
    }

    #[test]
    fn input_mute_forces_silent_vad_input_without_changing_the_saved_gain() {
        let mut config = ParapperConfig::default();
        config.input.volume_db = 6.0;
        config.input.muted = true;

        assert!(
            configured_input_gain(&config).abs() < f32::EPSILON,
            "mute must gate this source's VAD/ASR samples even when its configured gain remains nonzero"
        );
        assert!(
            input_volume_db_to_gain(config.input.volume_db) > 1.0,
            "the persisted volume must remain available after unmute"
        );
    }

    #[test]
    fn apply_input_gain_does_not_clip_audio_sample_range() {
        let mut samples = vec![-0.25, 0.25, 0.75];

        apply_input_gain(&mut samples, 2.0);

        assert_f32_slice_close(&samples, &[-0.5, 0.5, 1.5]);
    }

    #[test]
    fn input_level_peak_preserves_values_above_display_range() {
        let mut samples = vec![-0.5, 0.25, 0.75];

        apply_input_gain(&mut samples, 4.0);

        assert_f32_slice_close(&samples, &[-2.0, 1.0, 3.0]);
        assert!((peak_level(&samples) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn vad_only_noise_cancellation_sends_enhanced_audio_to_vad_and_aligned_raw_audio_to_asr() {
        let mut stage = NoiseCancellationStage::new(
            Box::new(DelayedMarkerNoiseCancellation),
            NoiseCancellationTarget::VadOnly,
        );

        let first = stage.process(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let second = stage.process(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(first.vad_samples(), [101.0, 102.0, 103.0, 104.0]);
        assert_eq!(first.into_asr_samples(), vec![0.0, 0.0, 1.0, 2.0]);
        assert_eq!(second.vad_samples(), [105.0, 106.0, 107.0, 108.0]);
        assert_eq!(second.into_asr_samples(), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn vad_and_asr_noise_cancellation_keeps_the_legacy_shared_enhanced_stream() {
        let mut stage = NoiseCancellationStage::new(
            Box::new(DelayedMarkerNoiseCancellation),
            NoiseCancellationTarget::VadAndAsr,
        );

        let processed = stage.process(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(processed.vad_samples(), [101.0, 102.0, 103.0, 104.0]);
        assert_eq!(
            processed.into_asr_samples(),
            vec![101.0, 102.0, 103.0, 104.0]
        );
    }

    #[test]
    fn audio_input_processor_fails_when_noise_cancellation_model_is_missing() {
        let handle = tauri_test_handle();
        let config = parapper_config! {
            noise_cancellation_enabled: true,
            noise_cancellation_model: NoiseCancellationModel::UlUnas,
            model_dir: Some(missing_model_dir("noise-cancellation-init-failure")),
            ..ParapperConfig::default()
        };

        let err = AudioInputProcessor::initialize(handle, &config, ASR_SAMPLE_RATE)
            .err()
            .expect("missing noise cancellation model should fail audio input startup");

        assert!(
            err.to_string()
                .contains("Noise cancellation model not found"),
            "unexpected noise cancellation init error: {err}"
        );
    }

    #[test]
    fn input_level_emitter_emits_every_three_chunks_and_resets_peaks() {
        let handle = tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<InputLevelEvent>();
        let _event_id = handle.listen("parapper://input-level", move |event| {
            let payload = serde_json::from_str::<InputLevelEvent>(event.payload())
                .expect("input level payload should decode");
            sender
                .send(payload)
                .expect("input level event should be recorded");
        });
        let mut emitter = InputLevelEmitter::default();

        emitter.push_for_source(&handle, 0.1, 0.2, Some("profile-a"));
        emitter.push_for_source(&handle, f32::NAN, f32::INFINITY, Some("profile-a"));
        assert!(
            receiver.recv_timeout(Duration::from_millis(50)).is_err(),
            "input level should not emit before three chunks"
        );
        emitter.push_for_source(&handle, 0.5, 0.6, Some("profile-a"));
        let first = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("third chunk should emit input level");
        assert_f32_close(first.pre_gain_level, 0.5);
        assert_f32_close(first.post_gain_level, 0.6);
        assert_eq!(first.source_id.as_deref(), Some("profile-a"));

        emitter.push(&handle, 0.1, 0.1);
        emitter.push(&handle, 0.2, 0.2);
        emitter.push(&handle, 0.3, 0.3);
        let second = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("sixth chunk should emit input level after reset");
        assert_f32_close(second.pre_gain_level, 0.3);
        assert_f32_close(second.post_gain_level, 0.3);
        assert_eq!(second.source_id, None);
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() < f32::EPSILON,
                "index={index}, actual={actual}, expected={expected}"
            );
        }
    }

    fn missing_model_dir(test_name: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "parapper-missing-audio-input-model-{test_name}-{}-{unique}",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }

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
