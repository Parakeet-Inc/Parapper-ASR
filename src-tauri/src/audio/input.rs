use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, sync_channel},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result};
use cpal::traits::StreamTrait;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::{
    config::ParapperConfig,
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    model::noise_cancellation_model_dir,
    recognition::{
        RecognitionPipeline,
        engines::{NoiseCancellationEngine, UlUnasNoiseCancellationEngine},
    },
};

use super::{
    device::selected_input_device,
    resampler::{MonoFastFixedInResampler, validated_vad_interval_ms},
    stream::{InputChunk, build_input_stream, peak_level},
};

pub const ASR_SAMPLE_RATE: u32 = 16_000;
const INPUT_QUEUE_SIZE: usize = 8;
const INPUT_LEVEL_EMIT_CHUNKS: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunkEvent {
    pub source_sample_rate: u32,
    pub sample_rate: u32,
    pub frames: usize,
    pub level: f32,
    pub pre_gain_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputLevelEvent {
    pub pre_gain_level: f32,
    pub post_gain_level: f32,
}

pub struct RunningAudioInput {
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

impl RunningAudioInput {
    pub fn start(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: Arc<RwLock<ParapperConfig>>,
    ) -> Result<Self> {
        let selection = selected_input_device(config)?;
        let source_sample_rate = selection.stream_config.sample_rate;
        let (sender, receiver) = sync_channel(INPUT_QUEUE_SIZE);
        let stream = build_input_stream(
            &selection.device,
            &selection.stream_config,
            selection.sample_format,
            sender,
        )?;
        stream.play().context("Failed to start input stream")?;

        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = stop_requested.clone();
        let recognition_config = config.clone();
        let join_handle = thread::Builder::new()
            .name("parapper-audio-input".to_string())
            .spawn(move || {
                let handle = handle;
                let recognition_config = recognition_config;
                let receiver = receiver;
                let worker_stop = worker_stop;
                run_audio_worker(
                    &handle,
                    &recognition_config,
                    &runtime_config,
                    &receiver,
                    source_sample_rate,
                    &worker_stop,
                );
                drop(stream);
            })
            .context("Failed to spawn audio worker")?;

        Ok(Self {
            stop_requested,
            join_handle: Some(join_handle),
        })
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        if let Some(join_handle) = self.join_handle.take()
            && let Err(err) = join_handle.join()
        {
            log::warn!("Audio worker thread panicked: {err:?}");
        }
    }
}

impl Drop for RunningAudioInput {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

fn run_audio_worker(
    handle: &AppHandle,
    config: &ParapperConfig,
    runtime_config: &Arc<RwLock<ParapperConfig>>,
    receiver: &Receiver<InputChunk>,
    source_sample_rate: u32,
    stop_requested: &AtomicBool,
) {
    let mut vad_interval_ms = validated_vad_interval_ms(config.vad_interval_ms);
    let mut resampler =
        match MonoFastFixedInResampler::new(source_sample_rate, ASR_SAMPLE_RATE, vad_interval_ms) {
            Ok(resampler) => resampler,
            Err(err) => {
                emit_parapper_error(
                    handle,
                    ParapperErrorType::Resampler,
                    ErrorSeverity::Fatal,
                    Some(err.to_string()),
                );
                return;
            }
        };
    let mut recognition = match RecognitionPipeline::new(handle.clone(), config, runtime_config) {
        Ok(recognition) => Some(recognition),
        Err(err) => {
            emit_parapper_error(
                handle,
                ParapperErrorType::AudioInput,
                ErrorSeverity::Fatal,
                Some(err.to_string()),
            );
            None
        }
    };
    let mut noise_cancellation = match create_noise_cancellation_engine(handle, config) {
        Ok(noise_cancellation) => noise_cancellation,
        Err(err) => {
            emit_parapper_error(
                handle,
                ParapperErrorType::AudioInput,
                ErrorSeverity::Fatal,
                Some(err.to_string()),
            );
            recognition = None;
            None
        }
    };
    let mut input_level_emitter = InputLevelEmitter::default();

    while !stop_requested.load(Ordering::Acquire) {
        let current_config = runtime_config
            .read()
            .map_or_else(|_| config.clone(), |config| config.clone());
        let current_vad_interval_ms = validated_vad_interval_ms(current_config.vad_interval_ms);
        if current_vad_interval_ms != vad_interval_ms {
            match MonoFastFixedInResampler::new(
                source_sample_rate,
                ASR_SAMPLE_RATE,
                current_vad_interval_ms,
            ) {
                Ok(next_resampler) => {
                    resampler = next_resampler;
                    vad_interval_ms = current_vad_interval_ms;
                }
                Err(err) => {
                    emit_parapper_error(
                        handle,
                        ParapperErrorType::Resampler,
                        ErrorSeverity::Warning,
                        Some(err.to_string()),
                    );
                }
            }
        }
        if let Some(recognition) = recognition.as_mut() {
            recognition.update_config(&current_config);
        }

        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                let input_gain = input_volume_db_to_gain(current_config.input_volume_db);
                InputChunkProcessor {
                    handle,
                    resampler: &mut resampler,
                    noise_cancellation: &mut noise_cancellation,
                    input_level_emitter: &mut input_level_emitter,
                    recognition: recognition.as_mut(),
                    source_sample_rate,
                    input_gain,
                }
                .process(&chunk);
                tick_recognition_after_worker_iteration(
                    recognition
                        .as_mut()
                        .map(|recognition| recognition as &mut dyn RecognitionTicker),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                tick_recognition_after_worker_iteration(
                    recognition
                        .as_mut()
                        .map(|recognition| recognition as &mut dyn RecognitionTicker),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    if let Some(recognition) = recognition {
        recognition.stop();
    }
}

trait RecognitionTicker {
    fn tick(&mut self);
}

impl RecognitionTicker for RecognitionPipeline {
    fn tick(&mut self) {
        RecognitionPipeline::tick(self);
    }
}

fn tick_recognition_after_worker_iteration(recognition: Option<&mut dyn RecognitionTicker>) {
    if let Some(recognition) = recognition {
        recognition.tick();
    }
}

fn create_noise_cancellation_engine(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<Option<Box<dyn NoiseCancellationEngine>>> {
    if !config.noise_cancellation_enabled {
        return Ok(None);
    }

    let model_dir = noise_cancellation_model_dir(handle, config.noise_cancellation_model)?;
    Ok(Some(Box::new(UlUnasNoiseCancellationEngine::new(
        &model_dir,
    )?)))
}

struct InputChunkProcessor<'a> {
    handle: &'a AppHandle,
    resampler: &'a mut MonoFastFixedInResampler,
    noise_cancellation: &'a mut Option<Box<dyn NoiseCancellationEngine>>,
    input_level_emitter: &'a mut InputLevelEmitter,
    recognition: Option<&'a mut RecognitionPipeline>,
    source_sample_rate: u32,
    input_gain: f32,
}

impl InputChunkProcessor<'_> {
    fn process(mut self, chunk: &InputChunk) {
        let Ok(resampled_chunks) = self.resampler.push(&chunk.samples) else {
            emit_parapper_error(
                self.handle,
                ParapperErrorType::Resampler,
                ErrorSeverity::Warning,
                Some("Failed to resample input audio".to_string()),
            );
            return;
        };
        for mut samples in resampled_chunks {
            let pre_gain_level = peak_level(&samples);
            apply_input_gain(&mut samples, self.input_gain);
            if let Some(noise_cancellation) = self.noise_cancellation.as_deref_mut() {
                samples = match noise_cancellation.process(&samples) {
                    Ok(samples) => samples,
                    Err(err) => {
                        emit_parapper_error(
                            self.handle,
                            ParapperErrorType::AudioInput,
                            ErrorSeverity::Warning,
                            Some(err.to_string()),
                        );
                        continue;
                    }
                };
            }
            let post_gain_level = peak_level(&samples);
            self.input_level_emitter
                .push(self.handle, pre_gain_level, post_gain_level);
            let event = AudioChunkEvent {
                source_sample_rate: self.source_sample_rate,
                sample_rate: ASR_SAMPLE_RATE,
                frames: samples.len(),
                level: post_gain_level,
                pre_gain_level,
            };
            let _ = self.handle.emit("parapper://audio-chunk", event);
            if let Some(recognition) = self.recognition.as_deref_mut()
                && let Err(err) = recognition.process_chunk(&samples)
            {
                emit_parapper_error(
                    self.handle,
                    ParapperErrorType::Vad,
                    ErrorSeverity::Warning,
                    Some(err.to_string()),
                );
            }
        }
    }
}

fn input_volume_db_to_gain(volume_db: f32) -> f32 {
    let volume_db = if volume_db.is_finite() {
        volume_db.clamp(-30.0, 30.0)
    } else {
        0.0
    };
    10.0_f32.powf(volume_db / 20.0)
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
    fn push(&mut self, handle: &AppHandle, pre_gain_level: f32, post_gain_level: f32) {
        self.chunks_since_emit += 1;
        self.pre_gain_peak_level = self.pre_gain_peak_level.max(pre_gain_level);
        self.post_gain_peak_level = self.post_gain_peak_level.max(post_gain_level);

        if self.chunks_since_emit >= INPUT_LEVEL_EMIT_CHUNKS {
            let _ = handle.emit(
                "parapper://input-level",
                InputLevelEvent {
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
    use super::{
        RecognitionTicker, apply_input_gain, input_volume_db_to_gain,
        tick_recognition_after_worker_iteration,
    };
    use crate::audio::stream::peak_level;

    #[derive(Default)]
    struct MockRecognitionTicker {
        ticks: usize,
    }

    impl RecognitionTicker for MockRecognitionTicker {
        fn tick(&mut self) {
            self.ticks += 1;
        }
    }

    #[test]
    fn input_volume_db_to_gain_uses_decibel_scale() {
        assert!((input_volume_db_to_gain(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((input_volume_db_to_gain(20.0) - 10.0).abs() < 0.0001);
        assert!((input_volume_db_to_gain(-20.0) - 0.1).abs() < 0.0001);
    }

    #[test]
    fn apply_input_gain_does_not_clip_audio_sample_range() {
        let mut samples = vec![-0.25, 0.25, 0.75];

        apply_input_gain(&mut samples, 2.0);

        assert_eq!(samples, vec![-0.5, 0.5, 1.5]);
    }

    #[test]
    fn input_level_peak_preserves_values_above_display_range() {
        let mut samples = vec![-0.5, 0.25, 0.75];

        apply_input_gain(&mut samples, 4.0);

        assert_eq!(samples, vec![-2.0, 1.0, 3.0]);
        assert!((peak_level(&samples) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn worker_iteration_ticks_recognition_after_input_chunk_is_processed() {
        let mut recognition = MockRecognitionTicker::default();

        tick_recognition_after_worker_iteration(Some(&mut recognition));

        assert_eq!(recognition.ticks, 1);
    }

    #[test]
    fn worker_iteration_without_recognition_does_not_panic() {
        tick_recognition_after_worker_iteration(None);
    }
}
