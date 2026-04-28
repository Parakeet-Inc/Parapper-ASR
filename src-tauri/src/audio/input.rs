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
    recognition::RecognitionPipeline,
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

        log::info!(
            "Started audio input: {} [{}] {} Hz, {} channel(s)",
            selection.device_info.display_name,
            selection.device_info.host,
            source_sample_rate,
            selection.stream_config.channels
        );

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
            Ok(chunk) => process_input_chunk(
                handle,
                &mut resampler,
                &mut input_level_emitter,
                recognition.as_mut(),
                source_sample_rate,
                &chunk,
            ),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    if let Some(recognition) = recognition {
        recognition.stop();
    }
}

fn process_input_chunk(
    handle: &AppHandle,
    resampler: &mut MonoFastFixedInResampler,
    input_level_emitter: &mut InputLevelEmitter,
    mut recognition: Option<&mut RecognitionPipeline>,
    source_sample_rate: u32,
    chunk: &InputChunk,
) {
    let Ok(resampled_chunks) = resampler.push(&chunk.samples) else {
        emit_parapper_error(
            handle,
            ParapperErrorType::Resampler,
            ErrorSeverity::Warning,
            Some("Failed to resample input audio".to_string()),
        );
        return;
    };
    for samples in resampled_chunks {
        let level = peak_level(&samples);
        input_level_emitter.push(handle, level);
        let event = AudioChunkEvent {
            source_sample_rate,
            sample_rate: ASR_SAMPLE_RATE,
            frames: samples.len(),
            level,
        };
        let _ = handle.emit("parapper://audio-chunk", event);
        if let Some(recognition) = recognition.as_deref_mut()
            && let Err(err) = recognition.process_chunk(&samples)
        {
            emit_parapper_error(
                handle,
                ParapperErrorType::Vad,
                ErrorSeverity::Warning,
                Some(err.to_string()),
            );
        }
    }
}

#[derive(Default)]
struct InputLevelEmitter {
    chunks_since_emit: u32,
    peak_level: f32,
}

impl InputLevelEmitter {
    fn push(&mut self, handle: &AppHandle, level: f32) {
        self.chunks_since_emit += 1;
        self.peak_level = self.peak_level.max(level);

        if self.chunks_since_emit >= INPUT_LEVEL_EMIT_CHUNKS {
            let _ = handle.emit("parapper://input-level", self.peak_level);
            self.chunks_since_emit = 0;
            self.peak_level = 0.0;
        }
    }
}
