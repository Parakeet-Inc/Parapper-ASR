use std::{
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use tauri::{AppHandle, Emitter};

use super::events::{
    AsrMissingEvent, ConnectionStateEvent, ConnectionTarget, OscMuteStateEvent, RecognizedTextEvent,
};
use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    connect::{NeoHttpTextTransport, TextTransport, query_current_mute_state},
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    model::{AsrEngine, SherpaOnnxAsrEngine, asr_model_dir},
};

pub(crate) struct AsrWorker {
    sender: Option<SyncSender<Vec<f32>>>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

impl AsrWorker {
    pub(crate) fn start(
        handle: AppHandle,
        config: ParapperConfig,
        runtime_config: &Arc<RwLock<ParapperConfig>>,
    ) -> Result<Self> {
        let (sender, receiver) = sync_channel(4);
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = stop_requested.clone();
        let worker_config = runtime_config.clone();
        let join_handle = thread::Builder::new()
            .name("parapper-asr".to_string())
            .spawn(move || {
                run_asr_worker(&handle, &config, &worker_config, &receiver, &worker_stop);
            })
            .context("Failed to spawn ASR worker")?;

        Ok(Self {
            sender: Some(sender),
            stop_requested,
            join_handle: Some(join_handle),
        })
    }

    pub(crate) fn send(&self, phrase: Vec<f32>) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };
        if let Err(err) = sender.try_send(phrase) {
            log::warn!("Dropping phrase because ASR queue is full: {err}");
        }
    }

    pub(crate) fn stop_inner(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        self.sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            if let Err(err) = join_handle.join() {
                log::warn!("ASR worker thread panicked: {err:?}");
            }
        }
    }
}

impl Drop for AsrWorker {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

fn run_asr_worker(
    handle: &AppHandle,
    config: &ParapperConfig,
    runtime_config: &Arc<RwLock<ParapperConfig>>,
    receiver: &Receiver<Vec<f32>>,
    stop_requested: &AtomicBool,
) {
    let mut asr = match build_asr_engine(handle, config) {
        Ok(Some(asr)) => Some(asr),
        Ok(None) => {
            let _ = handle.emit(
                "parapper://asr-missing",
                AsrMissingEvent {
                    reason: "ASR model dir is not configured".to_string(),
                },
            );
            None
        }
        Err(err) => {
            let _ = handle.emit(
                "parapper://asr-missing",
                AsrMissingEvent {
                    reason: err.to_string(),
                },
            );
            None
        }
    };

    while !stop_requested.load(Ordering::Acquire) {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(phrase) => {
                let Some(asr) = asr.as_mut() else {
                    continue;
                };
                let current_config = runtime_config
                    .read()
                    .map_or_else(|_| config.clone(), |config| config.clone());
                let mut text_transport = build_text_transport(&current_config);
                let mute_check = (current_config.vrc_osc_micmute
                    && ParapperConfig::vrc_osc_supported())
                .then(|| spawn_vrchat_mute_check(handle.clone()));
                let started_at = Instant::now();
                match asr.transcribe(&phrase) {
                    Ok(text) if !text.is_empty() => {
                        emit_recognized_text(
                            handle,
                            &current_config,
                            &mut *text_transport,
                            mute_check,
                            phrase,
                            text,
                            started_at,
                        );
                    }
                    Ok(_) => {}
                    Err(err) => {
                        emit_parapper_error(
                            handle,
                            ParapperErrorType::Asr,
                            ErrorSeverity::Warning,
                            Some(err.to_string()),
                        );
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn emit_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    text_transport: &mut dyn TextTransport,
    mute_check: Option<JoinHandle<bool>>,
    phrase: Vec<f32>,
    text: String,
    started_at: Instant,
) {
    let elapsed_millis = started_at.elapsed().as_millis();
    #[expect(clippy::cast_precision_loss)]
    let audio_seconds = phrase.len() as f64 / f64::from(ASR_SAMPLE_RATE);
    let recognized_at_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default();
    log::info!(
        "ASR completed in {} ms for {} frames",
        elapsed_millis,
        phrase.len()
    );
    let vrchat_muted = config.vrc_osc_micmute
        && ParapperConfig::vrc_osc_supported()
        && is_vrchat_muted_before_send(handle, mute_check);
    if config.neo_http_enabled && ParapperConfig::neo_http_supported() {
        if vrchat_muted {
            log::info!("Clearing NEO text because VRChat is muted");
            if let Err(err) = text_transport.send_text("") {
                log::warn!("Failed to clear NEO text while VRChat is muted: {err}");
                emit_connection_state(handle, ConnectionTarget::Neo, false, Some(err.to_string()));
            } else {
                emit_connection_state(handle, ConnectionTarget::Neo, true, None);
            }
        } else if let Err(err) = text_transport.send_text(&text) {
            log::warn!("Failed to send text to NEO API: {err}");
            emit_connection_state(handle, ConnectionTarget::Neo, false, Some(err.to_string()));
        } else {
            emit_connection_state(handle, ConnectionTarget::Neo, true, None);
        }
    }
    emit_recognized_text_event(
        handle,
        config,
        phrase,
        text,
        recognized_at_millis,
        audio_seconds,
        elapsed_millis,
    );
}

fn emit_recognized_text_event(
    handle: &AppHandle,
    config: &ParapperConfig,
    phrase: Vec<f32>,
    text: String,
    recognized_at_millis: u64,
    audio_seconds: f64,
    elapsed_millis: u128,
) {
    let _ = handle.emit(
        "parapper://recognized-text",
        RecognizedTextEvent {
            text,
            recognized_at_millis,
            audio_seconds,
            elapsed_millis,
            audio_frames: phrase.len(),
            debug_asr_audio_sample_rate: config.debug_asr_audio_playback.then_some(ASR_SAMPLE_RATE),
            debug_asr_audio_samples: config.debug_asr_audio_playback.then_some(phrase),
        },
    );
}

fn spawn_vrchat_mute_check(handle: AppHandle) -> JoinHandle<bool> {
    thread::spawn(move || query_vrchat_mute_state_with_cache(&handle))
}

fn is_vrchat_muted_before_send(handle: &AppHandle, mute_check: Option<JoinHandle<bool>>) -> bool {
    if let Some(mute_check) = mute_check {
        return if let Ok(is_muted) = mute_check.join() {
            is_muted
        } else {
            emit_osc_mute_state(handle, None);
            false
        };
    }

    query_vrchat_mute_state_with_cache(handle)
}

fn query_vrchat_mute_state_with_cache(handle: &AppHandle) -> bool {
    if let Ok(is_muted) = query_current_mute_state() {
        emit_connection_state(handle, ConnectionTarget::Vrchat, true, None);
        emit_osc_mute_state(handle, Some(is_muted));
        is_muted
    } else {
        emit_connection_state(handle, ConnectionTarget::Vrchat, false, None);
        emit_osc_mute_state(handle, None);
        false
    }
}

fn emit_osc_mute_state(handle: &AppHandle, muted: Option<bool>) {
    let _ = handle.emit("parapper://osc-mute-state", OscMuteStateEvent { muted });
}

fn emit_connection_state(
    handle: &AppHandle,
    target: ConnectionTarget,
    found: bool,
    detail: Option<String>,
) {
    let _ = handle.emit(
        "parapper://connection-state",
        ConnectionStateEvent {
            target,
            found,
            detail,
        },
    );
}

fn build_asr_engine(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<Option<Box<dyn AsrEngine>>> {
    let model_dir = asr_model_dir(handle, config)?;
    let engine = SherpaOnnxAsrEngine::new(
        &model_dir,
        config.asr_model,
        config.asr_precision,
        config.asr_num_threads,
    )?;
    Ok(Some(Box::new(engine)))
}

fn build_text_transport(config: &ParapperConfig) -> Box<dyn TextTransport> {
    Box::new(NeoHttpTextTransport::localhost(config.neo_http_port))
}
