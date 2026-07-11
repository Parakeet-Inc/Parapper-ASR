use std::{
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex, mpsc::Sender},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::{
    config::{
        ConfigPreset, DeveloperConnectionMode, InputSourceKind, ParapperConfig, SpeechMapping,
        StreamingRecognitionOutputMode, delete_config_preset, load_config_presets,
        save_config_preset,
    },
    model::{ModelStatus, any_model_installed_in, model_status_from_root, models_root},
    recognition::{
        RecognitionShutdownResult, RecognitionStartError, RecognitionStatus,
        RecognitionStreamEvent, RunningInputSource, RunningRecognitionInput, RuntimeConfigState,
        TurnOutputSink,
    },
    streaming_recognition::{
        NetworkOutputMode, StreamingRecognitionServer, StreamingRecognitionServerConfig,
    },
    synthesis::prewarm_local_tts_engines,
    translation::TranslationHttpListener,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranslationHttpListenerState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationHttpListenerStatus {
    pub state: TranslationHttpListenerState,
    pub port: Option<u16>,
    pub error: Option<String>,
}

impl Default for TranslationHttpListenerStatus {
    fn default() -> Self {
        Self {
            state: TranslationHttpListenerState::Stopped,
            port: None,
            error: None,
        }
    }
}

pub struct AppState {
    config_path: PathBuf,
    config_presets_path: PathBuf,
    models_root: PathBuf,
    config: Mutex<ParapperConfig>,
    runtime_config: Arc<RuntimeConfigState>,
    recognition_status: Mutex<RecognitionStatus>,
    recognition_session: Mutex<RecognitionSessionSlot<RunningRecognitionInput>>,
    streaming_recognition_server: Mutex<Option<StreamingRecognitionServer>>,
    translation_http_listener: StdMutex<Option<TranslationHttpListener>>,
    translation_http_listener_status: StdMutex<TranslationHttpListenerStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecognitionSessionOwner {
    Desktop,
    WebSocket { session_id: String },
}

struct RunningRecognitionSession<T> {
    owner: RecognitionSessionOwner,
    input: T,
}

struct RecognitionSessionSlot<T> {
    active: Option<RunningRecognitionSession<T>>,
}

impl<T> Default for RecognitionSessionSlot<T> {
    fn default() -> Self {
        Self { active: None }
    }
}

impl<T> RecognitionSessionSlot<T> {
    fn insert(
        &mut self,
        owner: RecognitionSessionOwner,
        input: T,
    ) -> Result<(), RecognitionSessionOwner> {
        if let Some(active) = &self.active {
            return Err(active.owner.clone());
        }
        self.active = Some(RunningRecognitionSession { owner, input });
        Ok(())
    }

    fn owner(&self) -> Option<&RecognitionSessionOwner> {
        self.active.as_ref().map(|active| &active.owner)
    }

    fn take(&mut self, owner: &RecognitionSessionOwner) -> Option<T> {
        if self.owner() != Some(owner) {
            return None;
        }
        self.active.take().map(|active| active.input)
    }
}

impl AppState {
    pub fn build(handle: &AppHandle) -> Result<Self> {
        let app_config_dir = handle
            .path()
            .app_config_dir()
            .context("Failed to resolve app config dir")?;
        let config_path = app_config_dir.join("config.json");
        let config_presets_path = app_config_dir.join("config-presets.json");
        let models_root = models_root(handle)?;
        let config = ParapperConfig::load(&config_path)?;
        Ok(Self {
            config_path,
            config_presets_path,
            models_root,
            runtime_config: Arc::new(RuntimeConfigState::new(config.clone())),
            config: Mutex::new(config),
            recognition_status: Mutex::new(RecognitionStatus::Idle),
            recognition_session: Mutex::new(RecognitionSessionSlot::default()),
            streaming_recognition_server: Mutex::new(None),
            translation_http_listener: StdMutex::new(None),
            translation_http_listener_status: StdMutex::new(
                TranslationHttpListenerStatus::default(),
            ),
        })
    }

    pub async fn get_config(&self) -> ParapperConfig {
        self.config.lock().await.clone()
    }

    pub async fn set_config(&self, config: ParapperConfig) -> Result<ParapperConfig> {
        let mut config = config.normalized();
        if matches!(
            *self.recognition_status.lock().await,
            RecognitionStatus::WaitingForClient
                | RecognitionStatus::Listening
                | RecognitionStatus::Draining
        ) {
            let previous = self.runtime_config_snapshot()?;
            if previous.input.source_kind != config.input.source_kind
                || previous.streaming_recognition != config.streaming_recognition
            {
                anyhow::bail!(
                    "recognition input source and listener settings cannot change while recognition is running"
                );
            }
            preserve_running_vad_interval(&previous, &mut config);
            config.speech.mappings = preserve_running_speech_model_mappings(
                &previous.speech.mappings,
                &config.speech.mappings,
            );
        }
        config.save(&self.config_path)?;
        self.runtime_config.replace(config.clone());
        *self.config.lock().await = config.clone();
        Ok(config)
    }

    pub fn config_presets(&self) -> Result<Vec<ConfigPreset>> {
        load_config_presets(&self.config_presets_path)
    }

    pub fn save_config_preset(
        &self,
        name: String,
        config: ParapperConfig,
    ) -> Result<Vec<ConfigPreset>> {
        save_config_preset(&self.config_presets_path, name, config)
    }

    pub fn delete_config_preset(&self, name: String) -> Result<Vec<ConfigPreset>> {
        delete_config_preset(&self.config_presets_path, name)
    }

    pub fn model_status(&self, config: &ParapperConfig) -> ModelStatus {
        model_status_from_root(&self.models_root, config)
    }

    pub fn runtime_config_snapshot(&self) -> Result<ParapperConfig> {
        self.runtime_config.snapshot()
    }

    pub fn has_any_model_installed(&self) -> Result<bool> {
        if !self.models_root.try_exists()? {
            return Ok(false);
        }
        Ok(any_model_installed_in(&self.models_root))
    }

    pub fn translation_http_listener_status(&self) -> TranslationHttpListenerStatus {
        self.translation_http_listener_status
            .lock()
            .expect("translation HTTP listener status lock poisoned")
            .clone()
    }

    pub fn start_translation_http_listener(
        &self,
        handle: AppHandle,
        port: u16,
        local_model: crate::config::LocalTranslationModel,
    ) -> Result<TranslationHttpListenerStatus> {
        let mut listener = self
            .translation_http_listener
            .lock()
            .expect("translation HTTP listener lock poisoned");
        if listener.is_some() {
            anyhow::bail!("translation HTTP listener is already running");
        }
        self.set_translation_http_listener_status(TranslationHttpListenerStatus {
            state: TranslationHttpListenerState::Starting,
            port: Some(port),
            error: None,
        });
        match TranslationHttpListener::start(handle, port, local_model) {
            Ok(started) => {
                let bound_port = started.local_addr().port();
                *listener = Some(started);
                Ok(
                    self.set_translation_http_listener_status(TranslationHttpListenerStatus {
                        state: TranslationHttpListenerState::Running,
                        port: Some(bound_port),
                        error: None,
                    }),
                )
            }
            Err(err) => {
                self.set_translation_http_listener_status(TranslationHttpListenerStatus {
                    state: TranslationHttpListenerState::Error,
                    port: Some(port),
                    error: Some(err.to_string()),
                });
                Err(err)
            }
        }
    }

    pub async fn stop_translation_http_listener(&self) -> Result<TranslationHttpListenerStatus> {
        let listener = self
            .translation_http_listener
            .lock()
            .expect("translation HTTP listener lock poisoned")
            .take();
        let Some(listener) = listener else {
            return Ok(
                self.set_translation_http_listener_status(TranslationHttpListenerStatus::default())
            );
        };
        let port = listener.local_addr().port();
        self.set_translation_http_listener_status(TranslationHttpListenerStatus {
            state: TranslationHttpListenerState::Stopping,
            port: Some(port),
            error: None,
        });
        match tauri::async_runtime::spawn_blocking(move || listener.stop()).await {
            Ok(Ok(())) => {
                Ok(self
                    .set_translation_http_listener_status(TranslationHttpListenerStatus::default()))
            }
            Ok(Err(err)) => {
                self.set_translation_http_listener_status(TranslationHttpListenerStatus {
                    state: TranslationHttpListenerState::Error,
                    port: Some(port),
                    error: Some(err.to_string()),
                });
                Err(err)
            }
            Err(err) => {
                let error = anyhow::anyhow!("translation HTTP listener stop task failed: {err}");
                self.set_translation_http_listener_status(TranslationHttpListenerStatus {
                    state: TranslationHttpListenerState::Error,
                    port: Some(port),
                    error: Some(error.to_string()),
                });
                Err(error)
            }
        }
    }

    fn set_translation_http_listener_status(
        &self,
        status: TranslationHttpListenerStatus,
    ) -> TranslationHttpListenerStatus {
        *self
            .translation_http_listener_status
            .lock()
            .expect("translation HTTP listener status lock poisoned") = status.clone();
        status
    }

    pub async fn get_recognition_status(&self) -> RecognitionStatus {
        *self.recognition_status.lock().await
    }

    pub async fn set_recognition_status(&self, status: RecognitionStatus) -> RecognitionStatus {
        *self.recognition_status.lock().await = status;
        status
    }

    pub async fn start_audio_input(
        &self,
        handle: AppHandle,
    ) -> Result<RecognitionStatus, RecognitionStartError> {
        let config = self.get_config().await;
        if config.input.source_kind == InputSourceKind::WebSocket {
            return self.start_streaming_recognition(handle, &config).await;
        }

        // Keep the listener slot locked through desktop session acquisition so a
        // concurrent WebSocket start cannot pass its reciprocal ownership check.
        let streaming_server = self.streaming_recognition_server.lock().await;
        if streaming_server.is_some() {
            return Err(RecognitionStartError::Busy);
        }
        let mut recognition_session = self.recognition_session.lock().await;
        if let Some(owner) = recognition_session.owner() {
            if *owner == RecognitionSessionOwner::Desktop {
                return Ok(self
                    .set_recognition_status(RecognitionStatus::Listening)
                    .await);
            }
            return Err(RecognitionStartError::Busy);
        }

        prewarm_local_tts_engines(&handle, &config);
        let running_recognition_input =
            match RunningRecognitionInput::start(handle, &config, self.runtime_config.clone()) {
                Ok(input) => input,
                Err(err) => {
                    self.set_recognition_status(RecognitionStatus::Error).await;
                    return Err(err);
                }
            };
        recognition_session
            .insert(RecognitionSessionOwner::Desktop, running_recognition_input)
            .map_err(|_| RecognitionStartError::Busy)?;
        drop(recognition_session);
        drop(streaming_server);

        Ok(self
            .set_recognition_status(RecognitionStatus::Listening)
            .await)
    }

    pub async fn stop_audio_input(&self) -> RecognitionStatus {
        let streaming_server = self.streaming_recognition_server.lock().await.take();
        if let Some(server) = streaming_server {
            self.set_recognition_status(RecognitionStatus::Draining)
                .await;
            if let Err(err) = tauri::async_runtime::spawn_blocking(move || server.stop()).await {
                log::warn!("Streaming recognition server stop task failed: {err}");
            }
            return self
                .set_recognition_status(RecognitionStatus::Stopped)
                .await;
        }

        let (running_recognition_input, another_owner_is_active) = {
            let mut recognition_session = self.recognition_session.lock().await;
            let another_owner_is_active = recognition_session
                .owner()
                .is_some_and(|owner| *owner != RecognitionSessionOwner::Desktop);
            (
                recognition_session.take(&RecognitionSessionOwner::Desktop),
                another_owner_is_active,
            )
        };

        if another_owner_is_active {
            return self.get_recognition_status().await;
        }

        if let Some(running_recognition_input) = running_recognition_input {
            match tauri::async_runtime::spawn_blocking(move || running_recognition_input.stop())
                .await
            {
                Ok(RecognitionShutdownResult::TimedOut) => {
                    return self.set_recognition_status(RecognitionStatus::Error).await;
                }
                Ok(RecognitionShutdownResult::Completed | RecognitionShutdownResult::Cancelled) => {
                }
                Err(err) => {
                    log::warn!("Recognition input stop task failed: {err}");
                    return self.set_recognition_status(RecognitionStatus::Error).await;
                }
            }
        }

        self.set_recognition_status(RecognitionStatus::Stopped)
            .await
    }

    pub(crate) async fn start_network_input(
        &self,
        handle: AppHandle,
        session_id: String,
        source: RunningInputSource,
        output_sink: Box<dyn TurnOutputSink>,
        activity_sender: Sender<RecognitionStreamEvent>,
    ) -> Result<(), RecognitionStartError> {
        let owner = RecognitionSessionOwner::WebSocket { session_id };
        let mut recognition_session = self.recognition_session.lock().await;
        if recognition_session.owner().is_some() {
            return Err(RecognitionStartError::Busy);
        }

        let config = self.get_config().await;
        let running = RunningRecognitionInput::start_with_source_and_sink(
            handle,
            &config,
            self.runtime_config.clone(),
            source,
            output_sink,
            Some(activity_sender),
        )?;
        recognition_session
            .insert(owner, running)
            .map_err(|_| RecognitionStartError::Busy)?;
        drop(recognition_session);
        self.set_recognition_status(RecognitionStatus::Listening)
            .await;
        Ok(())
    }

    pub(crate) async fn stop_network_input(
        &self,
        session_id: &str,
        cancel: bool,
    ) -> (RecognitionStatus, RecognitionShutdownResult) {
        let owner = RecognitionSessionOwner::WebSocket {
            session_id: session_id.to_string(),
        };
        let running = self.recognition_session.lock().await.take(&owner);
        if let Some(running) = running {
            let stop = tauri::async_runtime::spawn_blocking(move || {
                if cancel {
                    running.cancel()
                } else {
                    running.stop()
                }
            });
            let shutdown_result = match stop.await {
                Ok(result) => result,
                Err(err) => {
                    log::warn!("Network recognition stop task failed: {err}");
                    RecognitionShutdownResult::Cancelled
                }
            };
            let next = if self.streaming_recognition_server.lock().await.is_some() {
                RecognitionStatus::WaitingForClient
            } else {
                RecognitionStatus::Stopped
            };
            return (self.set_recognition_status(next).await, shutdown_result);
        }
        (
            self.get_recognition_status().await,
            RecognitionShutdownResult::Cancelled,
        )
    }

    async fn start_streaming_recognition(
        &self,
        handle: AppHandle,
        config: &ParapperConfig,
    ) -> Result<RecognitionStatus, RecognitionStartError> {
        if !config.streaming_recognition.enabled
            || config.streaming_recognition.mode != DeveloperConnectionMode::WebSocket
        {
            return Err(RecognitionStartError::AudioInput(anyhow::anyhow!(
                "WebSocket input is selected but external recognition input is disabled"
            )));
        }
        let mut server_slot = self.streaming_recognition_server.lock().await;
        if server_slot.is_some() {
            return Ok(self.get_recognition_status().await);
        }
        if self.recognition_session.lock().await.owner().is_some() {
            return Err(RecognitionStartError::Busy);
        }
        let bind_addr = config
            .streaming_recognition
            .validated_bind_addr()
            .map_err(RecognitionStartError::AudioInput)?;
        let output_mode = match config.streaming_recognition.output_mode {
            StreamingRecognitionOutputMode::WebSocketOnly => NetworkOutputMode::WebSocketOnly,
            StreamingRecognitionOutputMode::WebSocketAndDesktop => {
                NetworkOutputMode::WebSocketAndDesktop
            }
        };
        let server = StreamingRecognitionServer::start(
            handle,
            StreamingRecognitionServerConfig {
                bind_addr,
                api_key: config.streaming_recognition.api_key.clone(),
                output_mode,
            },
        )
        .map_err(RecognitionStartError::AudioInput)?;
        log::info!("Streaming recognition listening on {}", server.local_addr());
        *server_slot = Some(server);
        Ok(self
            .set_recognition_status(RecognitionStatus::WaitingForClient)
            .await)
    }
}

fn preserve_running_vad_interval(previous: &ParapperConfig, next: &mut ParapperConfig) {
    next.segmentation.vad_interval_ms = previous.segmentation.vad_interval_ms;
}

fn preserve_running_speech_model_mappings(
    previous: &[SpeechMapping],
    next: &[SpeechMapping],
) -> Vec<SpeechMapping> {
    previous
        .iter()
        .map(|previous_mapping| {
            let Some(next_mapping) = next
                .iter()
                .find(|mapping| mapping.id == previous_mapping.id)
            else {
                return previous_mapping.clone();
            };
            let mut mapping = previous_mapping.clone();
            mapping.talker.clone_from(&next_mapping.talker);
            mapping
                .local_tts_language
                .clone_from(&next_mapping.local_tts_language);
            mapping.local_tts_speaker_id = next_mapping.local_tts_speaker_id;
            mapping
                .output_device_id
                .clone_from(&next_mapping.output_device_id);
            mapping
                .output_device_host
                .clone_from(&next_mapping.output_device_host);
            mapping
                .output_device_name
                .clone_from(&next_mapping.output_device_name);
            mapping.muted = next_mapping.muted;
            mapping.volume = next_mapping.volume;
            mapping
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{RecognitionSessionOwner, RecognitionSessionSlot, preserve_running_vad_interval};
    use crate::config::{ParapperConfig, TurnDetector};

    #[test]
    fn running_vad_interval_is_preserved_but_timing_settings_can_update() {
        let previous = parapper_config! {
            vad_threshold: 0.7,
            vad_interval_ms: 32,
            segment_start_speech_ms: 128,
            turn_detector: TurnDetector::Namo,
            interim_result_enabled: false,
            interim_result_silence_ms: 640,
            turn_check_silence_ms: 960,
            namo_turn_confidence_threshold: 0.65,
            namo_context_max_tokens: 128,
            turn_rerecognize_full_on_complete: true,
            ..ParapperConfig::default()
        };
        let mut next = parapper_config! {
            vad_threshold: 0.1,
            vad_interval_ms: 999,
            segment_start_speech_ms: 32,
            turn_detector: TurnDetector::Simple,
            interim_result_enabled: true,
            interim_result_silence_ms: 32,
            turn_check_silence_ms: 32,
            namo_turn_confidence_threshold: 0.95,
            namo_context_max_tokens: 512,
            turn_rerecognize_full_on_complete: false,
            input_volume_db: 6.0,
            ..ParapperConfig::default()
        };

        preserve_running_vad_interval(&previous, &mut next);

        assert_eq!(
            next.segmentation.vad_interval_ms,
            previous.segmentation.vad_interval_ms
        );
        assert_f32_close(next.segmentation.vad_threshold, 0.1);
        assert_eq!(next.segmentation.segment_start_speech_ms, 32);
        assert_eq!(next.turn.detector, TurnDetector::Simple);
        assert!(next.turn.interim_result_enabled);
        assert_eq!(next.turn.interim_result_silence_ms, 32);
        assert_eq!(next.turn.check_silence_ms, 32);
        assert_f32_close(next.turn.namo_confidence_threshold, 0.95);
        assert_eq!(next.turn.namo_context_max_tokens, 512);
        assert!(!next.turn.rerecognize_full_on_complete);
        assert_f32_close(next.input.volume_db, 6.0);
    }

    #[test]
    fn recognition_slot_rejects_desktop_and_websocket_double_ownership_symmetrically() {
        let cases = [
            (
                RecognitionSessionOwner::Desktop,
                RecognitionSessionOwner::WebSocket {
                    session_id: "network".to_string(),
                },
            ),
            (
                RecognitionSessionOwner::WebSocket {
                    session_id: "network".to_string(),
                },
                RecognitionSessionOwner::Desktop,
            ),
        ];

        for (first, second) in cases {
            let mut slot = RecognitionSessionSlot::default();
            slot.insert(first.clone(), 1_u8).unwrap();

            let active = slot.insert(second, 2_u8).unwrap_err();

            assert_eq!(active, first);
            assert_eq!(slot.owner(), Some(&first));
        }
    }

    #[test]
    fn recognition_slot_releases_only_the_matching_owner() {
        let network = RecognitionSessionOwner::WebSocket {
            session_id: "network".to_string(),
        };
        let mut slot = RecognitionSessionSlot::default();
        slot.insert(network.clone(), 7_u8).unwrap();

        assert_eq!(slot.take(&RecognitionSessionOwner::Desktop), None);
        assert_eq!(slot.owner(), Some(&network));
        assert_eq!(slot.take(&network), Some(7));
        assert_eq!(slot.owner(), None);
    }

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "actual={actual}, expected={expected}"
        );
    }
}
