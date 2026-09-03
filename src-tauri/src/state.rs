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
        ConfigPreset, DeveloperConnectionMode, InputSourceKind, ParapperConfig, SpeechBackend,
        StreamingRecognitionOutputMode, TranslationBackend, delete_config_preset,
        load_config_presets, save_config_preset,
    },
    model::{ModelStatus, any_model_installed_in, model_status_from_root, models_root},
    recognition::{
        NetworkOutputMode, RecognitionShutdownResult, RecognitionStartError, RecognitionStatus,
        RecognitionStreamEvent, RunningInputSource, RunningRecognitionInput, RuntimeConfigState,
        StreamingRecognitionServer, StreamingRecognitionServerConfig, TurnOutputSink,
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
        config.validate()?;
        let config = config.normalized();
        if matches!(
            *self.recognition_status.lock().await,
            RecognitionStatus::WaitingForClient
                | RecognitionStatus::Listening
                | RecognitionStatus::Draining
        ) {
            let previous = self.runtime_config_snapshot()?;
            validate_running_config_change(&previous, &config)?;
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
        config
            .validate()
            .context("recognition input config is invalid")
            .map_err(RecognitionStartError::AudioInput)?;
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
        config
            .validate()
            .context("recognition input config is invalid")
            .map_err(RecognitionStartError::AudioInput)?;
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

fn validate_running_config_change(previous: &ParapperConfig, next: &ParapperConfig) -> Result<()> {
    if running_config_with_runtime_parameters(previous, next) == *next {
        return Ok(());
    }
    anyhow::bail!(
        "recognition must be stopped before changing session, resource, or pipeline settings"
    );
}

fn running_config_with_runtime_parameters(
    previous: &ParapperConfig,
    next: &ParapperConfig,
) -> ParapperConfig {
    let mut allowed = previous.clone();
    allowed.neo.http_enabled = next.neo.http_enabled;
    allowed.neo.http_port = next.neo.http_port;
    allowed.input.volume_db = next.input.volume_db;
    allowed
        .streaming_recognition
        .http_url
        .clone_from(&next.streaming_recognition.http_url);
    allowed.asr.normalize_input_audio = next.asr.normalize_input_audio;
    allowed.translation.enabled = next.translation.enabled;
    allowed.translation.ync_plugin_port = next.translation.ync_plugin_port;
    allowed.translation.send_timing = next.translation.send_timing;
    if !translation_mapping_change_requires_restart(previous, next) {
        allowed
            .translation
            .mappings
            .clone_from(&next.translation.mappings);
    }
    if !speech_mapping_change_requires_restart(previous, next) {
        allowed.speech.mappings.clone_from(&next.speech.mappings);
    }
    if previous.stt_profiles.is_empty() && next.stt_profiles.is_empty() {
        // Legacy single-input mode keeps its existing runtime-tunable fields.
        allowed.input.volume_db = next.input.volume_db;
        allowed.input.muted = next.input.muted;
        allowed.segmentation.vad_threshold = next.segmentation.vad_threshold;
        allowed.segmentation.segment_start_speech_ms = next.segmentation.segment_start_speech_ms;
        allowed.turn.interim_result_silence_ms = next.turn.interim_result_silence_ms;
        allowed.turn.check_silence_ms = next.turn.check_silence_ms;
        allowed.turn.namo_confidence_threshold = next.turn.namo_confidence_threshold;
        allowed.turn.namo_context_max_tokens = next.turn.namo_context_max_tokens;
        allowed.turn.rerecognize_full_on_complete = next.turn.rerecognize_full_on_complete;
    } else if stt_profile_runtime_updates_only(previous, next) {
        // In independent-profile mode, only each lane's live gain/mute may
        // change while running.  Device/channel and all pipeline settings are
        // startup-owned, even though the flattened legacy fields above remain
        // present for compatibility with older config files.
        allowed.stt_profiles = previous
            .stt_profiles
            .iter()
            .zip(&next.stt_profiles)
            .map(|(previous, next)| {
                let mut profile = previous.clone();
                profile.input.volume_percent = next.input.volume_percent;
                profile.input.muted = next.input.muted;
                profile
            })
            .collect();
    }
    allowed.vrc.clone_from(&next.vrc);
    allowed.debug.clone_from(&next.debug);
    allowed
}

fn stt_profile_runtime_updates_only(previous: &ParapperConfig, next: &ParapperConfig) -> bool {
    if previous.stt_profiles.len() != next.stt_profiles.len() {
        return false;
    }

    previous
        .stt_profiles
        .iter()
        .zip(&next.stt_profiles)
        .all(|(previous, next)| {
            previous.id == next.id
                && previous.name == next.name
                && previous.enabled == next.enabled
                && previous.display_color == next.display_color
                && previous.input.device_host == next.input.device_host
                && previous.input.device_id == next.input.device_id
                && previous.input.device_name == next.input.device_name
                && previous.input.channel_index == next.input.channel_index
                && previous.noise_cancellation == next.noise_cancellation
                && previous.segmentation == next.segmentation
                && previous.turn == next.turn
                && previous.asr == next.asr
                && previous.delivery_profile_id == next.delivery_profile_id
        })
}

fn translation_mapping_change_requires_restart(
    previous: &ParapperConfig,
    next: &ParapperConfig,
) -> bool {
    next.translation.mappings.iter().any(|next_mapping| {
        let previous_mapping = previous
            .translation
            .mappings
            .iter()
            .find(|mapping| mapping.id == next_mapping.id);
        match (previous_mapping, next_mapping.backend) {
            (Some(previous_mapping), TranslationBackend::Local) => {
                previous_mapping.backend != TranslationBackend::Local
                    || previous_mapping.local_model != next_mapping.local_model
            }
            (Some(previous_mapping), TranslationBackend::Ync) => {
                previous_mapping.backend == TranslationBackend::Local
            }
            (None, TranslationBackend::Local) => true,
            (None, TranslationBackend::Ync) => false,
        }
    })
}

fn speech_mapping_change_requires_restart(
    previous: &ParapperConfig,
    next: &ParapperConfig,
) -> bool {
    next.speech.mappings.iter().any(|next_mapping| {
        let previous_mapping = previous
            .speech
            .mappings
            .iter()
            .find(|mapping| mapping.id == next_mapping.id);
        match (previous_mapping, next_mapping.backend) {
            (Some(previous_mapping), SpeechBackend::LocalTts) => {
                previous_mapping.backend != SpeechBackend::LocalTts
                    || previous_mapping.local_tts_voice != next_mapping.local_tts_voice
            }
            (Some(previous_mapping), SpeechBackend::Ync) => {
                previous_mapping.backend == SpeechBackend::LocalTts
            }
            (None, SpeechBackend::LocalTts) => true,
            (None, SpeechBackend::Ync) => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, sync::Arc};

    use tokio::sync::Mutex;
    use uuid::Uuid;

    use super::{
        AppState, RecognitionSessionOwner, RecognitionSessionSlot, TranslationHttpListenerStatus,
    };
    use crate::{
        config::{
            AsrMode, AsrModel, AsrPrecision, InputSourceKind, LocalTranslationModel, LocalTtsVoice,
            NeoSendTiming, ParapperConfig, SpeechBackend, SpeechMapping, SpeechSourceKind,
            SttProfileConfig, SttProfileDisplayColor, SttProfileInputConfig, TranslationBackend,
            TranslationLanguage, TranslationMapping, TurnDetector,
        },
        recognition::{RecognitionStatus, RuntimeConfigState},
    };

    fn translation_mapping(backend: TranslationBackend) -> TranslationMapping {
        TranslationMapping {
            id: "translation-route".to_string(),
            source_asr_model: None,
            backend,
            local_model: LocalTranslationModel::default(),
            source_lang: TranslationLanguage::Ja,
            target_lang: TranslationLanguage::En,
        }
    }

    fn speech_mapping(backend: SpeechBackend) -> SpeechMapping {
        SpeechMapping {
            id: "speech-route".to_string(),
            source_kind: SpeechSourceKind::Recognition,
            source_asr_model: None,
            target_lang: None,
            backend,
            talker: "voice".to_string(),
            local_tts_voice: (backend == SpeechBackend::LocalTts)
                .then_some(LocalTtsVoice::Supertonic2Onnx),
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_id: None,
            output_device_host: None,
            output_device_name: None,
            muted: false,
            volume: 0.0,
        }
    }

    fn test_app_state(config: &ParapperConfig, status: RecognitionStatus) -> AppState {
        let root = std::env::temp_dir().join(format!("parapper-state-test-{}", Uuid::new_v4()));
        let config_path = root.join("config.json");
        config.save(&config_path).unwrap();
        AppState {
            config_path,
            config_presets_path: root.join("config-presets.json"),
            models_root: root.join("models"),
            config: Mutex::new(config.clone()),
            runtime_config: Arc::new(RuntimeConfigState::new(config.clone())),
            recognition_status: Mutex::new(status),
            recognition_session: Mutex::new(RecognitionSessionSlot::default()),
            streaming_recognition_server: Mutex::new(None),
            translation_http_listener: std::sync::Mutex::new(None),
            translation_http_listener_status: std::sync::Mutex::new(
                TranslationHttpListenerStatus::default(),
            ),
        }
    }

    fn persisted_config(path: &PathBuf) -> ParapperConfig {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    fn stt_profile(id: &str, device: &str, channel_index: u16) -> SttProfileConfig {
        let defaults = ParapperConfig::default();
        SttProfileConfig {
            id: id.to_owned(),
            name: id.to_owned(),
            enabled: true,
            neo_http_enabled: true,
            developer_http_enabled: true,
            display_color: SttProfileDisplayColor::Green,
            input: SttProfileInputConfig {
                device_host: Some("wasapi".to_owned()),
                device_id: Some(device.to_owned()),
                device_name: Some(device.to_owned()),
                channel_index,
                volume_percent: 100,
                muted: false,
            },
            noise_cancellation: defaults.noise_cancellation,
            segmentation: defaults.segmentation,
            turn: defaults.turn,
            asr: defaults.asr,
            delivery_profile_id: None,
        }
    }

    fn profile_config() -> ParapperConfig {
        ParapperConfig {
            stt_profiles: vec![
                stt_profile("profile-a", "device-a", 0),
                stt_profile("profile-b", "device-b", 0),
            ],
            ..ParapperConfig::default()
        }
    }

    fn remove_test_state(state: AppState) {
        let root = state
            .config_path
            .parent()
            .expect("test config must have a parent")
            .to_path_buf();
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    type ConfigMutation = fn(&mut ParapperConfig);

    #[tokio::test]
    async fn running_recognition_rejects_each_session_fixed_stt_change_without_partial_save() {
        let cases: [(&str, ConfigMutation); 16] = [
            ("primary ASR model", |config| {
                config.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
            }),
            ("ASR precision", |config| {
                config.asr.precision = AsrPrecision::Float32;
            }),
            ("interim ASR model", |config| {
                config.asr.interim_model = Some(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8);
            }),
            ("ASR thread count", |config| config.asr.num_threads = 2),
            ("multilingual ASR", |config| {
                config.asr.multilingual_enabled = true;
            }),
            ("enabled ASR models", |config| {
                config
                    .asr
                    .enabled_models
                    .push(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8);
            }),
            ("Turn Detector", |config| {
                config.turn.detector = TurnDetector::Namo;
            }),
            ("interim recognition topology", |config| {
                config.turn.interim_result_enabled = false;
            }),
            ("noise cancellation", |config| {
                config.noise_cancellation.enabled = true;
            }),
            ("noise cancellation target", |config| {
                config.noise_cancellation.target =
                    crate::config::NoiseCancellationTarget::VadAndAsr;
            }),
            ("model directory", |config| {
                config.models.dir = Some("other-models".to_string());
            }),
            ("input source", |config| {
                config.input.source_kind = InputSourceKind::WebSocket;
            }),
            ("ASR decoding mode", |config| {
                config.asr.mode = AsrMode::Accurate;
            }),
            ("developer listener", |config| {
                config.streaming_recognition.enabled = true;
            }),
            ("new local translation route", |config| {
                config
                    .translation
                    .mappings
                    .push(translation_mapping(TranslationBackend::Local));
            }),
            ("new local speech route", |config| {
                config
                    .speech
                    .mappings
                    .push(speech_mapping(SpeechBackend::LocalTts));
            }),
        ];

        for status in [
            RecognitionStatus::WaitingForClient,
            RecognitionStatus::Listening,
            RecognitionStatus::Draining,
        ] {
            for (name, mutate) in cases {
                let previous = ParapperConfig::default();
                let state = test_app_state(&previous, status);
                let config_path = state.config_path.clone();
                let mut next = previous.clone();
                mutate(&mut next);

                let error = state.set_config(next).await.expect_err(name).to_string();

                assert!(
                    error.contains("recognition must be stopped"),
                    "{name} returned an unclear error in {status:?}: {error}"
                );
                assert_eq!(state.get_config().await, previous, "{name} in {status:?}");
                assert_eq!(
                    state.runtime_config_snapshot().unwrap(),
                    previous,
                    "{name} in {status:?}"
                );
                assert_eq!(
                    persisted_config(&config_path),
                    previous,
                    "{name} in {status:?}"
                );
                remove_test_state(state);
            }
        }
    }

    #[tokio::test]
    async fn running_recognition_accepts_one_transaction_of_runtime_parameters() {
        let previous = ParapperConfig::default();
        let state = test_app_state(&previous, RecognitionStatus::Listening);
        let mut next = previous.clone();
        next.input.volume_db = 6.0;
        next.segmentation.vad_threshold = 0.25;
        next.segmentation.segment_start_speech_ms = 160;
        next.turn.interim_result_silence_ms = 192;
        next.turn.check_silence_ms = 640;
        next.turn.namo_confidence_threshold = 0.65;
        next.turn.namo_context_max_tokens = 128;
        next.turn.rerecognize_full_on_complete = true;
        next.asr.normalize_input_audio = false;
        next.neo.http_enabled = true;
        next.neo.http_port = 15521;
        next.streaming_recognition.http_url = "http://127.0.0.1:15523/events".to_string();
        next.translation.enabled = true;
        next.translation.ync_plugin_port = 8081;
        next.translation.send_timing = NeoSendTiming::Interim;
        next.translation
            .mappings
            .push(translation_mapping(TranslationBackend::Ync));
        next.speech
            .mappings
            .push(speech_mapping(SpeechBackend::Ync));

        let saved = state.set_config(next.clone()).await.unwrap();

        assert_eq!(saved, next.normalized());
        remove_test_state(state);
    }

    #[tokio::test]
    async fn running_profile_recognition_allows_only_profile_gain_and_mute_updates() {
        type ProfileMutation = fn(&mut ParapperConfig);

        let allowed: [(&str, ProfileMutation); 3] = [
            ("profile volume", |config| {
                config.stt_profiles[0].input.volume_percent = 35;
            }),
            ("profile mute", |config| {
                config.stt_profiles[1].input.muted = true;
            }),
            ("profile volume and mute", |config| {
                config.stt_profiles[0].input.volume_percent = 5;
                config.stt_profiles[1].input.muted = true;
            }),
        ];
        let rejected: [(&str, ProfileMutation); 9] = [
            ("profile device", |config| {
                config.stt_profiles[0].input.device_id = Some("other-device".to_owned());
            }),
            ("profile channel", |config| {
                config.stt_profiles[0].input.channel_index = 1;
            }),
            ("profile display color", |config| {
                config.stt_profiles[0].display_color = SttProfileDisplayColor::Blue;
            }),
            ("profile noise cancellation", |config| {
                config.stt_profiles[0].noise_cancellation.enabled = true;
            }),
            ("profile VAD", |config| {
                config.stt_profiles[0].segmentation.vad_threshold = 0.42;
            }),
            ("profile ASR", |config| {
                config.stt_profiles[0].asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
            }),
            ("profile set", |config| {
                config.stt_profiles.pop();
            }),
            ("profile add", |config| {
                config
                    .stt_profiles
                    .push(stt_profile("profile-c", "device-c", 0));
            }),
            ("profile reorder", |config| {
                config.stt_profiles.reverse();
            }),
        ];

        for (name, mutate) in allowed {
            let previous = profile_config();
            let state = test_app_state(&previous, RecognitionStatus::Listening);
            let mut next = previous.clone();
            mutate(&mut next);

            state
                .set_config(next.clone())
                .await
                .unwrap_or_else(|error| panic!("{name} should be allowed: {error}"));
            assert_eq!(state.get_config().await, next.normalized(), "{name}");
            remove_test_state(state);
        }

        for (name, mutate) in rejected {
            let previous = profile_config();
            let state = test_app_state(&previous, RecognitionStatus::Listening);
            let mut next = previous.clone();
            mutate(&mut next);

            let error = state.set_config(next).await.expect_err(name).to_string();
            assert!(
                error.contains("recognition must be stopped"),
                "{name}: {error}"
            );
            assert_eq!(state.get_config().await, previous, "{name}");
            remove_test_state(state);
        }
    }

    #[tokio::test]
    async fn stopped_recognition_accepts_session_fixed_config_for_the_next_start() {
        let previous = ParapperConfig::default();
        let state = test_app_state(&previous, RecognitionStatus::Stopped);
        let mut next = previous;
        next.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        next.turn.detector = TurnDetector::Namo;

        let saved = state.set_config(next.clone()).await.unwrap();

        assert_eq!(saved, next.normalized());
        assert_eq!(state.get_config().await, saved);
        assert_eq!(state.runtime_config_snapshot().unwrap(), saved);
        remove_test_state(state);
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
}
