use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result};
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::{
    audio::RunningAudioInput,
    config::{ParapperConfig, SpeechMapping},
    config_preset::{ConfigPreset, delete_config_preset, load_config_presets, save_config_preset},
    model::{ModelStatus, any_model_installed_in, model_status_from_root, models_root},
    recognition::RecognitionStatus,
    synthesis::prewarm_local_tts_engines,
};

pub struct AppState {
    config_path: PathBuf,
    config_presets_path: PathBuf,
    models_root: PathBuf,
    config: Mutex<ParapperConfig>,
    runtime_config: Arc<RwLock<ParapperConfig>>,
    recognition_status: Mutex<RecognitionStatus>,
    audio_input: Mutex<Option<RunningAudioInput>>,
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
            runtime_config: Arc::new(RwLock::new(config.clone())),
            config: Mutex::new(config),
            recognition_status: Mutex::new(RecognitionStatus::Idle),
            audio_input: Mutex::new(None),
        })
    }

    pub async fn get_config(&self) -> ParapperConfig {
        self.config.lock().await.clone()
    }

    pub async fn set_config(&self, config: ParapperConfig) -> Result<ParapperConfig> {
        let mut config = config.normalized();
        if *self.recognition_status.lock().await == RecognitionStatus::Listening {
            let previous = self.runtime_config_snapshot()?;
            config.speech_mappings = preserve_running_speech_model_mappings(
                &previous.speech_mappings,
                &config.speech_mappings,
            );
        }
        config.save(&self.config_path)?;
        if let Ok(mut runtime_config) = self.runtime_config.write() {
            *runtime_config = config.clone();
        }
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
        self.runtime_config
            .read()
            .map(|config| config.clone())
            .map_err(|_| anyhow::anyhow!("runtime config lock is poisoned"))
    }

    pub fn has_any_model_installed(&self) -> Result<bool> {
        if !self.models_root.try_exists()? {
            return Ok(false);
        }
        Ok(any_model_installed_in(&self.models_root))
    }

    pub async fn get_recognition_status(&self) -> RecognitionStatus {
        *self.recognition_status.lock().await
    }

    pub async fn set_recognition_status(&self, status: RecognitionStatus) -> RecognitionStatus {
        *self.recognition_status.lock().await = status;
        status
    }

    pub async fn start_audio_input(&self, handle: AppHandle) -> Result<RecognitionStatus> {
        let mut audio_input = self.audio_input.lock().await;
        if audio_input.is_some() {
            return Ok(self
                .set_recognition_status(RecognitionStatus::Listening)
                .await);
        }

        let config = self.get_config().await;
        prewarm_local_tts_engines(&handle, &config);
        let running_audio_input =
            RunningAudioInput::start(handle, &config, self.runtime_config.clone())?;
        *audio_input = Some(running_audio_input);
        drop(audio_input);

        Ok(self
            .set_recognition_status(RecognitionStatus::Listening)
            .await)
    }

    pub async fn stop_audio_input(&self) -> RecognitionStatus {
        let mut audio_input = self.audio_input.lock().await;
        if let Some(running_audio_input) = audio_input.take() {
            running_audio_input.stop();
        }
        drop(audio_input);

        self.set_recognition_status(RecognitionStatus::Stopped)
            .await
    }
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
