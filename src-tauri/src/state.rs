use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result};
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::{
    audio::RunningAudioInput,
    config::ParapperConfig,
    model::{ModelStatus, any_model_installed_in, model_status_from_root, models_root},
    recognition::RecognitionStatus,
};

pub struct AppState {
    config_path: PathBuf,
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
        let models_root = models_root(handle)?;
        let config = ParapperConfig::load(&config_path)?;

        Ok(Self {
            config_path,
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
        let config = config.normalized();
        config.save(&self.config_path)?;
        if let Ok(mut runtime_config) = self.runtime_config.write() {
            *runtime_config = config.clone();
        }
        *self.config.lock().await = config.clone();
        Ok(config)
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
