use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;

use crate::{
    config::{ParapperConfig, SpeechMapping},
    config_preset::{ConfigPreset, delete_config_preset, load_config_presets, save_config_preset},
    model::{ModelStatus, any_model_installed_in, model_status_from_root, models_root},
    recognition::{
        RecognitionStartError, RecognitionStatus, RunningRecognitionInput, RuntimeConfigState,
    },
    synthesis::prewarm_local_tts_engines,
};

pub struct AppState {
    config_path: PathBuf,
    config_presets_path: PathBuf,
    models_root: PathBuf,
    config: Mutex<ParapperConfig>,
    runtime_config: Arc<RuntimeConfigState>,
    recognition_status: Mutex<RecognitionStatus>,
    recognition_input: Mutex<Option<RunningRecognitionInput>>,
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
            recognition_input: Mutex::new(None),
        })
    }

    pub async fn get_config(&self) -> ParapperConfig {
        self.config.lock().await.clone()
    }

    pub async fn set_config(&self, config: ParapperConfig) -> Result<ParapperConfig> {
        let mut config = config.normalized();
        if *self.recognition_status.lock().await == RecognitionStatus::Listening {
            let previous = self.runtime_config_snapshot()?;
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
        let mut recognition_input = self.recognition_input.lock().await;
        if recognition_input.is_some() {
            return Ok(self
                .set_recognition_status(RecognitionStatus::Listening)
                .await);
        }

        let config = self.get_config().await;
        prewarm_local_tts_engines(&handle, &config);
        let running_recognition_input =
            match RunningRecognitionInput::start(handle, &config, self.runtime_config.clone()) {
                Ok(input) => input,
                Err(err) => {
                    self.set_recognition_status(RecognitionStatus::Error).await;
                    return Err(err);
                }
            };
        *recognition_input = Some(running_recognition_input);
        drop(recognition_input);

        Ok(self
            .set_recognition_status(RecognitionStatus::Listening)
            .await)
    }

    pub async fn stop_audio_input(&self) -> RecognitionStatus {
        let running_recognition_input = {
            let mut recognition_input = self.recognition_input.lock().await;
            recognition_input.take()
        };

        if let Some(running_recognition_input) = running_recognition_input
            && let Err(err) =
                tauri::async_runtime::spawn_blocking(move || running_recognition_input.stop()).await
        {
            log::warn!("Recognition input stop task failed: {err}");
        }

        self.set_recognition_status(RecognitionStatus::Stopped)
            .await
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
    use super::preserve_running_vad_interval;
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

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "actual={actual}, expected={expected}"
        );
    }
}
