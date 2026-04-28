use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrPrecision {
    Int8,
    Int8Float32,
    Float32,
}

impl Default for AsrPrecision {
    fn default() -> Self {
        Self::Int8Float32
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsrModel {
    #[serde(rename = "reazonspeech_k2_v2")]
    ReazonSpeechK2V2,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v2_int8")]
    NemoParakeetTdt0_6BV2Int8,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v3_int8")]
    NemoParakeetTdt0_6BV3Int8,
}

impl AsrModel {
    pub fn language(self) -> AsrLanguage {
        match self {
            Self::ReazonSpeechK2V2 => AsrLanguage::Japanese,
            Self::NemoParakeetTdt0_6BV2Int8 => AsrLanguage::English,
            Self::NemoParakeetTdt0_6BV3Int8 => AsrLanguage::EuropeanMultilingual,
        }
    }

    pub fn default_for_language(language: AsrLanguage) -> Self {
        match language {
            AsrLanguage::Japanese => Self::ReazonSpeechK2V2,
            AsrLanguage::English => Self::NemoParakeetTdt0_6BV2Int8,
            AsrLanguage::EuropeanMultilingual => Self::NemoParakeetTdt0_6BV3Int8,
        }
    }

    pub fn supports_precision(self, precision: AsrPrecision) -> bool {
        match self {
            Self::ReazonSpeechK2V2 => true,
            Self::NemoParakeetTdt0_6BV2Int8 | Self::NemoParakeetTdt0_6BV3Int8 => {
                precision == AsrPrecision::Int8
            }
        }
    }

    pub fn default_precision(self) -> AsrPrecision {
        match self {
            Self::ReazonSpeechK2V2 => AsrPrecision::Int8Float32,
            Self::NemoParakeetTdt0_6BV2Int8 | Self::NemoParakeetTdt0_6BV3Int8 => AsrPrecision::Int8,
        }
    }
}

impl Default for AsrModel {
    fn default() -> Self {
        Self::ReazonSpeechK2V2
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrLanguage {
    Japanese,
    English,
    EuropeanMultilingual,
}

impl Default for AsrLanguage {
    fn default() -> Self {
        Self::Japanese
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ParapperConfig {
    pub neo_http_enabled: bool,
    pub neo_http_port: u16,
    pub input_device_id: Option<String>,
    pub input_device_host: Option<String>,
    pub input_device_name: Option<String>,
    pub asr_language: AsrLanguage,
    pub asr_model: AsrModel,
    pub asr_precision: AsrPrecision,
    pub asr_num_threads: i32,
    pub model_dir: Option<String>,
    pub vad_threshold: f32,
    pub vad_interval_ms: u32,
    pub pause_threshold: u32,
    pub phrase_threshold: u32,
    pub vrc_osc_micmute: bool,
    pub debug_asr_audio_playback: bool,
    pub recognition_log_limit: Option<usize>,
    pub debug_audio_log_limit: Option<usize>,
}

impl Default for ParapperConfig {
    fn default() -> Self {
        Self {
            neo_http_enabled: true,
            neo_http_port: 15520,
            input_device_id: None,
            input_device_host: None,
            input_device_name: None,
            asr_language: AsrLanguage::Japanese,
            asr_model: AsrModel::ReazonSpeechK2V2,
            asr_precision: AsrPrecision::Int8Float32,
            asr_num_threads: 4,
            model_dir: None,
            vad_threshold: 0.5,
            vad_interval_ms: 32,
            pause_threshold: 10,
            phrase_threshold: 10,
            vrc_osc_micmute: false,
            debug_asr_audio_playback: false,
            recognition_log_limit: Some(500),
            debug_audio_log_limit: Some(20),
        }
        .normalized_for_platform()
    }
}

impl ParapperConfig {
    pub fn neo_http_supported() -> bool {
        !cfg!(target_os = "macos")
    }

    pub fn vrc_osc_supported() -> bool {
        !cfg!(target_os = "macos")
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        serde_json::from_str::<Self>(&content)
            .map(Self::normalized)
            .with_context(|| format!("Failed to parse config: {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config dir: {}", parent.display()))?;
        }

        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)
            .with_context(|| format!("Failed to write config: {}", path.display()))
    }

    pub fn normalized(mut self) -> Self {
        if self.vad_interval_ms != 32 {
            let previous_interval_ms = self.vad_interval_ms.max(1);
            self.pause_threshold = chunks_for_millis(
                self.pause_threshold.saturating_mul(previous_interval_ms),
                32,
            );
            self.phrase_threshold = chunks_for_millis(
                self.phrase_threshold.saturating_mul(previous_interval_ms),
                32,
            );
            self.vad_interval_ms = 32;
        }
        if self.asr_model.language() != self.asr_language {
            self.asr_model = AsrModel::default_for_language(self.asr_language);
        }
        if !self.asr_model.supports_precision(self.asr_precision) {
            self.asr_precision = self.asr_model.default_precision();
        }
        self.asr_num_threads = self.asr_num_threads.max(0);
        self = self.normalized_for_platform();
        self
    }

    fn normalized_for_platform(mut self) -> Self {
        if !Self::neo_http_supported() {
            self.neo_http_enabled = false;
        }
        if !Self::vrc_osc_supported() {
            self.vrc_osc_micmute = false;
        }
        self
    }
}

fn chunks_for_millis(threshold_ms: u32, interval_ms: u32) -> u32 {
    threshold_ms.div_ceil(interval_ms).max(1)
}

#[cfg(test)]
mod tests {
    use super::{AsrLanguage, AsrModel, AsrPrecision, ParapperConfig};

    #[test]
    fn default_config_uses_neo_http_port() {
        assert_eq!(ParapperConfig::default().neo_http_port, 15520);
    }

    #[test]
    fn default_config_sends_text_to_neo() {
        #[cfg(not(target_os = "macos"))]
        assert!(ParapperConfig::default().neo_http_enabled);
        #[cfg(target_os = "macos")]
        assert!(!ParapperConfig::default().neo_http_enabled);
    }

    #[test]
    fn legacy_config_defaults_to_japanese_asr() {
        let config = serde_json::from_str::<ParapperConfig>(r#"{"neo_http_port":15520}"#)
            .unwrap()
            .normalized();

        assert_eq!(config.asr_language, AsrLanguage::Japanese);
        assert_eq!(config.asr_model, AsrModel::ReazonSpeechK2V2);
    }

    #[test]
    fn legacy_english_config_defaults_to_parakeet_v2() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{"neo_http_port":15520,"asr_language":"english"}"#,
        )
        .unwrap()
        .normalized();

        assert_eq!(config.asr_model, AsrModel::NemoParakeetTdt0_6BV2Int8);
    }

    #[test]
    fn unsupported_model_precision_is_normalized() {
        let config = ParapperConfig {
            asr_language: AsrLanguage::English,
            asr_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            asr_precision: AsrPrecision::Float32,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.asr_precision, AsrPrecision::Int8);
    }

    #[test]
    fn european_multilingual_defaults_to_parakeet_v3() {
        let config = ParapperConfig {
            asr_language: AsrLanguage::EuropeanMultilingual,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.asr_model, AsrModel::NemoParakeetTdt0_6BV3Int8);
    }

    #[test]
    fn negative_asr_num_threads_is_normalized_to_auto() {
        let config = ParapperConfig {
            asr_num_threads: -1,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.asr_num_threads, 0);
    }

    #[test]
    fn legacy_vad_interval_is_normalized_to_v6_chunk_size() {
        let config = ParapperConfig {
            vad_interval_ms: 100,
            pause_threshold: 3,
            phrase_threshold: 3,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.vad_interval_ms, 32);
        assert_eq!(config.pause_threshold, 10);
        assert_eq!(config.phrase_threshold, 10);
    }
}
