use std::{
    fs,
    net::{IpAddr, SocketAddr},
    path::Path,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{
    AsrLanguage, AsrModel, AsrPrecision, LocalTranslationModel, LocalTtsVoice, NeoSendTiming,
    NoiseCancellationModel, SpeechBackend, SpeechMapping, TranslationMapping, TurnDetector,
};

#[cfg(test)]
use super::{
    AsrModelCapability, AsrModelImplementation, SpeechSourceKind, TranslationBackend,
    TranslationLanguage, TurnDetectorClass, TurnDetectorModel,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ParapperConfig {
    #[serde(flatten)]
    pub neo: NeoConfig,
    #[serde(flatten)]
    pub input: InputConfig,
    #[serde(flatten)]
    pub streaming_recognition: StreamingRecognitionConfig,
    #[serde(flatten)]
    pub asr: AsrConfig,
    #[serde(flatten)]
    pub translation: TranslationConfig,
    #[serde(flatten)]
    pub speech: SpeechConfig,
    #[serde(flatten)]
    pub models: ModelStorageConfig,
    #[serde(flatten)]
    pub segmentation: SegmentationConfig,
    #[serde(flatten)]
    pub turn: TurnConfig,
    #[serde(flatten)]
    pub noise_cancellation: NoiseCancellationConfig,
    #[serde(flatten)]
    pub vrc: VrcConfig,
    #[serde(flatten)]
    pub debug: DebugConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NeoConfig {
    #[serde(rename = "neo_http_enabled")]
    pub http_enabled: bool,
    #[serde(rename = "neo_http_port")]
    pub http_port: u16,
    #[serde(rename = "neo_send_timing", skip_serializing)]
    pub send_timing: NeoSendTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InputConfig {
    #[serde(rename = "input_source_kind")]
    pub source_kind: InputSourceKind,
    #[serde(rename = "input_device_id")]
    pub device_id: Option<String>,
    #[serde(rename = "input_device_host")]
    pub device_host: Option<String>,
    #[serde(rename = "input_device_name")]
    pub device_name: Option<String>,
    #[serde(rename = "input_volume_db")]
    pub volume_db: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum InputSourceKind {
    #[default]
    DesktopAudio,
    WebSocket,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StreamingRecognitionOutputMode {
    #[default]
    WebSocketOnly,
    WebSocketAndDesktop,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DeveloperConnectionMode {
    Http,
    #[default]
    WebSocket,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StreamingRecognitionConfig {
    #[serde(rename = "streaming_recognition_enabled")]
    pub enabled: bool,
    #[serde(rename = "developer_connection_mode")]
    pub mode: DeveloperConnectionMode,
    #[serde(rename = "developer_http_url")]
    pub http_url: String,
    #[serde(rename = "streaming_recognition_bind_address")]
    pub bind_address: String,
    #[serde(rename = "streaming_recognition_port")]
    pub port: u16,
    #[serde(rename = "streaming_recognition_api_key")]
    pub api_key: Option<String>,
    #[serde(rename = "streaming_recognition_output_mode")]
    pub output_mode: StreamingRecognitionOutputMode,
}

impl StreamingRecognitionConfig {
    pub(crate) fn validated_bind_addr(&self) -> Result<SocketAddr> {
        let ip = self
            .bind_address
            .trim()
            .parse::<IpAddr>()
            .with_context(|| {
                format!(
                    "invalid streaming recognition bind address: {}",
                    self.bind_address
                )
            })?;
        if self.port == 0 {
            anyhow::bail!("streaming recognition port must be between 1 and 65535");
        }
        if !ip.is_loopback()
            && self
                .api_key
                .as_deref()
                .is_none_or(|key| key.trim().is_empty())
        {
            anyhow::bail!(
                "an API key is required when streaming recognition accepts LAN connections"
            );
        }
        Ok(SocketAddr::new(ip, self.port))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AsrConfig {
    #[serde(rename = "asr_language")]
    pub language: AsrLanguage,
    #[serde(rename = "asr_model")]
    pub model: AsrModel,
    #[serde(rename = "interim_asr_model")]
    pub interim_model: Option<AsrModel>,
    #[serde(rename = "asr_precision")]
    pub precision: AsrPrecision,
    #[serde(rename = "asr_num_threads")]
    pub num_threads: i32,
    #[serde(rename = "asr_normalize_input_audio")]
    pub normalize_input_audio: bool,
    #[serde(rename = "multilingual_asr_enabled")]
    pub multilingual_enabled: bool,
    #[serde(rename = "enabled_asr_models")]
    pub enabled_models: Vec<AsrModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TranslationConfig {
    #[serde(rename = "translation_enabled")]
    pub enabled: bool,
    #[serde(rename = "ync_plugin_port", alias = "translation_plugin_http_port")]
    pub ync_plugin_port: u16,
    #[serde(rename = "translation_local_server_port")]
    pub local_server_port: u16,
    #[serde(rename = "translation_local_server_model")]
    pub local_server_model: LocalTranslationModel,
    #[serde(rename = "translation_send_timing")]
    pub send_timing: NeoSendTiming,
    #[serde(rename = "translation_mappings")]
    pub mappings: Vec<TranslationMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct SpeechConfig {
    #[serde(rename = "speech_mappings")]
    pub mappings: Vec<SpeechMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct ModelStorageConfig {
    #[serde(rename = "model_dir")]
    pub dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SegmentationConfig {
    #[serde(rename = "vad_threshold")]
    pub vad_threshold: f32,
    #[serde(rename = "vad_interval_ms")]
    pub vad_interval_ms: u32,
    #[serde(rename = "segment_start_speech_ms")]
    pub segment_start_speech_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TurnConfig {
    #[serde(rename = "turn_detector")]
    pub detector: TurnDetector,
    #[serde(rename = "interim_result_enabled")]
    pub interim_result_enabled: bool,
    #[serde(rename = "interim_result_silence_ms")]
    pub interim_result_silence_ms: u32,
    #[serde(rename = "turn_check_silence_ms")]
    pub check_silence_ms: u32,
    #[serde(rename = "namo_turn_confidence_threshold")]
    pub namo_confidence_threshold: f32,
    #[serde(rename = "namo_context_max_tokens")]
    pub namo_context_max_tokens: u32,
    #[serde(rename = "turn_rerecognize_full_on_complete")]
    pub rerecognize_full_on_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct NoiseCancellationConfig {
    #[serde(rename = "noise_cancellation_enabled")]
    pub enabled: bool,
    #[serde(rename = "noise_cancellation_model")]
    pub model: NoiseCancellationModel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct VrcConfig {
    #[serde(rename = "vrc_osc_micmute")]
    pub osc_micmute: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DebugConfig {
    #[serde(rename = "debug_asr_audio_playback")]
    pub asr_audio_playback: bool,
    pub recognition_log_limit: Option<usize>,
    pub debug_audio_log_limit: Option<usize>,
}

impl Default for NeoConfig {
    fn default() -> Self {
        Self {
            http_enabled: false,
            http_port: 15520,
            send_timing: NeoSendTiming::Interim,
        }
    }
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            source_kind: InputSourceKind::DesktopAudio,
            device_id: None,
            device_host: None,
            device_name: None,
            volume_db: 0.0,
        }
    }
}

impl Default for StreamingRecognitionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: DeveloperConnectionMode::WebSocket,
            http_url: "http://127.0.0.1:15522/api/events".to_string(),
            bind_address: "127.0.0.1".to_string(),
            port: 18082,
            api_key: None,
            output_mode: StreamingRecognitionOutputMode::WebSocketOnly,
        }
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            language: AsrLanguage::Japanese,
            model: AsrModel::ReazonSpeechK2V2,
            interim_model: None,
            precision: AsrPrecision::Int8Float32,
            num_threads: 4,
            normalize_input_audio: true,
            multilingual_enabled: false,
            enabled_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
        }
    }
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ync_plugin_port: 8080,
            local_server_port: 18081,
            local_server_model: LocalTranslationModel::default(),
            send_timing: NeoSendTiming::Final,
            mappings: Vec::new(),
        }
    }
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            vad_interval_ms: 32,
            segment_start_speech_ms: 96,
        }
    }
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            detector: TurnDetector::Simple,
            interim_result_enabled: true,
            interim_result_silence_ms: 96,
            check_silence_ms: 320,
            namo_confidence_threshold: 0.8,
            namo_context_max_tokens: 256,
            rerecognize_full_on_complete: false,
        }
    }
}

impl Default for NoiseCancellationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: NoiseCancellationModel::UlUnas,
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            asr_audio_playback: false,
            recognition_log_limit: Some(500),
            debug_audio_log_limit: Some(20),
        }
    }
}

impl Default for ParapperConfig {
    fn default() -> Self {
        Self {
            neo: NeoConfig::default(),
            input: InputConfig::default(),
            streaming_recognition: StreamingRecognitionConfig::default(),
            asr: AsrConfig::default(),
            translation: TranslationConfig::default(),
            speech: SpeechConfig::default(),
            models: ModelStorageConfig::default(),
            segmentation: SegmentationConfig::default(),
            turn: TurnConfig::default(),
            noise_cancellation: NoiseCancellationConfig::default(),
            vrc: VrcConfig::default(),
            debug: DebugConfig::default(),
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

    pub fn required_asr_models(&self) -> Vec<AsrModel> {
        let mut models = if self.asr.multilingual_enabled {
            self.asr.enabled_models.clone()
        } else {
            vec![self.asr.model]
        };
        push_unique_asr_model(&mut models, self.asr.interim_model);
        models
    }

    #[cfg(test)]
    pub(crate) fn completion_asr_model(&self) -> AsrModel {
        self.asr.model
    }

    pub(crate) fn effective_asr_num_threads(&self) -> i32 {
        if self.asr.num_threads > 0 {
            return self.asr.num_threads;
        }
        std::thread::available_parallelism()
            .map(usize::from)
            .ok()
            .and_then(|threads| i32::try_from(threads).ok())
            .filter(|threads| *threads > 0)
            .unwrap_or(1)
    }

    pub fn asr_precision_for(&self, model: AsrModel) -> AsrPrecision {
        if model == self.asr.model {
            self.asr.precision
        } else {
            model.default_precision()
        }
    }

    #[cfg(test)]
    pub fn turn_detector_class(&self) -> TurnDetectorClass {
        self.turn.detector.class()
    }

    #[cfg(test)]
    pub fn turn_detector_model(&self) -> Option<TurnDetectorModel> {
        self.turn_detector_class().model()
    }

    pub fn uses_namo_turn_detector(&self) -> bool {
        self.turn.detector.uses_namo_model()
    }

    pub fn uses_morph_turn_boundary(&self) -> bool {
        self.turn.detector.uses_morph_boundary()
    }

    pub fn confirms_normal_end_with_namo(&self) -> bool {
        self.turn.detector.confirms_normal_end_with_namo()
    }

    pub fn uses_deferred_turn_completion(&self) -> bool {
        self.turn.detector.uses_deferred_turn_completion()
    }

    pub fn can_connect_interim_after_completion(&self) -> bool {
        self.turn.detector.can_connect_interim_after_completion()
    }

    pub fn required_namo_turn_detector_languages(&self) -> Vec<AsrLanguage> {
        if !self.uses_namo_turn_detector() {
            return Vec::new();
        }
        let mut languages = self
            .required_asr_models()
            .into_iter()
            .map(AsrModel::language)
            .collect::<Vec<_>>();
        normalize_asr_languages(&mut languages);
        languages
    }

    pub fn requires_japanese_morph_analyzer(&self) -> bool {
        self.uses_morph_turn_boundary()
            && self
                .required_asr_models()
                .into_iter()
                .any(|model| model.language() == AsrLanguage::Japanese)
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        match serde_json::from_str::<Self>(&content) {
            Ok(config) => Ok(config.normalized()),
            Err(err) => {
                log::warn!(
                    "Failed to parse config: {}. Falling back to default config. Error: {err}",
                    path.display()
                );
                Ok(Self::default())
            }
        }
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
        self.segmentation.vad_interval_ms = 32;
        self.segmentation.segment_start_speech_ms = self
            .segmentation
            .segment_start_speech_ms
            .max(self.segmentation.vad_interval_ms.max(1));
        self.turn.interim_result_silence_ms = self
            .turn
            .interim_result_silence_ms
            .max(self.segmentation.vad_interval_ms.max(1));
        self.turn.check_silence_ms = self
            .turn
            .check_silence_ms
            .max(self.segmentation.vad_interval_ms.max(1));
        if self.turn.interim_result_enabled {
            self.turn.check_silence_ms = self
                .turn
                .check_silence_ms
                .max(self.turn.interim_result_silence_ms);
        } else {
            self.turn.check_silence_ms = self
                .turn
                .check_silence_ms
                .max(self.segmentation.vad_interval_ms.max(1));
        }
        self.turn.namo_confidence_threshold = self.turn.namo_confidence_threshold.clamp(0.0, 1.0);
        self.turn.namo_context_max_tokens = self.turn.namo_context_max_tokens.min(512);
        self.input.volume_db = normalize_input_volume_db(self.input.volume_db);
        self.streaming_recognition.bind_address =
            self.streaming_recognition.bind_address.trim().to_string();
        self.streaming_recognition.http_url =
            self.streaming_recognition.http_url.trim().to_string();
        self.streaming_recognition.api_key = self
            .streaming_recognition
            .api_key
            .take()
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty());
        if !self.asr.model.supports_completion() || self.asr.model.language() != self.asr.language {
            self.asr.model = AsrModel::default_for_language(self.asr.language);
            self.asr.language = self.asr.model.language();
        }
        if !self.asr.model.supports_precision(self.asr.precision) {
            self.asr.precision = self.asr.model.default_precision();
        }
        self.asr.interim_model =
            normalize_interim_asr_model(self.asr.model, self.asr.interim_model);
        normalize_enabled_asr_models(&mut self.asr.enabled_models);
        if !self.asr.enabled_models.contains(&self.asr.model) {
            self.asr.enabled_models.push(self.asr.model);
        }
        self.translation.mappings = normalize_translation_mappings(self.translation.mappings);
        if !self.translation.local_server_model.is_available() {
            self.translation.local_server_model = LocalTranslationModel::default();
        }
        self.speech.mappings = normalize_speech_mappings(self.speech.mappings);
        self.asr.num_threads = self.asr.num_threads.max(0);
        self = self.normalized_for_platform();
        self
    }

    fn normalized_for_platform(mut self) -> Self {
        if !Self::neo_http_supported() {
            self.neo.http_enabled = false;
        }
        if !Self::vrc_osc_supported() {
            self.vrc.osc_micmute = false;
        }
        self
    }
}

fn normalize_translation_mappings(mappings: Vec<TranslationMapping>) -> Vec<TranslationMapping> {
    mappings
        .into_iter()
        .filter_map(|mut mapping| {
            mapping.id = mapping.id.trim().to_string();
            if mapping.id.is_empty() || mapping.source_lang == mapping.target_lang {
                return None;
            }
            if !mapping.local_model.is_available() {
                mapping.local_model = LocalTranslationModel::default();
            }
            Some(mapping)
        })
        .collect()
}

fn normalize_speech_mappings(mappings: Vec<SpeechMapping>) -> Vec<SpeechMapping> {
    mappings
        .into_iter()
        .filter_map(|mut mapping| {
            mapping.id = mapping.id.trim().to_string();
            mapping.talker = mapping.talker.trim().to_string();
            mapping.target_lang = mapping
                .target_lang
                .take()
                .and_then(|target_lang| non_empty_trimmed(&target_lang));
            if mapping.backend == SpeechBackend::LocalTts && mapping.local_tts_voice.is_none() {
                mapping.local_tts_voice = Some(LocalTtsVoice::default());
            }
            mapping.local_tts_language = normalize_local_tts_language(
                mapping.local_tts_voice,
                mapping.local_tts_language.as_deref(),
            );
            mapping.local_tts_speaker_id = normalize_local_tts_speaker_id(
                mapping.local_tts_voice,
                mapping.local_tts_speaker_id,
            );
            mapping.output_device_id = mapping
                .output_device_id
                .take()
                .and_then(|id| non_empty_trimmed(&id));
            mapping.output_device_host = mapping
                .output_device_host
                .take()
                .and_then(|host| non_empty_trimmed(&host));
            mapping.output_device_name = mapping
                .output_device_name
                .take()
                .and_then(|name| non_empty_trimmed(&name));
            if mapping.output_device_id.is_none() || mapping.output_device_host.is_none() {
                mapping.output_device_id = None;
                mapping.output_device_host = None;
                mapping.output_device_name = None;
            }
            mapping.volume = normalize_speech_volume(mapping.volume);
            if mapping.id.is_empty() {
                return None;
            }
            Some(mapping)
        })
        .collect()
}

fn normalize_local_tts_language(
    voice: Option<LocalTtsVoice>,
    language: Option<&str>,
) -> Option<String> {
    let voice = voice?;
    let language = language
        .map(str::trim)
        .filter(|language| !language.is_empty())
        .unwrap_or("en");
    let normalized = language.to_ascii_lowercase();
    if let Some(languages) = voice.supported_language_codes() {
        if languages.contains(&normalized.as_str()) {
            return Some(normalized);
        }
        return Some("en".to_string());
    }
    None
}

fn normalize_local_tts_speaker_id(
    voice: Option<LocalTtsVoice>,
    speaker_id: Option<i32>,
) -> Option<i32> {
    match voice {
        Some(LocalTtsVoice::Supertonic2Onnx | LocalTtsVoice::Supertonic3Onnx) => {
            Some(speaker_id.unwrap_or(0).clamp(0, 9))
        }
        _ => None,
    }
}

fn normalize_speech_volume(volume: f32) -> f32 {
    if volume.is_finite() {
        volume.clamp(-20.0, 20.0)
    } else {
        0.0
    }
}

fn normalize_input_volume_db(volume_db: f32) -> f32 {
    if volume_db.is_finite() {
        volume_db.clamp(-30.0, 30.0)
    } else {
        0.0
    }
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn normalize_enabled_asr_models(models: &mut Vec<AsrModel>) {
    models.retain(|model| model.supports_completion());
    models.sort_by_key(|model| model.sort_key());
    models.dedup();
    if models.is_empty() {
        models.push(AsrModel::ReazonSpeechK2V2);
    }
}

fn push_unique_asr_model(models: &mut Vec<AsrModel>, model: Option<AsrModel>) {
    let Some(model) = model else {
        return;
    };
    if !models.contains(&model) {
        models.push(model);
    }
}

fn normalize_interim_asr_model(
    primary_model: AsrModel,
    model: Option<AsrModel>,
) -> Option<AsrModel> {
    model.filter(|model| model.is_interim_only() && *model != primary_model)
}

fn normalize_asr_languages(languages: &mut Vec<AsrLanguage>) {
    languages.sort_by_key(|language| match language {
        AsrLanguage::Japanese => 0,
        AsrLanguage::English => 1,
        AsrLanguage::EuropeanMultilingual => 2,
        AsrLanguage::Multilingual => 3,
    });
    languages.dedup();
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        AsrLanguage, AsrModel, AsrModelCapability, AsrModelImplementation, AsrPrecision,
        InputSourceKind, LocalTranslationModel, LocalTtsVoice, NeoSendTiming,
        NoiseCancellationModel, ParapperConfig, SpeechBackend, SpeechMapping, SpeechSourceKind,
        TranslationBackend, TranslationLanguage, TranslationMapping, TurnDetector,
        TurnDetectorClass, TurnDetectorModel,
    };

    #[test]
    fn default_config_uses_desktop_audio_without_opening_an_external_listener() {
        let config = ParapperConfig::default();

        assert_eq!(config.input.source_kind, InputSourceKind::DesktopAudio);
        assert!(!config.streaming_recognition.enabled);
        assert_eq!(
            config
                .streaming_recognition
                .validated_bind_addr()
                .expect("loopback defaults should be valid")
                .to_string(),
            "127.0.0.1:18082"
        );
    }

    #[test]
    fn lan_streaming_recognition_bind_requires_an_explicit_api_key() {
        let mut config = ParapperConfig::default().streaming_recognition;
        config.bind_address = "0.0.0.0".to_string();
        assert!(config.validated_bind_addr().is_err());
        config.api_key = Some("   ".to_string());
        assert!(config.validated_bind_addr().is_err());

        config.api_key = Some("secret".to_string());
        assert_eq!(
            config
                .validated_bind_addr()
                .expect("LAN bind with an API key should be valid")
                .to_string(),
            "0.0.0.0:18082"
        );
    }

    #[test]
    fn default_config_uses_neo_http_port() {
        assert_eq!(ParapperConfig::default().neo.http_port, 15520);
    }

    #[test]
    fn default_config_does_not_require_neo_for_normal_use() {
        assert!(!ParapperConfig::default().neo.http_enabled);
    }

    #[test]
    fn default_config_sends_interim_text_to_neo() {
        assert_eq!(
            ParapperConfig::default().neo.send_timing,
            NeoSendTiming::Interim
        );
    }

    #[test]
    fn default_config_has_ul_unas_noise_cancellation_available_but_disabled() {
        let config = ParapperConfig::default();

        assert!(!config.noise_cancellation.enabled);
        assert_eq!(
            config.noise_cancellation.model,
            NoiseCancellationModel::UlUnas
        );
    }

    #[test]
    fn save_keeps_flat_config_file_shape() {
        let path = temporary_config_path("flat-shape");
        let mut config = ParapperConfig::default();
        config.neo.http_port = 16620;
        config.input.device_name = Some("Desk Mic".to_string());
        config.asr.language = AsrLanguage::English;
        config.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        config.translation.enabled = true;
        config.turn.detector = TurnDetector::Namo;
        config.noise_cancellation.enabled = true;

        config
            .save(&path)
            .expect("flat config test should write config");
        let content = fs::read_to_string(&path).expect("flat config test should read config");
        let value =
            serde_json::from_str::<serde_json::Value>(&content).expect("saved config is json");
        let object = value.as_object().expect("saved config should be an object");

        for nested_key in [
            "neo",
            "input",
            "streaming_recognition",
            "asr",
            "translation",
            "speech",
            "models",
            "segmentation",
            "turn",
            "noise_cancellation",
            "vrc",
            "debug",
        ] {
            assert!(
                !object.contains_key(nested_key),
                "config file should not contain nested {nested_key} object"
            );
        }
        assert_eq!(object["neo_http_port"], serde_json::json!(16620));
        assert_eq!(object["input_device_name"], serde_json::json!("Desk Mic"));
        assert_eq!(
            object["input_source_kind"],
            serde_json::json!("desktop_audio")
        );
        assert_eq!(
            object["streaming_recognition_enabled"],
            serde_json::json!(false)
        );
        assert_eq!(object["asr_language"], serde_json::json!("english"));
        assert_eq!(
            object["asr_model"],
            serde_json::json!("nemo_parakeet_tdt_0_6b_v2_int8")
        );
        assert_eq!(object["translation_enabled"], serde_json::json!(true));
        assert!(!object.contains_key("translation_local_server_mode"));
        assert_eq!(object["ync_plugin_port"], serde_json::json!(8080));
        assert!(!object.contains_key("translation_plugin_http_port"));
        assert_eq!(
            object["translation_local_server_port"],
            serde_json::json!(18081)
        );
        assert_eq!(
            object["translation_local_server_model"],
            serde_json::json!("lfm2_q4")
        );
        assert_eq!(object["turn_detector"], serde_json::json!("namo"));
        assert_eq!(
            object["noise_cancellation_enabled"],
            serde_json::json!(true)
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn flat_config_file_shape_loads_into_grouped_runtime_config() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "neo_http_enabled": false,
                "neo_http_port": 16620,
                "neo_send_timing": "final",
                "input_device_id": "mic-1",
                "input_device_host": "wasapi",
                "input_device_name": "Desk Mic",
                "input_volume_db": 4.5,
                "asr_language": "english",
                "asr_model": "nemo_parakeet_tdt_0_6b_v2_int8",
                "asr_precision": "int8",
                "asr_num_threads": 2,
                "asr_normalize_input_audio": false,
                "multilingual_asr_enabled": true,
                "enabled_asr_models": [
                    "nemo_parakeet_tdt_0_6b_v2_int8",
                    "reazonspeech_k2_v2"
                ],
                "translation_enabled": true,
                "translation_plugin_http_port": 18080,
                "translation_local_server_mode": "on",
                "translation_local_server_port": 18081,
                "translation_local_server_model": "cat_translate_0_8b_q4_k_quant",
                "translation_send_timing": "interim",
                "translation_mappings": [{
                    "id": "translate-en",
                    "source_asr_model": "nemo_parakeet_tdt_0_6b_v2_int8",
                    "target_lang": "ja_JP"
                }],
                "speech_mappings": [],
                "model_dir": "models",
                "vad_threshold": 0.6,
                "vad_interval_ms": 32,
                "segment_start_speech_ms": 128,
                "turn_detector": "namo",
                "interim_result_enabled": false,
                "interim_result_silence_ms": 128,
                "turn_check_silence_ms": 640,
                "namo_turn_confidence_threshold": 0.7,
                "namo_context_max_tokens": 128,
                "turn_rerecognize_full_on_complete": true,
                "noise_cancellation_enabled": true,
                "noise_cancellation_model": "ul_unas",
                "vrc_osc_micmute": true,
                "debug_asr_audio_playback": true,
                "recognition_log_limit": 100,
                "debug_audio_log_limit": 5
            }"#,
        )
        .expect("flat config json should deserialize")
        .normalized();

        assert!(!config.neo.http_enabled);
        assert_eq!(config.neo.http_port, 16620);
        assert_eq!(config.neo.send_timing, NeoSendTiming::Final);
        assert_eq!(config.input.device_id.as_deref(), Some("mic-1"));
        assert_eq!(config.input.device_host.as_deref(), Some("wasapi"));
        assert_eq!(config.input.device_name.as_deref(), Some("Desk Mic"));
        assert!((config.input.volume_db - 4.5).abs() < f32::EPSILON);
        assert_eq!(config.asr.language, AsrLanguage::English);
        assert_eq!(config.asr.model, AsrModel::NemoParakeetTdt0_6BV2Int8);
        assert_eq!(config.asr.precision, AsrPrecision::Int8);
        assert_eq!(config.asr.num_threads, 2);
        assert!(!config.asr.normalize_input_audio);
        assert!(config.asr.multilingual_enabled);
        assert_eq!(
            config.asr.enabled_models,
            vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8
            ]
        );
        assert!(config.translation.enabled);
        assert_eq!(config.translation.ync_plugin_port, 18080);
        assert_eq!(config.translation.local_server_port, 18081);
        assert_eq!(
            config.translation.local_server_model,
            LocalTranslationModel::Lfm2Q4
        );
        assert_eq!(config.translation.send_timing, NeoSendTiming::Interim);
        assert_eq!(config.translation.mappings[0].id, "translate-en");
        assert_eq!(
            config.translation.mappings[0].backend,
            TranslationBackend::Ync
        );
        assert_eq!(
            config.translation.mappings[0].source_lang,
            TranslationLanguage::En
        );
        assert_eq!(
            config.translation.mappings[0].target_lang,
            TranslationLanguage::Ja
        );
        assert_eq!(config.models.dir.as_deref(), Some("models"));
        assert!((config.segmentation.vad_threshold - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.segmentation.segment_start_speech_ms, 128);
        assert_eq!(config.turn.detector, TurnDetector::Namo);
        assert!(!config.turn.interim_result_enabled);
        assert_eq!(config.turn.check_silence_ms, 640);
        assert!(config.turn.rerecognize_full_on_complete);
        assert!(config.noise_cancellation.enabled);
        #[cfg(not(target_os = "macos"))]
        assert!(config.vrc.osc_micmute);
        #[cfg(target_os = "macos")]
        assert!(!config.vrc.osc_micmute);
        assert!(config.debug.asr_audio_playback);
        assert_eq!(config.debug.recognition_log_limit, Some(100));
        assert_eq!(config.debug.debug_audio_log_limit, Some(5));
    }

    #[test]
    fn namo_turn_detector_value_loads_from_canonical_storage_value() {
        let config = serde_json::from_str::<ParapperConfig>(r#"{ "turn_detector": "namo" }"#)
            .expect("namo turn detector should deserialize")
            .normalized();

        assert_eq!(config.turn.detector, TurnDetector::Namo);
    }

    #[test]
    fn namo_turn_detector_serializes_with_canonical_storage_value() {
        let mut config = ParapperConfig::default();
        config.turn.detector = TurnDetector::Namo;
        let value = serde_json::to_value(config).expect("config should serialize");

        assert_eq!(value["turn_detector"], serde_json::json!("namo"));
    }

    #[test]
    fn load_invalid_legacy_config_falls_back_to_default() {
        let path = temporary_config_path("legacy-config");
        fs::write(
            &path,
            r#"{
                "asr_model": "removed_asr_model",
                "neo_http_port": 12345
            }"#,
        )
        .expect("failed to write test config");

        let config = ParapperConfig::load(&path).expect("legacy config should fall back");

        assert_eq!(config, ParapperConfig::default());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn unsupported_model_precision_is_normalized() {
        let config = config_with(|config| {
            config.asr.language = AsrLanguage::English;
            config.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
            config.asr.precision = AsrPrecision::Float32;
        });

        assert_eq!(config.asr.precision, AsrPrecision::Int8);
    }

    #[test]
    fn required_asr_models_include_interim_only_model_without_duplicates() {
        let config = config_with(|config| {
            config.asr.model = AsrModel::ReazonSpeechK2V2;
            config.asr.interim_model = Some(AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8);
            config.asr.multilingual_enabled = false;
        });

        assert_eq!(
            config.required_asr_models(),
            vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
            ]
        );
    }

    #[test]
    fn primary_asr_model_normalization_replaces_interim_only_nemotron() {
        let config = config_with(|config| {
            config.asr.language = AsrLanguage::Multilingual;
            config.asr.model = AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8;
            config.asr.multilingual_enabled = false;
        });

        assert_eq!(config.asr.language, AsrLanguage::EuropeanMultilingual);
        assert_eq!(config.asr.model, AsrModel::NemoParakeetTdt0_6BV3Int8);
        assert_eq!(
            config.completion_asr_model(),
            AsrModel::NemoParakeetTdt0_6BV3Int8,
            "Nemotron streaming models are restricted to interim display"
        );
        assert_eq!(
            config.required_asr_models(),
            vec![AsrModel::NemoParakeetTdt0_6BV3Int8]
        );
    }

    #[test]
    fn final_capable_interim_override_normalizes_to_primary_model() {
        let config = config_with(|config| {
            config.asr.model = AsrModel::ReazonSpeechK2V2;
            config.asr.interim_model = Some(AsrModel::NemoParakeetTdt0_6BV2Int8);
        });

        assert_eq!(config.asr.interim_model, None);
        assert_eq!(
            config.required_asr_models(),
            vec![AsrModel::ReazonSpeechK2V2]
        );
    }

    #[test]
    fn nemotron_models_are_int8_only_and_expose_streaming_languages() {
        let cases = [
            (
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
        ];

        for (model, language, required_language_code) in cases {
            assert!(model.is_nemotron());
            assert_eq!(model.implementation(), AsrModelImplementation::Nemotron);
            assert_eq!(model.capability(), AsrModelCapability::InterimOnly);
            assert_eq!(model.language(), language);
            assert!(
                model
                    .supported_language_codes()
                    .contains(&required_language_code)
            );
            assert_eq!(model.default_precision(), AsrPrecision::Int8);
            assert!(model.supports_precision(AsrPrecision::Int8));
            assert!(!model.supports_precision(AsrPrecision::Float32));
        }
    }

    #[test]
    fn european_multilingual_defaults_to_parakeet_v3() {
        let config = config_with(|config| {
            config.asr.language = AsrLanguage::EuropeanMultilingual;
        });

        assert_eq!(config.asr.model, AsrModel::NemoParakeetTdt0_6BV3Int8);
    }

    #[test]
    fn negative_asr_num_threads_is_normalized_to_auto() {
        let config = config_with(|config| {
            config.asr.num_threads = -1;
        });

        assert_eq!(config.asr.num_threads, 0);
    }

    #[test]
    fn auto_asr_num_threads_resolves_to_available_parallelism() {
        let config = config_with(|config| {
            config.asr.num_threads = 0;
        });
        let expected = std::thread::available_parallelism()
            .map(usize::from)
            .ok()
            .and_then(|threads| i32::try_from(threads).ok())
            .filter(|threads| *threads > 0)
            .unwrap_or(1);

        assert_eq!(config.effective_asr_num_threads(), expected);
    }

    #[test]
    fn explicit_asr_num_threads_is_used_as_effective_thread_count() {
        let config = config_with(|config| {
            config.asr.num_threads = 4;
        });

        assert_eq!(config.effective_asr_num_threads(), 4);
    }

    #[test]
    fn vad_interval_is_normalized_to_supported_chunk_size() {
        let config = config_with(|config| {
            config.segmentation.vad_interval_ms = 100;
            config.segmentation.segment_start_speech_ms = 300;
        });

        assert_eq!(config.segmentation.vad_interval_ms, 32);
        assert_eq!(config.segmentation.segment_start_speech_ms, 300);
    }

    #[test]
    fn default_vad_timing_keeps_short_speech_starts_responsive() {
        let config = ParapperConfig::default();

        assert_eq!(config.turn.interim_result_silence_ms, 96);
        assert_eq!(config.turn.check_silence_ms, 320);
        assert_eq!(config.segmentation.segment_start_speech_ms, 96);
    }

    #[test]
    fn turn_detector_thresholds_are_normalized() {
        let config = config_with(|config| {
            config.turn.interim_result_silence_ms = 1;
            config.turn.check_silence_ms = 1;
            config.turn.namo_confidence_threshold = 2.0;
            config.turn.namo_context_max_tokens = 999;
        });

        assert_eq!(config.turn.interim_result_silence_ms, 32);
        assert_eq!(config.turn.check_silence_ms, 32);
        assert!((config.turn.namo_confidence_threshold - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.turn.namo_context_max_tokens, 512);
    }

    #[test]
    fn namo_turn_detector_keeps_interim_and_check_silence_independent() {
        let config = config_with(|config| {
            config.turn.detector = TurnDetector::Namo;
            config.turn.interim_result_silence_ms = 96;
            config.turn.check_silence_ms = 320;
        });

        assert_eq!(config.turn.interim_result_silence_ms, 96);
        assert_eq!(config.turn.check_silence_ms, 320);
    }

    #[test]
    fn input_volume_is_normalized_to_supported_db_range() {
        let config = config_with(|config| {
            config.input.volume_db = 99.0;
        });

        assert!((config.input.volume_db - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn turn_detector_mode_capabilities_are_separated() {
        let mut namo_config = ParapperConfig::default();
        namo_config.turn.detector = TurnDetector::Namo;
        assert_eq!(
            namo_config.turn_detector_class(),
            TurnDetectorClass::Model(TurnDetectorModel::Namo)
        );
        assert_eq!(
            namo_config.turn_detector_model(),
            Some(TurnDetectorModel::Namo)
        );
        assert!(namo_config.uses_namo_turn_detector());
        assert!(namo_config.uses_morph_turn_boundary());
        assert!(namo_config.requires_japanese_morph_analyzer());

        let mut morph_config = ParapperConfig::default();
        morph_config.turn.detector = TurnDetector::Morph;
        assert_eq!(
            morph_config.turn_detector_class(),
            TurnDetectorClass::Simple
        );
        assert_eq!(morph_config.turn_detector_model(), None);
        assert!(!morph_config.uses_namo_turn_detector());
        assert!(morph_config.uses_morph_turn_boundary());
        assert!(morph_config.requires_japanese_morph_analyzer());

        assert!(TurnDetector::Namo.can_connect_interim_after_completion());
        assert!(TurnDetector::Morph.can_connect_interim_after_completion());
        assert!(!TurnDetector::Simple.can_connect_interim_after_completion());
    }

    #[test]
    fn namo_turn_detector_is_kept_for_english_and_multilingual_asr() {
        for (language, expected_model) in [
            (AsrLanguage::English, AsrModel::NemoParakeetTdt0_6BV2Int8),
            (
                AsrLanguage::EuropeanMultilingual,
                AsrModel::NemoParakeetTdt0_6BV3Int8,
            ),
        ] {
            let config = config_with(|config| {
                config.asr.language = language;
                config.asr.model = expected_model;
                config.turn.detector = TurnDetector::Namo;
                config.asr.multilingual_enabled = false;
            });

            assert_eq!(config.turn.detector, TurnDetector::Namo);
            assert_eq!(config.required_asr_models(), vec![expected_model]);
            assert_eq!(
                config.required_namo_turn_detector_languages(),
                vec![language],
                "language={language:?}"
            );
        }
    }

    #[test]
    fn translation_defaults_are_disabled_and_speech_mappings_default_empty() {
        let config = ParapperConfig::default();

        assert!(!config.translation.enabled);
        assert_eq!(config.translation.ync_plugin_port, 8080);
        assert_eq!(config.translation.local_server_port, 18081);
        assert_eq!(
            config.translation.local_server_model,
            LocalTranslationModel::Lfm2Q4
        );
        assert_eq!(config.translation.send_timing, NeoSendTiming::Final);
        assert!(config.translation.mappings.is_empty());
        assert!(config.speech.mappings.is_empty());
    }

    #[test]
    fn legacy_ync_plugin_port_loads_and_saves_only_the_canonical_key() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{"translation_plugin_http_port": 18080, "translation_local_server_mode": "on"}"#,
        )
        .expect("legacy config should load");
        assert_eq!(config.translation.ync_plugin_port, 18080);

        let value = serde_json::to_value(config).expect("config should serialize");
        assert_eq!(value["ync_plugin_port"], serde_json::json!(18080));
        assert!(value.get("translation_plugin_http_port").is_none());
        assert!(value.get("translation_local_server_mode").is_none());
    }

    #[test]
    fn speech_backend_serializes_as_ync() {
        assert_eq!(
            serde_json::to_string(&SpeechBackend::Ync).unwrap(),
            r#""ync""#
        );
    }

    #[test]
    fn local_translation_model_serializes_as_model_file_quantization_name() {
        assert_eq!(
            serde_json::to_string(&LocalTranslationModel::Lfm2Q4).unwrap(),
            r#""lfm2_q4""#
        );
        assert_eq!(
            serde_json::to_string(&LocalTranslationModel::CatTranslate0_8BQ4KQuant).unwrap(),
            r#""cat_translate_0_8b_q4_k_quant""#
        );
        for legacy_value in [
            "f32",
            "fp32",
            "model",
            "q4",
            "q4f16",
            "q4_f16",
            "model_quantized",
            "k_quant",
            "q4_k_quant",
            "lfm2_q4_k_quant",
        ] {
            assert_eq!(
                serde_json::from_str::<LocalTranslationModel>(&format!(r#""{legacy_value}""#))
                    .unwrap(),
                LocalTranslationModel::Lfm2Q4
            );
        }
        assert_eq!(
            serde_json::from_str::<LocalTranslationModel>(r#""cat-translate-0.8b-onnx-q4""#)
                .unwrap(),
            LocalTranslationModel::CatTranslate0_8BQ4KQuant
        );
    }

    #[test]
    fn disabled_cat_translation_config_migrates_to_onnx_community_lfm2_q4() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "translation_local_server_model": "cat_translate_0_8b_q4_k_quant",
                "translation_mappings": [{
                    "id": "legacy-cat",
                    "backend": "local",
                    "local_model": "cat_translate_0_8b_q4_k_quant",
                    "source_lang": "ja",
                    "target_lang": "en"
                }]
            }"#,
        )
        .expect("legacy CAT config should deserialize")
        .normalized();

        assert_eq!(
            config.translation.local_server_model,
            LocalTranslationModel::Lfm2Q4
        );
        assert_eq!(
            config.translation.mappings[0].local_model,
            LocalTranslationModel::Lfm2Q4
        );
    }

    #[test]
    fn legacy_speech_backend_config_value_loads_as_ync() {
        let old_backend_value = ["yuka", "kone_neo"].concat();
        let config_json = format!(
            r#"{{
                "speech_mappings": [{{
                    "id": "speech-legacy-backend",
                    "source_kind": "recognition",
                    "target_lang": null,
                    "backend": "{old_backend_value}",
                    "talker": "ずんだもん/VOICEVOX",
                    "muted": false,
                    "volume": 1.0
                }}]
            }}"#
        );

        let config = serde_json::from_str::<ParapperConfig>(&config_json)
            .unwrap()
            .normalized();

        assert_eq!(config.speech.mappings[0].backend, SpeechBackend::Ync);
    }

    #[test]
    fn translation_and_speech_mappings_are_normalized() {
        let config = config_with(|config| {
            config.asr.multilingual_enabled = true;
            config.asr.enabled_models = vec![AsrModel::ReazonSpeechK2V2];
            config.translation.mappings = vec![
                TranslationMapping {
                    id: " translate-ja ".to_string(),
                    source_asr_model: Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
                    backend: TranslationBackend::Local,
                    local_model: LocalTranslationModel::Lfm2Q4,
                    source_lang: TranslationLanguage::Ja,
                    target_lang: TranslationLanguage::En,
                },
                TranslationMapping {
                    id: "same-language".to_string(),
                    source_asr_model: None,
                    backend: TranslationBackend::Ync,
                    local_model: LocalTranslationModel::default(),
                    source_lang: TranslationLanguage::En,
                    target_lang: TranslationLanguage::En,
                },
            ];
            config.speech.mappings = vec![SpeechMapping {
                id: " speech-ja ".to_string(),
                source_kind: SpeechSourceKind::Translation,
                source_asr_model: None,
                target_lang: Some(" ".to_string()),
                backend: SpeechBackend::Ync,
                talker: " ずんだもん/VOICEVOX ".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: -99.0,
            }];
        });

        assert_eq!(config.translation.mappings.len(), 1);
        assert_eq!(config.translation.mappings[0].id, "translate-ja");
        assert_eq!(
            config.translation.mappings[0].backend,
            TranslationBackend::Local
        );
        assert_eq!(
            config.translation.mappings[0].source_lang,
            TranslationLanguage::Ja
        );
        assert_eq!(
            config.translation.mappings[0].target_lang,
            TranslationLanguage::En
        );
        assert_eq!(config.speech.mappings.len(), 1);
        assert_eq!(config.speech.mappings[0].id, "speech-ja");
        assert_eq!(config.speech.mappings[0].talker, "ずんだもん/VOICEVOX");
        assert!((config.speech.mappings[0].volume + 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn unsupported_legacy_translation_target_drops_mapping_without_dropping_config() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "translation_enabled": true,
                "translation_mappings": [{
                    "id": "translate-fr",
                    "target_lang": "fr_FR"
                }]
            }"#,
        )
        .expect("unsupported legacy translation mapping should not reject whole config")
        .normalized();

        assert!(config.translation.enabled);
        assert!(config.translation.mappings.is_empty());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn neo_text_input_disabled_keeps_translation_and_plugin_speech_available() {
        let config = config_with(|config| {
            config.neo.http_enabled = false;
            config.translation.enabled = true;
            config.speech.mappings = vec![SpeechMapping {
                id: "speech-neo".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::Ync,
                talker: "ずんだもん/VOICEVOX".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 0.0,
            }];
        });

        assert!(config.translation.enabled);
        assert_eq!(config.speech.mappings[0].backend, SpeechBackend::Ync);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unsupported_neo_http_platform_disables_text_input_translation_and_vrc_flags() {
        let config = config_with(|config| {
            config.neo.http_enabled = true;
            config.translation.enabled = true;
            config.vrc.osc_micmute = true;
        });

        assert!(!config.neo.http_enabled);
        assert!(!config.translation.enabled);
        assert!(!config.vrc.osc_micmute);
    }

    #[test]
    fn speech_mapping_without_talker_is_kept_but_incomplete() {
        let config = config_with(|config| {
            config.speech.mappings = vec![SpeechMapping {
                id: " speech-empty ".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::Ync,
                talker: " ".to_string(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }];
        });

        assert_eq!(config.speech.mappings.len(), 1);
        assert_eq!(config.speech.mappings[0].id, "speech-empty");
        assert!(config.speech.mappings[0].talker.is_empty());
    }

    #[test]
    fn local_tts_speech_mapping_defaults_voice() {
        let config = config_with(|config| {
            config.speech.mappings = vec![SpeechMapping {
                id: "speech-local-tts".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::LocalTts,
                talker: String::new(),
                local_tts_voice: None,
                local_tts_language: None,
                local_tts_speaker_id: None,
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }];
        });

        assert_eq!(
            config.speech.mappings[0].local_tts_voice,
            Some(LocalTtsVoice::Kristin)
        );
    }

    #[test]
    fn removed_local_tts_speech_mapping_voice_falls_back_to_default() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "speech_mappings": [{
                    "id": "speech-removed-voice",
                    "source_kind": "recognition",
                    "target_lang": null,
                    "backend": "local_tts",
                    "talker": "",
                    "local_tts_voice": "removed_voice",
                    "muted": false,
                    "volume": 1.0
                }]
            }"#,
        )
        .unwrap()
        .normalized();

        assert_eq!(
            config.speech.mappings[0].local_tts_voice,
            Some(LocalTtsVoice::Kristin)
        );
    }

    #[test]
    fn supertonic_speech_mapping_defaults_language_and_speaker() {
        let config = config_with(|config| {
            config.speech.mappings = vec![SpeechMapping {
                id: "speech-supertonic".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::LocalTts,
                talker: String::new(),
                local_tts_voice: Some(LocalTtsVoice::Supertonic2Onnx),
                local_tts_language: Some(" ES ".to_string()),
                local_tts_speaker_id: Some(99),
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }];
        });

        assert_eq!(
            config.speech.mappings[0].local_tts_language.as_deref(),
            Some("es")
        );
        assert_eq!(config.speech.mappings[0].local_tts_speaker_id, Some(9));
    }

    #[test]
    fn supertonic3_speech_mapping_accepts_extended_languages() {
        let config = config_with(|config| {
            config.speech.mappings = vec![SpeechMapping {
                id: "speech-supertonic3".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::LocalTts,
                talker: String::new(),
                local_tts_voice: Some(LocalTtsVoice::Supertonic3Onnx),
                local_tts_language: Some(" JA ".to_string()),
                local_tts_speaker_id: Some(0),
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }];
        });

        assert_eq!(
            config.speech.mappings[0].local_tts_language.as_deref(),
            Some("ja")
        );
    }

    fn temporary_config_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("parapper-{name}-{nanos}.json"))
    }

    fn config_with(update: impl FnOnce(&mut ParapperConfig)) -> ParapperConfig {
        let mut config = ParapperConfig::default();
        update(&mut config);
        config.normalized()
    }
}
