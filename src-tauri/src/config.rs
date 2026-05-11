use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NeoSendTiming {
    Interim,
    Final,
}

impl Default for NeoSendTiming {
    fn default() -> Self {
        Self::Interim
    }
}

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
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

    pub fn supported_language_codes(self) -> &'static [&'static str] {
        match self {
            Self::ReazonSpeechK2V2 => &["ja"],
            Self::NemoParakeetTdt0_6BV2Int8 => &["en"],
            Self::NemoParakeetTdt0_6BV3Int8 => PARAKEET_TDT_0_6B_V3_LANGUAGE_CODES,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnDetector {
    Simple,
    Namo,
}

impl Default for TurnDetector {
    fn default() -> Self {
        Self::Simple
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NoiseCancellationModel {
    #[serde(rename = "ul_unas")]
    UlUnas,
}

impl Default for NoiseCancellationModel {
    fn default() -> Self {
        Self::UlUnas
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorClass {
    Simple,
    Model(TurnDetectorModel),
}

impl TurnDetectorClass {
    pub fn model(self) -> Option<TurnDetectorModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::Simple => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorModel {
    Namo,
}

impl TurnDetector {
    pub fn class(self) -> TurnDetectorClass {
        match self {
            Self::Simple => TurnDetectorClass::Simple,
            Self::Namo => TurnDetectorClass::Model(TurnDetectorModel::Namo),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[expect(clippy::struct_excessive_bools)]
pub struct ParapperConfig {
    pub neo_http_enabled: bool,
    pub neo_http_port: u16,
    pub neo_send_timing: NeoSendTiming,
    pub input_device_id: Option<String>,
    pub input_device_host: Option<String>,
    pub input_device_name: Option<String>,
    pub input_volume_db: f32,
    pub asr_language: AsrLanguage,
    pub asr_model: AsrModel,
    pub asr_precision: AsrPrecision,
    pub asr_num_threads: i32,
    pub asr_normalize_input_audio: bool,
    pub multilingual_asr_enabled: bool,
    pub enabled_asr_models: Vec<AsrModel>,
    pub translation_enabled: bool,
    pub translation_plugin_http_port: u16,
    pub translation_send_timing: NeoSendTiming,
    pub translation_mappings: Vec<TranslationMapping>,
    pub speech_mappings: Vec<SpeechMapping>,
    pub model_dir: Option<String>,
    pub vad_threshold: f32,
    pub vad_interval_ms: u32,
    pub segment_start_speech_ms: u32,
    pub turn_detector: TurnDetector,
    pub interim_result_enabled: bool,
    pub interim_result_silence_ms: u32,
    pub turn_check_silence_ms: u32,
    pub namo_turn_confidence_threshold: f32,
    pub namo_context_max_tokens: u32,
    pub turn_rerecognize_full_on_complete: bool,
    pub noise_cancellation_enabled: bool,
    pub noise_cancellation_model: NoiseCancellationModel,
    pub vrc_osc_micmute: bool,
    pub debug_asr_audio_playback: bool,
    pub recognition_log_limit: Option<usize>,
    pub debug_audio_log_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationMapping {
    pub id: String,
    pub source_asr_model: Option<AsrModel>,
    pub target_lang: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechSourceKind {
    Recognition,
    Translation,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechBackend {
    Ync,
    LocalTts,
}

impl<'de> Deserialize<'de> for SpeechBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "ync" => Ok(Self::Ync),
            "local_tts" => Ok(Self::LocalTts),
            legacy if is_legacy_ync_backend_value(legacy) => Ok(Self::Ync),
            _ => Err(D::Error::unknown_variant(&value, &["ync", "local_tts"])),
        }
    }
}

impl Default for SpeechBackend {
    fn default() -> Self {
        Self::Ync
    }
}

fn is_legacy_ync_backend_value(value: &str) -> bool {
    value.len() == 12 && value.starts_with("yuka") && value.ends_with("kone_neo")
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LocalTtsVoice {
    #[serde(rename = "vits_piper_en_US_kristin_medium")]
    Kristin,
    #[serde(rename = "vits_piper_en_US_john_medium")]
    John,
    #[serde(rename = "vits_piper_en_US_norman_medium")]
    Norman,
    #[serde(rename = "supertonic_2_onnx")]
    Supertonic2Onnx,
    #[serde(rename = "supertonic_3_onnx")]
    Supertonic3Onnx,
}

impl Default for LocalTtsVoice {
    fn default() -> Self {
        Self::Kristin
    }
}

impl LocalTtsVoice {
    pub fn family(self) -> LocalTtsFamily {
        match self {
            Self::Kristin | Self::John | Self::Norman => LocalTtsFamily::Vits,
            Self::Supertonic2Onnx | Self::Supertonic3Onnx => LocalTtsFamily::Supertonic,
        }
    }

    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Kristin => "vits-piper-en_US-kristin-medium",
            Self::John => "vits-piper-en_US-john-medium",
            Self::Norman => "vits-piper-en_US-norman-medium",
            Self::Supertonic2Onnx => "supertonic-2-onnx",
            Self::Supertonic3Onnx => "supertonic-3-onnx",
        }
    }

    pub fn onnx_file_name(self) -> &'static str {
        match self {
            Self::Kristin => "en_US-kristin-medium.onnx",
            Self::John => "en_US-john-medium.onnx",
            Self::Norman => "en_US-norman-medium.onnx",
            Self::Supertonic2Onnx | Self::Supertonic3Onnx => "onnx/duration_predictor.onnx",
        }
    }

    pub fn supported_language_codes(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Supertonic2Onnx => Some(SUPERTONIC2_LANGUAGE_CODES),
            Self::Supertonic3Onnx => Some(SUPERTONIC3_LANGUAGE_CODES),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTtsFamily {
    Vits,
    Supertonic,
}

pub const ALL_LOCAL_TTS_VOICES: &[LocalTtsVoice] = &[
    LocalTtsVoice::Kristin,
    LocalTtsVoice::John,
    LocalTtsVoice::Norman,
    LocalTtsVoice::Supertonic2Onnx,
    LocalTtsVoice::Supertonic3Onnx,
];

pub const SUPERTONIC2_LANGUAGE_CODES: &[&str] = &["en", "ko", "es", "pt", "fr"];
pub const SUPERTONIC3_LANGUAGE_CODES: &[&str] = &[
    "en", "ko", "ja", "bg", "cs", "da", "el", "es", "et", "fi", "hu", "it", "nl", "pl", "pt", "ro",
    "ar", "de", "fr", "hi", "id", "ru", "vi",
];

const PARAKEET_TDT_0_6B_V3_LANGUAGE_CODES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechMapping {
    pub id: String,
    pub source_kind: SpeechSourceKind,
    #[serde(default)]
    pub source_asr_model: Option<AsrModel>,
    pub target_lang: Option<String>,
    #[serde(default)]
    pub backend: SpeechBackend,
    pub talker: String,
    #[serde(default, deserialize_with = "deserialize_optional_local_tts_voice")]
    pub local_tts_voice: Option<LocalTtsVoice>,
    #[serde(default)]
    pub local_tts_language: Option<String>,
    #[serde(default)]
    pub local_tts_speaker_id: Option<i32>,
    #[serde(default)]
    pub output_device_id: Option<String>,
    #[serde(default)]
    pub output_device_host: Option<String>,
    #[serde(default)]
    pub output_device_name: Option<String>,
    #[serde(default)]
    pub muted: bool,
    pub volume: f32,
}

impl Default for ParapperConfig {
    fn default() -> Self {
        Self {
            neo_http_enabled: true,
            neo_http_port: 15520,
            neo_send_timing: NeoSendTiming::Interim,
            input_device_id: None,
            input_device_host: None,
            input_device_name: None,
            input_volume_db: 0.0,
            asr_language: AsrLanguage::Japanese,
            asr_model: AsrModel::ReazonSpeechK2V2,
            asr_precision: AsrPrecision::Int8Float32,
            asr_num_threads: 4,
            asr_normalize_input_audio: true,
            multilingual_asr_enabled: false,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            translation_enabled: false,
            translation_plugin_http_port: 8080,
            translation_send_timing: NeoSendTiming::Final,
            translation_mappings: Vec::new(),
            speech_mappings: Vec::new(),
            model_dir: None,
            vad_threshold: 0.5,
            vad_interval_ms: 32,
            segment_start_speech_ms: 96,
            turn_detector: TurnDetector::Simple,
            interim_result_enabled: true,
            interim_result_silence_ms: 96,
            turn_check_silence_ms: 320,
            namo_turn_confidence_threshold: 0.8,
            namo_context_max_tokens: 256,
            turn_rerecognize_full_on_complete: false,
            noise_cancellation_enabled: false,
            noise_cancellation_model: NoiseCancellationModel::UlUnas,
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

    pub fn required_asr_models(&self) -> Vec<AsrModel> {
        if self.multilingual_asr_enabled {
            self.enabled_asr_models.clone()
        } else {
            vec![self.asr_model]
        }
    }

    pub fn asr_precision_for(&self, model: AsrModel) -> AsrPrecision {
        if model == self.asr_model {
            self.asr_precision
        } else {
            model.default_precision()
        }
    }

    pub fn turn_detector_class(&self) -> TurnDetectorClass {
        self.turn_detector.class()
    }

    pub fn turn_detector_model(&self) -> Option<TurnDetectorModel> {
        self.turn_detector_class().model()
    }

    pub fn uses_namo_turn_detector(&self) -> bool {
        self.turn_detector_model() == Some(TurnDetectorModel::Namo)
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
        self.vad_interval_ms = 32;
        self.segment_start_speech_ms = self
            .segment_start_speech_ms
            .max(self.vad_interval_ms.max(1));
        self.interim_result_silence_ms = self
            .interim_result_silence_ms
            .max(self.vad_interval_ms.max(1));
        self.turn_check_silence_ms = self.turn_check_silence_ms.max(self.vad_interval_ms.max(1));
        if self.interim_result_enabled {
            self.turn_check_silence_ms = self
                .turn_check_silence_ms
                .max(self.interim_result_silence_ms);
        } else {
            self.turn_check_silence_ms =
                self.turn_check_silence_ms.max(self.vad_interval_ms.max(1));
        }
        self.namo_turn_confidence_threshold = self.namo_turn_confidence_threshold.clamp(0.0, 1.0);
        self.namo_context_max_tokens = self.namo_context_max_tokens.min(512);
        self.input_volume_db = normalize_input_volume_db(self.input_volume_db);
        if self.asr_model.language() != self.asr_language {
            self.asr_model = AsrModel::default_for_language(self.asr_language);
        }
        if !self.asr_model.supports_precision(self.asr_precision) {
            self.asr_precision = self.asr_model.default_precision();
        }
        normalize_enabled_asr_models(&mut self.enabled_asr_models);
        if !self.enabled_asr_models.contains(&self.asr_model) {
            self.enabled_asr_models.push(self.asr_model);
        }
        self.translation_mappings = normalize_translation_mappings(self.translation_mappings);
        self.speech_mappings = normalize_speech_mappings(self.speech_mappings);
        self.asr_num_threads = self.asr_num_threads.max(0);
        self = self.normalized_for_platform();
        self
    }

    fn normalized_for_platform(mut self) -> Self {
        if !Self::neo_http_supported() {
            self.neo_http_enabled = false;
            self.translation_enabled = false;
        }
        if !Self::vrc_osc_supported() {
            self.vrc_osc_micmute = false;
        }
        self
    }
}

fn normalize_translation_mappings(mappings: Vec<TranslationMapping>) -> Vec<TranslationMapping> {
    mappings
        .into_iter()
        .filter_map(|mut mapping| {
            mapping.id = mapping.id.trim().to_string();
            mapping.target_lang = mapping.target_lang.trim().to_string();
            if mapping.id.is_empty() || mapping.target_lang.is_empty() {
                return None;
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

fn deserialize_optional_local_tts_voice<'de, D>(
    deserializer: D,
) -> Result<Option<LocalTtsVoice>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match value.as_str() {
        "vits_piper_en_US_kristin_medium" => Some(LocalTtsVoice::Kristin),
        "vits_piper_en_US_john_medium" => Some(LocalTtsVoice::John),
        "vits_piper_en_US_norman_medium" => Some(LocalTtsVoice::Norman),
        "supertonic_2_onnx" => Some(LocalTtsVoice::Supertonic2Onnx),
        "supertonic_3_onnx" => Some(LocalTtsVoice::Supertonic3Onnx),
        _ => None,
    }))
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
    models.sort_by_key(|model| match model {
        AsrModel::ReazonSpeechK2V2 => 0,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => 1,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => 2,
    });
    models.dedup();
    if models.is_empty() {
        models.push(AsrModel::ReazonSpeechK2V2);
    }
}

fn normalize_asr_languages(languages: &mut Vec<AsrLanguage>) {
    languages.sort_by_key(|language| match language {
        AsrLanguage::Japanese => 0,
        AsrLanguage::English => 1,
        AsrLanguage::EuropeanMultilingual => 2,
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
        AsrLanguage, AsrModel, AsrPrecision, LocalTtsVoice, NeoSendTiming, NoiseCancellationModel,
        ParapperConfig, SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationMapping,
        TurnDetector, TurnDetectorClass, TurnDetectorModel,
    };

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
    fn default_config_sends_interim_text_to_neo() {
        assert_eq!(
            ParapperConfig::default().neo_send_timing,
            NeoSendTiming::Interim
        );
    }

    #[test]
    fn default_config_has_ul_unas_noise_cancellation_available_but_disabled() {
        let config = ParapperConfig::default();

        assert!(!config.noise_cancellation_enabled);
        assert_eq!(
            config.noise_cancellation_model,
            NoiseCancellationModel::UlUnas
        );
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
    fn vad_interval_is_normalized_to_supported_chunk_size() {
        let config = ParapperConfig {
            vad_interval_ms: 100,
            segment_start_speech_ms: 300,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.vad_interval_ms, 32);
        assert_eq!(config.segment_start_speech_ms, 300);
    }

    #[test]
    fn default_vad_timing_keeps_short_speech_starts_responsive() {
        let config = ParapperConfig::default();

        assert_eq!(config.interim_result_silence_ms, 96);
        assert_eq!(config.turn_check_silence_ms, 320);
        assert_eq!(config.segment_start_speech_ms, 96);
    }

    #[test]
    fn turn_detector_thresholds_are_normalized() {
        let config = ParapperConfig {
            interim_result_silence_ms: 1,
            turn_check_silence_ms: 1,
            namo_turn_confidence_threshold: 2.0,
            namo_context_max_tokens: 999,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.interim_result_silence_ms, 32);
        assert_eq!(config.turn_check_silence_ms, 32);
        assert!((config.namo_turn_confidence_threshold - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.namo_context_max_tokens, 512);
    }

    #[test]
    fn namo_turn_detector_keeps_interim_and_check_silence_independent() {
        let config = ParapperConfig {
            turn_detector: TurnDetector::Namo,
            interim_result_silence_ms: 96,
            turn_check_silence_ms: 320,
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.interim_result_silence_ms, 96);
        assert_eq!(config.turn_check_silence_ms, 320);
    }

    #[test]
    fn input_volume_is_normalized_to_supported_db_range() {
        let config = ParapperConfig {
            input_volume_db: 99.0,
            ..ParapperConfig::default()
        }
        .normalized();

        assert!((config.input_volume_db - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn namo_setting_maps_to_model_turn_detector_class() {
        let config = ParapperConfig {
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };

        assert_eq!(
            config.turn_detector_class(),
            TurnDetectorClass::Model(TurnDetectorModel::Namo)
        );
        assert_eq!(config.turn_detector_model(), Some(TurnDetectorModel::Namo));
        assert!(config.uses_namo_turn_detector());
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
            let config = ParapperConfig {
                asr_language: language,
                asr_model: expected_model,
                turn_detector: TurnDetector::Namo,
                multilingual_asr_enabled: false,
                ..ParapperConfig::default()
            }
            .normalized();

            assert_eq!(config.turn_detector, TurnDetector::Namo);
            assert_eq!(config.required_asr_models(), vec![expected_model]);
            assert_eq!(
                config.required_namo_turn_detector_languages(),
                vec![language]
            );
        }
    }

    #[test]
    fn translation_defaults_are_disabled_and_speech_mappings_default_empty() {
        let config = ParapperConfig::default();

        assert!(!config.translation_enabled);
        assert_eq!(config.translation_plugin_http_port, 8080);
        assert_eq!(config.translation_send_timing, NeoSendTiming::Final);
        assert!(config.translation_mappings.is_empty());
        assert!(config.speech_mappings.is_empty());
    }

    #[test]
    fn speech_backend_serializes_as_ync() {
        assert_eq!(
            serde_json::to_string(&SpeechBackend::Ync).unwrap(),
            r#""ync""#
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

        assert_eq!(config.speech_mappings[0].backend, SpeechBackend::Ync);
    }

    #[test]
    fn translation_and_speech_mappings_are_normalized() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![AsrModel::ReazonSpeechK2V2],
            translation_mappings: vec![
                TranslationMapping {
                    id: " translate-ja ".to_string(),
                    source_asr_model: Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
                    target_lang: " en_US ".to_string(),
                },
                TranslationMapping {
                    id: "empty-target".to_string(),
                    source_asr_model: None,
                    target_lang: " ".to_string(),
                },
            ],
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.translation_mappings.len(), 1);
        assert_eq!(config.translation_mappings[0].id, "translate-ja");
        assert_eq!(config.translation_mappings[0].target_lang, "en_US");
        assert_eq!(config.speech_mappings.len(), 1);
        assert_eq!(config.speech_mappings[0].id, "speech-ja");
        assert_eq!(config.speech_mappings[0].talker, "ずんだもん/VOICEVOX");
        assert!((config.speech_mappings[0].volume + 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn neo_text_input_disabled_keeps_translation_and_plugin_speech_available() {
        let config = ParapperConfig {
            neo_http_enabled: false,
            translation_enabled: true,
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert!(config.translation_enabled);
        assert_eq!(config.speech_mappings[0].backend, SpeechBackend::Ync);
    }

    #[test]
    fn speech_mapping_without_talker_is_kept_but_incomplete() {
        let config = ParapperConfig {
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(config.speech_mappings.len(), 1);
        assert_eq!(config.speech_mappings[0].id, "speech-empty");
        assert!(config.speech_mappings[0].talker.is_empty());
    }

    #[test]
    fn local_tts_speech_mapping_defaults_voice() {
        let config = ParapperConfig {
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(
            config.speech_mappings[0].local_tts_voice,
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
            config.speech_mappings[0].local_tts_voice,
            Some(LocalTtsVoice::Kristin)
        );
    }

    #[test]
    fn supertonic_speech_mapping_defaults_language_and_speaker() {
        let config = ParapperConfig {
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(
            config.speech_mappings[0].local_tts_language.as_deref(),
            Some("es")
        );
        assert_eq!(config.speech_mappings[0].local_tts_speaker_id, Some(9));
    }

    #[test]
    fn supertonic3_speech_mapping_accepts_extended_languages() {
        let config = ParapperConfig {
            speech_mappings: vec![SpeechMapping {
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
            }],
            ..ParapperConfig::default()
        }
        .normalized();

        assert_eq!(
            config.speech_mappings[0].local_tts_language.as_deref(),
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
}
