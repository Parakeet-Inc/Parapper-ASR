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
    #[serde(rename = "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8")]
    NemoParakeetTdtCtc0_6BJa35000Int8,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v2_int8")]
    NemoParakeetTdt0_6BV2Int8,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v3_int8")]
    NemoParakeetTdt0_6BV3Int8,
}

impl AsrModel {
    pub fn language(self) -> AsrLanguage {
        match self {
            Self::ReazonSpeechK2V2 | Self::NemoParakeetTdtCtc0_6BJa35000Int8 => {
                AsrLanguage::Japanese
            }
            Self::NemoParakeetTdt0_6BV2Int8 => AsrLanguage::English,
            Self::NemoParakeetTdt0_6BV3Int8 => AsrLanguage::EuropeanMultilingual,
        }
    }

    pub fn supported_language_codes(self) -> &'static [&'static str] {
        match self {
            Self::ReazonSpeechK2V2 | Self::NemoParakeetTdtCtc0_6BJa35000Int8 => &["ja"],
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
            Self::NemoParakeetTdtCtc0_6BJa35000Int8
            | Self::NemoParakeetTdt0_6BV2Int8
            | Self::NemoParakeetTdt0_6BV3Int8 => precision == AsrPrecision::Int8,
        }
    }

    pub fn default_precision(self) -> AsrPrecision {
        match self {
            Self::ReazonSpeechK2V2 => AsrPrecision::Int8Float32,
            Self::NemoParakeetTdtCtc0_6BJa35000Int8
            | Self::NemoParakeetTdt0_6BV2Int8
            | Self::NemoParakeetTdt0_6BV3Int8 => AsrPrecision::Int8,
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
    Morph,
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

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorClass {
    Simple,
    Model(TurnDetectorModel),
}

#[cfg(test)]
impl TurnDetectorClass {
    pub fn model(self) -> Option<TurnDetectorModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::Simple => None,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorModel {
    Namo,
}

impl TurnDetector {
    #[cfg(test)]
    pub fn class(self) -> TurnDetectorClass {
        match self {
            Self::Simple | Self::Morph => TurnDetectorClass::Simple,
            Self::Namo => TurnDetectorClass::Model(TurnDetectorModel::Namo),
        }
    }

    pub fn uses_namo_model(self) -> bool {
        match self {
            Self::Namo => true,
            Self::Simple | Self::Morph => false,
        }
    }

    pub fn uses_morph_boundary(self) -> bool {
        matches!(self, Self::Namo | Self::Morph)
    }

    pub fn confirms_normal_end_with_namo(self) -> bool {
        matches!(self, Self::Namo)
    }

    pub fn uses_deferred_turn_completion(self) -> bool {
        !matches!(self, Self::Simple)
    }

    pub fn can_connect_interim_after_completion(self) -> bool {
        match self {
            // Simple は ASR 後に TD / grammar split で安全に戻せないので、
            // completion と interim を別々の ASR request として扱う。
            Self::Simple => false,
            // Namo / Morph は VAD 完了を確定境界にせず、
            // 後続 interim まで含めて TD / grammar boundary に判断させる。
            Self::Morph | Self::Namo => true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ParapperConfig {
    #[serde(flatten)]
    pub neo: NeoConfig,
    #[serde(flatten)]
    pub input: InputConfig,
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
    #[serde(rename = "neo_send_timing")]
    pub send_timing: NeoSendTiming,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InputConfig {
    #[serde(rename = "input_device_id")]
    pub device_id: Option<String>,
    #[serde(rename = "input_device_host")]
    pub device_host: Option<String>,
    #[serde(rename = "input_device_name")]
    pub device_name: Option<String>,
    #[serde(rename = "input_volume_db")]
    pub volume_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AsrConfig {
    #[serde(rename = "asr_language")]
    pub language: AsrLanguage,
    #[serde(rename = "asr_model")]
    pub model: AsrModel,
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
    #[serde(rename = "translation_plugin_http_port")]
    pub plugin_http_port: u16,
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

impl Default for NeoConfig {
    fn default() -> Self {
        Self {
            http_enabled: true,
            http_port: 15520,
            send_timing: NeoSendTiming::Interim,
        }
    }
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            device_host: None,
            device_name: None,
            volume_db: 0.0,
        }
    }
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            language: AsrLanguage::Japanese,
            model: AsrModel::ReazonSpeechK2V2,
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
            plugin_http_port: 8080,
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
        if self.asr.multilingual_enabled {
            self.asr.enabled_models.clone()
        } else {
            vec![self.asr.model]
        }
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
        if self.asr.model.language() != self.asr.language {
            self.asr.model = AsrModel::default_for_language(self.asr.language);
        }
        if !self.asr.model.supports_precision(self.asr.precision) {
            self.asr.precision = self.asr.model.default_precision();
        }
        normalize_enabled_asr_models(&mut self.asr.enabled_models);
        if !self.asr.enabled_models.contains(&self.asr.model) {
            self.asr.enabled_models.push(self.asr.model);
        }
        self.translation.mappings = normalize_translation_mappings(self.translation.mappings);
        self.speech.mappings = normalize_speech_mappings(self.speech.mappings);
        self.asr.num_threads = self.asr.num_threads.max(0);
        self = self.normalized_for_platform();
        self
    }

    fn normalized_for_platform(mut self) -> Self {
        if !Self::neo_http_supported() {
            self.neo.http_enabled = false;
            self.translation.enabled = false;
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
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 => 1,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => 2,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => 3,
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
        assert_eq!(ParapperConfig::default().neo.http_port, 15520);
    }

    #[test]
    fn default_config_sends_text_to_neo() {
        #[cfg(not(target_os = "macos"))]
        assert!(ParapperConfig::default().neo.http_enabled);
        #[cfg(target_os = "macos")]
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
        assert_eq!(object["asr_language"], serde_json::json!("english"));
        assert_eq!(
            object["asr_model"],
            serde_json::json!("nemo_parakeet_tdt_0_6b_v2_int8")
        );
        assert_eq!(object["translation_enabled"], serde_json::json!(true));
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
        #[cfg(not(target_os = "macos"))]
        assert!(config.translation.enabled);
        #[cfg(target_os = "macos")]
        assert!(!config.translation.enabled);
        assert_eq!(config.translation.plugin_http_port, 18080);
        assert_eq!(config.translation.send_timing, NeoSendTiming::Interim);
        assert_eq!(config.translation.mappings[0].id, "translate-en");
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
        assert_eq!(config.translation.plugin_http_port, 8080);
        assert_eq!(config.translation.send_timing, NeoSendTiming::Final);
        assert!(config.translation.mappings.is_empty());
        assert!(config.speech.mappings.is_empty());
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
                    target_lang: " en_US ".to_string(),
                },
                TranslationMapping {
                    id: "empty-target".to_string(),
                    source_asr_model: None,
                    target_lang: " ".to_string(),
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
        assert_eq!(config.translation.mappings[0].target_lang, "en_US");
        assert_eq!(config.speech.mappings.len(), 1);
        assert_eq!(config.speech.mappings[0].id, "speech-ja");
        assert_eq!(config.speech.mappings[0].talker, "ずんだもん/VOICEVOX");
        assert!((config.speech.mappings[0].volume + 20.0).abs() < f32::EPSILON);
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
