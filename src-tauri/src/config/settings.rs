use std::{
    collections::HashSet,
    fs,
    net::{IpAddr, SocketAddr},
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, Serializer};
use unicode_normalization::UnicodeNormalization;

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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delivery_profiles: Vec<DeliveryProfileConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub http_delivery_profiles: Vec<HttpDeliveryProfileConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stt_profiles: Vec<SttProfileConfig>,
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
    #[serde(rename = "input_muted", default, skip_serializing_if = "is_false")]
    pub muted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_endpoint: Option<CaptureEndpointConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recognition_sources: Vec<RecognitionSourceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaptureEndpointConfig {
    pub id: String,
    pub device_host: String,
    pub device_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecognitionSourceConfig {
    pub source_id: String,
    pub speaker_label: String,
    pub capture_endpoint_id: String,
    pub channel_index: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asr_route_policy: Option<AsrRoutePolicyConfig>,
}

/// The fixed palette used to visually identify an STT profile in the UI.
///
/// This is intentionally an enum rather than an arbitrary color string so a
/// malformed persisted config cannot be rendered as an unexpected CSS value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SttProfileDisplayColor {
    #[default]
    Green,
    Blue,
    Violet,
    Red,
    Orange,
    Yellow,
    White,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttProfileConfig {
    pub id: String,
    pub name: String,
    /// Disabled profiles remain stored for later use but are not started.
    pub enabled: bool,
    /// Whether this profile contributes recognized text to the YNC NEO text input.
    pub neo_http_enabled: bool,
    /// Whether this profile contributes events to the global Developer HTTP sink.
    pub developer_http_enabled: bool,
    pub display_color: SttProfileDisplayColor,
    pub input: SttProfileInputConfig,
    pub noise_cancellation: NoiseCancellationConfig,
    pub segmentation: SegmentationConfig,
    pub turn: TurnConfig,
    pub asr: AsrConfig,
    pub delivery_profile_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SttProfileConfigWire {
    id: String,
    name: String,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_true")]
    neo_http_enabled: bool,
    #[serde(default = "default_true")]
    developer_http_enabled: bool,
    #[serde(default)]
    display_color: SttProfileDisplayColor,
    input: SttProfileInputConfig,
    noise_cancellation: SttProfileNoiseCancellationWire,
    segmentation: SttProfileSegmentationWire,
    turn: SttProfileTurnWire,
    asr: SttProfileAsrWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    delivery_profile_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
struct SttProfileNoiseCancellationWire {
    enabled: bool,
    model: NoiseCancellationModel,
    target: NoiseCancellationTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct SttProfileSegmentationWire {
    vad_threshold: f32,
    vad_interval_ms: u32,
    segment_start_speech_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct SttProfileTurnWire {
    detector: TurnDetector,
    interim_result_enabled: bool,
    interim_result_silence_ms: u32,
    check_silence_ms: u32,
    namo_confidence_threshold: f32,
    namo_context_max_tokens: u32,
    rerecognize_full_on_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct SttProfileAsrWire {
    language: AsrLanguage,
    model: AsrModel,
    interim_model: Option<AsrModel>,
    precision: AsrPrecision,
    num_threads: i32,
    mode: AsrMode,
    hotwords_enabled: bool,
    hotwords: Vec<AsrHotword>,
    normalize_input_audio: bool,
    multilingual_enabled: bool,
    enabled_models: Vec<AsrModel>,
    #[serde(default)]
    runtime_profiles: Vec<AsrRuntimeProfileConfig>,
}

impl Serialize for SttProfileConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SttProfileConfigWire {
            id: self.id.clone(),
            name: self.name.clone(),
            enabled: self.enabled,
            neo_http_enabled: self.neo_http_enabled,
            developer_http_enabled: self.developer_http_enabled,
            display_color: self.display_color,
            input: self.input.clone(),
            noise_cancellation: SttProfileNoiseCancellationWire {
                enabled: self.noise_cancellation.enabled,
                model: self.noise_cancellation.model,
                target: self.noise_cancellation.target,
            },
            segmentation: SttProfileSegmentationWire {
                vad_threshold: self.segmentation.vad_threshold,
                vad_interval_ms: self.segmentation.vad_interval_ms,
                segment_start_speech_ms: self.segmentation.segment_start_speech_ms,
            },
            turn: SttProfileTurnWire {
                detector: self.turn.detector,
                interim_result_enabled: self.turn.interim_result_enabled,
                interim_result_silence_ms: self.turn.interim_result_silence_ms,
                check_silence_ms: self.turn.check_silence_ms,
                namo_confidence_threshold: self.turn.namo_confidence_threshold,
                namo_context_max_tokens: self.turn.namo_context_max_tokens,
                rerecognize_full_on_complete: self.turn.rerecognize_full_on_complete,
            },
            asr: SttProfileAsrWire {
                language: self.asr.language,
                model: self.asr.model,
                interim_model: self.asr.interim_model,
                precision: self.asr.precision,
                num_threads: self.asr.num_threads,
                mode: self.asr.mode,
                hotwords_enabled: self.asr.hotwords_enabled,
                hotwords: self.asr.hotwords.clone(),
                normalize_input_audio: self.asr.normalize_input_audio,
                multilingual_enabled: self.asr.multilingual_enabled,
                enabled_models: self.asr.enabled_models.clone(),
                runtime_profiles: self.asr.runtime_profiles.clone(),
            },
            delivery_profile_id: self.delivery_profile_id.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SttProfileConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = SttProfileConfigWire::deserialize(deserializer)?;
        Ok(Self {
            id: wire.id,
            name: wire.name,
            enabled: wire.enabled,
            neo_http_enabled: wire.neo_http_enabled,
            developer_http_enabled: wire.developer_http_enabled,
            display_color: wire.display_color,
            input: wire.input,
            noise_cancellation: NoiseCancellationConfig {
                enabled: wire.noise_cancellation.enabled,
                model: wire.noise_cancellation.model,
                target: wire.noise_cancellation.target,
            },
            segmentation: SegmentationConfig {
                vad_threshold: wire.segmentation.vad_threshold,
                vad_interval_ms: wire.segmentation.vad_interval_ms,
                segment_start_speech_ms: wire.segmentation.segment_start_speech_ms,
            },
            turn: TurnConfig {
                detector: wire.turn.detector,
                interim_result_enabled: wire.turn.interim_result_enabled,
                interim_result_silence_ms: wire.turn.interim_result_silence_ms,
                check_silence_ms: wire.turn.check_silence_ms,
                namo_confidence_threshold: wire.turn.namo_confidence_threshold,
                namo_context_max_tokens: wire.turn.namo_context_max_tokens,
                rerecognize_full_on_complete: wire.turn.rerecognize_full_on_complete,
            },
            asr: AsrConfig {
                language: wire.asr.language,
                model: wire.asr.model,
                interim_model: wire.asr.interim_model,
                precision: wire.asr.precision,
                num_threads: wire.asr.num_threads,
                mode: wire.asr.mode,
                legacy_beam_search_enabled: None,
                hotwords_enabled: wire.asr.hotwords_enabled,
                hotwords: wire.asr.hotwords,
                normalize_input_audio: wire.asr.normalize_input_audio,
                multilingual_enabled: wire.asr.multilingual_enabled,
                enabled_models: wire.asr.enabled_models,
                runtime_profiles: wire.asr.runtime_profiles,
            },
            delivery_profile_id: wire.delivery_profile_id,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SttProfileInputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    pub channel_index: u16,
    pub volume_percent: u8,
    #[serde(default)]
    pub muted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveryProfileConfig {
    pub id: String,
    pub gui_enabled: bool,
    #[serde(default)]
    pub translation_mapping_ids: Vec<String>,
    #[serde(default)]
    pub speech_mapping_ids: Vec<String>,
    #[serde(default)]
    pub http_profile_ids: Vec<String>,
    pub neo_text_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HttpDeliveryProfileConfig {
    pub id: String,
    pub url: String,
    pub payload_format: HttpPayloadFormat,
    pub artifact_kinds: Vec<HttpArtifactKind>,
    pub send_timing: NeoSendTiming,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HttpPayloadFormat {
    TextEventV1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum HttpArtifactKind {
    Recognition,
    Translation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryRouteSnapshot {
    pub profile_id: String,
    pub gui_enabled: bool,
    pub translation_mapping_ids: Vec<String>,
    pub speech_mapping_ids: Vec<String>,
    pub http_profiles: Vec<HttpDeliveryProfileConfig>,
    pub neo_text_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsrRoutePolicyConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interim_runtime_id: Option<String>,
    pub completion_runtime_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsrRuntimeProfileConfig {
    pub id: String,
    pub model: AsrModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedAsrRoutePolicy {
    pub interim_model: Option<AsrModel>,
    pub completion_model: AsrModel,
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
    #[serde(rename = "asr_mode", default = "missing_asr_mode")]
    pub mode: AsrMode,
    /// Read-only migration input for v0.4 configurations.  The normalized
    /// config translates `true` to [`AsrMode::Accurate`] and always clears
    /// this field, so all newly written configs use `asr_mode` exclusively.
    #[serde(rename = "asr_beam_search_enabled", default, skip_serializing)]
    legacy_beam_search_enabled: Option<bool>,
    #[serde(rename = "asr_hotwords_enabled")]
    pub hotwords_enabled: bool,
    /// Persisted context-bias entries used by `ReazonSpeech` beam decoding.
    ///
    /// The entries intentionally remain in the application configuration even
    /// when beam search is disabled.  The recognition model factory applies
    /// them only when the user opted into the Reazon beam mode.
    #[serde(rename = "asr_hotwords")]
    pub hotwords: Vec<AsrHotword>,
    #[serde(rename = "asr_normalize_input_audio")]
    pub normalize_input_audio: bool,
    #[serde(rename = "multilingual_asr_enabled")]
    pub multilingual_enabled: bool,
    #[serde(rename = "enabled_asr_models")]
    pub enabled_models: Vec<AsrModel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_profiles: Vec<AsrRuntimeProfileConfig>,
}

/// User-facing decoding preset.  Model-specific decoder tuning remains in
/// the host; this persisted value intentionally exposes no beam/gate knobs.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum AsrMode {
    #[default]
    Fast,
    Accurate,
    /// Internal deserialization sentinel used only before configuration
    /// normalization. It lets the v0.4 beam flag migrate old files without
    /// overriding an explicit v0.5 `asr_mode` value in mixed-version files.
    #[doc(hidden)]
    #[serde(skip)]
    Missing,
}

const fn missing_asr_mode() -> AsrMode {
    AsrMode::Missing
}

impl Serialize for AsrMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Accurate => serializer.serialize_str("accurate"),
            // A direct serde caller may serialize an old config before the
            // app's normalization boundary. Keep that write valid and safe;
            // normal loading still promotes legacy `true` to Accurate first.
            Self::Fast | Self::Missing => serializer.serialize_str("fast"),
        }
    }
}

/// A user-managed ASR context-bias entry.
///
/// `surface` is the text shown in the transcript.  `readings` contains
/// optional spoken forms; they are stored canonically as hiragana so that a
/// user may enter either hiragana or katakana without creating duplicate
/// graph branches. `score` is an optional positive phrase-level likelihood
/// multiplier that overrides the product default (x100), not a raw log
/// score. The model backend converts this DTO into its own tokenized graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrHotword {
    pub surface: String,
    #[serde(default)]
    pub readings: Vec<String>,
    #[serde(default)]
    pub score: Option<f32>,
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
    #[serde(
        rename = "noise_cancellation_target",
        default = "legacy_noise_cancellation_target"
    )]
    pub target: NoiseCancellationTarget,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NoiseCancellationTarget {
    #[default]
    VadOnly,
    VadAndAsr,
}

const fn legacy_noise_cancellation_target() -> NoiseCancellationTarget {
    NoiseCancellationTarget::VadAndAsr
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
            muted: false,
            capture_endpoint: None,
            recognition_sources: Vec::new(),
        }
    }
}

impl InputConfig {
    fn validate(&self) -> Result<()> {
        let Some(endpoint) = self.capture_endpoint.as_ref() else {
            if self.recognition_sources.is_empty() {
                if let Some(value) = self.device_id.as_deref() {
                    validate_non_empty("legacy input device id", value)?;
                }
                if let Some(value) = self.device_host.as_deref() {
                    validate_non_empty("legacy input device host", value)?;
                }
                if let Some(value) = self.device_name.as_deref() {
                    validate_non_empty("legacy input device name", value)?;
                }
                return Ok(());
            }
            anyhow::bail!("recognition_sources require an explicit capture_endpoint");
        };

        if self.device_id.as_deref().is_some()
            || self.device_host.as_deref().is_some()
            || self.device_name.as_deref().is_some()
        {
            anyhow::bail!(
                "explicit capture_endpoint cannot be mixed with legacy input device fields"
            );
        }

        validate_non_empty("capture endpoint id", &endpoint.id)?;
        validate_non_empty("capture endpoint device host", &endpoint.device_host)?;
        validate_non_empty("capture endpoint device id", &endpoint.device_id)?;
        if let Some(name) = endpoint.device_name.as_deref() {
            validate_non_empty("capture endpoint device name", name)?;
        }
        if self.source_kind == InputSourceKind::WebSocket {
            anyhow::bail!("explicit capture endpoints do not support WebSocket input yet");
        }
        if self.recognition_sources.is_empty() {
            anyhow::bail!("explicit capture_endpoint requires recognition_sources");
        }

        let mut source_ids = HashSet::new();
        let mut channel_indexes = HashSet::new();
        for source in &self.recognition_sources {
            validate_non_empty("recognition source id", &source.source_id)?;
            validate_non_empty("recognition source speaker label", &source.speaker_label)?;
            validate_non_empty(
                "recognition source capture endpoint id",
                &source.capture_endpoint_id,
            )?;
            if source.capture_endpoint_id.trim() != endpoint.id.trim() {
                anyhow::bail!(
                    "recognition source capture endpoint id does not match capture endpoint"
                );
            }
            if !source_ids.insert(source.source_id.trim()) {
                anyhow::bail!("recognition source ids must be unique");
            }
            if !channel_indexes.insert(source.channel_index) {
                anyhow::bail!("recognition source channel indexes must be unique");
            }
        }
        Ok(())
    }

    fn normalize_capture_sources(&mut self) {
        self.device_id = self.device_id.take().map(|value| value.trim().to_string());
        self.device_host = self
            .device_host
            .take()
            .map(|value| value.trim().to_string());
        self.device_name = self
            .device_name
            .take()
            .map(|value| normalize_input_display_name(&value));

        if let Some(endpoint) = self.capture_endpoint.as_mut() {
            endpoint.id = endpoint.id.trim().to_string();
            endpoint.device_host = endpoint.device_host.trim().to_string();
            endpoint.device_id = endpoint.device_id.trim().to_string();
            endpoint.device_name = endpoint
                .device_name
                .take()
                .map(|value| normalize_input_display_name(&value));
        }
        for source in &mut self.recognition_sources {
            source.source_id = source.source_id.trim().to_string();
            source.capture_endpoint_id = source.capture_endpoint_id.trim().to_string();
            let label = std::mem::take(&mut source.speaker_label);
            source.speaker_label = normalize_input_display_name(&label);
            source.delivery_profile_id = source
                .delivery_profile_id
                .take()
                .map(|id| id.trim().to_string());
            if let Some(route) = source.asr_route_policy.as_mut() {
                route.completion_runtime_id = route.completion_runtime_id.trim().to_string();
                route.interim_runtime_id = route
                    .interim_runtime_id
                    .take()
                    .map(|id| id.trim().to_string());
            }
        }
    }
}

fn validate_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{field} must not be empty");
    }
    Ok(())
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if predicates receive a shared reference"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

fn default_true() -> bool {
    true
}

impl Default for SttProfileInputConfig {
    fn default() -> Self {
        Self {
            device_host: None,
            device_id: None,
            device_name: None,
            channel_index: 0,
            volume_percent: 100,
            muted: false,
        }
    }
}

impl Default for SttProfileNoiseCancellationWire {
    fn default() -> Self {
        let config = NoiseCancellationConfig::default();
        Self {
            enabled: config.enabled,
            model: config.model,
            target: config.target,
        }
    }
}

impl Default for SttProfileSegmentationWire {
    fn default() -> Self {
        let config = SegmentationConfig::default();
        Self {
            vad_threshold: config.vad_threshold,
            vad_interval_ms: config.vad_interval_ms,
            segment_start_speech_ms: config.segment_start_speech_ms,
        }
    }
}

impl Default for SttProfileTurnWire {
    fn default() -> Self {
        let config = TurnConfig::default();
        Self {
            detector: config.detector,
            interim_result_enabled: config.interim_result_enabled,
            interim_result_silence_ms: config.interim_result_silence_ms,
            check_silence_ms: config.check_silence_ms,
            namo_confidence_threshold: config.namo_confidence_threshold,
            namo_context_max_tokens: config.namo_context_max_tokens,
            rerecognize_full_on_complete: config.rerecognize_full_on_complete,
        }
    }
}

impl Default for SttProfileAsrWire {
    fn default() -> Self {
        let config = AsrConfig::default();
        Self {
            language: config.language,
            model: config.model,
            interim_model: config.interim_model,
            precision: config.precision,
            num_threads: config.num_threads,
            mode: config.mode,
            hotwords_enabled: config.hotwords_enabled,
            hotwords: config.hotwords,
            normalize_input_audio: config.normalize_input_audio,
            multilingual_enabled: config.multilingual_enabled,
            enabled_models: config.enabled_models,
            runtime_profiles: config.runtime_profiles,
        }
    }
}

fn normalize_input_display_name(value: &str) -> String {
    value.nfkc().collect::<String>().trim().to_string()
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
            mode: AsrMode::Fast,
            legacy_beam_search_enabled: None,
            hotwords_enabled: false,
            hotwords: Vec::new(),
            normalize_input_audio: true,
            multilingual_enabled: false,
            enabled_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            runtime_profiles: Vec::new(),
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
            target: NoiseCancellationTarget::VadOnly,
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
            delivery_profiles: Vec::new(),
            http_delivery_profiles: Vec::new(),
            stt_profiles: Vec::new(),
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
        if !self.stt_profiles.is_empty() {
            let mut models = Vec::new();
            for profile in self.stt_profiles.iter().filter(|profile| profile.enabled) {
                for model in stt_profile_required_asr_models(profile) {
                    push_unique_asr_model(&mut models, Some(model));
                }
            }
            return models;
        }
        if self.input.capture_endpoint.is_some() {
            let mut models = Vec::new();
            for profile in &self.asr.runtime_profiles {
                push_unique_asr_model(&mut models, Some(profile.model));
            }
            return models;
        }
        let mut models = if self.asr.multilingual_enabled {
            self.asr.enabled_models.clone()
        } else {
            vec![self.asr.model]
        };
        push_unique_asr_model(&mut models, self.asr.interim_model);
        models
    }

    pub fn resolved_stt_profile(&self, profile_id: &str) -> Result<&SttProfileConfig> {
        self.stt_profiles
            .iter()
            .find(|profile| profile.id == profile_id)
            .with_context(|| format!("STT profile {profile_id:?} is not configured"))
    }

    pub fn developer_http_enabled_for_source(&self, source_id: &str) -> bool {
        self.stt_profiles
            .iter()
            .find(|profile| profile.id == source_id)
            .is_none_or(|profile| profile.developer_http_enabled)
    }

    pub fn neo_http_enabled_for_source(&self, source_id: &str) -> bool {
        self.stt_profiles
            .iter()
            .find(|profile| profile.id == source_id)
            .is_none_or(|profile| profile.neo_http_enabled)
    }

    pub fn config_for_stt_profile(&self, profile_id: &str) -> Result<Self> {
        let profile = self.resolved_stt_profile(profile_id)?;
        let mut config = self.clone();
        // The resolved lane is a flat runtime config. Keeping sibling profiles
        // here would make aggregate helpers inspect unrelated lanes.
        config.stt_profiles.clear();
        config.input.source_kind = InputSourceKind::DesktopAudio;
        config
            .input
            .device_host
            .clone_from(&profile.input.device_host);
        config.input.device_id.clone_from(&profile.input.device_id);
        config
            .input
            .device_name
            .clone_from(&profile.input.device_name);
        config.input.volume_db = input_volume_percent_to_db(profile.input.volume_percent);
        config.input.muted = profile.input.muted;
        config.noise_cancellation = profile.noise_cancellation.clone();
        config.segmentation = profile.segmentation.clone();
        config.turn = profile.turn.clone();
        config.asr = profile.asr.clone();
        Ok(config)
    }

    pub fn resolved_delivery_route_for_stt_profile(
        &self,
        profile_id: &str,
    ) -> Result<DeliveryRouteSnapshot> {
        let profile = self.resolved_stt_profile(profile_id)?;
        profile.delivery_profile_id.as_deref().map_or_else(
            || Ok(self.legacy_delivery_route()),
            |id| self.delivery_route_for_profile(id),
        )
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

    pub fn resolved_asr_route_for_source(&self, source_id: &str) -> Result<ResolvedAsrRoutePolicy> {
        let source = self
            .input
            .recognition_sources
            .iter()
            .find(|source| source.source_id == source_id)
            .with_context(|| format!("recognition source {source_id:?} is not configured"))?;
        let policy = source
            .asr_route_policy
            .as_ref()
            .with_context(|| format!("recognition source {source_id:?} has no ASR route policy"))?;
        let completion_model = self
            .asr_runtime_model(&policy.completion_runtime_id)
            .with_context(|| {
                format!(
                    "recognition source {source_id:?} references missing completion runtime {:?}",
                    policy.completion_runtime_id
                )
            })?;
        if !completion_model.supports_completion() {
            anyhow::bail!(
                "recognition source {source_id:?} completion runtime {:?} uses model {completion_model:?}, which does not support completion",
                policy.completion_runtime_id
            );
        }
        let interim_model = policy
            .interim_runtime_id
            .as_deref()
            .map(|runtime_id| {
                self.asr_runtime_model(runtime_id).with_context(|| {
                    format!(
                        "recognition source {source_id:?} references missing interim runtime {runtime_id:?}"
                    )
                })
            })
            .transpose()?;
        Ok(ResolvedAsrRoutePolicy {
            interim_model,
            completion_model,
        })
    }

    /// Produces the per-source session configuration used by the runtime pool
    /// boundary. It never substitutes the global flat model for a
    /// missing runtime reference.
    pub fn with_asr_route_for_source(&self, source_id: &str) -> Result<Self> {
        let route = self.resolved_asr_route_for_source(source_id)?;
        let mut config = self.clone();
        config.asr.model = route.completion_model;
        config.asr.language = route.completion_model.language();
        config.asr.interim_model = route.interim_model;
        config.asr.enabled_models = vec![route.completion_model];
        push_unique_asr_model(&mut config.asr.enabled_models, route.interim_model);
        Ok(config)
    }

    /// Resolves an explicit source's persisted delivery profile. Unlike the
    /// legacy wrapper, this never falls back to flat delivery settings.
    pub fn resolved_delivery_route_for_source(
        &self,
        source_id: &str,
    ) -> Result<DeliveryRouteSnapshot> {
        let source = self
            .input
            .recognition_sources
            .iter()
            .find(|source| source.source_id == source_id)
            .with_context(|| format!("recognition source {source_id:?} is not configured"))?;
        let profile_id = source
            .delivery_profile_id
            .as_deref()
            .with_context(|| format!("recognition source {source_id:?} has no delivery profile"))?;
        self.delivery_route_for_profile(profile_id)
    }

    /// Preserves the flat legacy output contract for the single-source
    /// wrapper. Explicit capture sources must use
    /// [`Self::resolved_delivery_route_for_source`] instead.
    pub fn legacy_delivery_route(&self) -> DeliveryRouteSnapshot {
        DeliveryRouteSnapshot {
            profile_id: "legacy-default".to_owned(),
            gui_enabled: true,
            translation_mapping_ids: unique_configured_ids(
                self.translation
                    .mappings
                    .iter()
                    .map(|mapping| mapping.id.as_str()),
            ),
            speech_mapping_ids: unique_configured_ids(
                self.speech
                    .mappings
                    .iter()
                    .map(|mapping| mapping.id.as_str()),
            ),
            http_profiles: Vec::new(),
            neo_text_enabled: self.neo.http_enabled,
        }
    }

    fn delivery_route_for_profile(&self, profile_id: &str) -> Result<DeliveryRouteSnapshot> {
        let profile = self
            .delivery_profiles
            .iter()
            .find(|profile| profile.id == profile_id)
            .with_context(|| format!("delivery profile {profile_id:?} is not configured"))?;
        let http_profiles = profile
            .http_profile_ids
            .iter()
            .map(|http_profile_id| {
                self.http_delivery_profiles
                    .iter()
                    .find(|profile| profile.id == *http_profile_id)
                    .cloned()
                    .with_context(|| {
                        format!(
                            "delivery profile {profile_id:?} references missing HTTP profile {http_profile_id:?}"
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DeliveryRouteSnapshot {
            profile_id: profile.id.clone(),
            gui_enabled: profile.gui_enabled,
            translation_mapping_ids: profile.translation_mapping_ids.clone(),
            speech_mapping_ids: profile.speech_mapping_ids.clone(),
            http_profiles,
            neo_text_enabled: profile.neo_text_enabled,
        })
    }

    fn asr_runtime_model(&self, runtime_id: &str) -> Option<AsrModel> {
        self.asr
            .runtime_profiles
            .iter()
            .find(|profile| profile.id == runtime_id)
            .map(|profile| profile.model)
    }

    /// Returns whether the selected model and user settings can apply the
    /// persisted context-bias list. The list itself is always preserved,
    /// including while fast mode is active.
    #[must_use]
    pub fn hotwords_enabled(&self) -> bool {
        self.asr.hotwords_enabled
            && self.asr.mode == AsrMode::Accurate
            && supports_accurate_asr_mode(self.asr.model)
            && !self.asr.hotwords.is_empty()
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
        if !self.stt_profiles.is_empty() {
            return self
                .stt_profiles
                .iter()
                .filter(|profile| profile.enabled)
                .any(|profile| profile.turn.detector.uses_namo_model());
        }
        self.turn.detector.uses_namo_model()
    }

    pub fn uses_morph_turn_boundary(&self) -> bool {
        if !self.stt_profiles.is_empty() {
            return self
                .stt_profiles
                .iter()
                .filter(|profile| profile.enabled)
                .any(|profile| profile.turn.detector.uses_morph_boundary());
        }
        self.turn.detector.uses_morph_boundary()
    }

    #[cfg(test)]
    pub fn confirms_normal_end_with_namo(&self) -> bool {
        self.turn.detector.confirms_normal_end_with_namo()
    }

    #[cfg(test)]
    pub fn uses_deferred_turn_completion(&self) -> bool {
        self.turn.detector.uses_deferred_turn_completion()
    }

    #[cfg(test)]
    pub fn can_connect_interim_after_completion(&self) -> bool {
        self.turn.detector.can_connect_interim_after_completion()
    }

    pub fn required_namo_turn_detector_languages(&self) -> Vec<AsrLanguage> {
        if !self.stt_profiles.is_empty() {
            let mut languages = self
                .stt_profiles
                .iter()
                .filter(|profile| profile.enabled)
                .filter(|profile| profile.turn.detector.uses_namo_model())
                .flat_map(stt_profile_required_asr_models)
                .map(AsrModel::language)
                .collect::<Vec<_>>();
            normalize_asr_languages(&mut languages);
            return languages;
        }
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
        if !self.stt_profiles.is_empty() {
            return self
                .stt_profiles
                .iter()
                .filter(|profile| profile.enabled)
                .filter(|profile| profile.turn.detector.uses_morph_boundary())
                .flat_map(stt_profile_required_asr_models)
                .any(|model| model.language() == AsrLanguage::Japanese);
        }
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
        let explicit_schema_requested = match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(serde_json::Value::Object(object)) => {
                object.contains_key("capture_endpoint")
                    || object.contains_key("recognition_sources")
                    || object.get("stt_profiles").is_some_and(|profiles| {
                        profiles
                            .as_array()
                            .is_none_or(|profiles| !profiles.is_empty())
                    })
            }
            Ok(_) | Err(_) => false,
        };
        match serde_json::from_str::<Self>(&content) {
            Ok(config) => {
                let config = config.normalized();
                if explicit_schema_requested {
                    if config.input.capture_endpoint.is_none()
                        && config.input.recognition_sources.is_empty()
                        && config.stt_profiles.is_empty()
                    {
                        return Err(anyhow::anyhow!(
                            "explicit capture config must include capture_endpoint and recognition_sources"
                        ))
                        .with_context(|| {
                            format!("Invalid explicit capture config: {}", path.display())
                        });
                    }
                    config.validate().with_context(|| {
                        format!("Invalid explicit capture config: {}", path.display())
                    })?;
                }
                Ok(config)
            }
            Err(err) if explicit_schema_requested => Err(err).with_context(|| {
                format!(
                    "Failed to parse explicit capture config: {}",
                    path.display()
                )
            }),
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
        self.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config dir: {}", parent.display()))?;
        }

        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)
            .with_context(|| format!("Failed to write config: {}", path.display()))
    }

    /// Validates user-authored values that cannot be repaired without
    /// changing the intended context-bias rule.  Callers that accept config
    /// updates should run this before `normalized()` so the UI receives an
    /// actionable error instead of silently losing an entry.
    pub fn validate(&self) -> Result<()> {
        self.validate_stt_profiles()?;
        self.input.validate()?;
        self.validate_asr_runtime_profiles_and_routes()?;
        self.validate_delivery_profiles()?;
        validate_asr_hotwords(&self.asr.hotwords)
    }

    fn validate_stt_profiles(&self) -> Result<()> {
        if self.stt_profiles.is_empty() {
            return Ok(());
        }
        if self.input.capture_endpoint.is_some() || !self.input.recognition_sources.is_empty() {
            bail!("stt_profiles cannot be mixed with explicit capture");
        }
        if self.input.source_kind != InputSourceKind::DesktopAudio {
            bail!("stt_profiles require desktop audio input");
        }
        if !self.stt_profiles.iter().any(|profile| profile.enabled) {
            bail!("STT profile mode requires at least one enabled STT profile");
        }

        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        let mut device_channels = HashSet::new();
        for profile in &self.stt_profiles {
            validate_non_empty("STT profile id", &profile.id)?;
            validate_non_empty("STT profile name", &profile.name)?;
            if !ids.insert(profile.id.trim()) {
                bail!("STT profile ids must be unique");
            }
            if !names.insert(profile.name.trim()) {
                bail!("STT profile names must be unique");
            }
            if profile.input.volume_percent > 100 {
                bail!(
                    "STT profile {:?} volume percent must be between 0 and 100",
                    profile.id
                );
            }
            match (
                profile.input.device_host.as_deref(),
                profile.input.device_id.as_deref(),
            ) {
                (None, None) if self.stt_profiles.len() == 1 => {
                    if profile.input.channel_index != 0 {
                        bail!("a device-less STT profile must use channel index 0");
                    }
                }
                (Some(host), Some(device)) => {
                    validate_non_empty("STT profile device host", host)?;
                    validate_non_empty("STT profile device id", device)?;
                    if let Some(name) = profile.input.device_name.as_deref() {
                        validate_non_empty("STT profile device name", name)?;
                    }
                    if !device_channels.insert((
                        host.trim(),
                        device.trim(),
                        profile.input.channel_index,
                    )) {
                        bail!("STT profile device/channel combination must be unique");
                    }
                }
                (None, None) => {
                    bail!("multiple STT profiles require device host and id");
                }
                _ => {
                    bail!("STT profile device host and id must be provided together");
                }
            }
            if !profile.asr.runtime_profiles.is_empty() {
                bail!(
                    "STT profile {:?} must not define nested ASR runtime profiles",
                    profile.id
                );
            }
            validate_asr_hotwords(&profile.asr.hotwords)?;
            if let Some(delivery_profile_id) = profile.delivery_profile_id.as_deref() {
                validate_non_empty("STT profile delivery profile id", delivery_profile_id)?;
                if !self
                    .delivery_profiles
                    .iter()
                    .any(|delivery| delivery.id == delivery_profile_id)
                {
                    bail!(
                        "STT profile {:?} references unknown delivery profile {delivery_profile_id:?}",
                        profile.id
                    );
                }
            }
        }
        Ok(())
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
        self.input.normalize_capture_sources();
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
        let legacy_beam_search_enabled = self.asr.legacy_beam_search_enabled.take();
        if self.asr.mode == AsrMode::Missing {
            self.asr.mode = if legacy_beam_search_enabled == Some(true) {
                AsrMode::Accurate
            } else {
                AsrMode::Fast
            };
        }
        if !supports_accurate_asr_mode(self.asr.model) {
            self.asr.mode = AsrMode::Fast;
        }
        self.asr.interim_model =
            normalize_interim_asr_model(self.asr.model, self.asr.interim_model);
        normalize_asr_runtime_profiles(&mut self.asr.runtime_profiles);
        self.migrate_explicit_asr_routes_from_flat_config();
        normalize_enabled_asr_models(&mut self.asr.enabled_models);
        self.asr.hotwords = normalize_asr_hotwords(std::mem::take(&mut self.asr.hotwords));
        if !self.asr.enabled_models.contains(&self.asr.model) {
            self.asr.enabled_models.push(self.asr.model);
        }
        self.translation.mappings = normalize_translation_mappings(self.translation.mappings);
        if !self.translation.local_server_model.is_available() {
            self.translation.local_server_model = LocalTranslationModel::default();
        }
        self.speech.mappings = normalize_speech_mappings(self.speech.mappings);
        normalize_delivery_profiles(&mut self.delivery_profiles);
        normalize_http_delivery_profiles(&mut self.http_delivery_profiles);
        normalize_stt_profiles(&mut self.stt_profiles);
        self.asr.num_threads = self.asr.num_threads.max(0);
        self = self.normalized_for_platform();
        self.migrate_explicit_delivery_profiles_from_legacy();
        self
    }

    fn migrate_explicit_asr_routes_from_flat_config(&mut self) {
        if self.input.capture_endpoint.is_none()
            || self.input.recognition_sources.is_empty()
            || self
                .input
                .recognition_sources
                .iter()
                .any(|source| source.asr_route_policy.is_some())
            || !self.asr.runtime_profiles.is_empty()
        {
            return;
        }
        let completion_runtime_id = ensure_runtime_profile(
            &mut self.asr.runtime_profiles,
            "legacy-completion",
            self.asr.model,
        );
        let interim_runtime_id = self.asr.interim_model.map(|model| {
            ensure_runtime_profile(&mut self.asr.runtime_profiles, "legacy-interim", model)
        });
        for source in &mut self.input.recognition_sources {
            if source.asr_route_policy.is_none() {
                source.asr_route_policy = Some(AsrRoutePolicyConfig {
                    interim_runtime_id: interim_runtime_id.clone(),
                    completion_runtime_id: completion_runtime_id.clone(),
                });
            }
        }
    }

    fn migrate_explicit_delivery_profiles_from_legacy(&mut self) {
        if self.input.capture_endpoint.is_none()
            || self.input.recognition_sources.is_empty()
            || self
                .input
                .recognition_sources
                .iter()
                .any(|source| source.delivery_profile_id.is_some())
            || !self.delivery_profiles.is_empty()
            || !self.http_delivery_profiles.is_empty()
        {
            return;
        }
        let legacy = self.legacy_delivery_route();
        self.delivery_profiles.push(DeliveryProfileConfig {
            id: legacy.profile_id,
            gui_enabled: legacy.gui_enabled,
            translation_mapping_ids: legacy.translation_mapping_ids,
            speech_mapping_ids: legacy.speech_mapping_ids,
            http_profile_ids: Vec::new(),
            neo_text_enabled: legacy.neo_text_enabled,
        });
        for source in &mut self.input.recognition_sources {
            source.delivery_profile_id = Some("legacy-default".to_owned());
        }
    }

    fn validate_asr_runtime_profiles_and_routes(&self) -> Result<()> {
        if self.input.capture_endpoint.is_none() {
            return Ok(());
        }
        let mut ids = HashSet::new();
        for profile in &self.asr.runtime_profiles {
            validate_non_empty("ASR runtime profile id", &profile.id)?;
            if !ids.insert(profile.id.as_str()) {
                anyhow::bail!("ASR runtime profile ids must be unique");
            }
        }
        for source in &self.input.recognition_sources {
            let route = source.asr_route_policy.as_ref().with_context(|| {
                format!(
                    "recognition source {:?} requires an ASR route policy",
                    source.source_id
                )
            })?;
            validate_non_empty("ASR completion runtime id", &route.completion_runtime_id)?;
            if let Some(interim_id) = route.interim_runtime_id.as_deref() {
                validate_non_empty("ASR interim runtime id", interim_id)?;
                if self.asr_runtime_model(interim_id).is_none() {
                    anyhow::bail!(
                        "recognition source {:?} references unknown interim ASR runtime {interim_id:?}",
                        source.source_id
                    );
                }
            }
            let completion = self
                .asr_runtime_model(&route.completion_runtime_id)
                .with_context(|| {
                    format!(
                        "recognition source {:?} references unknown completion ASR runtime {:?}",
                        source.source_id, route.completion_runtime_id
                    )
                })?;
            if !completion.supports_completion() {
                anyhow::bail!(
                    "recognition source {:?} completion ASR runtime {:?} uses {completion:?}, which does not support completion",
                    source.source_id,
                    route.completion_runtime_id
                );
            }
        }
        Ok(())
    }

    fn validate_delivery_profiles(&self) -> Result<()> {
        if self.input.capture_endpoint.is_none() && self.stt_profiles.is_empty() {
            return Ok(());
        }
        let translation_mapping_ids = self
            .translation
            .mappings
            .iter()
            .map(|mapping| mapping.id.as_str())
            .collect::<HashSet<_>>();
        let speech_mapping_ids = self
            .speech
            .mappings
            .iter()
            .map(|mapping| mapping.id.as_str())
            .collect::<HashSet<_>>();
        let mut profile_ids = HashSet::new();
        for profile in &self.delivery_profiles {
            validate_non_empty("delivery profile id", &profile.id)?;
            if !profile_ids.insert(profile.id.as_str()) {
                bail!("delivery profile ids must be unique");
            }
            for mapping_id in &profile.translation_mapping_ids {
                validate_non_empty("delivery profile translation mapping id", mapping_id)?;
                if !translation_mapping_ids.contains(mapping_id.as_str()) {
                    bail!(
                        "delivery profile {:?} references unknown translation mapping {mapping_id:?}",
                        profile.id
                    );
                }
            }
            for mapping_id in &profile.speech_mapping_ids {
                validate_non_empty("delivery profile speech mapping id", mapping_id)?;
                if !speech_mapping_ids.contains(mapping_id.as_str()) {
                    bail!(
                        "delivery profile {:?} references unknown speech mapping {mapping_id:?}",
                        profile.id
                    );
                }
            }
            for http_id in &profile.http_profile_ids {
                validate_non_empty("delivery profile HTTP profile id", http_id)?;
            }
        }
        let http_profile_ids = self.validated_http_delivery_profile_ids()?;
        for profile in &self.delivery_profiles {
            for http_id in &profile.http_profile_ids {
                if !http_profile_ids.contains(http_id.as_str()) {
                    bail!(
                        "delivery profile {:?} references unknown HTTP profile {http_id:?}",
                        profile.id
                    );
                }
            }
        }
        for source in &self.input.recognition_sources {
            let profile_id = source.delivery_profile_id.as_deref().with_context(|| {
                format!(
                    "recognition source {:?} requires a delivery profile",
                    source.source_id
                )
            })?;
            validate_non_empty("recognition source delivery profile id", profile_id)?;
            if !profile_ids.contains(profile_id) {
                bail!(
                    "recognition source {:?} references unknown delivery profile {profile_id:?}",
                    source.source_id
                );
            }
        }
        for profile in &self.stt_profiles {
            if let Some(delivery_profile_id) = profile.delivery_profile_id.as_deref()
                && !profile_ids.contains(delivery_profile_id)
            {
                bail!(
                    "STT profile {:?} references unknown delivery profile {delivery_profile_id:?}",
                    profile.id
                );
            }
        }
        Ok(())
    }

    fn validated_http_delivery_profile_ids(&self) -> Result<HashSet<&str>> {
        let mut http_profile_ids = HashSet::new();
        for profile in &self.http_delivery_profiles {
            validate_non_empty("HTTP delivery profile id", &profile.id)?;
            if !http_profile_ids.insert(profile.id.as_str()) {
                bail!("HTTP delivery profile ids must be unique");
            }
            let url = reqwest::Url::parse(&profile.url).with_context(|| {
                format!("HTTP delivery profile {:?} has invalid URL", profile.id)
            })?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                bail!(
                    "HTTP delivery profile {:?} must use an HTTP(S) URL",
                    profile.id
                );
            }
            if profile.artifact_kinds.is_empty() {
                bail!(
                    "HTTP delivery profile {:?} must select at least one artifact kind",
                    profile.id
                );
            }
        }
        Ok(http_profile_ids)
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

fn normalize_stt_profiles(profiles: &mut [SttProfileConfig]) {
    for profile in profiles {
        profile.id = profile.id.trim().to_string();
        profile.name = normalize_input_display_name(&profile.name);
        profile.input.device_host = profile
            .input
            .device_host
            .take()
            .map(|value| value.trim().to_string());
        profile.input.device_id = profile
            .input
            .device_id
            .take()
            .map(|value| value.trim().to_string());
        profile.input.device_name = profile
            .input
            .device_name
            .take()
            .map(|value| normalize_input_display_name(&value));
        profile.delivery_profile_id = profile
            .delivery_profile_id
            .take()
            .map(|id| id.trim().to_string())
            .filter(|id| !id.is_empty());

        profile.segmentation.vad_interval_ms = 32;
        profile.segmentation.segment_start_speech_ms = profile
            .segmentation
            .segment_start_speech_ms
            .max(profile.segmentation.vad_interval_ms.max(1));
        profile.turn.interim_result_silence_ms = profile
            .turn
            .interim_result_silence_ms
            .max(profile.segmentation.vad_interval_ms.max(1));
        profile.turn.check_silence_ms = profile
            .turn
            .check_silence_ms
            .max(profile.segmentation.vad_interval_ms.max(1));
        if profile.turn.interim_result_enabled {
            profile.turn.check_silence_ms = profile
                .turn
                .check_silence_ms
                .max(profile.turn.interim_result_silence_ms);
        }
        profile.turn.namo_confidence_threshold =
            profile.turn.namo_confidence_threshold.clamp(0.0, 1.0);
        profile.turn.namo_context_max_tokens = profile.turn.namo_context_max_tokens.min(512);

        if !profile.asr.model.supports_completion()
            || profile.asr.model.language() != profile.asr.language
        {
            profile.asr.model = AsrModel::default_for_language(profile.asr.language);
            profile.asr.language = profile.asr.model.language();
        }
        if !profile.asr.model.supports_precision(profile.asr.precision) {
            profile.asr.precision = profile.asr.model.default_precision();
        }
        let legacy_beam_search_enabled = profile.asr.legacy_beam_search_enabled.take();
        if profile.asr.mode == AsrMode::Missing {
            profile.asr.mode = if legacy_beam_search_enabled == Some(true) {
                AsrMode::Accurate
            } else {
                AsrMode::Fast
            };
        }
        if !supports_accurate_asr_mode(profile.asr.model) {
            profile.asr.mode = AsrMode::Fast;
        }
        profile.asr.interim_model =
            normalize_interim_asr_model(profile.asr.model, profile.asr.interim_model);
        normalize_asr_runtime_profiles(&mut profile.asr.runtime_profiles);
        normalize_enabled_asr_models(&mut profile.asr.enabled_models);
        profile.asr.hotwords = normalize_asr_hotwords(std::mem::take(&mut profile.asr.hotwords));
        if !profile.asr.enabled_models.contains(&profile.asr.model) {
            profile.asr.enabled_models.push(profile.asr.model);
        }
        profile.asr.num_threads = profile.asr.num_threads.max(0);
    }
}

/// Only the two Japanese engines have a productized accurate preset. Hidden
/// and multilingual engines remain greedy even when a stale config requests
/// accurate mode.
#[must_use]
fn supports_accurate_asr_mode(model: AsrModel) -> bool {
    matches!(
        model,
        AsrModel::ReazonSpeechK2V2 | AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
    )
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

fn normalize_delivery_profiles(profiles: &mut [DeliveryProfileConfig]) {
    for profile in profiles {
        profile.id = profile.id.trim().to_string();
        normalize_profile_references(&mut profile.translation_mapping_ids);
        normalize_profile_references(&mut profile.speech_mapping_ids);
        normalize_profile_references(&mut profile.http_profile_ids);
    }
}

fn normalize_http_delivery_profiles(profiles: &mut [HttpDeliveryProfileConfig]) {
    for profile in profiles {
        profile.id = profile.id.trim().to_string();
        profile.url = profile.url.trim().to_string();
    }
}

fn normalize_profile_references(references: &mut [String]) {
    for reference in references {
        *reference = reference.trim().to_string();
    }
}

fn unique_configured_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut unique = HashSet::new();
    ids.into_iter()
        .filter_map(non_empty_trimmed)
        .filter(|id| unique.insert(id.clone()))
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
        Some(
            LocalTtsVoice::Supertonic2Onnx
            | LocalTtsVoice::Supertonic3Onnx
            | LocalTtsVoice::Supertonic3OnnxQuantized,
        ) => Some(speaker_id.unwrap_or(0).clamp(0, 9)),
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

fn input_volume_percent_to_db(volume_percent: u8) -> f32 {
    if volume_percent == 0 {
        return -30.0;
    }
    normalize_input_volume_db(20.0 * (f32::from(volume_percent) / 100.0).log10())
}

fn non_empty_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn normalize_asr_hotwords(hotwords: Vec<AsrHotword>) -> Vec<AsrHotword> {
    hotwords
        .into_iter()
        .filter_map(|mut hotword| {
            hotword.surface = non_empty_trimmed(&hotword.surface)?;
            hotword.readings = hotword
                .readings
                .into_iter()
                .filter_map(|reading| non_empty_trimmed(&reading))
                .map(|reading| katakana_to_hiragana(&reading))
                .filter(|reading| !reading.is_empty())
                .collect();
            hotword.readings.sort();
            hotword.readings.dedup();
            hotword.score = hotword
                .score
                .filter(|score| score.is_finite() && *score > 0.0);
            Some(hotword)
        })
        .collect()
}

/// Converts standard full-width katakana to hiragana without changing other
/// scripts.  Keeping this small and dependency-free is sufficient for the
/// persisted Japanese hotword form; the model tokenizer handles the final
/// vocabulary lookup.
fn katakana_to_hiragana(value: &str) -> String {
    value
        .nfkc()
        .map(|character| {
            let code = character as u32;
            if (0x30a1..=0x30f6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(character)
            } else {
                character
            }
        })
        .collect()
}

fn normalize_hotword_reading(value: &str) -> String {
    katakana_to_hiragana(value.trim())
}

fn validate_asr_hotwords(hotwords: &[AsrHotword]) -> Result<()> {
    let mut paths = Vec::<(String, &str)>::new();
    for (index, hotword) in hotwords.iter().enumerate() {
        let surface = hotword.surface.trim();
        if surface.is_empty() {
            bail!("asr_hotwords[{index}].surface must not be empty");
        }
        paths.push((hotword_path(surface), surface));
        if let Some(score) = hotword.score
            && (!score.is_finite() || score <= 0.0)
        {
            bail!("asr_hotwords[{index}].score must be a finite positive number");
        }
        for (reading_index, reading) in hotword.readings.iter().enumerate() {
            let normalized = normalize_hotword_reading(reading);
            if normalized.is_empty() {
                bail!("asr_hotwords[{index}].readings[{reading_index}] must not be empty");
            }
            paths.push((hotword_path(&normalized), surface));
            let katakana = hiragana_to_katakana(&normalized);
            if katakana != normalized {
                paths.push((hotword_path(&katakana), surface));
            }
        }
    }

    let mut path_owners = std::collections::HashMap::<&str, &str>::new();
    for (path, surface) in &paths {
        if let Some(previous_surface) = path_owners.insert(path, surface)
            && previous_surface != *surface
        {
            bail!(
                "ASR hotword token path {path:?} conflicts between {previous_surface:?} and {surface:?}"
            );
        }
    }
    for (short_index, (short, short_surface)) in paths.iter().enumerate() {
        for (long_index, (long, long_surface)) in paths.iter().enumerate() {
            if short_index != long_index && short.len() < long.len() && long.starts_with(short) {
                bail!(
                    "ASR hotword path for {short_surface:?} is a terminal prefix of {long_surface:?}"
                );
            }
        }
    }
    Ok(())
}

fn hotword_path(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn hiragana_to_katakana(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            let code = character as u32;
            if (0x3041..=0x3096).contains(&code) {
                char::from_u32(code + 0x60).unwrap_or(character)
            } else {
                character
            }
        })
        .collect()
}

fn normalize_enabled_asr_models(models: &mut Vec<AsrModel>) {
    models.retain(|model| model.supports_completion());
    models.sort_by_key(|model| model.sort_key());
    models.dedup();
    if models.is_empty() {
        models.push(AsrModel::ReazonSpeechK2V2);
    }
}

fn normalize_asr_runtime_profiles(profiles: &mut [AsrRuntimeProfileConfig]) {
    for profile in profiles {
        profile.id = profile.id.trim().to_string();
    }
}

fn ensure_runtime_profile(
    profiles: &mut Vec<AsrRuntimeProfileConfig>,
    preferred_id: &str,
    model: AsrModel,
) -> String {
    if let Some(profile) = profiles.iter().find(|profile| profile.model == model) {
        return profile.id.clone();
    }
    let mut id = preferred_id.to_owned();
    let mut suffix = 2_u32;
    while profiles.iter().any(|profile| profile.id == id) {
        id = format!("{preferred_id}-{suffix}");
        suffix += 1;
    }
    profiles.push(AsrRuntimeProfileConfig {
        id: id.clone(),
        model,
    });
    id
}

fn push_unique_asr_model(models: &mut Vec<AsrModel>, model: Option<AsrModel>) {
    let Some(model) = model else {
        return;
    };
    if !models.contains(&model) {
        models.push(model);
    }
}

fn stt_profile_required_asr_models(profile: &SttProfileConfig) -> Vec<AsrModel> {
    let mut models = Vec::new();
    push_unique_asr_model(&mut models, Some(profile.asr.model));
    push_unique_asr_model(&mut models, profile.asr.interim_model);
    if profile.asr.multilingual_enabled {
        for model in &profile.asr.enabled_models {
            push_unique_asr_model(&mut models, Some(*model));
        }
    }
    models
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
        AsrConfig, AsrHotword, AsrLanguage, AsrMode, AsrModel, AsrModelCapability,
        AsrModelImplementation, AsrPrecision, AsrRoutePolicyConfig, AsrRuntimeProfileConfig,
        CaptureEndpointConfig, DeliveryProfileConfig, DeveloperConnectionMode, HttpArtifactKind,
        HttpDeliveryProfileConfig, HttpPayloadFormat, InputSourceKind, LocalTranslationModel,
        LocalTtsVoice, NeoSendTiming, NoiseCancellationConfig, NoiseCancellationModel,
        NoiseCancellationTarget, ParapperConfig, RecognitionSourceConfig, ResolvedAsrRoutePolicy,
        SegmentationConfig, SpeechBackend, SpeechMapping, SpeechSourceKind, SttProfileConfig,
        SttProfileDisplayColor, SttProfileInputConfig, TranslationBackend, TranslationLanguage,
        TranslationMapping, TurnConfig, TurnDetector, TurnDetectorClass, TurnDetectorModel,
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
        assert_eq!(
            config.noise_cancellation.target,
            NoiseCancellationTarget::VadOnly
        );
    }

    #[test]
    fn legacy_enabled_noise_cancellation_without_target_keeps_vad_and_asr_processing() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "noise_cancellation_enabled": true,
                "noise_cancellation_model": "ul_unas"
            }"#,
        )
        .expect("legacy noise cancellation config should deserialize");

        assert!(config.noise_cancellation.enabled);
        assert_eq!(
            config.noise_cancellation.target,
            NoiseCancellationTarget::VadAndAsr
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
        assert_eq!(
            object["noise_cancellation_target"],
            serde_json::json!("vad_only")
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the compatibility fixture intentionally lists the complete legacy flat shape"
    )]
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
            LocalTranslationModel::CatTranslate0_8BQ4KQuant
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
        assert_eq!(
            config.noise_cancellation.target,
            NoiseCancellationTarget::VadAndAsr,
            "legacy files without an application target must preserve the old shared NC stream"
        );
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
    fn legacy_beam_search_setting_migrates_to_accurate_and_is_not_saved_again() {
        let legacy = serde_json::from_str::<ParapperConfig>(
            r#"{ "asr_model": "reazonspeech_k2_v2", "asr_beam_search_enabled": true }"#,
        )
        .expect("legacy config should deserialize")
        .normalized();

        assert_eq!(legacy.asr.mode, AsrMode::Accurate);
        let persisted = serde_json::to_value(&legacy).expect("config should serialize");
        assert_eq!(persisted["asr_mode"], serde_json::json!("accurate"));
        assert!(
            persisted.get("asr_beam_search_enabled").is_none(),
            "new saves must not resurrect the deprecated setting"
        );
    }

    #[test]
    fn explicit_asr_mode_wins_when_a_legacy_beam_setting_is_also_present() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{
                "asr_model": "reazonspeech_k2_v2",
                "asr_mode": "fast",
                "asr_beam_search_enabled": true
            }"#,
        )
        .expect("mixed-version config should deserialize")
        .normalized();

        assert_eq!(config.asr.mode, AsrMode::Fast);
    }

    #[test]
    fn asr_mode_serializes_with_fast_and_accurate_wire_values() {
        let mut config = ParapperConfig::default();
        assert_eq!(
            serde_json::to_value(&config).expect("default should serialize")["asr_mode"],
            serde_json::json!("fast")
        );

        config.asr.mode = AsrMode::Accurate;
        assert_eq!(
            serde_json::to_value(&config).expect("accurate should serialize")["asr_mode"],
            serde_json::json!("accurate")
        );
    }

    #[test]
    fn hotwords_are_persisted_with_surface_readings_and_optional_score() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![AsrHotword {
            surface: " 斎藤 ".to_string(),
            readings: vec!["ｻｲﾄｳ".to_string(), "さいとう".to_string()],
            score: Some(2.5),
        }];

        let value = serde_json::to_value(config.normalized()).expect("config should serialize");

        assert_eq!(
            value["asr_hotwords"][0],
            serde_json::json!({
                "surface": "斎藤",
                "readings": ["さいとう"],
                "score": 2.5
            })
        );
    }

    #[test]
    fn hotword_validation_rejects_empty_score_and_cross_surface_reading_conflicts() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![
            AsrHotword {
                surface: "有効".to_string(),
                readings: vec!["ゆうこう".to_string()],
                score: None,
            },
            AsrHotword {
                surface: "別表記".to_string(),
                readings: vec!["ユウコウ".to_string()],
                score: Some(1.0),
            },
        ];

        let error = config
            .validate()
            .expect_err("same normalized reading must not target two surfaces");
        assert!(error.to_string().contains("conflicts"));

        config.asr.hotwords[1].readings = vec!["べつひょうき".to_string()];
        config.asr.hotwords[1].score = Some(0.0);
        let error = config
            .validate()
            .expect_err("non-positive score must be rejected before save");
        assert!(error.to_string().contains("score"));

        config.asr.hotwords[1].score = None;
        config.asr.hotwords[1].surface.clear();
        let error = config
            .validate()
            .expect_err("empty surface must be rejected before save");
        assert!(error.to_string().contains("surface"));
    }

    #[test]
    fn hotword_normalization_drops_empty_surfaces_and_invalid_scores() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![
            AsrHotword {
                surface: "  ".to_string(),
                readings: vec!["不要".to_string()],
                score: Some(1.0),
            },
            AsrHotword {
                surface: "有効".to_string(),
                readings: vec![
                    "  ユウコウ ".to_string(),
                    "ゆうこう".to_string(),
                    " ".to_string(),
                ],
                score: Some(-1.0),
            },
        ];

        let normalized = config.normalized();

        assert_eq!(normalized.asr.hotwords.len(), 1);
        assert_eq!(normalized.asr.hotwords[0].surface, "有効");
        assert_eq!(normalized.asr.hotwords[0].readings, vec!["ゆうこう"]);
        assert_eq!(normalized.asr.hotwords[0].score, None);
    }

    #[test]
    fn hotword_validation_rejects_surface_reading_aliases_for_different_outputs() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![
            AsrHotword {
                surface: "サイトウ".to_string(),
                readings: Vec::new(),
                score: None,
            },
            AsrHotword {
                surface: "斎藤".to_string(),
                readings: vec!["さいとう".to_string()],
                score: None,
            },
        ];

        assert!(config.validate().is_err());
    }

    #[test]
    fn hotword_validation_rejects_terminal_prefixes_even_for_the_same_surface() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![AsrHotword {
            surface: "東京".to_string(),
            readings: vec!["とうきょう".to_string(), "とうきょうと".to_string()],
            score: None,
        }];

        assert!(config.validate().is_err());
    }

    #[test]
    fn effective_hotwords_require_opt_in_accurate_mode_supported_model_and_entries() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![AsrHotword {
            surface: "固有名詞".to_string(),
            readings: vec!["こゆうめいし".to_string()],
            score: None,
        }];

        assert!(!config.hotwords_enabled());

        config.asr.mode = AsrMode::Accurate;
        assert!(
            !config.hotwords_enabled(),
            "the separate checkbox is required"
        );
        config.asr.hotwords_enabled = true;
        assert!(config.hotwords_enabled());

        config.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        assert!(
            config.hotwords_enabled(),
            "Parakeet JA accurate supports hotwords"
        );

        config.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        assert!(
            !config.hotwords_enabled(),
            "unsupported models must never receive hotwords"
        );
    }

    #[test]
    fn unsupported_primary_models_normalize_to_fast_mode_without_losing_hotword_list() {
        let mut config = ParapperConfig::default();
        config.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        config.asr.language = config.asr.model.language();
        config.asr.mode = AsrMode::Accurate;
        config.asr.hotwords_enabled = true;
        config.asr.hotwords = vec![AsrHotword {
            surface: "保存する語".to_string(),
            readings: vec![],
            score: None,
        }];

        let normalized = config.normalized();
        assert_eq!(normalized.asr.mode, AsrMode::Fast);
        assert!(normalized.asr.hotwords_enabled);
        assert_eq!(normalized.asr.hotwords[0].surface, "保存する語");
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
    fn load_rejects_well_formed_explicit_config_with_invalid_field_types() {
        let path = temporary_config_path("explicit-invalid-type");
        fs::write(
            &path,
            r#"{
                "capture_endpoint": {
                    "id": "interface-1",
                    "device_host": "wasapi",
                    "device_id": "mic-1"
                },
                "recognition_sources": [{
                    "source_id": "channel-1",
                    "speaker_label": "A",
                    "capture_endpoint_id": "interface-1",
                    "channel_index": "zero"
                }]
            }"#,
        )
        .expect("failed to write test config");

        let error = ParapperConfig::load(&path)
            .expect_err("invalid explicit field types must not fall back to defaults");

        assert_eq!(
            error.to_string(),
            format!(
                "Failed to parse explicit capture config: {}",
                path.display()
            )
        );
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("invalid type")),
            "serde field error must remain in the error chain: {error:#}"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_rejects_invalid_stt_profile_instead_of_falling_back_to_legacy_defaults() {
        let path = temporary_config_path("stt-profile-invalid-type");
        fs::write(
            &path,
            r#"{
                "stt_profiles": [{
                    "id": "profile-1",
                    "name": "Profile 1",
                    "input": {
                        "channel_index": "zero",
                        "volume_percent": 100,
                        "muted": false
                    },
                    "noise_cancellation": {
                        "enabled": false,
                        "model": "ul_unas",
                        "target": "vad_only"
                    },
                    "segmentation": {
                        "vad_threshold": 0.5,
                        "vad_interval_ms": 32,
                        "segment_start_speech_ms": 96
                    },
                    "turn": {
                        "detector": "simple",
                        "interim_result_enabled": true,
                        "interim_result_silence_ms": 96,
                        "check_silence_ms": 320,
                        "namo_confidence_threshold": 0.8,
                        "namo_context_max_tokens": 256,
                        "rerecognize_full_on_complete": false
                    },
                    "asr": {
                        "language": "japanese",
                        "model": "reazonspeech_k2_v2",
                        "interim_model": null,
                        "precision": "int8_float32",
                        "num_threads": 4,
                        "mode": "fast",
                        "hotwords_enabled": false,
                        "hotwords": [],
                        "normalize_input_audio": true,
                        "multilingual_enabled": false,
                        "enabled_models": ["reazonspeech_k2_v2"],
                        "runtime_profiles": []
                    }
                }]
            }"#,
        )
        .expect("failed to write test config");

        let error = ParapperConfig::load(&path)
            .expect_err("invalid STT profile must not fall back to legacy defaults");
        assert!(
            error
                .to_string()
                .contains("Failed to parse explicit capture config")
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_keeps_an_older_beta_stt_profile_when_new_nested_fields_are_absent() {
        let path = temporary_config_path("stt-profile-forward-compatible-defaults");
        fs::write(
            &path,
            r#"{
                "stt_profiles": [{
                    "id": "profile-1",
                    "name": "Profile 1",
                    "input": {
                        "channel_index": 0,
                        "volume_percent": 100
                    },
                    "noise_cancellation": {
                        "enabled": false,
                        "model": "ul_unas"
                    },
                    "segmentation": {
                        "vad_threshold": 0.5,
                        "vad_interval_ms": 32
                    },
                    "turn": {
                        "detector": "simple",
                        "interim_result_enabled": true,
                        "interim_result_silence_ms": 96,
                        "check_silence_ms": 320,
                        "namo_confidence_threshold": 0.8,
                        "namo_context_max_tokens": 256
                    },
                    "asr": {
                        "language": "japanese",
                        "model": "reazonspeech_k2_v2",
                        "precision": "int8_float32",
                        "num_threads": 4,
                        "mode": "fast"
                    }
                }]
            }"#,
        )
        .expect("failed to write older beta config");

        let loaded = ParapperConfig::load(&path)
            .expect("an older beta STT profile should remain usable after fields are added");
        assert_eq!(
            loaded.stt_profiles,
            vec![stt_profile("profile-1", "Profile 1", None, None, 0)]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_rejects_semantically_invalid_explicit_capture_shapes() {
        for (name, content, expected_cause) in [
            (
                "endpoint-only",
                r#"{
                    "capture_endpoint": {
                        "id": "interface-1",
                        "device_host": "wasapi",
                        "device_id": "mic-1"
                    }
                }"#,
                "explicit capture_endpoint requires recognition_sources",
            ),
            (
                "duplicate-source",
                r#"{
                    "capture_endpoint": {
                        "id": "interface-1",
                        "device_host": "wasapi",
                        "device_id": "mic-1"
                    },
                    "recognition_sources": [
                        {
                            "source_id": "channel-1",
                            "speaker_label": "A",
                            "capture_endpoint_id": "interface-1",
                            "channel_index": 0
                        },
                        {
                            "source_id": "channel-1",
                            "speaker_label": "B",
                            "capture_endpoint_id": "interface-1",
                            "channel_index": 1
                        }
                    ]
                }"#,
                "recognition source ids must be unique",
            ),
            (
                "empty-label",
                r#"{
                    "capture_endpoint": {
                        "id": "interface-1",
                        "device_host": "wasapi",
                        "device_id": "mic-1"
                    },
                    "recognition_sources": [{
                        "source_id": "channel-1",
                        "speaker_label": "  ",
                        "capture_endpoint_id": "interface-1",
                        "channel_index": 0
                    }]
                }"#,
                "recognition source speaker label must not be empty",
            ),
            (
                "empty-explicit-shape",
                r#"{
                    "capture_endpoint": null,
                    "recognition_sources": []
                }"#,
                "explicit capture config must include capture_endpoint and recognition_sources",
            ),
        ] {
            let path = temporary_config_path(name);
            fs::write(&path, content).expect("failed to write test config");

            let error = ParapperConfig::load(&path)
                .expect_err("semantic errors in explicit config must be returned");

            assert_eq!(
                error.to_string(),
                format!("Invalid explicit capture config: {}", path.display()),
                "case={name}"
            );
            assert!(
                error
                    .chain()
                    .any(|cause| cause.to_string() == expected_cause),
                "expected cause `{expected_cause}` for case={name}: {error:#}"
            );
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn load_normalizes_and_validates_valid_explicit_capture_config() {
        let path = temporary_config_path("explicit-valid");
        fs::write(
            &path,
            r#"{
                "capture_endpoint": {
                    "id": "  interface-1  ",
                    "device_host": "  wasapi  ",
                    "device_id": "  mic-1  ",
                    "device_name": "  Ｍｉｃ　Ａ  "
                },
                "recognition_sources": [{
                    "source_id": "  channel-1  ",
                    "speaker_label": "  Ａさん  ",
                    "capture_endpoint_id": "  interface-1  ",
                    "channel_index": 0
                }]
            }"#,
        )
        .expect("failed to write test config");

        let loaded = ParapperConfig::load(&path).expect("valid explicit config should load");

        assert_eq!(
            loaded.input.capture_endpoint,
            Some(CaptureEndpointConfig {
                id: "interface-1".to_string(),
                device_host: "wasapi".to_string(),
                device_id: "mic-1".to_string(),
                device_name: Some("Mic A".to_string()),
            })
        );
        assert_eq!(
            loaded.input.recognition_sources,
            vec![RecognitionSourceConfig {
                source_id: "channel-1".to_string(),
                speaker_label: "Aさん".to_string(),
                capture_endpoint_id: "interface-1".to_string(),
                channel_index: 0,
                asr_route_policy: Some(AsrRoutePolicyConfig {
                    interim_runtime_id: None,
                    completion_runtime_id: "legacy-completion".to_string(),
                }),
                delivery_profile_id: Some("legacy-default".to_string()),
            }]
        );
        assert_eq!(
            loaded.asr.runtime_profiles,
            vec![AsrRuntimeProfileConfig {
                id: "legacy-completion".to_string(),
                model: AsrModel::ReazonSpeechK2V2,
            }]
        );
        loaded
            .validate()
            .expect("loaded explicit config must be valid");
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
                AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
                AsrLanguage::English,
                "en",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
                AsrLanguage::Multilingual,
                "ja",
            ),
            (
                AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
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
    fn stt_profile_turn_detector_asset_requirements_are_scoped_to_each_profile() {
        let mut simple_japanese = stt_profile("ja-simple", "Japanese", None, None, 0);
        simple_japanese.turn.detector = TurnDetector::Simple;
        simple_japanese.asr.language = AsrLanguage::Japanese;
        simple_japanese.asr.model = AsrModel::ReazonSpeechK2V2;

        let mut namo_english = stt_profile("en-namo", "English", Some("host"), Some("id"), 0);
        namo_english.turn.detector = TurnDetector::Namo;
        namo_english.asr.language = AsrLanguage::English;
        namo_english.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        namo_english.asr.enabled_models = vec![namo_english.asr.model];

        let config = ParapperConfig {
            stt_profiles: vec![simple_japanese.clone(), namo_english.clone()],
            ..ParapperConfig::default()
        };

        assert!(config.uses_namo_turn_detector());
        // Namo also exposes a morph-aware boundary implementation, but only
        // its own English ASR lane contributes to the morph asset requirement.
        assert!(config.uses_morph_turn_boundary());
        assert_eq!(
            config.required_namo_turn_detector_languages(),
            vec![AsrLanguage::English]
        );
        assert!(!config.requires_japanese_morph_analyzer());

        let mut morph_japanese = simple_japanese;
        morph_japanese.id = "ja-morph".to_owned();
        morph_japanese.name = "Japanese morph".to_owned();
        morph_japanese.turn.detector = TurnDetector::Morph;
        let config = ParapperConfig {
            stt_profiles: vec![namo_english, morph_japanese],
            ..ParapperConfig::default()
        };
        assert!(config.uses_morph_turn_boundary());
        assert!(config.requires_japanese_morph_analyzer());
    }

    #[test]
    fn disabled_stt_profiles_do_not_require_turn_detector_assets() {
        let enabled = stt_profile("enabled", "Enabled", None, None, 0);
        let mut disabled = stt_profile("disabled", "Disabled", Some("host"), Some("id"), 0);
        disabled.enabled = false;
        disabled.turn.detector = TurnDetector::Namo;
        disabled.asr.language = AsrLanguage::English;
        disabled.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        disabled.asr.enabled_models = vec![disabled.asr.model];

        let config = ParapperConfig {
            stt_profiles: vec![enabled, disabled],
            ..ParapperConfig::default()
        };

        assert!(!config.uses_namo_turn_detector());
        assert!(!config.uses_morph_turn_boundary());
        assert!(config.required_namo_turn_detector_languages().is_empty());
        assert!(!config.requires_japanese_morph_analyzer());
    }

    #[test]
    fn resolved_stt_profile_is_flat_and_does_not_inherit_sibling_turn_detectors() {
        let mut simple = stt_profile("simple", "Simple", None, None, 0);
        simple.turn.detector = TurnDetector::Simple;
        let mut namo = stt_profile("namo", "Namo", Some("host"), Some("id"), 0);
        namo.turn.detector = TurnDetector::Namo;
        let config = ParapperConfig {
            stt_profiles: vec![simple, namo],
            ..ParapperConfig::default()
        };

        let simple_config = config
            .config_for_stt_profile("simple")
            .expect("simple profile should resolve");
        assert!(simple_config.stt_profiles.is_empty());
        assert!(!simple_config.uses_namo_turn_detector());

        let namo_config = config
            .config_for_stt_profile("namo")
            .expect("namo profile should resolve");
        assert!(namo_config.stt_profiles.is_empty());
        assert!(namo_config.uses_namo_turn_detector());
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
    fn published_cat_translation_config_remains_selected_after_normalization() {
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
            LocalTranslationModel::CatTranslate0_8BQ4KQuant
        );
        assert_eq!(
            config.translation.mappings[0].local_model,
            LocalTranslationModel::CatTranslate0_8BQ4KQuant
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
            Some(LocalTtsVoice::Supertonic2Onnx)
        );
    }

    #[test]
    fn removed_piper_speech_mapping_voices_migrate_to_supertonic2() {
        for removed_voice in [
            "vits_piper_en_US_kristin_medium",
            "vits_piper_en_US_john_medium",
            "vits_piper_en_US_norman_medium",
        ] {
            let config_json = r#"{
                "speech_mappings": [{
                    "id": "speech-removed-voice",
                    "source_kind": "recognition",
                    "target_lang": null,
                    "backend": "local_tts",
                    "talker": "",
                    "local_tts_voice": "REMOVED_VOICE",
                    "muted": false,
                    "volume": 1.0
                }]
            }"#
            .replace("REMOVED_VOICE", removed_voice);
            let config = serde_json::from_str::<ParapperConfig>(&config_json)
                .unwrap()
                .normalized();

            assert_eq!(
                config.speech.mappings[0].local_tts_voice,
                Some(LocalTtsVoice::Supertonic2Onnx),
                "removed voice {removed_voice} should migrate"
            );
            assert_eq!(
                config.speech.mappings[0].local_tts_language.as_deref(),
                Some("en")
            );
            assert_eq!(config.speech.mappings[0].local_tts_speaker_id, Some(0));
        }
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

    #[test]
    fn quantized_supertonic3_preserves_its_model_identity_and_normalizes_voice_options() {
        let config = config_with(|config| {
            config.speech.mappings = vec![SpeechMapping {
                id: "speech-supertonic3-quantized".to_string(),
                source_kind: SpeechSourceKind::Recognition,
                source_asr_model: None,
                target_lang: None,
                backend: SpeechBackend::LocalTts,
                talker: String::new(),
                local_tts_voice: Some(LocalTtsVoice::Supertonic3OnnxQuantized),
                local_tts_language: Some(" JA ".to_string()),
                local_tts_speaker_id: Some(99),
                output_device_id: None,
                output_device_host: None,
                output_device_name: None,
                muted: false,
                volume: 1.0,
            }];
        });

        assert_eq!(
            config.speech.mappings[0].local_tts_voice,
            Some(LocalTtsVoice::Supertonic3OnnxQuantized)
        );
        assert_eq!(
            config.speech.mappings[0].local_tts_language.as_deref(),
            Some("ja")
        );
        assert_eq!(config.speech.mappings[0].local_tts_speaker_id, Some(9));
        let value = serde_json::to_value(&config).expect("config should serialize");
        assert_eq!(
            value["speech_mappings"][0]["local_tts_voice"],
            serde_json::json!("supertonic_3_onnx_quantized")
        );
    }

    #[test]
    fn explicit_capture_endpoint_and_sources_round_trip_without_legacy_fields() {
        let mut config = ParapperConfig::default();
        config.input.capture_endpoint = Some(CaptureEndpointConfig {
            id: "  interface-1  ".to_string(),
            device_host: "  wasapi  ".to_string(),
            device_id: "  mic-1  ".to_string(),
            device_name: Some("  Ｍｉｃ　Ａ  ".to_string()),
        });
        config.input.recognition_sources = vec![RecognitionSourceConfig {
            source_id: "  channel-1  ".to_string(),
            speaker_label: "  Ａさん  ".to_string(),
            capture_endpoint_id: "  interface-1  ".to_string(),
            channel_index: 0,
            asr_route_policy: None,
            delivery_profile_id: None,
        }];

        let normalized = config.normalized();
        normalized
            .validate()
            .expect("explicit capture config should validate");
        assert_eq!(
            normalized.input.capture_endpoint.as_ref().unwrap().id,
            "interface-1"
        );
        assert_eq!(
            normalized
                .input
                .capture_endpoint
                .as_ref()
                .unwrap()
                .device_name,
            Some("Mic A".to_string())
        );
        assert_eq!(
            normalized.input.recognition_sources[0].speaker_label,
            "Aさん"
        );

        let value = serde_json::to_value(&normalized).expect("explicit config should serialize");
        assert!(value.get("capture_endpoint").is_some());
        assert!(value.get("recognition_sources").is_some());
        assert_eq!(value["input_device_id"], serde_json::Value::Null);
        let restored = serde_json::from_value::<ParapperConfig>(value)
            .expect("explicit config should deserialize")
            .normalized();
        assert_eq!(restored, normalized);
    }

    #[test]
    fn explicit_capture_validation_rejects_invalid_shapes_without_repairing_to_legacy() {
        for name in [
            "endpoint id is empty",
            "device host is empty",
            "device id is empty",
            "device name is empty",
            "source id is empty",
            "speaker label is empty",
            "source endpoint id is empty",
            "endpoint reference differs",
            "source ids are duplicated",
            "channel indexes are duplicated",
            "explicit websocket is unsupported",
        ] {
            let mut config = explicit_capture_config();
            let expected_error = match name {
                "endpoint id is empty" => {
                    config.input.capture_endpoint.as_mut().unwrap().id = "  ".to_string();
                    "capture endpoint id must not be empty"
                }
                "device host is empty" => {
                    config.input.capture_endpoint.as_mut().unwrap().device_host = "  ".to_string();
                    "capture endpoint device host must not be empty"
                }
                "device id is empty" => {
                    config.input.capture_endpoint.as_mut().unwrap().device_id = "  ".to_string();
                    "capture endpoint device id must not be empty"
                }
                "device name is empty" => {
                    config.input.capture_endpoint.as_mut().unwrap().device_name =
                        Some("  ".to_string());
                    "capture endpoint device name must not be empty"
                }
                "source id is empty" => {
                    config.input.recognition_sources[0].source_id = "  ".to_string();
                    "recognition source id must not be empty"
                }
                "speaker label is empty" => {
                    config.input.recognition_sources[0].speaker_label = "  ".to_string();
                    "recognition source speaker label must not be empty"
                }
                "source endpoint id is empty" => {
                    config.input.recognition_sources[0].capture_endpoint_id = "  ".to_string();
                    "recognition source capture endpoint id must not be empty"
                }
                "endpoint reference differs" => {
                    config.input.recognition_sources[0].capture_endpoint_id = "other".to_string();
                    "recognition source capture endpoint id does not match capture endpoint"
                }
                "source ids are duplicated" => {
                    config
                        .input
                        .recognition_sources
                        .push(RecognitionSourceConfig {
                            source_id: "channel-1".to_string(),
                            speaker_label: "B".to_string(),
                            capture_endpoint_id: "interface-1".to_string(),
                            channel_index: 1,
                            asr_route_policy: None,
                            delivery_profile_id: None,
                        });
                    "recognition source ids must be unique"
                }
                "channel indexes are duplicated" => {
                    config
                        .input
                        .recognition_sources
                        .push(RecognitionSourceConfig {
                            source_id: "channel-2".to_string(),
                            speaker_label: "B".to_string(),
                            capture_endpoint_id: "interface-1".to_string(),
                            channel_index: 0,
                            asr_route_policy: None,
                            delivery_profile_id: None,
                        });
                    "recognition source channel indexes must be unique"
                }
                "explicit websocket is unsupported" => {
                    config.input.source_kind = InputSourceKind::WebSocket;
                    "explicit capture endpoints do not support WebSocket input yet"
                }
                _ => unreachable!(),
            };
            let normalized = config.normalized();
            assert_eq!(
                normalized.validate().unwrap_err().to_string(),
                expected_error,
                "case={name}"
            );
            assert!(
                normalized.input.capture_endpoint.is_some(),
                "invalid explicit config must not silently become legacy: case={name}"
            );
        }
    }

    #[test]
    fn explicit_and_legacy_input_fields_cannot_be_mixed() {
        let mut config = explicit_capture_config();
        config.input.device_id = Some("legacy-device".to_string());
        assert_eq!(
            config.validate().unwrap_err().to_string(),
            "explicit capture_endpoint cannot be mixed with legacy input device fields"
        );
    }

    #[test]
    fn endpoint_only_and_sources_only_are_rejected_with_stable_errors() {
        let mut endpoint_only = explicit_capture_config();
        endpoint_only.input.recognition_sources.clear();
        assert_eq!(
            endpoint_only.validate().unwrap_err().to_string(),
            "explicit capture_endpoint requires recognition_sources"
        );

        let mut sources_only = explicit_capture_config();
        sources_only.input.capture_endpoint = None;
        assert_eq!(
            sources_only.validate().unwrap_err().to_string(),
            "recognition_sources require an explicit capture_endpoint"
        );
    }

    #[test]
    fn legacy_flat_config_without_new_fields_remains_legacy_and_flat() {
        let config = serde_json::from_str::<ParapperConfig>(
            r#"{"input_device_host":"wasapi","input_device_id":"mic-1"}"#,
        )
        .expect("legacy config should deserialize")
        .normalized();

        config.validate().expect("legacy config should validate");
        assert!(config.input.capture_endpoint.is_none());
        assert!(config.input.recognition_sources.is_empty());
        let value = serde_json::to_value(config).expect("legacy config should serialize");
        assert!(value.get("capture_endpoint").is_none());
        assert!(value.get("recognition_sources").is_none());
        assert_eq!(value["input_device_host"], "wasapi");
    }

    fn explicit_capture_config() -> ParapperConfig {
        let mut config = ParapperConfig::default();
        config.input.capture_endpoint = Some(CaptureEndpointConfig {
            id: "interface-1".to_string(),
            device_host: "wasapi".to_string(),
            device_id: "mic-1".to_string(),
            device_name: None,
        });
        config.input.recognition_sources = vec![RecognitionSourceConfig {
            source_id: "channel-1".to_string(),
            speaker_label: "A".to_string(),
            capture_endpoint_id: "interface-1".to_string(),
            channel_index: 0,
            asr_route_policy: None,
            delivery_profile_id: None,
        }];
        config
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

    #[test]
    fn explicit_sources_resolve_independent_completion_and_interim_runtime_profiles() {
        let mut config = explicit_capture_config();
        config.asr.runtime_profiles = vec![
            AsrRuntimeProfileConfig {
                id: "ja-completion".to_owned(),
                model: AsrModel::ReazonSpeechK2V2,
            },
            AsrRuntimeProfileConfig {
                id: "en-completion".to_owned(),
                model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            },
            AsrRuntimeProfileConfig {
                id: "en-interim".to_owned(),
                model: AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            },
        ];
        config.input.recognition_sources[0].asr_route_policy = Some(AsrRoutePolicyConfig {
            interim_runtime_id: Some("en-interim".to_owned()),
            completion_runtime_id: "ja-completion".to_owned(),
        });
        config
            .input
            .recognition_sources
            .push(RecognitionSourceConfig {
                source_id: "channel-2".to_owned(),
                speaker_label: "B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: Some(AsrRoutePolicyConfig {
                    interim_runtime_id: None,
                    completion_runtime_id: "en-completion".to_owned(),
                }),
                delivery_profile_id: None,
            });

        let config = config.normalized();

        config
            .validate()
            .expect("all referenced runtimes are valid");
        assert_eq!(
            config.required_asr_models(),
            vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            ]
        );
        assert_eq!(
            config.resolved_asr_route_for_source("channel-2").unwrap(),
            ResolvedAsrRoutePolicy {
                interim_model: None,
                completion_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            }
        );
    }

    #[test]
    fn explicit_config_without_routes_migrates_flat_models_to_explicit_runtime_profiles() {
        let mut config = explicit_capture_config();
        config.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        config.asr.language = config.asr.model.language();
        config.asr.interim_model = Some(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8);

        let config = config.normalized();

        assert_eq!(
            config.asr.runtime_profiles,
            vec![
                AsrRuntimeProfileConfig {
                    id: "legacy-completion".to_owned(),
                    model: AsrModel::NemoParakeetTdt0_6BV2Int8,
                },
                AsrRuntimeProfileConfig {
                    id: "legacy-interim".to_owned(),
                    model: AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
                },
            ]
        );
        assert_eq!(
            config.input.recognition_sources[0].asr_route_policy,
            Some(AsrRoutePolicyConfig {
                interim_runtime_id: Some("legacy-interim".to_owned()),
                completion_runtime_id: "legacy-completion".to_owned(),
            })
        );
        config
            .validate()
            .expect("normalized migration must be runnable without fallback");
    }

    #[test]
    fn explicit_runtime_profiles_without_source_routes_are_not_completed_from_flat_asr_settings() {
        let mut config = explicit_capture_config();
        config.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: "configured-completion".to_owned(),
            model: AsrModel::NemoParakeetTdt0_6BV2Int8,
        }];

        let config = config.normalized();

        assert_eq!(
            config.asr.runtime_profiles,
            vec![AsrRuntimeProfileConfig {
                id: "configured-completion".to_owned(),
                model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            }]
        );
        assert_eq!(config.input.recognition_sources[0].asr_route_policy, None);
        assert!(
            config
                .validate()
                .expect_err("a partially authored ASR schema must not fall back")
                .to_string()
                .contains("requires an ASR route policy")
        );
    }

    #[test]
    fn explicit_routes_reject_empty_duplicate_unknown_and_interim_only_completion_runtimes() {
        let mut empty_profile = explicit_capture_config();
        empty_profile.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: " ".to_owned(),
            model: AsrModel::ReazonSpeechK2V2,
        }];
        empty_profile.input.recognition_sources[0].asr_route_policy = Some(AsrRoutePolicyConfig {
            interim_runtime_id: None,
            completion_runtime_id: "completion".to_owned(),
        });

        let mut duplicate_profile = explicit_capture_config();
        duplicate_profile.asr.runtime_profiles = vec![
            AsrRuntimeProfileConfig {
                id: "completion".to_owned(),
                model: AsrModel::ReazonSpeechK2V2,
            },
            AsrRuntimeProfileConfig {
                id: "completion".to_owned(),
                model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            },
        ];
        duplicate_profile.input.recognition_sources[0].asr_route_policy =
            Some(AsrRoutePolicyConfig {
                interim_runtime_id: None,
                completion_runtime_id: "completion".to_owned(),
            });

        let mut unknown_reference = explicit_capture_config();
        unknown_reference.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: "completion".to_owned(),
            model: AsrModel::ReazonSpeechK2V2,
        }];
        unknown_reference.input.recognition_sources[0].asr_route_policy =
            Some(AsrRoutePolicyConfig {
                interim_runtime_id: None,
                completion_runtime_id: "missing-completion".to_owned(),
            });

        let mut unknown_interim = explicit_capture_config();
        unknown_interim.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: "completion".to_owned(),
            model: AsrModel::ReazonSpeechK2V2,
        }];
        unknown_interim.input.recognition_sources[0].asr_route_policy =
            Some(AsrRoutePolicyConfig {
                interim_runtime_id: Some("missing-interim".to_owned()),
                completion_runtime_id: "completion".to_owned(),
            });

        let mut nemotron_completion = explicit_capture_config();
        nemotron_completion.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: "streaming-only".to_owned(),
            model: AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
        }];
        nemotron_completion.input.recognition_sources[0].asr_route_policy =
            Some(AsrRoutePolicyConfig {
                interim_runtime_id: None,
                completion_runtime_id: "streaming-only".to_owned(),
            });

        for (config, expected_error) in [
            (empty_profile, "ASR runtime profile id must not be empty"),
            (duplicate_profile, "ASR runtime profile ids must be unique"),
            (
                unknown_reference,
                "references unknown completion ASR runtime \"missing-completion\"",
            ),
            (
                unknown_interim,
                "references unknown interim ASR runtime \"missing-interim\"",
            ),
            (nemotron_completion, "which does not support completion"),
        ] {
            let error = config
                .validate()
                .expect_err("invalid explicit route must fail");
            assert!(
                error.to_string().contains(expected_error),
                "expected {expected_error:?} in {error:#}"
            );
        }
    }

    #[test]
    fn source_session_config_clone_uses_its_explicit_route_without_mutating_global_model() {
        let mut config = explicit_capture_config();
        config.asr.model = AsrModel::ReazonSpeechK2V2;
        config.asr.interim_model = None;
        config.asr.runtime_profiles = vec![
            AsrRuntimeProfileConfig {
                id: "en-completion".to_owned(),
                model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            },
            AsrRuntimeProfileConfig {
                id: "en-interim".to_owned(),
                model: AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            },
        ];
        config.input.recognition_sources[0].asr_route_policy = Some(AsrRoutePolicyConfig {
            interim_runtime_id: Some("en-interim".to_owned()),
            completion_runtime_id: "en-completion".to_owned(),
        });

        let source_config = config
            .with_asr_route_for_source("channel-1")
            .expect("configured source route must resolve");

        assert_eq!(config.asr.model, AsrModel::ReazonSpeechK2V2);
        assert_eq!(config.asr.interim_model, None);
        assert_eq!(source_config.asr.model, AsrModel::NemoParakeetTdt0_6BV2Int8);
        assert_eq!(
            source_config.asr.interim_model,
            Some(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8)
        );
        assert_eq!(
            source_config.asr.enabled_models,
            vec![
                AsrModel::NemoParakeetTdt0_6BV2Int8,
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            ]
        );
    }

    #[test]
    fn explicit_sources_can_share_the_same_completion_runtime_profile() {
        let mut config = explicit_capture_config();
        config.asr.runtime_profiles = vec![AsrRuntimeProfileConfig {
            id: "shared-ja".to_owned(),
            model: AsrModel::ReazonSpeechK2V2,
        }];
        let shared_policy = AsrRoutePolicyConfig {
            interim_runtime_id: None,
            completion_runtime_id: "shared-ja".to_owned(),
        };
        config.input.recognition_sources[0].asr_route_policy = Some(shared_policy.clone());
        config
            .input
            .recognition_sources
            .push(RecognitionSourceConfig {
                source_id: "channel-2".to_owned(),
                speaker_label: "B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: Some(shared_policy),
                delivery_profile_id: None,
            });

        let config = config.normalized();

        config
            .validate()
            .expect("more than one source can use one runtime profile");
        assert_eq!(
            config.required_asr_models(),
            vec![AsrModel::ReazonSpeechK2V2]
        );
    }

    #[test]
    fn explicit_source_without_delivery_profile_migrates_legacy_outputs_to_persisted_default() {
        let mut config = explicit_capture_config();
        config.translation.mappings = vec![TranslationMapping {
            id: " translate-ja-en ".to_owned(),
            source_asr_model: None,
            backend: TranslationBackend::Ync,
            local_model: LocalTranslationModel::default(),
            source_lang: TranslationLanguage::Ja,
            target_lang: TranslationLanguage::En,
        }];
        let duplicate_translation = config.translation.mappings[0].clone();
        config.translation.mappings.push(duplicate_translation);
        config.translation.mappings.push(TranslationMapping {
            id: "same-language".to_owned(),
            source_asr_model: None,
            backend: TranslationBackend::Ync,
            local_model: LocalTranslationModel::default(),
            source_lang: TranslationLanguage::En,
            target_lang: TranslationLanguage::En,
        });
        config.speech.mappings = vec![SpeechMapping {
            id: " speech-en ".to_owned(),
            source_kind: SpeechSourceKind::Translation,
            source_asr_model: None,
            target_lang: Some("en".to_owned()),
            backend: SpeechBackend::Ync,
            talker: "talker".to_owned(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_id: None,
            output_device_host: None,
            output_device_name: None,
            muted: false,
            volume: 1.0,
        }];
        let duplicate_speech = config.speech.mappings[0].clone();
        config.speech.mappings.push(duplicate_speech);
        let mut discarded_speech = config.speech.mappings[0].clone();
        discarded_speech.id = " ".to_owned();
        config.speech.mappings.push(discarded_speech);

        let config = config.normalized();

        assert_eq!(
            config.input.recognition_sources[0]
                .delivery_profile_id
                .as_deref(),
            Some("legacy-default")
        );
        assert_eq!(
            config.delivery_profiles,
            vec![DeliveryProfileConfig {
                id: "legacy-default".to_owned(),
                gui_enabled: true,
                translation_mapping_ids: vec!["translate-ja-en".to_owned()],
                speech_mapping_ids: vec!["speech-en".to_owned()],
                http_profile_ids: Vec::new(),
                neo_text_enabled: false,
            }]
        );
        assert_eq!(
            config
                .resolved_delivery_route_for_source("channel-1")
                .expect("normalized explicit source must resolve a persisted delivery profile")
                .profile_id,
            "legacy-default"
        );
    }

    #[test]
    fn explicit_multisource_accepts_global_developer_http_and_rejects_invalid_delivery_profiles() {
        let mut legacy_developer_http = explicit_capture_config();
        legacy_developer_http
            .input
            .recognition_sources
            .push(RecognitionSourceConfig {
                source_id: "channel-2".to_owned(),
                speaker_label: "B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: None,
                delivery_profile_id: None,
            });
        let mut legacy_developer_http = legacy_developer_http.normalized();
        legacy_developer_http.streaming_recognition.enabled = true;
        legacy_developer_http.streaming_recognition.mode = DeveloperConnectionMode::Http;
        legacy_developer_http
            .validate()
            .expect("global Developer HTTP must accept every recognition source");

        let mut invalid_http_profile = explicit_capture_config().normalized();
        invalid_http_profile.delivery_profiles[0].http_profile_ids = vec!["events".to_owned()];
        invalid_http_profile.http_delivery_profiles = vec![HttpDeliveryProfileConfig {
            id: "events".to_owned(),
            url: "ftp://invalid.example/events".to_owned(),
            payload_format: HttpPayloadFormat::TextEventV1,
            artifact_kinds: vec![],
            send_timing: NeoSendTiming::Final,
        }];

        let mut missing_artifact = explicit_capture_config().normalized();
        missing_artifact.delivery_profiles[0].http_profile_ids = vec!["events".to_owned()];
        missing_artifact.http_delivery_profiles = vec![HttpDeliveryProfileConfig {
            id: "events".to_owned(),
            url: "https://example.invalid/events".to_owned(),
            payload_format: HttpPayloadFormat::TextEventV1,
            artifact_kinds: Vec::new(),
            send_timing: NeoSendTiming::Final,
        }];

        for (config, expected_error) in [
            (invalid_http_profile, "must use an HTTP(S) URL"),
            (missing_artifact, "must select at least one artifact kind"),
        ] {
            let error = config
                .validate()
                .expect_err("unsafe delivery config must fail");
            assert!(
                error.to_string().contains(expected_error),
                "expected {expected_error:?} in {error:#}"
            );
        }
    }

    #[test]
    fn explicit_delivery_profiles_reject_partial_source_and_unknown_consumer_references() {
        let base = explicit_capture_config().normalized();

        let mut partial_source = base.clone();
        partial_source
            .input
            .recognition_sources
            .push(RecognitionSourceConfig {
                source_id: "channel-2".to_owned(),
                speaker_label: "B".to_owned(),
                capture_endpoint_id: "interface-1".to_owned(),
                channel_index: 1,
                asr_route_policy: Some(AsrRoutePolicyConfig {
                    interim_runtime_id: None,
                    completion_runtime_id: "legacy-completion".to_owned(),
                }),
                delivery_profile_id: None,
            });
        let partial_source = partial_source.normalized();

        let mut unknown_translation = base.clone();
        unknown_translation.delivery_profiles[0].translation_mapping_ids =
            vec!["missing-mt".to_owned()];

        let mut unknown_speech = base.clone();
        unknown_speech.delivery_profiles[0].speech_mapping_ids = vec!["missing-tts".to_owned()];

        let mut unknown_http = base;
        unknown_http.delivery_profiles[0].http_profile_ids = vec!["missing-http".to_owned()];

        for (config, expected_error) in [
            (partial_source, "requires a delivery profile"),
            (
                unknown_translation,
                "references unknown translation mapping \"missing-mt\"",
            ),
            (
                unknown_speech,
                "references unknown speech mapping \"missing-tts\"",
            ),
            (
                unknown_http,
                "references unknown HTTP profile \"missing-http\"",
            ),
        ] {
            let error = config
                .validate()
                .expect_err("unknown or partial delivery route must fail closed");
            assert!(
                error.to_string().contains(expected_error),
                "expected {expected_error:?} in {error:#}"
            );
        }
    }

    #[test]
    fn explicit_profiles_without_source_associations_are_not_replaced_by_legacy_default() {
        let mut config = explicit_capture_config();
        config.delivery_profiles = vec![DeliveryProfileConfig {
            id: "new-profile".to_owned(),
            gui_enabled: false,
            translation_mapping_ids: Vec::new(),
            speech_mapping_ids: Vec::new(),
            http_profile_ids: Vec::new(),
            neo_text_enabled: false,
        }];

        let config = config.normalized();

        assert_eq!(config.delivery_profiles[0].id, "new-profile");
        assert_eq!(config.delivery_profiles.len(), 1);
        assert_eq!(
            config.input.recognition_sources[0].delivery_profile_id,
            None
        );
        assert!(
            config
                .validate()
                .expect_err("a partially authored delivery schema must not fall back")
                .to_string()
                .contains("requires a delivery profile")
        );
    }

    fn stt_profile(
        id: &str,
        name: &str,
        device_host: Option<&str>,
        device_id: Option<&str>,
        channel_index: u16,
    ) -> SttProfileConfig {
        SttProfileConfig {
            id: id.to_owned(),
            name: name.to_owned(),
            enabled: true,
            neo_http_enabled: true,
            developer_http_enabled: true,
            display_color: SttProfileDisplayColor::Green,
            input: SttProfileInputConfig {
                device_host: device_host.map(str::to_owned),
                device_id: device_id.map(str::to_owned),
                device_name: None,
                channel_index,
                volume_percent: 100,
                muted: false,
            },
            noise_cancellation: NoiseCancellationConfig::default(),
            segmentation: SegmentationConfig::default(),
            turn: TurnConfig::default(),
            asr: AsrConfig::default(),
            delivery_profile_id: None,
        }
    }

    #[test]
    fn stt_profiles_round_trip_as_nested_top_level_json_without_flat_legacy_fields() {
        let config = ParapperConfig {
            stt_profiles: vec![stt_profile(
                "profile-1",
                "Profile 1",
                Some("host"),
                Some("device"),
                0,
            )],
            ..ParapperConfig::default()
        };

        let value = serde_json::to_value(&config).expect("profile config should serialize");
        assert!(value.get("stt_profiles").is_some());
        assert!(value.get("stt_profiles").unwrap()[0].get("input").is_some());
        assert_eq!(value["stt_profiles"][0]["enabled"], true);
        assert_eq!(value["stt_profiles"][0]["neo_http_enabled"], true);
        assert_eq!(value["stt_profiles"][0]["developer_http_enabled"], true);
        assert!(
            value.get("stt_profiles").unwrap()[0]
                .get("noise_cancellation")
                .unwrap()
                .get("enabled")
                .is_some()
        );
        assert!(
            value.get("stt_profiles").unwrap()[0]
                .get("asr")
                .unwrap()
                .get("language")
                .is_some()
        );
        let decoded: ParapperConfig =
            serde_json::from_value(value).expect("profile config should deserialize");
        assert_eq!(decoded.stt_profiles, config.stt_profiles);
    }

    #[test]
    fn stt_profile_display_color_round_trips_new_palette_and_rejects_legacy_values() {
        let config = ParapperConfig {
            stt_profiles: vec![stt_profile(
                "profile-1",
                "Profile 1",
                Some("host"),
                Some("device"),
                0,
            )],
            ..ParapperConfig::default()
        };
        let mut value = serde_json::to_value(&config).expect("profile config should serialize");
        assert_eq!(value["stt_profiles"][0]["display_color"], "green");

        value["stt_profiles"][0]
            .as_object_mut()
            .expect("profile should be an object")
            .remove("display_color");
        let migrated: ParapperConfig =
            serde_json::from_value(value.clone()).expect("missing color migrates to green");
        assert_eq!(
            migrated.stt_profiles[0].display_color,
            SttProfileDisplayColor::Green
        );

        for color in [
            "green", "blue", "violet", "red", "orange", "yellow", "white",
        ] {
            value["stt_profiles"][0]["display_color"] = serde_json::json!(color);
            let parsed: ParapperConfig = serde_json::from_value(value.clone())
                .expect("every current profile color should deserialize");
            let serialized = serde_json::to_value(parsed).expect("profile config should serialize");
            assert_eq!(serialized["stt_profiles"][0]["display_color"], color);
        }

        for invalid_color in ["cyan", "pink", "not-a-profile-color"] {
            value["stt_profiles"][0]["display_color"] = serde_json::json!(invalid_color);
            assert!(serde_json::from_value::<ParapperConfig>(value.clone()).is_err());
        }
    }

    #[test]
    fn missing_stt_profile_enabled_migrates_to_enabled() {
        let config = ParapperConfig {
            stt_profiles: vec![stt_profile("profile-1", "Profile 1", None, None, 0)],
            ..ParapperConfig::default()
        };
        let mut value = serde_json::to_value(config).expect("profile config should serialize");
        value["stt_profiles"][0]
            .as_object_mut()
            .expect("profile should be an object")
            .remove("enabled");

        let migrated: ParapperConfig =
            serde_json::from_value(value).expect("missing enabled migrates to enabled");
        assert!(migrated.stt_profiles[0].enabled);
    }

    #[test]
    fn missing_stt_profile_developer_http_setting_migrates_to_send_enabled() {
        let config = ParapperConfig {
            stt_profiles: vec![stt_profile("profile-1", "Profile 1", None, None, 0)],
            ..ParapperConfig::default()
        };
        let mut value = serde_json::to_value(config).expect("profile config should serialize");
        value["stt_profiles"][0]
            .as_object_mut()
            .expect("profile should be an object")
            .remove("developer_http_enabled");

        let migrated: ParapperConfig = serde_json::from_value(value)
            .expect("missing Developer HTTP setting migrates to send enabled");
        assert!(migrated.stt_profiles[0].developer_http_enabled);
    }

    #[test]
    fn missing_stt_profile_neo_http_setting_migrates_to_send_enabled() {
        let config = ParapperConfig {
            stt_profiles: vec![stt_profile("profile-1", "Profile 1", None, None, 0)],
            ..ParapperConfig::default()
        };
        let mut value = serde_json::to_value(config).expect("profile config should serialize");
        value["stt_profiles"][0]
            .as_object_mut()
            .expect("profile should be an object")
            .remove("neo_http_enabled");

        let migrated: ParapperConfig =
            serde_json::from_value(value).expect("missing NEO setting migrates to send enabled");
        assert!(migrated.stt_profiles[0].neo_http_enabled);
    }

    #[test]
    fn stt_profile_mode_requires_at_least_one_enabled_profile() {
        let mut profile = stt_profile("profile-1", "Profile 1", None, None, 0);
        profile.enabled = false;
        let config = ParapperConfig {
            stt_profiles: vec![profile],
            ..ParapperConfig::default()
        };

        assert!(
            config
                .validate()
                .expect_err("profile mode must not allow every profile to be disabled")
                .to_string()
                .contains("at least one enabled STT profile")
        );
    }

    #[test]
    fn legacy_flat_config_keeps_empty_stt_profiles_and_does_not_emit_profile_schema() {
        let config = ParapperConfig::default().normalized();
        assert!(config.stt_profiles.is_empty());
        config.validate().expect("legacy config remains valid");
        let value = serde_json::to_value(config).expect("legacy config should serialize");
        assert!(value.get("stt_profiles").is_none());
    }

    #[test]
    fn stt_profile_mode_rejects_mixing_with_legacy_explicit_capture() {
        let mut config = ParapperConfig::default();
        config.input.capture_endpoint = Some(CaptureEndpointConfig {
            id: "endpoint".to_owned(),
            device_host: "host".to_owned(),
            device_id: "device".to_owned(),
            device_name: None,
        });
        config.stt_profiles = vec![stt_profile(
            "profile-1",
            "Profile 1",
            Some("host"),
            Some("device"),
            0,
        )];

        let error = config
            .validate()
            .expect_err("new profile mode must not mix with explicit capture");
        assert!(
            error
                .to_string()
                .contains("stt_profiles cannot be mixed with explicit capture")
        );
    }

    #[test]
    fn stt_profile_mode_rejects_websocket_input_until_profile_sessions_are_supported() {
        let mut config = ParapperConfig {
            stt_profiles: vec![stt_profile("profile-1", "Profile 1", None, None, 0)],
            ..ParapperConfig::default()
        };
        config.input.source_kind = InputSourceKind::WebSocket;

        let error = config
            .validate()
            .expect_err("profile mode must not be shadowed by WebSocket input startup");
        assert!(
            error
                .to_string()
                .contains("stt_profiles require desktop audio input")
        );
    }

    #[test]
    fn multiple_stt_profiles_route_developer_http_and_neo_by_source_profile() {
        let mut config = ParapperConfig {
            stt_profiles: vec![
                stt_profile(
                    "profile-1",
                    "Profile 1",
                    Some("host-a"),
                    Some("device-a"),
                    0,
                ),
                stt_profile(
                    "profile-2",
                    "Profile 2",
                    Some("host-b"),
                    Some("device-b"),
                    0,
                ),
            ],
            ..ParapperConfig::default()
        };
        config.streaming_recognition.enabled = true;
        config.streaming_recognition.mode = DeveloperConnectionMode::Http;
        config.stt_profiles[1].developer_http_enabled = false;
        config.stt_profiles[0].neo_http_enabled = false;

        config
            .validate()
            .expect("global destinations must accept multiple STT profiles");
        assert!(config.developer_http_enabled_for_source("profile-1"));
        assert!(!config.developer_http_enabled_for_source("profile-2"));
        assert!(config.developer_http_enabled_for_source("unknown-source"));
        assert!(!config.neo_http_enabled_for_source("profile-1"));
        assert!(config.neo_http_enabled_for_source("profile-2"));
        assert!(config.neo_http_enabled_for_source("unknown-source"));
    }

    #[test]
    fn multiple_stt_profiles_accept_source_aware_http_delivery_after_legacy_migration() {
        let mut first = stt_profile("profile-1", "Profile 1", Some("host"), Some("device"), 0);
        first.delivery_profile_id = Some("legacy-default".to_owned());
        let mut second = stt_profile("profile-2", "Profile 2", Some("host"), Some("device"), 1);
        second.delivery_profile_id = Some("legacy-default".to_owned());
        let config = ParapperConfig {
            stt_profiles: vec![first, second],
            delivery_profiles: vec![DeliveryProfileConfig {
                id: "legacy-default".to_owned(),
                gui_enabled: true,
                translation_mapping_ids: Vec::new(),
                speech_mapping_ids: Vec::new(),
                http_profile_ids: vec!["legacy-developer-http".to_owned()],
                neo_text_enabled: true,
            }],
            http_delivery_profiles: vec![HttpDeliveryProfileConfig {
                id: "legacy-developer-http".to_owned(),
                url: "http://127.0.0.1:15522/api/events".to_owned(),
                payload_format: HttpPayloadFormat::TextEventV1,
                artifact_kinds: vec![HttpArtifactKind::Recognition],
                send_timing: NeoSendTiming::Interim,
            }],
            ..ParapperConfig::default()
        };

        config
            .validate()
            .expect("source-aware HTTP delivery must support multiple STT profiles");
        for profile_id in ["profile-1", "profile-2"] {
            let route = config
                .resolved_delivery_route_for_stt_profile(profile_id)
                .expect("each STT profile must resolve the migrated HTTP route");
            assert_eq!(route.profile_id, "legacy-default");
            assert_eq!(route.http_profiles, config.http_delivery_profiles);
        }
    }

    #[test]
    fn stt_profile_resolution_isolated_from_global_flat_config_and_converts_volume() {
        let mut config = ParapperConfig::default();
        config.input.volume_db = 6.0;
        let mut first = stt_profile(
            "profile-1",
            "Profile 1",
            Some("host-a"),
            Some("device-a"),
            0,
        );
        first.input.volume_percent = 50;
        let mut second = stt_profile(
            "profile-2",
            "Profile 2",
            Some("host-b"),
            Some("device-b"),
            1,
        );
        second.input.volume_percent = 25;
        second.input.muted = true;
        second.segmentation.vad_threshold = 0.9;
        second.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        let second_model = second.asr.model;
        config.stt_profiles = vec![first, second];

        let resolved = config
            .resolved_stt_profile("profile-2")
            .expect("profile should resolve");
        assert_eq!(resolved.name, "Profile 2");
        let source_config = config
            .config_for_stt_profile("profile-2")
            .expect("profile config should resolve");
        assert_eq!(source_config.input.device_host.as_deref(), Some("host-b"));
        assert_eq!(source_config.input.device_id.as_deref(), Some("device-b"));
        assert!(source_config.input.muted);
        assert!((source_config.input.volume_db - (-12.0412)).abs() < 0.01);
        assert!((source_config.segmentation.vad_threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(source_config.asr.model, second_model);
        assert!((config.input.volume_db - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn stt_profile_validation_rejects_duplicate_ids_names_devices_and_invalid_volume() {
        let duplicate_id = ParapperConfig {
            stt_profiles: vec![
                stt_profile("same", "A", Some("host-a"), Some("device-a"), 0),
                stt_profile("same", "B", Some("host-b"), Some("device-b"), 1),
            ],
            ..ParapperConfig::default()
        };
        let duplicate_id_error = duplicate_id
            .validate()
            .expect_err("duplicate profile id must fail");
        assert!(
            duplicate_id_error
                .to_string()
                .contains("STT profile ids must be unique")
        );

        let duplicate_name = ParapperConfig {
            stt_profiles: vec![
                stt_profile("one", "Same", Some("host-a"), Some("device-a"), 0),
                stt_profile("two", "Same", Some("host-b"), Some("device-b"), 1),
            ],
            ..ParapperConfig::default()
        };
        let duplicate_name_error = duplicate_name
            .validate()
            .expect_err("duplicate profile name must fail");
        assert!(
            duplicate_name_error
                .to_string()
                .contains("STT profile names must be unique")
        );

        let duplicate_device = ParapperConfig {
            stt_profiles: vec![
                stt_profile("one", "One", Some("host"), Some("device"), 0),
                stt_profile("two", "Two", Some("host"), Some("device"), 0),
            ],
            ..ParapperConfig::default()
        };
        let duplicate_device_error = duplicate_device
            .validate()
            .expect_err("duplicate device channel must fail");
        assert!(
            duplicate_device_error
                .to_string()
                .contains("device/channel combination must be unique")
        );

        let mut invalid_volume_profile = stt_profile("one", "One", Some("host"), Some("device"), 0);
        invalid_volume_profile.input.volume_percent = 101;
        let invalid_volume = ParapperConfig {
            stt_profiles: vec![invalid_volume_profile],
            ..ParapperConfig::default()
        };
        let invalid_volume_error = invalid_volume
            .validate()
            .expect_err("volume over 100 must fail");
        assert!(
            invalid_volume_error
                .to_string()
                .contains("volume percent must be between 0 and 100")
        );
    }

    #[test]
    fn multiple_stt_profiles_require_devices_but_single_profile_can_use_os_default() {
        let single = ParapperConfig {
            stt_profiles: vec![stt_profile("one", "One", None, None, 0)],
            ..ParapperConfig::default()
        };
        single
            .validate()
            .expect("one profile may use OS default input channel 0");

        let multiple = ParapperConfig {
            stt_profiles: vec![
                stt_profile("one", "One", None, None, 0),
                stt_profile("two", "Two", Some("host"), Some("device"), 1),
            ],
            ..ParapperConfig::default()
        };
        let error = multiple
            .validate()
            .expect_err("multiple profiles need explicit devices");
        assert!(
            error
                .to_string()
                .contains("multiple STT profiles require device host and id")
        );
    }

    #[test]
    fn stt_profile_delivery_route_requires_an_existing_profile_and_preserves_legacy_fallback() {
        let mut config = ParapperConfig::default();
        let mut profile = stt_profile("one", "One", None, None, 0);
        profile.delivery_profile_id = Some("missing".to_owned());
        config.stt_profiles = vec![profile];
        let error = config
            .validate()
            .expect_err("unknown delivery profile must fail closed");
        assert!(
            error
                .to_string()
                .contains("references unknown delivery profile")
        );

        let valid = ParapperConfig {
            stt_profiles: vec![stt_profile("one", "One", None, None, 0)],
            ..ParapperConfig::default()
        };
        let route = valid
            .resolved_delivery_route_for_stt_profile("one")
            .expect("profile without delivery route should use legacy outputs");
        assert_eq!(route.profile_id, "legacy-default");
    }

    #[test]
    fn stt_profile_nested_settings_use_the_same_normalization_rules_as_flat_settings() {
        let mut profile = stt_profile(" profile-1 ", "  Profile 1 ", None, None, 0);
        profile.segmentation.vad_interval_ms = 999;
        profile.segmentation.segment_start_speech_ms = 1;
        profile.turn.interim_result_silence_ms = 1;
        profile.turn.check_silence_ms = 1;
        profile.turn.namo_confidence_threshold = 2.0;
        profile.turn.namo_context_max_tokens = 999;
        profile.asr.language = AsrLanguage::Japanese;
        profile.asr.model = AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8;
        let normalized = ParapperConfig {
            stt_profiles: vec![profile],
            ..ParapperConfig::default()
        }
        .normalized();
        let profile = &normalized.stt_profiles[0];
        assert_eq!(profile.id, "profile-1");
        assert_eq!(profile.name, "Profile 1");
        assert_eq!(profile.segmentation.vad_interval_ms, 32);
        assert_eq!(profile.segmentation.segment_start_speech_ms, 32);
        assert_eq!(profile.turn.interim_result_silence_ms, 32);
        assert_eq!(profile.turn.check_silence_ms, 32);
        assert!((profile.turn.namo_confidence_threshold - 1.0).abs() < f32::EPSILON);
        assert_eq!(profile.turn.namo_context_max_tokens, 512);
        assert_eq!(profile.asr.model, AsrModel::ReazonSpeechK2V2);
    }

    #[test]
    fn required_asr_models_union_all_stt_profiles() {
        let mut config = ParapperConfig::default();
        let mut first = stt_profile("one", "One", None, None, 0);
        first.asr.model = AsrModel::ReazonSpeechK2V2;
        first.asr.enabled_models = vec![first.asr.model];
        let mut second = stt_profile("two", "Two", Some("host"), Some("device"), 1);
        second.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        second.asr.enabled_models = vec![second.asr.model];
        config.stt_profiles = vec![first, second];

        assert_eq!(
            config.required_asr_models(),
            vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            ]
        );
    }

    #[test]
    fn monolingual_stt_profile_requires_only_its_selected_models() {
        let mut profile = stt_profile("japanese", "Japanese", None, None, 0);
        profile.asr.model = AsrModel::ReazonSpeechK2V2;
        profile.asr.interim_model = None;
        profile.asr.multilingual_enabled = false;
        profile.asr.enabled_models = vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ];
        profile.turn.detector = TurnDetector::Namo;
        let config = ParapperConfig {
            stt_profiles: vec![profile],
            ..ParapperConfig::default()
        };

        assert_eq!(
            config.required_asr_models(),
            vec![AsrModel::ReazonSpeechK2V2]
        );
        assert_eq!(
            config.required_namo_turn_detector_languages(),
            vec![AsrLanguage::Japanese]
        );
    }

    #[test]
    fn required_asr_models_exclude_disabled_stt_profiles() {
        let mut config = ParapperConfig::default();
        let mut enabled = stt_profile("enabled", "Enabled", None, None, 0);
        enabled.asr.model = AsrModel::ReazonSpeechK2V2;
        enabled.asr.enabled_models = vec![enabled.asr.model];
        let mut disabled = stt_profile("disabled", "Disabled", Some("host"), Some("device"), 1);
        disabled.enabled = false;
        disabled.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        disabled.asr.enabled_models = vec![disabled.asr.model];
        config.stt_profiles = vec![enabled, disabled];

        assert_eq!(
            config.required_asr_models(),
            vec![AsrModel::ReazonSpeechK2V2]
        );
    }
}
