use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};

use super::{AsrLanguage, AsrModel};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TranslationMapping {
    pub id: String,
    pub source_asr_model: Option<AsrModel>,
    pub backend: TranslationBackend,
    pub local_model: LocalTranslationModel,
    pub source_lang: TranslationLanguage,
    pub target_lang: TranslationLanguage,
}

impl<'de> Deserialize<'de> for TranslationMapping {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TranslationMappingWire {
            id: String,
            source_asr_model: Option<AsrModel>,
            #[serde(default)]
            backend: TranslationBackend,
            #[serde(default)]
            local_model: LocalTranslationModel,
            source_lang: Option<String>,
            target_lang: Option<String>,
        }

        let wire = TranslationMappingWire::deserialize(deserializer)?;
        let Some(target_lang) = wire
            .target_lang
            .as_deref()
            .and_then(TranslationLanguage::from_code)
        else {
            return Ok(Self {
                id: String::new(),
                source_asr_model: wire.source_asr_model,
                backend: wire.backend,
                local_model: wire.local_model,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            });
        };
        let source_lang = wire
            .source_lang
            .as_deref()
            .and_then(TranslationLanguage::from_code)
            .or_else(|| {
                wire.source_asr_model
                    .and_then(|model| TranslationLanguage::from_asr_language(model.language()))
                    .filter(|source_lang| *source_lang != target_lang)
            })
            .unwrap_or_else(|| target_lang.other());

        Ok(Self {
            id: wire.id,
            source_asr_model: wire.source_asr_model,
            backend: wire.backend,
            local_model: wire.local_model,
            source_lang,
            target_lang,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TranslationBackend {
    Ync,
    Local,
}

impl Default for TranslationBackend {
    fn default() -> Self {
        Self::Ync
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LocalTranslationModel {
    #[serde(
        rename = "lfm2_q4",
        alias = "f32",
        alias = "fp32",
        alias = "model",
        alias = "q4",
        alias = "q4f16",
        alias = "q4_f16",
        alias = "model_quantized",
        alias = "k_quant",
        alias = "q4_k_quant",
        alias = "lfm2_q4_k_quant"
    )]
    Lfm2Q4,
    // CAT-Translate is kept only to migrate existing config. It is not exposed
    // as an available model while its distribution is disabled.
    #[serde(
        rename = "cat_translate_0_8b_q4_k_quant",
        alias = "cat_translate_0_8b_q4",
        alias = "cat-translate-0.8b-onnx-q4"
    )]
    CatTranslate0_8BQ4KQuant,
}

impl Default for LocalTranslationModel {
    fn default() -> Self {
        Self::Lfm2Q4
    }
}

impl LocalTranslationModel {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Lfm2Q4)
    }

    pub fn onnx_file_name(self) -> &'static str {
        match self {
            Self::Lfm2Q4 => "onnx/model_q4.onnx",
            Self::CatTranslate0_8BQ4KQuant => "model_q4.onnx",
        }
    }

    pub fn sort_key(self) -> u8 {
        match self {
            Self::Lfm2Q4 => 0,
            Self::CatTranslate0_8BQ4KQuant => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum TranslationLanguage {
    #[serde(rename = "en")]
    En,
    #[serde(rename = "ja")]
    Ja,
}

impl<'de> Deserialize<'de> for TranslationLanguage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_code(&value).ok_or_else(|| D::Error::unknown_variant(&value, &["en", "ja"]))
    }
}

impl TranslationLanguage {
    pub fn as_code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ja => "ja",
        }
    }

    pub fn other(self) -> Self {
        match self {
            Self::En => Self::Ja,
            Self::Ja => Self::En,
        }
    }

    pub fn from_code(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.starts_with("en") {
            return Some(Self::En);
        }
        if normalized.starts_with("ja") {
            return Some(Self::Ja);
        }
        None
    }

    pub fn from_asr_language(language: AsrLanguage) -> Option<Self> {
        match language {
            AsrLanguage::English => Some(Self::En),
            AsrLanguage::Japanese => Some(Self::Ja),
            AsrLanguage::EuropeanMultilingual | AsrLanguage::Multilingual => None,
        }
    }
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
