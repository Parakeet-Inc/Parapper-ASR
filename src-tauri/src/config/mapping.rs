use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};

pub use parapper_models::{
    mt::{LocalTranslationModel, TranslationLanguage},
    tts::{
        ALL_LOCAL_TTS_VOICES, LocalTtsVoice, SUPERTONIC2_LANGUAGE_CODES, SUPERTONIC3_LANGUAGE_CODES,
    },
};

use super::AsrModel;

pub(crate) const fn translation_language_from_asr_language(
    language: parapper_models::asr::AsrLanguage,
) -> Option<TranslationLanguage> {
    match language {
        parapper_models::asr::AsrLanguage::English => Some(TranslationLanguage::En),
        parapper_models::asr::AsrLanguage::Japanese => Some(TranslationLanguage::Ja),
        parapper_models::asr::AsrLanguage::EuropeanMultilingual
        | parapper_models::asr::AsrLanguage::Multilingual => None,
    }
}

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
                    .and_then(|model| translation_language_from_asr_language(model.language()))
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
#[derive(Default)]
pub enum TranslationBackend {
    #[default]
    Ync,
    Local,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechSourceKind {
    Recognition,
    Translation,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SpeechBackend {
    #[default]
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

fn is_legacy_ync_backend_value(value: &str) -> bool {
    value.len() == 12 && value.starts_with("yuka") && value.ends_with("kone_neo")
}

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
        "supertonic_2_onnx" => Some(LocalTtsVoice::Supertonic2Onnx),
        "supertonic_3_onnx" => Some(LocalTtsVoice::Supertonic3Onnx),
        "supertonic_3_onnx_quantized" => Some(LocalTtsVoice::Supertonic3OnnxQuantized),
        _ => None,
    }))
}
