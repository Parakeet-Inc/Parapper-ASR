//! Local machine-translation models.

#[cfg(feature = "mt-ort")]
mod cache;
#[cfg(feature = "mt-ort")]
mod engine;
#[cfg(feature = "mt-ort")]
mod service;

use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};

#[cfg(feature = "mt-ort")]
pub use service::LocalTranslationService;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
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
    #[default]
    Lfm2Q4,
    #[serde(
        rename = "cat_translate_0_8b_q4_k_quant",
        alias = "cat_translate_0_8b_q4",
        alias = "cat-translate-0.8b-onnx-q4"
    )]
    CatTranslate0_8BQ4KQuant,
}

impl LocalTranslationModel {
    #[must_use]
    pub const fn is_available(self) -> bool {
        matches!(self, Self::Lfm2Q4 | Self::CatTranslate0_8BQ4KQuant)
    }

    #[must_use]
    pub const fn sort_key(self) -> u8 {
        match self {
            Self::Lfm2Q4 => 0,
            Self::CatTranslate0_8BQ4KQuant => 1,
        }
    }

    #[must_use]
    pub const fn onnx_file_name(self) -> &'static str {
        match self {
            Self::Lfm2Q4 => "onnx/model_q4.onnx",
            Self::CatTranslate0_8BQ4KQuant => "model_q4.onnx",
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
    #[must_use]
    pub const fn as_code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ja => "ja",
        }
    }

    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::En => Self::Ja,
            Self::Ja => Self::En,
        }
    }

    #[must_use]
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
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn persisted_translation_model_aliases_remain_backward_compatible() {
        for alias in [
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
                serde_json::from_str::<LocalTranslationModel>(&format!("\"{alias}\""))
                    .expect("legacy model name should deserialize"),
                LocalTranslationModel::Lfm2Q4,
                "alias={alias}"
            );
        }
    }

    #[test]
    fn translation_language_accepts_locale_codes_but_serializes_canonical_codes() {
        assert_eq!(
            serde_json::from_str::<TranslationLanguage>("\"en_US\"").unwrap(),
            TranslationLanguage::En
        );
        assert_eq!(
            serde_json::from_str::<TranslationLanguage>("\"ja-JP\"").unwrap(),
            TranslationLanguage::Ja
        );
        assert_eq!(
            serde_json::to_string(&TranslationLanguage::En).unwrap(),
            "\"en\""
        );
    }
}
