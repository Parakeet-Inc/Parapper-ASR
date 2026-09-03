use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrPrecision {
    Int8,
    #[default]
    Int8Float32,
    Float32,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AsrModel {
    #[serde(rename = "reazonspeech_k2_v2")]
    #[default]
    ReazonSpeechK2V2,
    #[serde(rename = "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8")]
    NemoParakeetTdtCtc0_6BJa35000Int8,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v2_int8")]
    NemoParakeetTdt0_6BV2Int8,
    #[serde(rename = "nemo_parakeet_tdt_0_6b_v3_int8")]
    NemoParakeetTdt0_6BV3Int8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_80ms_int8")]
    NemotronSpeechStreamingEn0_6B80MsInt8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_160ms_int8")]
    NemotronSpeechStreamingEn0_6B160MsInt8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_320ms_int8")]
    NemotronSpeechStreamingEn0_6B320MsInt8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_560ms_int8")]
    NemotronSpeechStreamingEn0_6B560MsInt8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_1120ms_int8")]
    NemotronSpeechStreamingEn0_6B1120MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_80ms_int8")]
    Nemotron3_5AsrStreaming0_6B80MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_160ms_int8")]
    Nemotron3_5AsrStreaming0_6B160MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_320ms_int8")]
    Nemotron3_5AsrStreaming0_6B320MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_560ms_int8")]
    Nemotron3_5AsrStreaming0_6B560MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_1120ms_int8")]
    Nemotron3_5AsrStreaming0_6B1120MsInt8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrModelImplementation {
    ReazonSpeechK2,
    NemoParakeetTdtCtc,
    NemoParakeetTdt,
    Nemotron,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrModelCapability {
    CompletionAndInterim,
    InterimOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrStreamLanguage {
    None,
    Nemotron35Auto,
}

#[derive(Debug, Clone, Copy)]
pub struct AsrModelInfo {
    pub language: AsrLanguage,
    pub supported_language_codes: &'static [&'static str],
    pub supported_precisions: &'static [AsrPrecision],
    pub default_precision: AsrPrecision,
    pub implementation: AsrModelImplementation,
    pub capability: AsrModelCapability,
    pub stream_language: AsrStreamLanguage,
    pub sort_key: u8,
}

impl AsrModel {
    #[must_use]
    pub fn info(self) -> AsrModelInfo {
        match self {
            Self::ReazonSpeechK2V2 => AsrModelInfo {
                language: AsrLanguage::Japanese,
                supported_language_codes: &["ja"],
                supported_precisions: &[
                    AsrPrecision::Int8,
                    AsrPrecision::Int8Float32,
                    AsrPrecision::Float32,
                ],
                default_precision: AsrPrecision::Int8Float32,
                implementation: AsrModelImplementation::ReazonSpeechK2,
                capability: AsrModelCapability::CompletionAndInterim,
                stream_language: AsrStreamLanguage::None,
                sort_key: 0,
            },
            Self::NemoParakeetTdtCtc0_6BJa35000Int8 => AsrModelInfo {
                language: AsrLanguage::Japanese,
                supported_language_codes: &["ja"],
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::NemoParakeetTdtCtc,
                capability: AsrModelCapability::CompletionAndInterim,
                stream_language: AsrStreamLanguage::None,
                sort_key: 1,
            },
            Self::NemoParakeetTdt0_6BV2Int8 => AsrModelInfo {
                language: AsrLanguage::English,
                supported_language_codes: &["en"],
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::NemoParakeetTdt,
                capability: AsrModelCapability::CompletionAndInterim,
                stream_language: AsrStreamLanguage::None,
                sort_key: 2,
            },
            Self::NemoParakeetTdt0_6BV3Int8 => AsrModelInfo {
                language: AsrLanguage::EuropeanMultilingual,
                supported_language_codes: PARAKEET_TDT_0_6B_V3_LANGUAGE_CODES,
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::NemoParakeetTdt,
                capability: AsrModelCapability::CompletionAndInterim,
                stream_language: AsrStreamLanguage::None,
                sort_key: 3,
            },
            Self::NemotronSpeechStreamingEn0_6B80MsInt8
            | Self::NemotronSpeechStreamingEn0_6B160MsInt8
            | Self::NemotronSpeechStreamingEn0_6B320MsInt8
            | Self::NemotronSpeechStreamingEn0_6B560MsInt8
            | Self::NemotronSpeechStreamingEn0_6B1120MsInt8 => AsrModelInfo {
                language: AsrLanguage::English,
                supported_language_codes: &["en"],
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::None,
                sort_key: 4,
            },
            Self::Nemotron3_5AsrStreaming0_6B80MsInt8
            | Self::Nemotron3_5AsrStreaming0_6B160MsInt8
            | Self::Nemotron3_5AsrStreaming0_6B320MsInt8
            | Self::Nemotron3_5AsrStreaming0_6B560MsInt8
            | Self::Nemotron3_5AsrStreaming0_6B1120MsInt8 => AsrModelInfo {
                language: AsrLanguage::Multilingual,
                supported_language_codes: NEMOTRON_3_5_LANGUAGE_CODES,
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::Nemotron35Auto,
                sort_key: 5,
            },
        }
    }

    #[must_use]
    pub fn language(self) -> AsrLanguage {
        self.info().language
    }

    #[must_use]
    pub fn supported_language_codes(self) -> &'static [&'static str] {
        self.info().supported_language_codes
    }

    #[must_use]
    pub fn default_for_language(language: AsrLanguage) -> Self {
        match language {
            AsrLanguage::Japanese => Self::ReazonSpeechK2V2,
            AsrLanguage::English => Self::NemoParakeetTdt0_6BV2Int8,
            AsrLanguage::EuropeanMultilingual | AsrLanguage::Multilingual => {
                Self::NemoParakeetTdt0_6BV3Int8
            }
        }
    }

    #[must_use]
    pub fn supports_precision(self, precision: AsrPrecision) -> bool {
        self.info().supported_precisions.contains(&precision)
    }

    #[must_use]
    pub fn default_precision(self) -> AsrPrecision {
        self.info().default_precision
    }

    #[must_use]
    pub fn implementation(self) -> AsrModelImplementation {
        self.info().implementation
    }

    #[must_use]
    pub fn capability(self) -> AsrModelCapability {
        self.info().capability
    }

    #[must_use]
    pub fn supports_completion(self) -> bool {
        self.capability() == AsrModelCapability::CompletionAndInterim
    }

    #[must_use]
    pub fn is_interim_only(self) -> bool {
        self.capability() == AsrModelCapability::InterimOnly
    }

    #[must_use]
    pub fn stream_language(self) -> AsrStreamLanguage {
        self.info().stream_language
    }

    #[must_use]
    pub fn sort_key(self) -> u8 {
        self.info().sort_key
    }

    #[must_use]
    pub fn is_nemotron(self) -> bool {
        self.implementation() == AsrModelImplementation::Nemotron
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsrLanguage {
    #[default]
    Japanese,
    English,
    EuropeanMultilingual,
    Multilingual,
}

const PARAKEET_TDT_0_6B_V3_LANGUAGE_CODES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];
const NEMOTRON_3_5_LANGUAGE_CODES: &[&str] = &[
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hi", "hr", "hu", "it", "ja",
    "ko", "nl", "nb", "pl", "pt", "ro", "ru", "sk", "sv", "tr", "uk", "vi", "zh",
];

#[cfg(test)]
mod tests {
    use super::{AsrLanguage, AsrModel, AsrPrecision};

    const MODEL_WIRE_CASES: &[(AsrModel, &str)] = &[
        (AsrModel::ReazonSpeechK2V2, "reazonspeech_k2_v2"),
        (
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8",
        ),
        (
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            "nemo_parakeet_tdt_0_6b_v2_int8",
        ),
        (
            AsrModel::NemoParakeetTdt0_6BV3Int8,
            "nemo_parakeet_tdt_0_6b_v3_int8",
        ),
        (
            AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
            "nemotron_speech_streaming_en_0_6b_80ms_int8",
        ),
        (
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            "nemotron_speech_streaming_en_0_6b_160ms_int8",
        ),
        (
            AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
            "nemotron_speech_streaming_en_0_6b_320ms_int8",
        ),
        (
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            "nemotron_speech_streaming_en_0_6b_560ms_int8",
        ),
        (
            AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
            "nemotron_speech_streaming_en_0_6b_1120ms_int8",
        ),
        (
            AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
            "nemotron_3_5_asr_streaming_0_6b_80ms_int8",
        ),
        (
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            "nemotron_3_5_asr_streaming_0_6b_160ms_int8",
        ),
        (
            AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
            "nemotron_3_5_asr_streaming_0_6b_320ms_int8",
        ),
        (
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
            "nemotron_3_5_asr_streaming_0_6b_560ms_int8",
        ),
        (
            AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
            "nemotron_3_5_asr_streaming_0_6b_1120ms_int8",
        ),
    ];

    #[test]
    fn every_asr_config_enum_keeps_its_existing_wire_name() {
        for &(model, wire_name) in MODEL_WIRE_CASES {
            let serialized = serde_json::to_string(&model).expect("ASR model must serialize");
            assert_eq!(serialized, format!("\"{wire_name}\""));
            assert_eq!(
                serde_json::from_str::<AsrModel>(&serialized).expect("ASR model must deserialize"),
                model
            );
        }
        for &(precision, wire_name) in &[
            (AsrPrecision::Int8, "int8"),
            (AsrPrecision::Int8Float32, "int8_float32"),
            (AsrPrecision::Float32, "float32"),
        ] {
            let json = format!("\"{wire_name}\"");
            assert_eq!(serde_json::to_string(&precision).unwrap(), json);
            assert_eq!(
                serde_json::from_str::<AsrPrecision>(&json).unwrap(),
                precision
            );
        }
        for &(language, wire_name) in &[
            (AsrLanguage::Japanese, "japanese"),
            (AsrLanguage::English, "english"),
            (AsrLanguage::EuropeanMultilingual, "european_multilingual"),
            (AsrLanguage::Multilingual, "multilingual"),
        ] {
            let json = format!("\"{wire_name}\"");
            assert_eq!(serde_json::to_string(&language).unwrap(), json);
            assert_eq!(
                serde_json::from_str::<AsrLanguage>(&json).unwrap(),
                language
            );
        }
    }
}
