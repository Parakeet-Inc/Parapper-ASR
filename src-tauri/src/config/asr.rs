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
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_160ms_int8")]
    NemotronSpeechStreamingEn0_6B160MsInt8,
    #[serde(rename = "nemotron_speech_streaming_en_0_6b_560ms_int8")]
    NemotronSpeechStreamingEn0_6B560MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_160ms_int8")]
    Nemotron3_5AsrStreaming0_6B160MsInt8,
    #[serde(rename = "nemotron_3_5_asr_streaming_0_6b_560ms_int8")]
    Nemotron3_5AsrStreaming0_6B560MsInt8,
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
            Self::NemotronSpeechStreamingEn0_6B160MsInt8 => AsrModelInfo {
                language: AsrLanguage::English,
                supported_language_codes: &["en"],
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::None,
                sort_key: 4,
            },
            Self::NemotronSpeechStreamingEn0_6B560MsInt8 => AsrModelInfo {
                language: AsrLanguage::English,
                supported_language_codes: &["en"],
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::None,
                sort_key: 5,
            },
            Self::Nemotron3_5AsrStreaming0_6B160MsInt8 => AsrModelInfo {
                language: AsrLanguage::Multilingual,
                supported_language_codes: NEMOTRON_3_5_LANGUAGE_CODES,
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::Nemotron35Auto,
                sort_key: 6,
            },
            Self::Nemotron3_5AsrStreaming0_6B560MsInt8 => AsrModelInfo {
                language: AsrLanguage::Multilingual,
                supported_language_codes: NEMOTRON_3_5_LANGUAGE_CODES,
                supported_precisions: &[AsrPrecision::Int8],
                default_precision: AsrPrecision::Int8,
                implementation: AsrModelImplementation::Nemotron,
                capability: AsrModelCapability::InterimOnly,
                stream_language: AsrStreamLanguage::Nemotron35Auto,
                sort_key: 7,
            },
        }
    }

    pub fn language(self) -> AsrLanguage {
        self.info().language
    }

    pub fn supported_language_codes(self) -> &'static [&'static str] {
        self.info().supported_language_codes
    }

    pub fn default_for_language(language: AsrLanguage) -> Self {
        match language {
            AsrLanguage::Japanese => Self::ReazonSpeechK2V2,
            AsrLanguage::English => Self::NemoParakeetTdt0_6BV2Int8,
            AsrLanguage::EuropeanMultilingual | AsrLanguage::Multilingual => {
                Self::NemoParakeetTdt0_6BV3Int8
            }
        }
    }

    pub fn supports_precision(self, precision: AsrPrecision) -> bool {
        self.info().supported_precisions.contains(&precision)
    }

    pub fn default_precision(self) -> AsrPrecision {
        self.info().default_precision
    }

    pub fn implementation(self) -> AsrModelImplementation {
        self.info().implementation
    }

    pub fn capability(self) -> AsrModelCapability {
        self.info().capability
    }

    pub fn supports_completion(self) -> bool {
        self.capability() == AsrModelCapability::CompletionAndInterim
    }

    pub fn is_interim_only(self) -> bool {
        self.capability() == AsrModelCapability::InterimOnly
    }

    pub fn stream_language(self) -> AsrStreamLanguage {
        self.info().stream_language
    }

    pub fn sort_key(self) -> u8 {
        self.info().sort_key
    }

    pub fn is_nemotron(self) -> bool {
        self.implementation() == AsrModelImplementation::Nemotron
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
    Multilingual,
}

impl Default for AsrLanguage {
    fn default() -> Self {
        Self::Japanese
    }
}

const PARAKEET_TDT_0_6B_V3_LANGUAGE_CODES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];
const NEMOTRON_3_5_LANGUAGE_CODES: &[&str] = &[
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hi", "hr", "hu", "it", "ja",
    "ko", "nl", "nb", "pl", "pt", "ro", "ru", "sk", "sv", "tr", "uk", "vi", "zh",
];
