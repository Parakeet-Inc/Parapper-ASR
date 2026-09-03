//! Local text-to-speech models.

#[cfg(feature = "tts-supertonic-ort")]
mod engine;
#[cfg(feature = "tts-supertonic-ort")]
mod supertonic_onnx;

#[cfg(feature = "tts-supertonic-ort")]
pub use engine::{LocalTtsEngine, SynthesizedTtsAudio};
#[cfg(feature = "tts-supertonic-ort")]
pub use supertonic_onnx::SupertonicOnnxTtsEngine;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum LocalTtsVoice {
    #[serde(rename = "supertonic_2_onnx")]
    #[default]
    Supertonic2Onnx,
    #[serde(rename = "supertonic_3_onnx")]
    Supertonic3Onnx,
    #[serde(rename = "supertonic_3_onnx_quantized")]
    Supertonic3OnnxQuantized,
}

impl LocalTtsVoice {
    #[must_use]
    pub const fn dir_name(self) -> &'static str {
        match self {
            Self::Supertonic2Onnx => "supertonic-2-onnx",
            Self::Supertonic3Onnx => "supertonic-3-onnx",
            Self::Supertonic3OnnxQuantized => "supertonic-3-onnx-quantized",
        }
    }

    #[must_use]
    pub const fn onnx_file_name(self) -> &'static str {
        "onnx/duration_predictor.onnx"
    }

    #[must_use]
    pub const fn supported_language_codes(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Supertonic2Onnx => Some(SUPERTONIC2_LANGUAGE_CODES),
            Self::Supertonic3Onnx | Self::Supertonic3OnnxQuantized => {
                Some(SUPERTONIC3_LANGUAGE_CODES)
            }
        }
    }
}

pub const ALL_LOCAL_TTS_VOICES: &[LocalTtsVoice] = &[
    LocalTtsVoice::Supertonic2Onnx,
    LocalTtsVoice::Supertonic3Onnx,
    LocalTtsVoice::Supertonic3OnnxQuantized,
];

pub const SUPERTONIC2_LANGUAGE_CODES: &[&str] = &["en", "ko", "es", "pt", "fr"];
pub const SUPERTONIC3_LANGUAGE_CODES: &[&str] = &[
    "en", "ko", "ja", "bg", "cs", "da", "el", "es", "et", "fi", "hu", "it", "nl", "pl", "pt", "ro",
    "ar", "de", "fr", "hi", "id", "ru", "vi",
];

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn local_tts_voice_distribution_metadata_is_complete_and_stable() {
        let actual = ALL_LOCAL_TTS_VOICES
            .iter()
            .map(|voice| {
                (
                    serde_json::to_string(voice).expect("voice should serialize"),
                    voice.dir_name(),
                    voice.onnx_file_name(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (
                    "\"supertonic_2_onnx\"".to_string(),
                    "supertonic-2-onnx",
                    "onnx/duration_predictor.onnx",
                ),
                (
                    "\"supertonic_3_onnx\"".to_string(),
                    "supertonic-3-onnx",
                    "onnx/duration_predictor.onnx",
                ),
                (
                    "\"supertonic_3_onnx_quantized\"".to_string(),
                    "supertonic-3-onnx-quantized",
                    "onnx/duration_predictor.onnx",
                ),
            ]
        );
        assert_eq!(LocalTtsVoice::default(), LocalTtsVoice::Supertonic2Onnx);
    }

    #[test]
    fn supertonic_language_contract_is_versioned_by_voice() {
        assert!(!SUPERTONIC2_LANGUAGE_CODES.contains(&"ja"));
        assert!(SUPERTONIC3_LANGUAGE_CODES.contains(&"ja"));
        assert_eq!(
            LocalTtsVoice::Supertonic3Onnx.supported_language_codes(),
            LocalTtsVoice::Supertonic3OnnxQuantized.supported_language_codes()
        );
    }
}
