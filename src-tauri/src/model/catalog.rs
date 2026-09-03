use crate::config::{AsrModel, LocalTranslationModel, LocalTtsVoice, NoiseCancellationModel};
pub use parapper_stt_engine::NamoTurnDetectorModel;

pub(crate) const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.0/src/silero_vad/data/silero_vad.onnx";
const ASR_MODEL_BASE_URL: &str = "https://huggingface.co/reazon-research/reazonspeech-k2-v2/resolve/291488c8151be24d7da4bf7af26e533fad96e407";
const ASR_MODEL_DIR_NAME_JA: &str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01";
const REAZON_STATIC_EMBEDDING_BASE_URL: &str = "https://huggingface.co/hotchpotch/static-embedding-japanese/resolve/95b3d9c80a7ccf604e2b5daee7b1b3eed6b1a9d3";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_CTC_0_6B_JA_35000_INT8: &str = "https://huggingface.co/nadare/parakeet-tdt_ctc-0.6b-ja-onnx-dynamic-int8/resolve/ab9073e4b457a4eb3df4e362946404be8adc0b1e";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_CTC_0_6B_JA_35000_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/2bda32ec70b097a55adaa07d9a7173915b43cc78";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const ASR_MODEL_BASE_URL_SHERPA_ONNX_RELEASES: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";
const ASR_MODEL_DIR_NAME_NEMOTRON_SPEECH_STREAMING_EN_0_6B_560MS_INT8: &str =
    "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25";
const ASR_MODEL_DIR_NAME_NEMOTRON_3_5_ASR_STREAMING_0_6B_560MS_INT8: &str =
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11";
const ASR_MODEL_ARCHIVE_INTEGRITY_NEMOTRON_SPEECH_STREAMING_EN_0_6B_560MS_INT8: FileIntegrity =
    FileIntegrity {
        size: 463_945_051,
        sha256: "78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358",
    };
const ASR_MODEL_ARCHIVE_INTEGRITY_NEMOTRON_3_5_ASR_STREAMING_0_6B_560MS_INT8: FileIntegrity =
    FileIntegrity {
        size: 475_271_763,
        sha256: "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
    };
const SPEECHBRAIN_ECAPA_MODEL_DIR: &str = "speechbrain-lang-id-voxlingua107-ecapa-onnx";
const SPEECHBRAIN_ECAPA_BASE_URL: &str =
    "https://huggingface.co/drakulavich/SpeechBrain-coreml/resolve/main";
const SPEECHBRAIN_ECAPA_FILES: &[&str] = &[
    "lang-id-ecapa.onnx",
    "lang-id-ecapa.onnx.data",
    "labels.json",
];
const NAMO_TURN_DETECTOR_BASE_URL: &str =
    "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Japanese/resolve/main";
const NAMO_TURN_DETECTOR_DIR_NAME: &str = "namo-turn-detector-v1-japanese";
const NAMO_TURN_DETECTOR_BASE_URL_ENGLISH: &str =
    "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English/resolve/main";
const NAMO_TURN_DETECTOR_DIR_NAME_ENGLISH: &str = "namo-turn-detector-v1-english";
const NAMO_TURN_DETECTOR_BASE_URL_MULTILINGUAL: &str =
    "https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual/resolve/main";
const NAMO_TURN_DETECTOR_DIR_NAME_MULTILINGUAL: &str = "namo-turn-detector-v1-multilingual";
const NAMO_TURN_DETECTOR_FILES_JAPANESE: &[&str] = &[
    "config.json",
    "model_quant.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
];
const NAMO_TURN_DETECTOR_FILES_ENGLISH: &[&str] = NAMO_TURN_DETECTOR_FILES_JAPANESE;
const NAMO_TURN_DETECTOR_FILES_MULTILINGUAL: &[&str] = &[
    "config.json",
    "model_quant.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
];
const SUPERTONIC2_MODEL_BASE_URL: &str =
    "https://huggingface.co/Supertone/supertonic-2/resolve/main";
const SUPERTONIC3_MODEL_BASE_URL: &str =
    "https://huggingface.co/Supertone/supertonic-3/resolve/main";
const SUPERTONIC3_QUANTIZED_MODEL_BASE_URL: &str = "https://huggingface.co/nadare/supertonic-3-onnx-q4/resolve/0831a17d4f7de14ade46364ec447d50e24ff1f82";
const LOCAL_TRANSLATION_MODEL_BASE_URL_LFM2_Q4: Option<&str> =
    Some("https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX/resolve/main");
const LOCAL_TRANSLATION_MODEL_BASE_URL_LFM2_LICENSE: &str = "https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT/resolve/80367784d525777ad7565b24534ba5810eeac59f";
const LOCAL_TRANSLATION_MODEL_BASE_URL_CAT_TRANSLATE_0_8B_Q4_K_QUANT: Option<&str> = Some(
    "https://huggingface.co/nadare/CAT-Translate-0.8b-onnx-q4-k-quant/resolve/a6369bfcaa1f7c9a8df7294c6b2011286e5dc843",
);
const LOCAL_TRANSLATION_MODEL_DIR_NAME_LFM2_Q4: &str = "lfm2-350m-enjp-mt-onnx-q4";
const LOCAL_TRANSLATION_MODEL_DIR_NAME_CAT_TRANSLATE_0_8B_Q4_K_QUANT: &str =
    "cat-translate-0.8b-onnx-q4-k-quant";
const LOCAL_TRANSLATION_MODEL_FILES_LFM2_Q4: &[&str] = &[
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model_q4.onnx",
    "onnx/model_q4.onnx_data",
];
const LOCAL_TRANSLATION_MODEL_FILES_CAT_TRANSLATE_0_8B_Q4_K_QUANT: &[&str] = &[
    "chat_template.jinja",
    "genai_config.json",
    "model_q4.onnx",
    "model_q4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "build-metadata.json",
    "distribution-manifest.json",
    "SHA256SUMS",
];
const SUPERTONIC_ONNX_TTS_REQUIRED_FILES: &[&str] = &[
    "LICENSE",
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
];
const SUPERTONIC3_QUANTIZED_ONNX_TTS_REQUIRED_FILES: &[&str] = &[
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "voice_styles/F1.json",
    "voice_styles/F2.json",
    "voice_styles/F3.json",
    "voice_styles/F4.json",
    "voice_styles/F5.json",
    "voice_styles/M1.json",
    "voice_styles/M2.json",
    "voice_styles/M3.json",
    "voice_styles/M4.json",
    "voice_styles/M5.json",
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "MODIFICATIONS.md",
    "build-metadata.json",
    "quantization-report.json",
    "distribution-manifest.json",
    "SHA256SUMS",
];
const NOISE_CANCELLATION_MODEL_BASE_URL_UL_UNAS: &str = "https://raw.githubusercontent.com/Xiaobin-Rong/ul-unas/refs/heads/main/ulunas_onnx/onnx_models";
const NOISE_CANCELLATION_MODEL_DIR_NAME_UL_UNAS: &str = "ul-unas";
const NOISE_CANCELLATION_MODEL_FILES_UL_UNAS: &[&str] = &["ulunas_stream_simple.onnx"];

pub(crate) const ALL_ASR_MODELS: &[AsrModel] = &[
    AsrModel::ReazonSpeechK2V2,
    AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
    AsrModel::NemoParakeetTdt0_6BV2Int8,
    AsrModel::NemoParakeetTdt0_6BV3Int8,
    AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
    AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
    AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
    AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
    AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
    AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
    AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
    AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
    AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
    AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
];

pub(crate) const ALL_NAMO_TURN_DETECTOR_MODELS: &[NamoTurnDetectorModel] = &[
    NamoTurnDetectorModel::Japanese,
    NamoTurnDetectorModel::English,
    NamoTurnDetectorModel::Multilingual,
];

pub(crate) const ALL_NOISE_CANCELLATION_MODELS: &[NoiseCancellationModel] =
    &[NoiseCancellationModel::UlUnas];
pub(crate) const ALL_LOCAL_TRANSLATION_MODELS: &[LocalTranslationModel] = &[
    LocalTranslationModel::Lfm2Q4,
    LocalTranslationModel::CatTranslate0_8BQ4KQuant,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileIntegrity {
    pub size: u64,
    pub sha256: &'static str,
}

pub(crate) fn asr_model_base_url(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_BASE_URL,
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 => {
            ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_CTC_0_6B_JA_35000_INT8
        }
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
        AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => {
            ASR_MODEL_BASE_URL_SHERPA_ONNX_RELEASES
        }
    }
}

pub(crate) fn asr_model_dir_name(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_DIR_NAME_JA,
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 => {
            ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_CTC_0_6B_JA_35000_INT8
        }
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
        AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8 => {
            ASR_MODEL_DIR_NAME_NEMOTRON_SPEECH_STREAMING_EN_0_6B_560MS_INT8
        }
        AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => {
            ASR_MODEL_DIR_NAME_NEMOTRON_3_5_ASR_STREAMING_0_6B_560MS_INT8
        }
    }
}

pub(crate) fn asr_model_archive_name(model: AsrModel) -> Option<String> {
    model
        .is_nemotron()
        .then(|| format!("{}.tar.bz2", asr_model_dir_name(model)))
}

pub(crate) fn asr_model_archive_integrity(model: AsrModel) -> Option<FileIntegrity> {
    match model {
        AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
        | AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8 => {
            Some(ASR_MODEL_ARCHIVE_INTEGRITY_NEMOTRON_SPEECH_STREAMING_EN_0_6B_560MS_INT8)
        }
        AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8
        | AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8 => {
            Some(ASR_MODEL_ARCHIVE_INTEGRITY_NEMOTRON_3_5_ASR_STREAMING_0_6B_560MS_INT8)
        }
        AsrModel::ReazonSpeechK2V2
        | AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
        | AsrModel::NemoParakeetTdt0_6BV2Int8
        | AsrModel::NemoParakeetTdt0_6BV3Int8 => None,
    }
}

pub(crate) fn asr_model_required_file_names(
    model: AsrModel,
    precision: crate::config::AsrPrecision,
) -> &'static [&'static str] {
    parapper_models::asr::backend::required_model_file_names(model, precision)
}

pub(crate) fn asr_model_file_integrity(model: AsrModel, file_name: &str) -> Option<FileIntegrity> {
    if model != AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8 {
        return None;
    }
    let integrity = match file_name {
        "encoder-model.int8.onnx" => FileIntegrity {
            size: 3_387_454,
            sha256: "3c4f14996c134b1e6ca2853230ccc0d36c66085af0b6272e002dfba0f25c5f5a",
        },
        "encoder-model.int8.onnx.data" => FileIntegrity {
            size: 878_444_544,
            sha256: "a39578a7c16db2d7024dbe8f90ae99892cef69a8bb833b5d1223f9122a7e87bc",
        },
        "decoder_joint-model.onnx" => FileIntegrity {
            size: 46_286_012,
            sha256: "64e64c50b7df62707dc7ac1e8b1ea804b9f590db88d84e4abcdf4d1188cf9b8a",
        },
        "ctc-head-model.onnx" => FileIntegrity {
            size: 959,
            sha256: "23f65702c75a984fa16c95bd284b66b60f9d6693bf1169dc2947b56734cc72d9",
        },
        "ctc-head-model.onnx_data" => FileIntegrity {
            size: 12_599_300,
            sha256: "662b5f6b91dd4987ac9d139cc185501979b4de51947b5ba47b4d09cae75287b9",
        },
        "vocab.txt" => FileIntegrity {
            size: 28_557,
            sha256: "732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a",
        },
        _ => return None,
    };
    Some(integrity)
}

pub(crate) const fn reazon_static_embedding_base_url() -> &'static str {
    REAZON_STATIC_EMBEDDING_BASE_URL
}

pub(crate) fn reazon_static_embedding_file_integrity(file_name: &str) -> Option<FileIntegrity> {
    match file_name {
        "0_StaticEmbedding/tokenizer.json" => Some(FileIntegrity {
            size: 2_127_941,
            sha256: "833add01c9eb44e78ffb2d9195caace320de0fcf64d1f4d95bc541b6e30a9fc9",
        }),
        "0_StaticEmbedding/model.safetensors" => Some(FileIntegrity {
            size: 134_217_824,
            sha256: "f0c60b3d2952fb89e67a063ac4aa558ff4b02facaac5fd674d637b9e2c52ccca",
        }),
        _ => None,
    }
}

pub(crate) fn language_id_model_dir_name() -> &'static str {
    SPEECHBRAIN_ECAPA_MODEL_DIR
}

pub(crate) fn language_id_model_base_url() -> &'static str {
    SPEECHBRAIN_ECAPA_BASE_URL
}

pub(crate) fn language_id_model_files() -> &'static [&'static str] {
    SPEECHBRAIN_ECAPA_FILES
}

pub(crate) fn namo_turn_detector_base_url(model: NamoTurnDetectorModel) -> &'static str {
    match model {
        NamoTurnDetectorModel::Japanese => NAMO_TURN_DETECTOR_BASE_URL,
        NamoTurnDetectorModel::English => NAMO_TURN_DETECTOR_BASE_URL_ENGLISH,
        NamoTurnDetectorModel::Multilingual => NAMO_TURN_DETECTOR_BASE_URL_MULTILINGUAL,
    }
}

pub(crate) fn namo_turn_detector_dir_name(model: NamoTurnDetectorModel) -> &'static str {
    match model {
        NamoTurnDetectorModel::Japanese => NAMO_TURN_DETECTOR_DIR_NAME,
        NamoTurnDetectorModel::English => NAMO_TURN_DETECTOR_DIR_NAME_ENGLISH,
        NamoTurnDetectorModel::Multilingual => NAMO_TURN_DETECTOR_DIR_NAME_MULTILINGUAL,
    }
}

pub(crate) fn namo_turn_detector_files(model: NamoTurnDetectorModel) -> &'static [&'static str] {
    match model {
        NamoTurnDetectorModel::Japanese => NAMO_TURN_DETECTOR_FILES_JAPANESE,
        NamoTurnDetectorModel::English => NAMO_TURN_DETECTOR_FILES_ENGLISH,
        NamoTurnDetectorModel::Multilingual => NAMO_TURN_DETECTOR_FILES_MULTILINGUAL,
    }
}

pub(crate) fn supertonic_tts_model_base_url(voice: LocalTtsVoice) -> &'static str {
    match voice {
        LocalTtsVoice::Supertonic2Onnx => SUPERTONIC2_MODEL_BASE_URL,
        LocalTtsVoice::Supertonic3Onnx => SUPERTONIC3_MODEL_BASE_URL,
        LocalTtsVoice::Supertonic3OnnxQuantized => SUPERTONIC3_QUANTIZED_MODEL_BASE_URL,
    }
}

pub(crate) fn local_tts_model_required_file_names(voice: LocalTtsVoice) -> Vec<&'static str> {
    if voice == LocalTtsVoice::Supertonic3OnnxQuantized {
        return SUPERTONIC3_QUANTIZED_ONNX_TTS_REQUIRED_FILES.to_vec();
    }
    SUPERTONIC_ONNX_TTS_REQUIRED_FILES.to_vec()
}

// This is a direct, auditable mapping of every file in the immutable release manifest.
#[allow(clippy::too_many_lines)]
pub(crate) fn local_tts_model_file_integrity(
    voice: LocalTtsVoice,
    file_name: &str,
) -> Option<FileIntegrity> {
    if voice != LocalTtsVoice::Supertonic3OnnxQuantized {
        return None;
    }
    let integrity = match file_name {
        "onnx/duration_predictor.onnx" => FileIntegrity {
            size: 3_700_147,
            sha256: "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
        },
        "onnx/text_encoder.onnx" => FileIntegrity {
            size: 36_416_150,
            sha256: "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
        },
        "onnx/vector_estimator.onnx" => FileIntegrity {
            size: 51_663_166,
            sha256: "1564c34bdb897c0006349213655979f9a7c573f27effe7ea1417f984d2315b04",
        },
        "onnx/vocoder.onnx" => FileIntegrity {
            size: 40_688_430,
            sha256: "cc4c42b8cb107cd63b352f037308c783589112535d51800e0a3680aad7bb8850",
        },
        "onnx/tts.json" => FileIntegrity {
            size: 8_253,
            sha256: "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
        },
        "onnx/unicode_indexer.json" => FileIntegrity {
            size: 277_676,
            sha256: "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
        },
        "voice_styles/F1.json" => FileIntegrity {
            size: 292_046,
            sha256: "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
        },
        "voice_styles/F2.json" => FileIntegrity {
            size: 292_423,
            sha256: "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6",
        },
        "voice_styles/F3.json" => FileIntegrity {
            size: 290_794,
            sha256: "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab",
        },
        "voice_styles/F4.json" => FileIntegrity {
            size: 291_808,
            sha256: "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b",
        },
        "voice_styles/F5.json" => FileIntegrity {
            size: 291_479,
            sha256: "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d",
        },
        "voice_styles/M1.json" => FileIntegrity {
            size: 291_748,
            sha256: "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b",
        },
        "voice_styles/M2.json" => FileIntegrity {
            size: 292_055,
            sha256: "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50",
        },
        "voice_styles/M3.json" => FileIntegrity {
            size: 290_198,
            sha256: "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b",
        },
        "voice_styles/M4.json" => FileIntegrity {
            size: 291_522,
            sha256: "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303",
        },
        "voice_styles/M5.json" => FileIntegrity {
            size: 291_469,
            sha256: "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2",
        },
        "LICENSE" => FileIntegrity {
            size: 15_007,
            sha256: "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
        },
        "MODEL_CARD.md" => FileIntegrity {
            size: 2_017,
            sha256: "8aed0ef6c691939f187160cdc589310ebd29057ebc55675ab77712c49ff1e27c",
        },
        "THIRD_PARTY_NOTICES.md" => FileIntegrity {
            size: 535,
            sha256: "8a88f972c84ba9f01980045d101ee03296e6517fdc5b659c9398e1cf6d3dc962",
        },
        "MODIFICATIONS.md" => FileIntegrity {
            size: 954,
            sha256: "36109f45382350855739589fb8b874716427ee11bca4d11613771244daf4b6e9",
        },
        "build-metadata.json" => FileIntegrity {
            size: 691,
            sha256: "6d96ef06289c2543eb3601ce0d4135cee6c8c9099b5fffe6708f26272a60c021",
        },
        "quantization-report.json" => FileIntegrity {
            size: 2_561,
            sha256: "6d5c06a480d87ecc858cf198f09878064ee2f683685f9f81fae58505f14f27fd",
        },
        "distribution-manifest.json" => FileIntegrity {
            size: 3_294,
            sha256: "75aebf93f7a2e24658e123c9b1e64101bbf05d2f0f68e61b3023449b53fa73b6",
        },
        "SHA256SUMS" => FileIntegrity {
            size: 1_906,
            sha256: "56371dd70960c8c65ac58d0210c04691e9d91814c19839285d808581581a3414",
        },
        _ => return None,
    };
    Some(integrity)
}

pub(crate) const fn local_tts_model_required_dir_names(
    _voice: LocalTtsVoice,
) -> &'static [&'static str] {
    &[]
}

pub(crate) fn local_translation_model_base_url(
    model: LocalTranslationModel,
) -> Option<&'static str> {
    match model {
        LocalTranslationModel::Lfm2Q4 => LOCAL_TRANSLATION_MODEL_BASE_URL_LFM2_Q4,
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => {
            LOCAL_TRANSLATION_MODEL_BASE_URL_CAT_TRANSLATE_0_8B_Q4_K_QUANT
        }
    }
}

pub(crate) fn local_translation_model_file_base_url(
    model: LocalTranslationModel,
    file_name: &str,
) -> Option<&'static str> {
    match (model, file_name) {
        (LocalTranslationModel::Lfm2Q4, "LICENSE") => {
            Some(LOCAL_TRANSLATION_MODEL_BASE_URL_LFM2_LICENSE)
        }
        _ => local_translation_model_base_url(model),
    }
}

pub(crate) fn local_translation_model_dir_name(model: LocalTranslationModel) -> &'static str {
    match model {
        LocalTranslationModel::Lfm2Q4 => LOCAL_TRANSLATION_MODEL_DIR_NAME_LFM2_Q4,
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => {
            LOCAL_TRANSLATION_MODEL_DIR_NAME_CAT_TRANSLATE_0_8B_Q4_K_QUANT
        }
    }
}

pub(crate) fn local_translation_model_required_file_names(
    model: LocalTranslationModel,
) -> &'static [&'static str] {
    match model {
        LocalTranslationModel::Lfm2Q4 => LOCAL_TRANSLATION_MODEL_FILES_LFM2_Q4,
        LocalTranslationModel::CatTranslate0_8BQ4KQuant => {
            LOCAL_TRANSLATION_MODEL_FILES_CAT_TRANSLATE_0_8B_Q4_K_QUANT
        }
    }
}

pub(crate) fn local_translation_model_file_integrity(
    model: LocalTranslationModel,
    file_name: &str,
) -> Option<FileIntegrity> {
    if model != LocalTranslationModel::CatTranslate0_8BQ4KQuant {
        return None;
    }

    let integrity = match file_name {
        "chat_template.jinja" => FileIntegrity {
            size: 2_583,
            sha256: "5a83f4ff2e2b57f292109af59619af3548ac4e88d3ceffaff56a9f12772e8db3",
        },
        "genai_config.json" => FileIntegrity {
            size: 1_509,
            sha256: "3a846b05b094eff00930a19ff128abf9fee807a2dfec8a21034f1a5de89bd9cb",
        },
        "model_q4.onnx" => FileIntegrity {
            size: 211_164,
            sha256: "af6fac6bb8df46ce7cffecde2fca833a92b64d4e46c5d873abb7fc3d60423fc3",
        },
        "model_q4.onnx.data" => FileIntegrity {
            size: 596_894_720,
            sha256: "66839e48f81021eb3f6cf888b57411021914555f705024b15bd76a15e0956480",
        },
        "special_tokens_map.json" => FileIntegrity {
            size: 1_019,
            sha256: "e4342cff1d582bc32705b2cad654071dc45810045f7c85eae35a220f20d69fbb",
        },
        "tokenizer.json" => FileIntegrity {
            size: 6_724_007,
            sha256: "0197cf057f75f7033095c881fc6c3b055aded47d0b59f67ed06c29ba5c80eed1",
        },
        "tokenizer.model" => FileIntegrity {
            size: 1_831_879,
            sha256: "008293028e1a9d9a1038d9b63d989a2319797dfeaa03f171093a57b33a3a8277",
        },
        "tokenizer_config.json" => FileIntegrity {
            size: 3_951,
            sha256: "8cf1949a4c2beab24a4c6b8ac4274a826c3b49078f04ef79227654515eda2be5",
        },
        "LICENSE" => FileIntegrity {
            size: 1_074,
            sha256: "b117dfdeb28b464adf227207c9129bdd6a1ec9de5852a01bd0ffaa6a7ab0d4f0",
        },
        "MODEL_CARD.md" => FileIntegrity {
            size: 2_401,
            sha256: "97d75205e1a8f97574581f78ed379e9bfc5da9ee9f7adf3f15d09b986995b368",
        },
        "THIRD_PARTY_NOTICES.md" => FileIntegrity {
            size: 799,
            sha256: "17b5d8f9790f0110c5622b84bd21d4b4519eda59b63513f769d705a8f23f67fc",
        },
        "build-metadata.json" => FileIntegrity {
            size: 1_463,
            sha256: "079bf61946eef4607bda2f0395effc2ea9c6a0dc52e4ea6d49751a6746bf9e35",
        },
        "distribution-manifest.json" => FileIntegrity {
            size: 3_332,
            sha256: "49b40a1eb59f6ce712cc98af02e2617d3589b0021a9f38aa3d1ef2b8da862b5e",
        },
        "SHA256SUMS" => FileIntegrity {
            size: 1_098,
            sha256: "8ea104c89c36ac3ade07ca90ff25d9edc294faa3273c2d09886a229ae786a571",
        },
        _ => return None,
    };
    Some(integrity)
}

pub(crate) fn noise_cancellation_model_base_url(model: NoiseCancellationModel) -> &'static str {
    match model {
        NoiseCancellationModel::UlUnas => NOISE_CANCELLATION_MODEL_BASE_URL_UL_UNAS,
    }
}

pub(crate) fn noise_cancellation_model_dir_name(model: NoiseCancellationModel) -> &'static str {
    match model {
        NoiseCancellationModel::UlUnas => NOISE_CANCELLATION_MODEL_DIR_NAME_UL_UNAS,
    }
}

pub(crate) fn noise_cancellation_model_required_file_names(
    model: NoiseCancellationModel,
) -> &'static [&'static str] {
    match model {
        NoiseCancellationModel::UlUnas => NOISE_CANCELLATION_MODEL_FILES_UL_UNAS,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        asr_model_archive_integrity, asr_model_base_url,
        local_translation_model_required_file_names, local_tts_model_required_file_names,
        reazon_static_embedding_base_url, reazon_static_embedding_file_integrity,
    };
    use crate::config::{AsrModel, LocalTranslationModel, LocalTtsVoice};

    #[test]
    fn hugging_face_asr_downloads_resolve_an_immutable_revision() {
        for model in [
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrModel::NemoParakeetTdt0_6BV3Int8,
        ] {
            let url = asr_model_base_url(model);
            assert!(!url.ends_with("/resolve/main"), "unpinned ASR URL: {url}");
            let revision = url
                .rsplit_once("/resolve/")
                .expect("Hugging Face ASR URL must include a revision")
                .1;
            assert_eq!(revision.len(), 40, "ASR revision must be a full SHA");
            assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn reazon_static_reranker_download_resolves_the_evaluated_revision() {
        assert_eq!(
            reazon_static_embedding_base_url(),
            "https://huggingface.co/hotchpotch/static-embedding-japanese/resolve/95b3d9c80a7ccf604e2b5daee7b1b3eed6b1a9d3"
        );
        assert!(
            parapper_models::asr::backend::REAZON_STATIC_EMBEDDING_REQUIRED_FILES
                .iter()
                .all(|file| reazon_static_embedding_file_integrity(file).is_some())
        );
    }

    #[test]
    fn every_nemotron_archive_has_a_pinned_size_and_digest() {
        for model in [
            AsrModel::NemotronSpeechStreamingEn0_6B80MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B320MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B1120MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B80MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B320MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B1120MsInt8,
        ] {
            let integrity = asr_model_archive_integrity(model)
                .expect("Nemotron release assets must have an integrity contract");
            assert!(integrity.size > 0);
            assert_eq!(integrity.sha256.len(), 64);
            assert!(
                integrity
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            );
        }
    }

    #[test]
    fn downloaded_model_distributions_include_their_required_license_documents() {
        assert!(
            local_translation_model_required_file_names(LocalTranslationModel::Lfm2Q4)
                .contains(&"LICENSE"),
            "the LFM2 distribution must retain its license beside the downloaded model"
        );

        for voice in [
            LocalTtsVoice::Supertonic2Onnx,
            LocalTtsVoice::Supertonic3Onnx,
        ] {
            assert!(
                local_tts_model_required_file_names(voice).contains(&"LICENSE"),
                "{voice:?} must retain its OpenRAIL-M license beside the downloaded model"
            );
        }
    }

    #[test]
    fn upstream_asr_attributions_reach_every_public_license_surface() {
        let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("src-tauri must have a repository parent");
        let japanese_readme = std::fs::read_to_string(repository_root.join("README.md"))
            .expect("Japanese README must be readable");
        let english_readme =
            std::fs::read_to_string(repository_root.join("documents/README.en.md"))
                .expect("English README must be readable");
        let license_ui =
            std::fs::read_to_string(repository_root.join("src/components/ui/licenses.tsx"))
                .expect("application license UI must be readable");
        let notice_path = repository_root.join("public/licenses/THIRD_PARTY_NOTICES.md");
        let notice = std::fs::read_to_string(&notice_path)
            .expect("third-party notice must be shipped as an application asset");
        let bundle_config = include_str!("../../tauri.conf.json");
        let macos_bundle_config = include_str!("../../tauri.macos.conf.json");

        for (name, surface) in [
            ("Japanese README", japanese_readme.as_str()),
            ("English README", english_readme.as_str()),
            ("application license UI", license_ui.as_str()),
            ("third-party notice", notice.as_str()),
        ] {
            for upstream in [
                "https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja",
                "https://github.com/NVIDIA/NeMo",
                "https://github.com/k2-fsa/sherpa-onnx",
            ] {
                assert!(surface.contains(upstream), "{name} omits {upstream}");
            }
        }
        assert!(notice.contains("CC-BY-4.0"));
        assert!(notice.contains("Apache-2.0"));
        assert!(license_ui.contains("/licenses/THIRD_PARTY_NOTICES.md"));
        for config in [bundle_config, macos_bundle_config] {
            assert!(config.contains("../public/licenses/THIRD_PARTY_NOTICES.md"));
            assert!(config.contains("licenses/THIRD_PARTY_NOTICES.md"));
        }
    }

    #[test]
    fn every_embedded_dictionary_license_reaches_the_installer_and_application() {
        let bundled_notice = include_str!("../../resources/hotword-reading/NOTICE.md");
        let displayed_notice = include_str!("../../../public/licenses/hotword-reading/NOTICE.md");
        let apache_license =
            include_str!("../../../public/licenses/hotword-reading/LICENSE-APACHE-2.0.txt");
        let bundled_morph_notice = include_str!("../../resources/morph/NOTICE");
        let bundled_morph_bsd = include_str!("../../resources/morph/BSD");
        let bundled_morph_authors = include_str!("../../resources/morph/AUTHORS");
        let displayed_morph_notice = include_str!("../../../public/licenses/morph/NOTICE");
        let displayed_morph_bsd = include_str!("../../../public/licenses/morph/BSD");
        let displayed_morph_authors = include_str!("../../../public/licenses/morph/AUTHORS");
        let bundle_config = include_str!("../../tauri.conf.json");
        let macos_bundle_config = include_str!("../../tauri.macos.conf.json");

        for notice in [bundled_notice, displayed_notice] {
            assert!(notice.contains("SudachiDict"));
            assert!(notice.contains("UniDic Consortium"));
            assert!(notice.contains("Carnegie Mellon Pronouncing"));
        }
        // `include_str!` preserves checkout line endings. `str::lines` accepts
        // both the LF form in the repository and a CRLF checkout on Windows.
        assert_eq!(
            apache_license.lines().next(),
            Some("Apache License"),
            "the bundled hotword dictionary license must be the Apache License 2.0 text"
        );
        assert!(apache_license.contains("TERMS AND CONDITIONS FOR USE"));
        assert_eq!(bundled_morph_notice, displayed_morph_notice);
        assert_eq!(bundled_morph_bsd, displayed_morph_bsd);
        assert_eq!(bundled_morph_authors, displayed_morph_authors);
        assert!(bundled_morph_notice.contains("UniDic for Contemporary"));
        assert!(bundled_morph_notice.contains("Written Japanese 3.1.1"));
        assert!(bundled_morph_bsd.contains("Redistribution and use in source and binary forms"));
        assert!(bundled_morph_authors.contains("The UniDic Consortium"));

        for config in [bundle_config, macos_bundle_config] {
            let config: serde_json::Value =
                serde_json::from_str(config).expect("bundle configuration must be valid JSON");
            let resources = config
                .pointer("/bundle/resources")
                .and_then(serde_json::Value::as_object)
                .expect("bundle configuration must declare installed resources");
            for installed_document in [
                "licenses/hotword-reading/NOTICE.md",
                "licenses/hotword-reading/LICENSE-APACHE-2.0.txt",
                "licenses/morph/NOTICE",
                "licenses/morph/BSD",
                "licenses/morph/AUTHORS",
            ] {
                assert!(
                    resources
                        .values()
                        .any(|destination| destination.as_str() == Some(installed_document)),
                    "bundle configuration must install {installed_document}"
                );
            }
        }
    }
}
