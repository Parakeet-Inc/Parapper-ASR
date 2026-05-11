use crate::config::{AsrLanguage, AsrModel, LocalTtsFamily, LocalTtsVoice, NoiseCancellationModel};

pub(crate) const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/refs/tags/v6.0/src/silero_vad/data/silero_vad.onnx";
const ASR_MODEL_BASE_URL: &str =
    "https://huggingface.co/reazon-research/reazonspeech-k2-v2/resolve/main";
const ASR_MODEL_DIR_NAME_JA: &str = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
const ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";
const ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8: &str =
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
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
const LOCAL_TTS_MODEL_BASE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";
const SUPERTONIC2_MODEL_BASE_URL: &str =
    "https://huggingface.co/Supertone/supertonic-2/resolve/main";
const SUPERTONIC3_MODEL_BASE_URL: &str =
    "https://huggingface.co/Supertone/supertonic-3/resolve/main";
const LOCAL_TTS_MODEL_REQUIRED_FILES: &[&str] = &["tokens.txt"];
const SUPERTONIC_ONNX_TTS_REQUIRED_FILES: &[&str] = &[
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
const NOISE_CANCELLATION_MODEL_BASE_URL_UL_UNAS: &str = "https://raw.githubusercontent.com/Xiaobin-Rong/ul-unas/refs/heads/main/ulunas_onnx/onnx_models";
const NOISE_CANCELLATION_MODEL_DIR_NAME_UL_UNAS: &str = "ul-unas";
const NOISE_CANCELLATION_MODEL_FILES_UL_UNAS: &[&str] = &["ulunas_stream_simple.onnx"];

pub(crate) const ALL_ASR_MODELS: &[AsrModel] = &[
    AsrModel::ReazonSpeechK2V2,
    AsrModel::NemoParakeetTdt0_6BV2Int8,
    AsrModel::NemoParakeetTdt0_6BV3Int8,
];

pub(crate) const ALL_NAMO_TURN_DETECTOR_MODELS: &[NamoTurnDetectorModel] = &[
    NamoTurnDetectorModel::Japanese,
    NamoTurnDetectorModel::English,
    NamoTurnDetectorModel::Multilingual,
];

pub(crate) const ALL_NOISE_CANCELLATION_MODELS: &[NoiseCancellationModel] =
    &[NoiseCancellationModel::UlUnas];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamoTurnDetectorModel {
    Japanese,
    English,
    Multilingual,
}

impl NamoTurnDetectorModel {
    pub fn for_asr_language(language: AsrLanguage) -> Self {
        match language {
            AsrLanguage::Japanese => Self::Japanese,
            AsrLanguage::English => Self::English,
            AsrLanguage::EuropeanMultilingual => Self::Multilingual,
        }
    }
}

pub(crate) fn asr_model_base_url(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_BASE_URL,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_BASE_URL_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
    }
}

pub(crate) fn asr_model_dir_name(model: AsrModel) -> &'static str {
    match model {
        AsrModel::ReazonSpeechK2V2 => ASR_MODEL_DIR_NAME_JA,
        AsrModel::NemoParakeetTdt0_6BV2Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V2_INT8,
        AsrModel::NemoParakeetTdt0_6BV3Int8 => ASR_MODEL_DIR_NAME_NEMO_PARAKEET_TDT_0_6B_V3_INT8,
    }
}

pub(crate) fn asr_model_required_file_names(
    model: AsrModel,
    precision: crate::config::AsrPrecision,
) -> &'static [&'static str] {
    match model {
        AsrModel::ReazonSpeechK2V2 => match precision {
            crate::config::AsrPrecision::Int8 => &[
                "encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.int8.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
                "tokens.txt",
            ],
            crate::config::AsrPrecision::Int8Float32 => &[
                "encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
                "tokens.txt",
            ],
            crate::config::AsrPrecision::Float32 => &[
                "encoder-epoch-99-avg-1.onnx",
                "decoder-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.onnx",
                "tokens.txt",
            ],
        },
        AsrModel::NemoParakeetTdt0_6BV2Int8 | AsrModel::NemoParakeetTdt0_6BV3Int8 => &[
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ],
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

pub(crate) fn local_tts_model_base_url() -> &'static str {
    LOCAL_TTS_MODEL_BASE_URL
}

pub(crate) fn supertonic_tts_model_base_url(voice: LocalTtsVoice) -> &'static str {
    match voice {
        LocalTtsVoice::Supertonic3Onnx => SUPERTONIC3_MODEL_BASE_URL,
        _ => SUPERTONIC2_MODEL_BASE_URL,
    }
}

pub(crate) fn local_tts_model_archive_name(voice: LocalTtsVoice) -> String {
    format!("{}.tar.bz2", voice.dir_name())
}

pub(crate) fn local_tts_model_required_file_names(voice: LocalTtsVoice) -> Vec<&'static str> {
    if voice.family() == LocalTtsFamily::Supertonic {
        return SUPERTONIC_ONNX_TTS_REQUIRED_FILES.to_vec();
    }

    let mut files = Vec::with_capacity(LOCAL_TTS_MODEL_REQUIRED_FILES.len() + 1);
    files.push(voice.onnx_file_name());
    files.extend_from_slice(LOCAL_TTS_MODEL_REQUIRED_FILES);
    files
}

pub(crate) fn local_tts_model_required_dir_names(voice: LocalTtsVoice) -> &'static [&'static str] {
    match voice.family() {
        LocalTtsFamily::Vits => &["espeak-ng-data"],
        LocalTtsFamily::Supertonic => &[],
    }
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
