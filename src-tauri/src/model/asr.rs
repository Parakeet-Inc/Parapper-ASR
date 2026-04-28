use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::{
    audio::ASR_SAMPLE_RATE,
    config::{AsrModel, AsrPrecision},
};

pub trait AsrEngine: Send {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String>;
}

pub struct SherpaOnnxAsrEngine {
    recognizer: OfflineRecognizer,
}

unsafe impl Send for SherpaOnnxAsrEngine {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SherpaOnnxTransducerModelFiles {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct SherpaOnnxTransducerFileNames {
    pub encoder: &'static str,
    pub decoder: &'static str,
    pub joiner: &'static str,
    pub tokens: &'static str,
}

impl SherpaOnnxTransducerModelFiles {
    pub fn from_dir(model_dir: &Path, model: AsrModel, precision: AsrPrecision) -> Result<Self> {
        let names = Self::file_names(model, precision);
        let files = Self {
            encoder: model_dir.join(names.encoder),
            decoder: model_dir.join(names.decoder),
            joiner: model_dir.join(names.joiner),
            tokens: model_dir.join(names.tokens),
        };
        files.validate()?;
        Ok(files)
    }

    pub fn required_file_names(
        model: AsrModel,
        precision: AsrPrecision,
    ) -> &'static [&'static str] {
        match model {
            AsrModel::ReazonSpeechK2V2 => match precision {
                AsrPrecision::Int8 => &[
                    "encoder-epoch-99-avg-1.int8.onnx",
                    "decoder-epoch-99-avg-1.int8.onnx",
                    "joiner-epoch-99-avg-1.int8.onnx",
                    "tokens.txt",
                ],
                AsrPrecision::Int8Float32 => &[
                    "encoder-epoch-99-avg-1.int8.onnx",
                    "decoder-epoch-99-avg-1.onnx",
                    "joiner-epoch-99-avg-1.int8.onnx",
                    "tokens.txt",
                ],
                AsrPrecision::Float32 => &[
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

    fn file_names(model: AsrModel, precision: AsrPrecision) -> SherpaOnnxTransducerFileNames {
        match model {
            AsrModel::ReazonSpeechK2V2 => match precision {
                AsrPrecision::Int8 => SherpaOnnxTransducerFileNames {
                    encoder: "encoder-epoch-99-avg-1.int8.onnx",
                    decoder: "decoder-epoch-99-avg-1.int8.onnx",
                    joiner: "joiner-epoch-99-avg-1.int8.onnx",
                    tokens: "tokens.txt",
                },
                AsrPrecision::Int8Float32 => SherpaOnnxTransducerFileNames {
                    encoder: "encoder-epoch-99-avg-1.int8.onnx",
                    decoder: "decoder-epoch-99-avg-1.onnx",
                    joiner: "joiner-epoch-99-avg-1.int8.onnx",
                    tokens: "tokens.txt",
                },
                AsrPrecision::Float32 => SherpaOnnxTransducerFileNames {
                    encoder: "encoder-epoch-99-avg-1.onnx",
                    decoder: "decoder-epoch-99-avg-1.onnx",
                    joiner: "joiner-epoch-99-avg-1.onnx",
                    tokens: "tokens.txt",
                },
            },
            AsrModel::NemoParakeetTdt0_6BV2Int8 | AsrModel::NemoParakeetTdt0_6BV3Int8 => {
                SherpaOnnxTransducerFileNames {
                    encoder: "encoder.int8.onnx",
                    decoder: "decoder.int8.onnx",
                    joiner: "joiner.int8.onnx",
                    tokens: "tokens.txt",
                }
            }
        }
    }

    #[cfg(test)]
    pub fn expected_file_names(
        model: AsrModel,
        precision: AsrPrecision,
    ) -> SherpaOnnxTransducerFileNames {
        Self::file_names(model, precision)
    }

    fn validate(&self) -> Result<()> {
        for path in [&self.encoder, &self.decoder, &self.joiner, &self.tokens] {
            if !path.is_file() {
                return Err(anyhow!("ASR model file not found: {}", path.display()));
            }
        }
        Ok(())
    }
}

impl SherpaOnnxAsrEngine {
    pub fn new(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
    ) -> Result<Self> {
        let files = SherpaOnnxTransducerModelFiles::from_dir(model_dir, model, precision)?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(files.encoder.display().to_string()),
            decoder: Some(files.decoder.display().to_string()),
            joiner: Some(files.joiner.display().to_string()),
        };
        config.model_config.tokens = Some(files.tokens.display().to_string());
        config.model_config.provider = Some("cpu".to_string());
        config.model_config.model_type = model_type(model).map(str::to_string);
        config.model_config.modeling_unit = modeling_unit(model).map(str::to_string);
        config.model_config.num_threads = num_threads;
        config.decoding_method = Some("greedy_search".to_string());

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("Failed to create sherpa-onnx recognizer"))?;
        Ok(Self { recognizer })
    }
}

fn model_type(model: AsrModel) -> Option<&'static str> {
    match model {
        AsrModel::NemoParakeetTdt0_6BV2Int8 | AsrModel::NemoParakeetTdt0_6BV3Int8 => {
            Some("nemo_transducer")
        }
        AsrModel::ReazonSpeechK2V2 => None,
    }
}

fn modeling_unit(model: AsrModel) -> Option<&'static str> {
    match model {
        AsrModel::ReazonSpeechK2V2 => Some("cjkchar"),
        AsrModel::NemoParakeetTdt0_6BV2Int8 | AsrModel::NemoParakeetTdt0_6BV3Int8 => None,
    }
}

impl AsrEngine for SherpaOnnxAsrEngine {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(
            i32::try_from(ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32"),
            samples,
        );
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("Failed to fetch sherpa-onnx result"))?;
        Ok(result.text.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::SherpaOnnxTransducerModelFiles;
    use crate::config::{AsrModel, AsrPrecision};

    #[test]
    fn nemo_parakeet_uses_nemo_file_names() {
        for model in [
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrModel::NemoParakeetTdt0_6BV3Int8,
        ] {
            let names =
                SherpaOnnxTransducerModelFiles::expected_file_names(model, AsrPrecision::Int8);

            assert_eq!(names.encoder, "encoder.int8.onnx");
            assert_eq!(names.decoder, "decoder.int8.onnx");
            assert_eq!(names.joiner, "joiner.int8.onnx");
            assert_eq!(names.tokens, "tokens.txt");
        }
    }

    #[test]
    fn reazonspeech_uses_epoch_file_names() {
        let names = SherpaOnnxTransducerModelFiles::expected_file_names(
            AsrModel::ReazonSpeechK2V2,
            AsrPrecision::Int8Float32,
        );

        assert_eq!(names.encoder, "encoder-epoch-99-avg-1.int8.onnx");
        assert_eq!(names.decoder, "decoder-epoch-99-avg-1.onnx");
        assert_eq!(names.joiner, "joiner-epoch-99-avg-1.int8.onnx");
        assert_eq!(names.tokens, "tokens.txt");
    }
}
