use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
#[cfg(any(not(test), feature = "real-asr-tests"))]
use sherpa_onnx::{
    OfflineNemoEncDecCtcModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineRecognizerResult, OfflineTransducerModelConfig, OnlineRecognizer,
    OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig,
    RecognizerResult as OnlineRecognizerResult,
};

#[cfg(any(not(test), feature = "real-asr-tests"))]
use crate::audio::ASR_SAMPLE_RATE;
use crate::config::{AsrModel, AsrModelImplementation, AsrPrecision, AsrStreamLanguage};
use crate::recognition::transcription::asr::task::AsrStreamingSessionKey;

pub trait AsrEngine: Send {
    fn transcribe(&mut self, samples: &[f32]) -> Result<AsrTranscript>;

    fn transcribe_streaming_delta(
        &mut self,
        _session: AsrStreamingSessionKey,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        self.transcribe(samples)
    }

    fn clear_streaming_session(&mut self, _session: AsrStreamingSessionKey) {
        self.clear_streaming_sessions();
    }

    fn clear_streaming_sessions(&mut self) {}
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrTranscript {
    pub text: String,
    pub tokens: Vec<AsrToken>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsrToken {
    pub text: String,
    pub start_sec: Option<f32>,
    pub duration_sec: Option<f32>,
    pub char_range: Option<std::ops::Range<usize>>,
}

impl AsrTranscript {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
        }
    }

    pub fn from_parts(
        text: impl Into<String>,
        token_texts: Vec<String>,
        timestamps: Option<&[f32]>,
        durations: Option<&[f32]>,
    ) -> Self {
        let text = text.into().trim().to_string();
        let token_ranges = token_char_ranges_relative_to_trimmed_text(&token_texts);
        let tokens = token_texts
            .into_iter()
            .enumerate()
            .map(|(index, token_text)| AsrToken {
                text: token_text,
                start_sec: timestamps
                    .as_ref()
                    .and_then(|timestamps| timestamps.get(index))
                    .copied(),
                duration_sec: durations
                    .as_ref()
                    .and_then(|durations| durations.get(index))
                    .copied(),
                char_range: token_ranges.get(index).cloned().flatten(),
            })
            .collect();
        Self { text, tokens }
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn from_sherpa_result(result: OfflineRecognizerResult) -> Self {
        Self::from_parts(
            result.text,
            result.tokens,
            result.timestamps.as_deref(),
            result.durations.as_deref(),
        )
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn from_sherpa_online_result(result: OnlineRecognizerResult) -> Self {
        Self::from_parts(
            result.text,
            result.tokens,
            result.timestamps.as_deref(),
            None,
        )
    }
}

fn token_char_ranges_relative_to_trimmed_text(
    token_texts: &[String],
) -> Vec<Option<std::ops::Range<usize>>> {
    let joined = token_texts.concat();
    let trimmed_start_bytes = joined.len() - joined.trim_start().len();
    let trimmed_end_bytes = joined.len().saturating_sub(joined.trim_end().len());
    let trimmed_start = joined[..trimmed_start_bytes].chars().count();
    let visible_end_bytes = joined.len().saturating_sub(trimmed_end_bytes);
    let trimmed_end = joined[..visible_end_bytes].chars().count();

    let mut cursor = 0;
    token_texts
        .iter()
        .map(|token| {
            let start = cursor;
            let end = start + token.chars().count();
            cursor = end;
            let visible_start = start.max(trimmed_start);
            let visible_end = end.min(trimmed_end);
            (visible_start < visible_end)
                .then(|| visible_start - trimmed_start..visible_end - trimmed_start)
        })
        .collect()
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
pub struct SherpaOnnxAsrEngine {
    recognizer: SherpaOnnxRecognizer,
    model: AsrModel,
    streaming_sessions: HashMap<AsrStreamingSessionKey, OnlineStream>,
}

#[cfg(all(test, not(feature = "real-asr-tests")))]
pub struct SherpaOnnxAsrEngine;

#[cfg(any(not(test), feature = "real-asr-tests"))]
enum SherpaOnnxRecognizer {
    Offline(OfflineRecognizer),
    Online(OnlineRecognizer),
}

// The recognizer is owned by the ASR worker thread after construction. Runtime
// access stays behind `&mut self`, so the wrapper is never shared concurrently.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SherpaOnnxNemoCtcModelFiles {
    pub model: PathBuf,
    pub tokens: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct SherpaOnnxNemoCtcFileNames {
    pub model: &'static str,
    pub tokens: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SherpaOnnxModelFiles {
    Transducer(SherpaOnnxTransducerModelFiles),
    NemoCtc(SherpaOnnxNemoCtcModelFiles),
}

impl SherpaOnnxTransducerModelFiles {
    #[cfg(any(not(test), feature = "real-asr-tests"))]
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

    fn file_names(model: AsrModel, precision: AsrPrecision) -> SherpaOnnxTransducerFileNames {
        match model.implementation() {
            AsrModelImplementation::ReazonSpeechK2 => match precision {
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
            AsrModelImplementation::NemoParakeetTdtCtc => {
                unreachable!("NeMo CTC models do not use transducer file names")
            }
            AsrModelImplementation::NemoParakeetTdt | AsrModelImplementation::Nemotron => {
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

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn validate(&self) -> Result<()> {
        for path in [&self.encoder, &self.decoder, &self.joiner, &self.tokens] {
            if !path.is_file() {
                return Err(anyhow!("ASR model file not found: {}", path.display()));
            }
        }
        Ok(())
    }
}

impl SherpaOnnxNemoCtcModelFiles {
    #[cfg(any(not(test), feature = "real-asr-tests"))]
    pub fn from_dir(model_dir: &Path, model: AsrModel) -> Result<Self> {
        let names = Self::file_names(model);
        let files = Self {
            model: model_dir.join(names.model),
            tokens: model_dir.join(names.tokens),
        };
        files.validate()?;
        Ok(files)
    }

    fn file_names(model: AsrModel) -> SherpaOnnxNemoCtcFileNames {
        match model.implementation() {
            AsrModelImplementation::NemoParakeetTdtCtc => SherpaOnnxNemoCtcFileNames {
                model: "model.int8.onnx",
                tokens: "tokens.txt",
            },
            AsrModelImplementation::ReazonSpeechK2
            | AsrModelImplementation::NemoParakeetTdt
            | AsrModelImplementation::Nemotron => {
                unreachable!("transducer models do not use NeMo CTC file names")
            }
        }
    }

    #[cfg(test)]
    pub fn expected_file_names(model: AsrModel) -> SherpaOnnxNemoCtcFileNames {
        Self::file_names(model)
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn validate(&self) -> Result<()> {
        for path in [&self.model, &self.tokens] {
            if !path.is_file() {
                return Err(anyhow!("ASR model file not found: {}", path.display()));
            }
        }
        Ok(())
    }
}

impl SherpaOnnxModelFiles {
    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn from_dir(model_dir: &Path, model: AsrModel, precision: AsrPrecision) -> Result<Self> {
        match model.implementation() {
            AsrModelImplementation::NemoParakeetTdtCtc => Ok(Self::NemoCtc(
                SherpaOnnxNemoCtcModelFiles::from_dir(model_dir, model)?,
            )),
            AsrModelImplementation::ReazonSpeechK2
            | AsrModelImplementation::NemoParakeetTdt
            | AsrModelImplementation::Nemotron => Ok(Self::Transducer(
                SherpaOnnxTransducerModelFiles::from_dir(model_dir, model, precision)?,
            )),
        }
    }
}

impl SherpaOnnxAsrEngine {
    #[cfg(any(not(test), feature = "real-asr-tests"))]
    pub fn new(
        model_dir: &Path,
        model: AsrModel,
        precision: AsrPrecision,
        num_threads: i32,
    ) -> Result<Self> {
        let files = SherpaOnnxModelFiles::from_dir(model_dir, model, precision)?;
        let recognizer = create_recognizer(&files, model, num_threads)?;

        Ok(Self {
            recognizer,
            model,
            streaming_sessions: HashMap::new(),
        })
    }

    #[cfg(all(test, not(feature = "real-asr-tests")))]
    pub fn new(
        _model_dir: &Path,
        _model: AsrModel,
        _precision: AsrPrecision,
        _num_threads: i32,
    ) -> Result<Self> {
        Err(anyhow!("Sherpa ONNX ASR is unavailable in unit tests"))
    }
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn create_recognizer(
    files: &SherpaOnnxModelFiles,
    model: AsrModel,
    num_threads: i32,
) -> Result<SherpaOnnxRecognizer> {
    if model.is_nemotron() {
        return create_online_recognizer(files, model, num_threads)
            .map(SherpaOnnxRecognizer::Online);
    }
    create_offline_recognizer(files, model, num_threads).map(SherpaOnnxRecognizer::Offline)
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn create_offline_recognizer(
    files: &SherpaOnnxModelFiles,
    model: AsrModel,
    num_threads: i32,
) -> Result<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    match files {
        SherpaOnnxModelFiles::Transducer(files) => {
            config.model_config.transducer = OfflineTransducerModelConfig {
                encoder: Some(files.encoder.display().to_string()),
                decoder: Some(files.decoder.display().to_string()),
                joiner: Some(files.joiner.display().to_string()),
            };
            config.model_config.tokens = Some(files.tokens.display().to_string());
        }
        SherpaOnnxModelFiles::NemoCtc(files) => {
            config.model_config.nemo_ctc = OfflineNemoEncDecCtcModelConfig {
                model: Some(files.model.display().to_string()),
            };
            config.model_config.tokens = Some(files.tokens.display().to_string());
        }
    }
    config.model_config.provider = Some("cpu".to_string());
    config.model_config.model_type = model_type(model).map(str::to_string);
    config.model_config.modeling_unit = modeling_unit(model).map(str::to_string);
    config.model_config.num_threads = num_threads;
    config.decoding_method = Some("greedy_search".to_string());
    config.max_active_paths = 1;

    OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow!("Failed to create sherpa-onnx recognizer"))
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn create_online_recognizer(
    files: &SherpaOnnxModelFiles,
    model: AsrModel,
    num_threads: i32,
) -> Result<OnlineRecognizer> {
    let SherpaOnnxModelFiles::Transducer(files) = files else {
        return Err(anyhow!("Nemotron ASR requires transducer model files"));
    };
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.transducer = OnlineTransducerModelConfig {
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
    config.max_active_paths = 1;
    config.enable_endpoint = true;

    OnlineRecognizer::create(&config)
        .ok_or_else(|| anyhow!("Failed to create sherpa-onnx online recognizer"))
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn model_type(model: AsrModel) -> Option<&'static str> {
    match model.implementation() {
        AsrModelImplementation::NemoParakeetTdt | AsrModelImplementation::Nemotron => {
            Some("nemo_transducer")
        }
        AsrModelImplementation::ReazonSpeechK2 | AsrModelImplementation::NemoParakeetTdtCtc => None,
    }
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn modeling_unit(model: AsrModel) -> Option<&'static str> {
    match model.implementation() {
        AsrModelImplementation::ReazonSpeechK2
        | AsrModelImplementation::NemoParakeetTdtCtc
        | AsrModelImplementation::Nemotron => Some("cjkchar"),
        AsrModelImplementation::NemoParakeetTdt => None,
    }
}

impl AsrEngine for SherpaOnnxAsrEngine {
    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn transcribe(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        if samples.is_empty() {
            return Ok(AsrTranscript::from_text(""));
        }

        match &self.recognizer {
            SherpaOnnxRecognizer::Offline(recognizer) => {
                let stream = recognizer.create_stream();
                stream.accept_waveform(
                    i32::try_from(ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32"),
                    samples,
                );
                recognizer.decode(&stream);
                let result = stream
                    .get_result()
                    .ok_or_else(|| anyhow!("Failed to fetch sherpa-onnx result"))?;
                Ok(AsrTranscript::from_sherpa_result(result))
            }
            SherpaOnnxRecognizer::Online(recognizer) => {
                transcribe_online(recognizer, self.model, samples)
            }
        }
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn transcribe_streaming_delta(
        &mut self,
        session: AsrStreamingSessionKey,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        if samples.is_empty() {
            return Ok(AsrTranscript::from_text(""));
        }
        let SherpaOnnxRecognizer::Online(recognizer) = &self.recognizer else {
            return self.transcribe(samples);
        };
        transcribe_online_streaming_delta(
            recognizer,
            self.model,
            &mut self.streaming_sessions,
            session,
            samples,
        )
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn clear_streaming_session(&mut self, session: AsrStreamingSessionKey) {
        let SherpaOnnxRecognizer::Online(recognizer) = &self.recognizer else {
            return;
        };
        if let Some(stream) = self.streaming_sessions.remove(&session) {
            recognizer.reset(&stream);
        }
    }

    #[cfg(any(not(test), feature = "real-asr-tests"))]
    fn clear_streaming_sessions(&mut self) {
        let SherpaOnnxRecognizer::Online(recognizer) = &self.recognizer else {
            return;
        };
        for (_, stream) in self.streaming_sessions.drain() {
            recognizer.reset(&stream);
        }
    }

    #[cfg(all(test, not(feature = "real-asr-tests")))]
    fn transcribe(&mut self, _samples: &[f32]) -> Result<AsrTranscript> {
        Err(anyhow!("Sherpa ONNX ASR is unavailable in unit tests"))
    }
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn transcribe_online(
    recognizer: &OnlineRecognizer,
    model: AsrModel,
    samples: &[f32],
) -> Result<AsrTranscript> {
    const NEMOTRON_CHUNK_SAMPLES: usize = ASR_SAMPLE_RATE as usize * 160 / 1000;
    let stream = recognizer.create_stream();
    if let Some(language) = nemotron_stream_language_option(model) {
        stream.set_option("language", language);
    }

    for chunk in samples.chunks(NEMOTRON_CHUNK_SAMPLES) {
        stream.accept_waveform(
            i32::try_from(ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32"),
            chunk,
        );
        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }
    }
    stream.input_finished();
    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }
    let result = recognizer
        .get_result(&stream)
        .ok_or_else(|| anyhow!("Failed to fetch sherpa-onnx online result"))?;
    Ok(AsrTranscript::from_sherpa_online_result(result))
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn transcribe_online_streaming_delta(
    recognizer: &OnlineRecognizer,
    model: AsrModel,
    streams: &mut HashMap<AsrStreamingSessionKey, OnlineStream>,
    session: AsrStreamingSessionKey,
    samples: &[f32],
) -> Result<AsrTranscript> {
    let stream = streams.entry(session).or_insert_with(|| {
        let stream = recognizer.create_stream();
        if let Some(language) = nemotron_stream_language_option(model) {
            stream.set_option("language", language);
        }
        stream
    });
    stream.accept_waveform(
        i32::try_from(ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32"),
        samples,
    );
    while recognizer.is_ready(stream) {
        recognizer.decode(stream);
    }
    let result = recognizer
        .get_result(stream)
        .ok_or_else(|| anyhow!("Failed to fetch sherpa-onnx online streaming result"))?;
    Ok(AsrTranscript::from_sherpa_online_result(result))
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn nemotron_stream_language_option(model: AsrModel) -> Option<&'static str> {
    match model.stream_language() {
        AsrStreamLanguage::Nemotron35Auto => Some(nemotron_35_multilingual_language_option()),
        AsrStreamLanguage::None => None,
    }
}

#[cfg(all(
    any(not(test), feature = "real-asr-tests"),
    test,
    feature = "real-asr-tests"
))]
fn nemotron_35_multilingual_language_option() -> &'static str {
    "ja-JP"
}

#[cfg(all(
    any(not(test), feature = "real-asr-tests"),
    not(all(test, feature = "real-asr-tests"))
))]
fn nemotron_35_multilingual_language_option() -> &'static str {
    "auto"
}

#[cfg(test)]
mod tests {
    use super::{
        AsrEngine, AsrTranscript, SherpaOnnxAsrEngine, SherpaOnnxNemoCtcModelFiles,
        SherpaOnnxTransducerModelFiles,
    };
    use crate::config::{AsrModel, AsrPrecision};

    #[test]
    fn nemo_parakeet_transducer_models_use_transducer_file_names() {
        for model in [
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrModel::NemoParakeetTdt0_6BV3Int8,
            AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8,
            AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8,
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
    fn japanese_parakeet_tdt_ctc_uses_nemo_ctc_file_names() {
        let names = SherpaOnnxNemoCtcModelFiles::expected_file_names(
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
        );

        assert_eq!(names.model, "model.int8.onnx");
        assert_eq!(names.tokens, "tokens.txt");
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

    #[cfg(feature = "real-asr-tests")]
    #[test]
    fn real_asr_tests_pin_nemotron_35_stream_language_to_ja_jp() {
        assert_eq!(
            super::nemotron_stream_language_option(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
            Some("ja-JP")
        );
        assert_eq!(
            super::nemotron_stream_language_option(AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8),
            Some("ja-JP")
        );
        assert_eq!(
            super::nemotron_stream_language_option(
                AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8
            ),
            None
        );
        assert_eq!(
            super::nemotron_stream_language_option(
                AsrModel::NemotronSpeechStreamingEn0_6B560MsInt8
            ),
            None
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "requires downloaded Nemotron 3.5 ASR 160ms model"]
    fn downloaded_nemotron_35_streaming_model_transcribes_archive_test_wavs() {
        let model_dir =
            std::env::var_os("PARAPPER_NEMOTRON_MODEL_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(std::env::var_os("APPDATA").expect(
                        "APPDATA or PARAPPER_NEMOTRON_MODEL_DIR is required for real ASR test",
                    ))
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                    .join("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11")
                });
        let wav_path = model_dir.join("test_wavs").join("ja.wav");
        let wave = sherpa_onnx::Wave::read(&wav_path.display().to_string())
            .unwrap_or_else(|| panic!("failed to read {}", wav_path.display()));
        assert_eq!(wave.sample_rate(), 16_000);
        let mut engine = SherpaOnnxAsrEngine::new(
            &model_dir,
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            AsrPrecision::Int8,
            2,
        )
        .expect("Nemotron engine should load from the downloaded model dir");

        let transcript = engine
            .transcribe(wave.samples())
            .expect("Nemotron engine should transcribe the archive test wav");

        assert!(
            !transcript.text.trim().is_empty(),
            "Nemotron archive test wav should produce non-empty text"
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "diagnostic: requires downloaded Nemotron 3.5 and Parakeet TDT CTC JA models"]
    fn measure_cpu4_rtf_nemotron_35_vs_parakeet_tdt_ctc_ja() {
        use std::time::{Duration, Instant};

        fn models_root() -> std::path::PathBuf {
            std::env::var_os("PARAPPER_MODELS_ROOT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(
                        std::env::var_os("APPDATA")
                            .expect("APPDATA or PARAPPER_MODELS_ROOT is required"),
                    )
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                })
        }

        fn measure(
            label: &str,
            model: AsrModel,
            model_dir: &std::path::Path,
            samples: &[f32],
            audio_sec: f64,
        ) -> (f64, String) {
            let mut engine = SherpaOnnxAsrEngine::new(model_dir, model, AsrPrecision::Int8, 4)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to load {label} from {}: {err:#}",
                        model_dir.display()
                    )
                });
            let warmup = engine
                .transcribe(samples)
                .unwrap_or_else(|err| panic!("{label} warmup transcription failed: {err:#}"));
            println!("{label} warmup text: {:?}", warmup.text);

            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 1..=repeats {
                let started_at = Instant::now();
                let transcript = engine.transcribe(samples).unwrap_or_else(|err| {
                    panic!("{label} transcription iteration {iteration} failed: {err:#}")
                });
                let elapsed = started_at.elapsed();
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "{label} iter {iteration}: elapsed_ms={:.1} rtf={:.3} text={:?}",
                    elapsed.as_secs_f64() * 1000.0,
                    rtf,
                    transcript.text
                );
                total += elapsed;
                last_text = transcript.text;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!("{label} avg_rtf={avg_rtf:.3} repeats={repeats}");
            (avg_rtf, last_text)
        }

        let models_root = models_root();
        let nemotron_dir = models_root.join(crate::model::catalog::asr_model_dir_name(
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        ));
        let parakeet_dir = models_root.join(crate::model::catalog::asr_model_dir_name(
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
        ));
        let wav_path = std::env::var_os("PARAPPER_ASR_RTF_WAV")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| nemotron_dir.join("test_wavs").join("ja.wav"));
        let wave = sherpa_onnx::Wave::read(&wav_path.display().to_string())
            .unwrap_or_else(|| panic!("failed to read {}", wav_path.display()));
        assert_eq!(
            wave.sample_rate(),
            i32::try_from(crate::audio::ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32")
        );
        let audio_sec = wave.samples().len() as f64 / f64::from(crate::audio::ASR_SAMPLE_RATE);
        println!(
            "RTF input: {} samples={} audio_sec={:.3}",
            wav_path.display(),
            wave.samples().len(),
            audio_sec
        );

        let (nemotron_rtf, nemotron_text) = measure(
            "nemotron_3_5_asr_streaming_0_6b_160ms_int8_cpu4",
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
            &nemotron_dir,
            wave.samples(),
            audio_sec,
        );
        let (parakeet_rtf, parakeet_text) = measure(
            "nemo_parakeet_tdt_ctc_0_6b_ja_35000_int8_cpu4",
            AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
            &parakeet_dir,
            wave.samples(),
            audio_sec,
        );

        println!(
            "RTF comparison cpu4: nemotron={nemotron_rtf:.3} parakeet_tdt_ctc_ja={parakeet_rtf:.3} ratio={:.3}",
            nemotron_rtf / parakeet_rtf
        );
        assert!(
            !nemotron_text.trim().is_empty(),
            "Nemotron should produce non-empty text for the RTF input"
        );
        assert!(
            !parakeet_text.trim().is_empty(),
            "Parakeet TDT CTC JA should produce non-empty text for the RTF input"
        );
    }

    #[test]
    fn transcript_tokens_keep_ranges_relative_to_trimmed_text() {
        let transcript = AsrTranscript::from_parts(
            "Well, I don't.".to_string(),
            vec![
                " We".to_string(),
                "ll".to_string(),
                ",".to_string(),
                " I".to_string(),
                " don".to_string(),
                "'".to_string(),
                "t".to_string(),
                ".".to_string(),
            ],
            Some(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            None,
        );

        assert_eq!(transcript.text, "Well, I don't.");
        assert_eq!(transcript.tokens[0].char_range, Some(0..2));
        assert_eq!(transcript.tokens[2].char_range, Some(4..5));
        assert_eq!(transcript.tokens[7].char_range, Some(13..14));
        assert_eq!(transcript.tokens[7].start_sec, Some(0.7));
    }

    #[test]
    fn transcript_tokens_keep_char_ranges_for_multibyte_cjk_and_emoji_text() {
        let transcript = AsrTranscript::from_parts(
            "漢字🙂かな".to_string(),
            vec![
                "  漢".to_string(),
                "字".to_string(),
                "🙂".to_string(),
                "か".to_string(),
                "な  ".to_string(),
            ],
            Some(&[0.0, 0.1, 0.2, 0.3, 0.4]),
            None,
        );

        assert_eq!(transcript.text, "漢字🙂かな");
        assert_eq!(
            transcript
                .tokens
                .iter()
                .map(|token| token.char_range.clone())
                .collect::<Vec<_>>(),
            vec![Some(0..1), Some(1..2), Some(2..3), Some(3..4), Some(4..5)]
        );
        assert_eq!(
            transcript
                .tokens
                .iter()
                .filter_map(|token| token.char_range.clone())
                .map(|range| slice_chars_for_test(&transcript.text, range))
                .collect::<Vec<_>>(),
            vec!["漢", "字", "🙂", "か", "な"]
        );
    }

    fn slice_chars_for_test(text: &str, range: std::ops::Range<usize>) -> String {
        text.chars()
            .skip(range.start)
            .take(range.end - range.start)
            .collect()
    }
}
