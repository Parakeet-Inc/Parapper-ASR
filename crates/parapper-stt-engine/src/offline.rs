use std::borrow::Cow;

use parapper_models::asr::{AsrModel, AsrTranscript, SAMPLE_RATE_HZ, StreamingSessionId};
use serde::{Deserialize, Serialize};

use crate::{AsrModelRegistry, transcription::preprocess::prepare_asr_input_audio};

const MAX_OFFLINE_DURATION_SECONDS: usize = 30;
const STREAMING_FILE_CHUNK_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 160 / 1_000;

/// Applies the model-specific boundary conditioning used by offline file jobs.
///
/// `ReazonSpeech K2 V2` uses the same 320 ms edge-silence and 10 ms source-edge
/// fade as the live completion path. Other completion models receive the
/// original samples unchanged.
#[must_use]
pub fn prepare_offline_model_input_audio(model: AsrModel, audio: &[f32]) -> Cow<'_, [f32]> {
    if model == AsrModel::ReazonSpeechK2V2 {
        prepare_asr_input_audio(audio, &[]).audio
    } else {
        Cow::Borrowed(audio)
    }
}

#[derive(Debug)]
pub enum OfflineTranscriptionError {
    EmptyJobId,
    EmptyAudio,
    UnsupportedSampleRate {
        actual: u32,
    },
    TooLong {
        actual_samples: usize,
        max_samples: usize,
    },
    UnsupportedModel {
        model: AsrModel,
    },
    UnsupportedStreamingModel {
        model: AsrModel,
    },
    NonFiniteSample {
        index: usize,
    },
    ModelUnavailable {
        model: AsrModel,
        source: anyhow::Error,
    },
    Inference {
        source: anyhow::Error,
    },
}

impl std::fmt::Display for OfflineTranscriptionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyJobId => write!(formatter, "offline transcription job ID is empty"),
            Self::EmptyAudio => write!(formatter, "offline transcription audio is empty"),
            Self::UnsupportedSampleRate { actual } => write!(
                formatter,
                "offline transcription requires {SAMPLE_RATE_HZ} Hz mono PCM, got {actual} Hz"
            ),
            Self::TooLong {
                actual_samples,
                max_samples,
            } => write!(
                formatter,
                "offline transcription audio exceeds {MAX_OFFLINE_DURATION_SECONDS} seconds: {actual_samples} samples (maximum {max_samples})"
            ),
            Self::UnsupportedModel { model } => write!(
                formatter,
                "{model:?} does not support offline completion transcription"
            ),
            Self::UnsupportedStreamingModel { model } => write!(
                formatter,
                "{model:?} does not support streaming file transcription"
            ),
            Self::NonFiniteSample { index } => write!(
                formatter,
                "offline transcription audio contains a non-finite sample at index {index}"
            ),
            Self::ModelUnavailable { model, source } => {
                write!(
                    formatter,
                    "{model:?} offline model is unavailable: {source}"
                )
            }
            Self::Inference { source } => {
                write!(
                    formatter,
                    "offline transcription inference failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for OfflineTranscriptionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelUnavailable { source, .. } | Self::Inference { source } => {
                Some(source.as_ref())
            }
            Self::EmptyJobId
            | Self::EmptyAudio
            | Self::UnsupportedSampleRate { .. }
            | Self::TooLong { .. }
            | Self::UnsupportedModel { .. }
            | Self::UnsupportedStreamingModel { .. }
            | Self::NonFiniteSample { .. } => None,
        }
    }
}

/// One already-decoded mono PCM job for offline transcription.
///
/// File/container decoding and resampling are host adapter responsibilities.
/// The service accepts only the canonical ASR sample rate so evaluation and
/// desktop file transcription reach the exact same model input contract.
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineTranscriptionRequest {
    pub job_id: String,
    pub model: AsrModel,
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
}

/// Host-neutral result produced before UI events or evaluation aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OfflineTranscriptionResult {
    pub job_id: String,
    pub source_duration_samples: u64,
    pub source_sample_rate_hz: u32,
    pub model: AsrModel,
    pub transcript: AsrTranscript,
}

/// Reuses preloaded model sessions across independent offline jobs.
pub struct OfflineTranscriptionService {
    models: AsrModelRegistry,
}

impl OfflineTranscriptionService {
    #[must_use]
    pub fn new(models: AsrModelRegistry) -> Self {
        Self { models }
    }

    /// Transcribes one complete, canonical PCM input with model-specific edge
    /// conditioning.
    ///
    /// # Errors
    ///
    /// Returns an error before inference for an empty input, a non-16 kHz
    /// input, audio longer than 30 seconds, a streaming-only model, or a model
    /// that was not preloaded. Model inference errors are returned unchanged.
    pub fn transcribe(
        &mut self,
        request: OfflineTranscriptionRequest,
    ) -> Result<OfflineTranscriptionResult, OfflineTranscriptionError> {
        validate_request(&request)?;
        let transcript = {
            let model_input =
                prepare_offline_model_input_audio(request.model, request.samples.as_slice());
            self.models
                .engine_mut(request.model)
                .map_err(|source| OfflineTranscriptionError::ModelUnavailable {
                    model: request.model,
                    source,
                })?
                .recognize(model_input.as_ref())
                .map_err(|source| OfflineTranscriptionError::Inference { source })?
        };
        Ok(OfflineTranscriptionResult {
            job_id: request.job_id,
            source_duration_samples: request.samples.len() as u64,
            source_sample_rate_hz: request.sample_rate_hz,
            model: request.model,
            transcript,
        })
    }
}

/// Replays complete files through the production 160 ms streaming lifecycle.
///
/// This adapter exists for streaming-only Nemotron models. It deliberately
/// does not route them through the live VAD runtime and does not zero-pad the
/// final partial chunk.
pub struct StreamingFileTranscriptionService {
    models: AsrModelRegistry,
    next_session_id: u64,
}

impl StreamingFileTranscriptionService {
    #[must_use]
    pub fn new(models: AsrModelRegistry) -> Self {
        Self {
            models,
            next_session_id: 0,
        }
    }

    /// Transcribes one canonical PCM file using start/push/finish operations.
    ///
    /// # Errors
    ///
    /// Returns a typed preflight error before inference, or an inference error
    /// from streaming initialization, a chunk push, or finalization.
    pub fn transcribe(
        &mut self,
        request: OfflineTranscriptionRequest,
    ) -> Result<OfflineTranscriptionResult, OfflineTranscriptionError> {
        validate_audio_request(&request)?;
        if !request.model.is_nemotron() {
            return Err(OfflineTranscriptionError::UnsupportedStreamingModel {
                model: request.model,
            });
        }

        let session = StreamingSessionId::new(self.next_session_id, None);
        self.next_session_id = self.next_session_id.wrapping_add(1);
        let model = request.model;
        self.engine_mut(model)?
            .start_stream(
                session,
                parapper_models::asr::AsrStreamConfig {
                    speech_range_samples: Some(parapper_models::asr::AsrSpeechRangeSamples {
                        start: 0,
                        end: request.samples.len(),
                    }),
                    language_hint: None,
                },
            )
            .map_err(|source| OfflineTranscriptionError::Inference { source })?;
        for chunk in request.samples.chunks(STREAMING_FILE_CHUNK_SAMPLES) {
            if let Err(source) = self.engine_mut(model)?.push_stream(session, chunk) {
                self.cancel(model, session);
                return Err(OfflineTranscriptionError::Inference { source });
            }
        }
        let transcript = match self.engine_mut(model)?.finish_stream(session) {
            Ok(transcript) => transcript,
            Err(source) => {
                self.cancel(model, session);
                return Err(OfflineTranscriptionError::Inference { source });
            }
        };
        Ok(OfflineTranscriptionResult {
            job_id: request.job_id,
            source_duration_samples: request.samples.len() as u64,
            source_sample_rate_hz: request.sample_rate_hz,
            model,
            transcript,
        })
    }

    fn engine_mut(
        &mut self,
        model: AsrModel,
    ) -> Result<&mut dyn parapper_models::asr::AsrEngine, OfflineTranscriptionError> {
        self.models
            .engine_mut(model)
            .map_err(|source| OfflineTranscriptionError::ModelUnavailable { model, source })
    }

    fn cancel(&mut self, model: AsrModel, session: StreamingSessionId) {
        if let Ok(engine) = self.models.engine_mut(model) {
            engine.cancel_stream(session);
        }
    }
}

fn validate_request(
    request: &OfflineTranscriptionRequest,
) -> Result<(), OfflineTranscriptionError> {
    validate_audio_request(request)?;
    if !request.model.supports_completion() {
        return Err(OfflineTranscriptionError::UnsupportedModel {
            model: request.model,
        });
    }
    Ok(())
}

fn validate_audio_request(
    request: &OfflineTranscriptionRequest,
) -> Result<(), OfflineTranscriptionError> {
    if request.job_id.trim().is_empty() {
        return Err(OfflineTranscriptionError::EmptyJobId);
    }
    if request.samples.is_empty() {
        return Err(OfflineTranscriptionError::EmptyAudio);
    }
    if request.sample_rate_hz != SAMPLE_RATE_HZ {
        return Err(OfflineTranscriptionError::UnsupportedSampleRate {
            actual: request.sample_rate_hz,
        });
    }
    let max_samples = SAMPLE_RATE_HZ as usize * MAX_OFFLINE_DURATION_SECONDS;
    if request.samples.len() > max_samples {
        return Err(OfflineTranscriptionError::TooLong {
            actual_samples: request.samples.len(),
            max_samples,
        });
    }
    if let Some(index) = request
        .samples
        .iter()
        .position(|sample| !sample.is_finite())
    {
        return Err(OfflineTranscriptionError::NonFiniteSample { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use parapper_models::asr::{AsrEngine, AsrModel, AsrTranscript, SAMPLE_RATE_HZ};

    use crate::{
        AsrModelRegistry, OfflineTranscriptionError, OfflineTranscriptionRequest,
        OfflineTranscriptionService, StreamingFileTranscriptionService,
        prepare_offline_model_input_audio,
    };

    struct RecordingEngine {
        calls: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl AsrEngine for RecordingEngine {
        fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
            self.calls.lock().unwrap().push(samples.to_vec());
            Ok(AsrTranscript::from_text("認識結果"))
        }
    }

    fn service() -> (OfflineTranscriptionService, Arc<Mutex<Vec<Vec<f32>>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut models = AsrModelRegistry::default();
        models
            .insert(
                AsrModel::ReazonSpeechK2V2,
                Box::new(RecordingEngine {
                    calls: Arc::clone(&calls),
                }),
            )
            .unwrap();
        (OfflineTranscriptionService::new(models), calls)
    }

    fn request(job_id: &str, sample_count: usize) -> OfflineTranscriptionRequest {
        OfflineTranscriptionRequest {
            job_id: job_id.to_owned(),
            model: AsrModel::ReazonSpeechK2V2,
            sample_rate_hz: SAMPLE_RATE_HZ,
            samples: vec![0.25; sample_count],
        }
    }

    #[test]
    fn thirty_second_reazon_job_adds_production_edge_silence_and_preserves_source_duration() {
        let (mut service, calls) = service();
        let sample_count = SAMPLE_RATE_HZ as usize * 30;
        let edge_samples = SAMPLE_RATE_HZ as usize * 320 / 1_000;

        let result = service
            .transcribe(request("sample-30s", sample_count))
            .unwrap();

        assert_eq!(
            result,
            crate::OfflineTranscriptionResult {
                job_id: "sample-30s".to_owned(),
                source_duration_samples: sample_count as u64,
                source_sample_rate_hz: SAMPLE_RATE_HZ,
                model: AsrModel::ReazonSpeechK2V2,
                transcript: AsrTranscript::from_text("認識結果"),
            }
        );
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let model_input = &calls[0];
        assert_eq!(model_input.len(), sample_count + edge_samples * 2);
        assert!(
            model_input[..edge_samples]
                .iter()
                .all(|&sample| sample == 0.0)
        );
        assert!(
            model_input[edge_samples + sample_count..]
                .iter()
                .all(|&sample| sample == 0.0)
        );
        assert!(
            model_input[edge_samples].abs() < f32::EPSILON,
            "the source edge is faded in"
        );
        assert!(
            (model_input[edge_samples + SAMPLE_RATE_HZ as usize / 100] - 0.25).abs() < f32::EPSILON,
            "the signal reaches full amplitude after the 10 ms fade"
        );
    }

    #[test]
    fn consecutive_jobs_reuse_one_engine_and_preserve_job_identity() {
        let (mut service, calls) = service();

        let first = service.transcribe(request("first", 16_000)).unwrap();
        let second = service.transcribe(request("second", 8_000)).unwrap();

        assert_eq!(first.job_id, "first");
        assert_eq!(second.job_id, "second");
        assert_eq!(
            calls
                .lock()
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![26_240, 18_240]
        );
    }

    #[test]
    fn offline_edge_silence_is_reazonspeech_specific() {
        let audio = [0.25, -0.5];

        let parakeet =
            prepare_offline_model_input_audio(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, &audio);

        assert!(matches!(parakeet, std::borrow::Cow::Borrowed(_)));
        assert_eq!(parakeet.as_ref(), audio);
    }

    #[test]
    fn empty_over_limit_wrong_rate_and_streaming_only_jobs_fail_before_inference() {
        let (mut service, calls) = service();
        assert!(matches!(
            service.transcribe(OfflineTranscriptionRequest {
                job_id: " ".to_owned(),
                ..request("ignored", 16_000)
            }),
            Err(OfflineTranscriptionError::EmptyJobId)
        ));
        assert!(matches!(
            service.transcribe(request("empty", 0)),
            Err(OfflineTranscriptionError::EmptyAudio)
        ));
        assert!(matches!(
            service.transcribe(request("too-long", SAMPLE_RATE_HZ as usize * 30 + 1)),
            Err(OfflineTranscriptionError::TooLong { .. })
        ));
        assert!(matches!(
            service.transcribe(OfflineTranscriptionRequest {
                sample_rate_hz: 48_000,
                ..request("wrong-rate", 48_000)
            }),
            Err(OfflineTranscriptionError::UnsupportedSampleRate { actual: 48_000 })
        ));
        assert!(matches!(
            service.transcribe(OfflineTranscriptionRequest {
                model: AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
                ..request("streaming-only", 16_000)
            }),
            Err(OfflineTranscriptionError::UnsupportedModel {
                model: AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8
            })
        ));
        assert!(matches!(
            service.transcribe(OfflineTranscriptionRequest {
                samples: vec![0.0, f32::NAN],
                ..request("non-finite", 2)
            }),
            Err(OfflineTranscriptionError::NonFiniteSample { index: 1 })
        ));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum StreamingEvent {
        Started,
        Pushed(usize),
        Finished,
        Cancelled,
    }

    struct RecordingStreamingEngine {
        events: Arc<Mutex<Vec<StreamingEvent>>>,
    }

    impl AsrEngine for RecordingStreamingEngine {
        fn recognize(&mut self, _samples: &[f32]) -> Result<AsrTranscript> {
            anyhow::bail!("file adapter must not use one-shot recognize for streaming models")
        }

        fn start_stream(
            &mut self,
            _session: parapper_models::asr::StreamingSessionId,
            _config: parapper_models::asr::AsrStreamConfig,
        ) -> Result<()> {
            self.events.lock().unwrap().push(StreamingEvent::Started);
            Ok(())
        }

        fn push_stream(
            &mut self,
            _session: parapper_models::asr::StreamingSessionId,
            samples: &[f32],
        ) -> Result<AsrTranscript> {
            self.events
                .lock()
                .unwrap()
                .push(StreamingEvent::Pushed(samples.len()));
            Ok(AsrTranscript::from_text("interim"))
        }

        fn finish_stream(
            &mut self,
            _session: parapper_models::asr::StreamingSessionId,
        ) -> Result<AsrTranscript> {
            self.events.lock().unwrap().push(StreamingEvent::Finished);
            Ok(AsrTranscript::from_text("final"))
        }

        fn cancel_stream(&mut self, _session: parapper_models::asr::StreamingSessionId) {
            self.events.lock().unwrap().push(StreamingEvent::Cancelled);
        }
    }

    #[test]
    fn streaming_file_job_uses_160ms_chunks_and_finalizes_unpadded_tail() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let model = AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8;
        let mut models = AsrModelRegistry::default();
        models
            .insert(
                model,
                Box::new(RecordingStreamingEngine {
                    events: Arc::clone(&events),
                }),
            )
            .unwrap();
        let mut service = StreamingFileTranscriptionService::new(models);

        let result = service
            .transcribe(OfflineTranscriptionRequest {
                job_id: "streaming-file".to_owned(),
                model,
                sample_rate_hz: SAMPLE_RATE_HZ,
                samples: vec![0.25; 2_560 * 2 + 100],
            })
            .unwrap();

        assert_eq!(result.transcript, AsrTranscript::from_text("final"));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                StreamingEvent::Started,
                StreamingEvent::Pushed(2_560),
                StreamingEvent::Pushed(2_560),
                StreamingEvent::Pushed(100),
                StreamingEvent::Finished,
            ]
        );
    }
}
