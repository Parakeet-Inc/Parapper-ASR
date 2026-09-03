use std::collections::HashMap;

use anyhow::{Result, anyhow};
use parapper_models::asr::{
    AsrEngine, AsrLanguage, AsrModel, AsrStreamConfig, AsrStreamLanguage, AsrTranscript,
    StreamingSessionId,
};

use crate::{
    SegmentCloseReason, SttAsrConfig,
    transcription::{
        preprocess::{
            asr_stream_config_for_source_audio,
            maybe_shift_transcript_timestamps_for_leading_padding, normalize_asr_input_audio,
            prepare_asr_input_audio, prepare_nemotron_input_audio,
        },
        task::{AsrRequest, AsrStreamingSessionKey, AsrTaskKind},
    },
};

/// Constructed ASR models available to one STT execution runtime.
///
/// A host resolves paths and constructs concrete model implementations before
/// handing them to this registry. ORT Sessions and model-specific cache tensors
/// remain encapsulated inside each [`AsrEngine`].
#[derive(Default)]
pub struct AsrModelRegistry {
    engines: HashMap<AsrModel, Box<dyn AsrEngine>>,
}

impl AsrModelRegistry {
    /// Registers one preconstructed model.
    ///
    /// # Errors
    ///
    /// Returns an error when the same model was already registered. Silently
    /// replacing a live model could discard its model-specific stream state.
    pub fn insert(&mut self, model: AsrModel, engine: Box<dyn AsrEngine>) -> Result<()> {
        if self.engines.contains_key(&model) {
            return Err(anyhow!("{model:?} ASR engine was already registered"));
        }
        self.engines.insert(model, engine);
        Ok(())
    }

    #[must_use]
    pub fn contains(&self, model: AsrModel) -> bool {
        self.engines.contains_key(&model)
    }

    pub(crate) fn engine_mut(&mut self, model: AsrModel) -> Result<&mut (dyn AsrEngine + '_)> {
        match self.engines.get_mut(&model) {
            Some(engine) => Ok(engine.as_mut()),
            None => Err(anyhow!("{model:?} ASR engine was not preloaded")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AsrStreamingState {
    backend_session: StreamingSessionId,
}

/// Host-neutral ASR request executor.
///
/// This owns STT-level model selection, streaming lifecycle, and input padding.
/// Worker threads, clocks, path resolution, and application events stay in the
/// host. Model-specific inference state stays inside `parapper-models` engines.
pub struct AsrExecutionRuntime {
    models: AsrModelRegistry,
    streams: HashMap<AsrStreamingSessionKey, AsrStreamingState>,
    next_model_session_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrStreamingLifecycleError {
    AlreadyStarted(AsrStreamingSessionKey),
    NotStarted(AsrStreamingSessionKey),
    Backend {
        operation: &'static str,
        message: String,
    },
}

impl std::fmt::Display for AsrStreamingLifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyStarted(key) => write!(formatter, "stream already started: {key:?}"),
            Self::NotStarted(key) => write!(formatter, "stream not started: {key:?}"),
            Self::Backend { operation, message } => {
                write!(formatter, "stream {operation} failed: {message}")
            }
        }
    }
}

impl std::error::Error for AsrStreamingLifecycleError {}

impl AsrExecutionRuntime {
    #[must_use]
    pub fn new(models: AsrModelRegistry) -> Self {
        Self {
            models,
            streams: HashMap::new(),
            next_model_session_id: 1,
        }
    }

    /// Executes one request without owning a clock or application error sink.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested model was not registered or the
    /// model rejects recognition or an explicit streaming lifecycle operation.
    pub fn execute(
        &mut self,
        config: &SttAsrConfig,
        request: &AsrRequest,
    ) -> Result<AsrTranscript> {
        if is_nemotron_streaming_interim_request(request) {
            self.execute_streaming(config, request)
        } else {
            self.reset_streaming_sessions_for_source(&request.target.source_session);
            if request.route.model.is_nemotron() {
                return Err(anyhow!(
                    "Nemotron streaming models are interim-only and cannot execute completion/offline requests"
                ));
            }
            self.execute_offline(config, request)
        }
    }

    pub fn reset_streaming_sessions(&mut self) {
        for (key, state) in std::mem::take(&mut self.streams) {
            if let Ok(engine) = self.models.engine_mut(key.model) {
                engine.cancel_stream(state.backend_session);
            }
        }
    }

    /// Cancel only one `SourceSession`'s decoder state.
    pub fn reset_streaming_sessions_for_source(
        &mut self,
        source_session: &crate::SourceSessionKey,
    ) {
        let keys: Vec<_> = self
            .streams
            .keys()
            .filter(|key| &key.source_session == source_session)
            .cloned()
            .collect();
        for key in keys {
            let Some(state) = self.streams.remove(&key) else {
                continue;
            };
            if let Ok(engine) = self.models.engine_mut(key.model) {
                engine.cancel_stream(state.backend_session);
            }
        }
    }

    ///
    /// # Errors
    ///
    /// Returns a structured lifecycle error for duplicate starts or backend
    /// initialization failures.
    pub fn start_stream(
        &mut self,
        key: AsrStreamingSessionKey,
        config: AsrStreamConfig,
    ) -> Result<(), AsrStreamingLifecycleError> {
        if self.streams.contains_key(&key) {
            return Err(AsrStreamingLifecycleError::AlreadyStarted(key));
        }
        let backend_session = self.allocate_backend_session(&key);
        self.models
            .engine_mut(key.model)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "start",
                message: error.to_string(),
            })?
            .start_stream(backend_session, config)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "start",
                message: error.to_string(),
            })?;
        self.streams
            .insert(key, AsrStreamingState { backend_session });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns a structured lifecycle error when the key is not active or the
    /// backend rejects the audio.
    pub fn push_stream(
        &mut self,
        key: &AsrStreamingSessionKey,
        samples: &[f32],
    ) -> Result<AsrTranscript, AsrStreamingLifecycleError> {
        let state = self
            .streams
            .get(key)
            .copied()
            .ok_or_else(|| AsrStreamingLifecycleError::NotStarted(key.clone()))?;
        self.models
            .engine_mut(key.model)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "push",
                message: error.to_string(),
            })?
            .push_stream(state.backend_session, samples)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "push",
                message: error.to_string(),
            })
    }

    ///
    /// # Errors
    ///
    /// Returns a structured lifecycle error when the key is not active or the
    /// backend cannot finish it.
    pub fn finish_stream(
        &mut self,
        key: &AsrStreamingSessionKey,
    ) -> Result<AsrTranscript, AsrStreamingLifecycleError> {
        let state = self
            .streams
            .remove(key)
            .ok_or_else(|| AsrStreamingLifecycleError::NotStarted(key.clone()))?;
        self.models
            .engine_mut(key.model)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "finish",
                message: error.to_string(),
            })?
            .finish_stream(state.backend_session)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "finish",
                message: error.to_string(),
            })
    }

    ///
    /// # Errors
    ///
    /// Returns a structured lifecycle error when the key is not active or the
    /// backend is unavailable.
    pub fn cancel_stream(
        &mut self,
        key: &AsrStreamingSessionKey,
    ) -> Result<(), AsrStreamingLifecycleError> {
        let Some(state) = self.streams.remove(key) else {
            return Err(AsrStreamingLifecycleError::NotStarted(key.clone()));
        };
        self.models
            .engine_mut(key.model)
            .map_err(|error| AsrStreamingLifecycleError::Backend {
                operation: "cancel",
                message: error.to_string(),
            })?
            .cancel_stream(state.backend_session);
        Ok(())
    }

    fn allocate_backend_session(&mut self, key: &AsrStreamingSessionKey) -> StreamingSessionId {
        if key.source_session.source_id.as_str() == crate::SourceId::LEGACY_SINGLE_SOURCE {
            return legacy_model_session_id(key);
        }
        let session_id = StreamingSessionId::new(self.next_model_session_id, None);
        self.next_model_session_id = self.next_model_session_id.saturating_add(1);
        session_id
    }

    fn execute_streaming(
        &mut self,
        config: &SttAsrConfig,
        request: &AsrRequest,
    ) -> Result<AsrTranscript> {
        let key = request.streaming_session_key();
        if !self.streams.contains_key(&key) {
            let source_audio = if request.source_audio.is_empty() {
                request.audio.as_slice()
            } else {
                request.source_audio.as_slice()
            };
            let source_vad_results = if request.source_vad_results.is_empty() {
                request.vad_results.as_slice()
            } else {
                request.source_vad_results.as_slice()
            };
            let mut stream_config =
                asr_stream_config_for_source_audio(source_audio, source_vad_results);
            stream_config.language_hint = nemotron_language_hint(config, request.route.model);
            self.start_stream(key.clone(), stream_config)
                .map_err(anyhow::Error::from)?;
        }

        let audio = normalize_asr_input_audio(config.normalize_input_audio, &request.audio);
        match self.push_stream(&key, audio.as_ref()) {
            Ok(transcript) => Ok(transcript),
            Err(error) => {
                let _ = self.cancel_stream(&key);
                Err(anyhow::Error::from(error))
            }
        }
    }

    fn execute_offline(
        &mut self,
        config: &SttAsrConfig,
        request: &AsrRequest,
    ) -> Result<AsrTranscript> {
        let prepared = if request.route.model.is_nemotron() {
            prepare_nemotron_input_audio(&request.audio, &request.vad_results)
        } else {
            prepare_asr_input_audio(&request.audio, &request.vad_results)
        };
        let audio =
            normalize_asr_input_audio(config.normalize_input_audio, prepared.audio.as_ref());
        let mut transcript = self
            .models
            .engine_mut(request.route.model)?
            .recognize(audio.as_ref())?;
        maybe_shift_transcript_timestamps_for_leading_padding(
            &mut transcript,
            prepared.leading_padding_samples,
        );
        Ok(transcript)
    }
}

fn nemotron_language_hint(config: &SttAsrConfig, streaming_model: AsrModel) -> Option<AsrLanguage> {
    (streaming_model.stream_language() == AsrStreamLanguage::Nemotron35Auto
        && !config.multilingual_enabled
        && config.model.language() == AsrLanguage::Japanese)
        .then_some(AsrLanguage::Japanese)
}

impl Drop for AsrExecutionRuntime {
    fn drop(&mut self) {
        self.reset_streaming_sessions();
    }
}

fn is_nemotron_streaming_interim_request(request: &AsrRequest) -> bool {
    request.route.model.is_nemotron()
        && request.kind == AsrTaskKind::InterimDisplay
        && request.close_reason == Some(SegmentCloseReason::InterimChunkReached)
}

fn legacy_model_session_id(session: &AsrStreamingSessionKey) -> StreamingSessionId {
    StreamingSessionId::new(
        session.turn_id.0,
        session.segment_id.map(|segment| segment.0),
    )
}

#[cfg(test)]
#[path = "asr_regression_tests.rs"]
mod tests;
