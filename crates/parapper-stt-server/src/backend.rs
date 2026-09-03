use std::{num::NonZeroUsize, sync::mpsc};

use parapper_stt_engine::{RecognitionShutdownResult, RecognizedTextOutput};

use crate::protocol::AudioFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendStartError {
    Busy,
    ModelUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSendError {
    Overrun,
    Disconnected,
}

/// Immutable decoder settings selected by the host when starting the server.
///
/// The WebSocket protocol deliberately has no equivalent fields: a client cannot
/// change the decoder for an established server. A backend may ignore settings it
/// does not support.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecognitionBackendConfig {
    pub tdt_dag: Option<TdtDagDecodingConfig>,
}

/// Host-selected settings for a TDT DAG decoder.
///
/// This type describes a transport-neutral configuration only. The host owns
/// choosing a compatible model implementation and validating any model-specific
/// restrictions on the gate threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdtDagDecodingConfig {
    pub beam_size: NonZeroUsize,
    pub ctc_gate_threshold: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdtDagDecodingConfigError {
    NonFiniteCtcGateThreshold,
}

impl std::fmt::Display for TdtDagDecodingConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteCtcGateThreshold => {
                formatter.write_str("CTC gate threshold must be finite")
            }
        }
    }
}

impl std::error::Error for TdtDagDecodingConfigError {}

impl TdtDagDecodingConfig {
    /// Creates server-fixed TDT DAG settings.
    ///
    /// `NonZeroUsize` makes a zero beam width unrepresentable at the transport
    /// boundary. Model-specific bounds remain the responsibility of the host.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied CTC gate threshold is not finite.
    pub fn new(
        beam_size: NonZeroUsize,
        ctc_gate_threshold: Option<f32>,
    ) -> Result<Self, TdtDagDecodingConfigError> {
        if ctc_gate_threshold.is_some_and(|threshold| !threshold.is_finite()) {
            return Err(TdtDagDecodingConfigError::NonFiniteCtcGateThreshold);
        }
        Ok(Self {
            beam_size,
            ctc_gate_threshold,
        })
    }
}

/// Session-scoped PCM input created by the host backend.
pub trait AudioInput: Send {
    /// Queues one PCM chunk without blocking.
    ///
    /// # Errors
    ///
    /// Returns [`InputSendError::Overrun`] when the bounded audio queue is full, or
    /// [`InputSendError::Disconnected`] when its consumer has stopped.
    fn try_send(&self, samples: Vec<f32>) -> Result<(), InputSendError>;
}

pub trait ActiveRecognitionSession: Send {
    fn stop(&mut self) -> RecognitionShutdownResult;
    fn cancel(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecognitionStreamEvent {
    SpeechStarted,
    Output(Box<RecognizedTextOutput>),
}

pub struct StartedRecognitionSession {
    pub(crate) input: Option<Box<dyn AudioInput>>,
    pub(crate) active: Option<Box<dyn ActiveRecognitionSession>>,
    pub(crate) event_receiver: mpsc::Receiver<RecognitionStreamEvent>,
}

impl StartedRecognitionSession {
    #[must_use]
    pub fn new(
        input: Box<dyn AudioInput>,
        active: Box<dyn ActiveRecognitionSession>,
        event_receiver: mpsc::Receiver<RecognitionStreamEvent>,
    ) -> Self {
        Self {
            input: Some(input),
            active: Some(active),
            event_receiver,
        }
    }
}

/// Host adapter used by the transport to create a recognition session.
pub trait RecognitionBackend: Send + Sync {
    /// Starts one recognition session for the negotiated audio format.
    ///
    /// # Errors
    ///
    /// Returns [`BackendStartError::Busy`] if another session owns the backend, or
    /// [`BackendStartError::ModelUnavailable`] when required models are unavailable.
    fn start(
        &self,
        session_id: &str,
        audio: &AudioFormat,
        config: RecognitionBackendConfig,
    ) -> Result<StartedRecognitionSession, BackendStartError>;
}
