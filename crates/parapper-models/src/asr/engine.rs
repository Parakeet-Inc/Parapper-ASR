use anyhow::{Result, anyhow};

use crate::{AsrLanguage, AsrTranscript, StreamingSessionId};

/// A speech interval within the first source PCM delta of an ASR stream.
///
/// Both offsets are sample offsets in the 16 kHz source PCM, not model-frame
/// or window indices. `end` is exclusive and may precede trailing silence in
/// the same delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrSpeechRangeSamples {
    pub start: usize,
    pub end: usize,
}

/// Model-neutral facts captured when an ASR stream begins.
///
/// The host reports only the source-audio speech boundary. Model backends own
/// every native window, padding, and buffering decision that follows from it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsrStreamConfig {
    /// Segmenter-derived speech bounds in the first PCM delta, when known.
    /// Model backends must not receive the VAD frame array itself.
    pub speech_range_samples: Option<AsrSpeechRangeSamples>,
    /// Optional language selected by the host for a prompt-aware streaming model.
    /// `None` keeps the model's own automatic language selection.
    pub language_hint: Option<AsrLanguage>,
}

/// Synchronous ASR boundary. A host may own an engine on its own worker thread.
/// Implementations are `Send`, not `Sync`, and calls require exclusive access.
pub trait AsrEngine: Send {
    /// Recognizes one complete audio buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot accept or decode the audio.
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript>;

    /// Creates backend state for a streaming recognition session.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot create the session.
    fn start_stream(
        &mut self,
        _session: StreamingSessionId,
        _config: AsrStreamConfig,
    ) -> Result<()> {
        Err(anyhow!("streaming recognition is not supported"))
    }

    /// Adds audio to a streaming session and returns its current transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is invalid or decoding fails.
    fn push_stream(
        &mut self,
        _session: StreamingSessionId,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        let _ = samples;
        Err(anyhow!("streaming recognition is not supported"))
    }

    /// Finishes a streaming session and returns its final transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is invalid or final decoding fails.
    fn finish_stream(&mut self, _session: StreamingSessionId) -> Result<AsrTranscript> {
        Err(anyhow!("streaming recognition is not supported"))
    }

    fn cancel_stream(&mut self, _session: StreamingSessionId) {}

    fn cancel_all_streams(&mut self) {}
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::AsrEngine;
    use crate::{AsrStreamConfig, AsrTranscript, StreamingSessionId};

    struct OfflineOnlyEngine;

    impl AsrEngine for OfflineOnlyEngine {
        fn recognize(&mut self, _samples: &[f32]) -> Result<AsrTranscript> {
            Ok(AsrTranscript::from_text("offline"))
        }
    }

    #[test]
    fn offline_implementations_reject_every_streaming_operation_by_default() {
        let mut engine = OfflineOnlyEngine;
        let session = StreamingSessionId::new(1, Some(2));

        assert!(
            engine
                .start_stream(session, AsrStreamConfig::default())
                .is_err()
        );
        assert!(engine.push_stream(session, &[0.0]).is_err());
        assert!(engine.finish_stream(session).is_err());
    }
}
