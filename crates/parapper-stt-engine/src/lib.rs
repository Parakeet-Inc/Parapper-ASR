//! Tauri-free recognition-engine contracts and segment state machine.

mod asr;
mod config;
#[cfg(test)]
#[path = "core_regression_tests/mod.rs"]
mod core_regression_tests;
mod driver;
mod offline;
mod output;
pub mod ports;
pub mod runtime;
mod segmentation;
mod session;
mod source;
pub mod transcription;
pub mod turn;

pub use asr::{AsrExecutionRuntime, AsrModelRegistry, AsrStreamingLifecycleError};
pub use config::{
    SttAsrConfig, SttEngineConfig, SttRuntimeParameters, SttSegmentationConfig, SttTurnConfig,
};
pub use driver::RecognitionDriver;
pub use offline::{
    OfflineTranscriptionError, OfflineTranscriptionRequest, OfflineTranscriptionResult,
    OfflineTranscriptionService, StreamingFileTranscriptionService,
    prepare_offline_model_input_audio,
};
pub use output::{
    RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput, RecognizedTextUpdateMode,
    RecognizedTurn, continuing_turn_text, finalize_turn_text, join_turn_segments,
    take_next_output_sequence, trim_continuation_marker, turn_event_id,
};
pub use segmentation::{
    RecognitionConfig, RecognitionFrame, RecognitionSegmentEngine, SegmentBuilder,
    SegmentBuilderEvent, SegmentCloseReason, VadResult,
};
pub use session::{RecognitionSession, RuntimeIo};
pub use source::{SourceId, SourceIdentitySnapshot, SourceSessionKey};

pub use parapper_models::asr::{AsrLanguage, AsrModel, AsrTranscript, SAMPLE_RATE_HZ};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub u64);

/// Result of draining or cancelling one recognition session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionShutdownResult {
    Completed,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamoTurnDetectorModel {
    Japanese,
    English,
    Multilingual,
}

impl NamoTurnDetectorModel {
    #[must_use]
    pub const fn for_asr_language(language: parapper_models::asr::AsrLanguage) -> Self {
        match language {
            parapper_models::asr::AsrLanguage::Japanese => Self::Japanese,
            parapper_models::asr::AsrLanguage::English => Self::English,
            parapper_models::asr::AsrLanguage::EuropeanMultilingual
            | parapper_models::asr::AsrLanguage::Multilingual => Self::Multilingual,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleRange {
    pub start: u64,
    pub end: u64,
}

impl SampleRange {
    #[must_use]
    pub const fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub const fn len(self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RecognitionEvent {
    Interim {
        turn_id: TurnId,
        segment_id: SegmentId,
        transcript: AsrTranscript,
    },
    Final {
        turn_id: TurnId,
        transcript: AsrTranscript,
    },
    Error {
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::SampleRange;

    #[test]
    fn sample_range_length_never_underflows_for_invalid_host_input() {
        assert_eq!(SampleRange::new(20, 10).len(), 0);
        assert!(SampleRange::new(20, 10).is_empty());
    }
}
