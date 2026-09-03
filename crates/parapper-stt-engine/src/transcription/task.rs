use parapper_models::asr::{AsrModel, AsrTranscript};

use crate::{
    SegmentId, SourceSessionKey, TurnId, VadResult, transcription::route::RecognitionRoute,
};

pub use crate::{SegmentId as RequestSegmentId, TurnId as RequestTurnId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AsrRequestId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnRevision(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalSampleIndex(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VadFrameIndex(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AsrTurnTarget {
    /// A request which may create a turn when its first usable result arrives.
    New,
    /// A request extending a turn that already exists.
    Existing(TurnId),
}

impl From<TurnId> for AsrTurnTarget {
    fn from(value: TurnId) -> Self {
        Self::Existing(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AsrStreamingSessionKey {
    pub model: AsrModel,
    pub turn_id: TurnId,
    pub segment_id: Option<SegmentId>,
    pub source_session: SourceSessionKey,
}

impl AsrStreamingSessionKey {
    #[must_use]
    pub fn new(model: AsrModel, turn_id: TurnId, segment_id: Option<SegmentId>) -> Self {
        Self {
            model,
            turn_id,
            segment_id,
            source_session: SourceSessionKey::legacy_single_source(0),
        }
    }

    #[must_use]
    pub fn for_source(
        model: AsrModel,
        turn_id: TurnId,
        segment_id: Option<SegmentId>,
        source_session: SourceSessionKey,
    ) -> Self {
        Self {
            model,
            turn_id,
            segment_id,
            source_session,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsrTaskKind {
    InterimDisplay,
    CompletionCheck,
    Rerecognition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioRange {
    pub start_sample: GlobalSampleIndex,
    pub end_sample: GlobalSampleIndex,
}

impl AudioRange {
    #[must_use]
    /// Creates a non-empty sample range.
    ///
    /// # Panics
    ///
    /// Panics unless `start_sample < end_sample`.
    pub fn new(start_sample: GlobalSampleIndex, end_sample: GlobalSampleIndex) -> Self {
        assert!(
            start_sample < end_sample,
            "ASR audio range must have a non-empty duration"
        );
        Self {
            start_sample,
            end_sample,
        }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            start_sample: self.start_sample.min(other.start_sample),
            end_sample: self.end_sample.max(other.end_sample),
        }
    }

    #[must_use]
    pub fn contains(self, other: Self) -> bool {
        self.start_sample <= other.start_sample && other.end_sample <= self.end_sample
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsrTarget {
    /// Whether this request creates a turn on first usable result or extends
    /// an already-created turn.
    pub turn_target: AsrTurnTarget,
    /// Compatibility projection for older observers. For `New` requests this
    /// may contain a segment-derived value, but it is never an allocated turn
    /// id; consumers must inspect [`Self::turn_target`].
    pub turn_id: TurnId,
    pub turn_revision: TurnRevision,
    pub range: AudioRange,
    pub first_segment_id: Option<SegmentId>,
    pub last_segment_id: Option<SegmentId>,
    pub source_session: SourceSessionKey,
}

impl AsrTarget {
    #[must_use]
    /// Creates an ASR target over a turn and optional contiguous segment range.
    ///
    /// # Panics
    ///
    /// Panics when both segment identifiers are present and the first exceeds the last.
    pub fn new<T: Into<AsrTurnTarget>>(
        turn_target: T,
        turn_revision: TurnRevision,
        range: AudioRange,
        first_segment_id: Option<SegmentId>,
        last_segment_id: Option<SegmentId>,
    ) -> Self {
        if let (Some(first), Some(last)) = (first_segment_id, last_segment_id) {
            assert!(
                first <= last,
                "ASR target segment ids must be a contiguous forward range"
            );
        }
        let turn_target = turn_target.into();
        let turn_id = match turn_target {
            AsrTurnTarget::New => TurnId(0),
            AsrTurnTarget::Existing(turn_id) => turn_id,
        };
        Self {
            turn_target,
            turn_id,
            turn_revision,
            range,
            first_segment_id,
            last_segment_id,
            source_session: SourceSessionKey::legacy_single_source(0),
        }
    }

    #[must_use]
    pub fn with_source_session(mut self, source_session: SourceSessionKey) -> Self {
        self.source_session = source_session;
        self
    }

    pub fn set_source_session(&mut self, source_session: SourceSessionKey) {
        self.source_session = source_session;
    }
}

#[derive(Clone, Debug)]
pub struct AsrRequest {
    pub request_id: AsrRequestId,
    pub kind: AsrTaskKind,
    pub target: AsrTarget,
    pub route: RecognitionRoute,
    pub detected_language: Option<String>,
    pub audio: Vec<f32>,
    pub vad_results: Vec<VadResult>,
    pub source_audio: Vec<f32>,
    pub source_vad_results: Vec<VadResult>,
    pub close_reason: Option<crate::SegmentCloseReason>,
    pub created_at_frame: VadFrameIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AsrResult {
    pub request_id: AsrRequestId,
    pub kind: AsrTaskKind,
    pub target: AsrTarget,
    pub route: RecognitionRoute,
    pub status: AsrResultStatus,
    pub completed_at_frame: VadFrameIndex,
    pub elapsed_millis: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsrResultStatus {
    Ok(AsrTranscript),
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsrInFlight {
    pub request_id: AsrRequestId,
    pub kind: AsrTaskKind,
    pub target: AsrTarget,
}

impl From<&AsrRequest> for AsrInFlight {
    fn from(request: &AsrRequest) -> Self {
        Self {
            request_id: request.request_id,
            kind: request.kind,
            target: request.target.clone(),
        }
    }
}

impl AsrRequest {
    #[must_use]
    pub fn streaming_session_key(&self) -> AsrStreamingSessionKey {
        AsrStreamingSessionKey::for_source(
            self.route.model,
            self.target.turn_id,
            self.target.last_segment_id,
            self.target.source_session.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "non-empty duration")]
    fn empty_asr_audio_range_is_rejected_at_the_engine_boundary() {
        let _ = AudioRange::new(GlobalSampleIndex(4), GlobalSampleIndex(4));
    }

    #[test]
    fn streaming_session_identity_includes_model_turn_and_latest_segment() {
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let request = AsrRequest {
            request_id: AsrRequestId(1),
            kind: AsrTaskKind::InterimDisplay,
            target: AsrTarget::new(
                TurnId(7),
                TurnRevision(2),
                AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(10)),
                Some(SegmentId(3)),
                Some(SegmentId(4)),
            ),
            route,
            detected_language: None,
            audio: vec![0.0; 10],
            vad_results: Vec::new(),
            source_audio: vec![0.0; 10],
            source_vad_results: Vec::new(),
            close_reason: None,
            created_at_frame: VadFrameIndex(1),
        };

        assert_eq!(
            request.streaming_session_key(),
            AsrStreamingSessionKey::new(route.model, TurnId(7), Some(SegmentId(4)))
        );
    }

    #[test]
    fn streaming_session_identity_includes_source_session() {
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let target = AsrTarget::new(
            TurnId(7),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(10)),
            Some(SegmentId(4)),
            Some(SegmentId(4)),
        );
        let mut source_a = AsrRequest {
            request_id: AsrRequestId(1),
            kind: AsrTaskKind::InterimDisplay,
            target: target
                .clone()
                .with_source_session(SourceSessionKey::new(1, crate::SourceId::from("a"))),
            route,
            detected_language: None,
            audio: vec![0.0; 10],
            vad_results: Vec::new(),
            source_audio: vec![0.0; 10],
            source_vad_results: Vec::new(),
            close_reason: None,
            created_at_frame: VadFrameIndex(1),
        };
        let key_a = source_a.streaming_session_key();
        source_a.target.source_session = SourceSessionKey::new(1, crate::SourceId::from("b"));
        assert_ne!(key_a, source_a.streaming_session_key());
    }
}
