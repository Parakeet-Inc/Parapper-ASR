use crate::recognition::{
    segmentation::{segment::builder::SegmentCloseReason, vad::engine::VadResult},
    transcription::{asr::engine::AsrTranscript, route::RecognitionRoute},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct AsrRequestId(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TurnId(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TurnRevision(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SegmentId(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct GlobalSampleIndex(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct VadFrameIndex(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AsrStreamingSessionKey {
    pub(crate) model: crate::config::AsrModel,
    pub(crate) turn_id: TurnId,
    pub(crate) segment_id: Option<SegmentId>,
}

impl AsrStreamingSessionKey {
    pub(crate) fn new(
        model: crate::config::AsrModel,
        turn_id: TurnId,
        segment_id: Option<SegmentId>,
    ) -> Self {
        Self {
            model,
            turn_id,
            segment_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsrTaskKind {
    InterimDisplay,
    CompletionCheck,
    Rerecognition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AudioRange {
    pub(crate) start_sample: GlobalSampleIndex,
    pub(crate) end_sample: GlobalSampleIndex,
}

impl AudioRange {
    pub(crate) fn new(start_sample: GlobalSampleIndex, end_sample: GlobalSampleIndex) -> Self {
        assert!(
            start_sample < end_sample,
            "ASR audio range must have a non-empty duration"
        );
        Self {
            start_sample,
            end_sample,
        }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            start_sample: self.start_sample.min(other.start_sample),
            end_sample: self.end_sample.max(other.end_sample),
        }
    }

    pub(crate) fn contains(self, other: Self) -> bool {
        self.start_sample <= other.start_sample && other.end_sample <= self.end_sample
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AsrTarget {
    pub(crate) turn_id: TurnId,
    pub(crate) turn_revision: TurnRevision,
    pub(crate) range: AudioRange,
    pub(crate) first_segment_id: Option<SegmentId>,
    pub(crate) last_segment_id: Option<SegmentId>,
}

impl AsrTarget {
    pub(crate) fn new(
        turn_id: TurnId,
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
        Self {
            turn_id,
            turn_revision,
            range,
            first_segment_id,
            last_segment_id,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AsrRequest {
    pub(crate) request_id: AsrRequestId,
    pub(crate) kind: AsrTaskKind,
    pub(crate) target: AsrTarget,
    pub(crate) route: RecognitionRoute,
    pub(crate) detected_language: Option<String>,
    // ASR input may include copied leading/trailing padding for a local segment.
    pub(crate) audio: Vec<f32>,
    pub(crate) vad_results: Vec<VadResult>,
    // Source audio is the continuous turn audio to preserve for output and rerecognition.
    pub(crate) source_audio: Vec<f32>,
    pub(crate) source_vad_results: Vec<VadResult>,
    // Kept for diagnostics and real-ASR system reports; production routing uses
    // the close reason before the request is materialized.
    #[allow(dead_code)]
    pub(crate) close_reason: Option<SegmentCloseReason>,
    pub(crate) created_at_frame: VadFrameIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AsrResult {
    pub(crate) request_id: AsrRequestId,
    pub(crate) kind: AsrTaskKind,
    pub(crate) target: AsrTarget,
    pub(crate) route: RecognitionRoute,
    pub(crate) status: AsrResultStatus,
    pub(crate) completed_at_frame: VadFrameIndex,
    pub(crate) elapsed_millis: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AsrResultStatus {
    Ok(AsrTranscript),
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AsrInFlight {
    pub(crate) request_id: AsrRequestId,
    pub(crate) kind: AsrTaskKind,
    pub(crate) target: AsrTarget,
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
    pub(crate) fn streaming_session_key(&self) -> AsrStreamingSessionKey {
        AsrStreamingSessionKey::new(
            self.route.model,
            self.target.turn_id,
            self.target.last_segment_id,
        )
    }
}
