use crate::recognition::segment_builder::SegmentCloseReason;

pub(super) enum AsrJob {
    SegmentClosed {
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        reason: SegmentCloseReason,
    },
    TurnCheckSilenceReached {
        previous_segment_id: u64,
    },
}
