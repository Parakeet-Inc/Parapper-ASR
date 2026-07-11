use std::collections::VecDeque;

use crate::{
    config::ParapperConfig,
    recognition::{
        segmentation::{segment::builder::SegmentCloseReason, vad::engine::VadResult},
        transcription::{
            asr::{
                input::{AsrRequestEdgePadding, ensure_asr_request_edge_silence},
                task::{
                    AsrRequest, AsrRequestId, AsrTarget, AsrTaskKind, AudioRange, SegmentId,
                    TurnId, TurnRevision, VadFrameIndex,
                },
            },
            route::RecognitionRouteSelection,
        },
    },
};

#[derive(Clone)]
pub(in crate::recognition) struct PendingAsrSegment {
    pub(in crate::recognition) segment_id: u64,
    pub(in crate::recognition) previous_segment_id: Option<u64>,
    pub(in crate::recognition) audio: Vec<f32>,
    pub(in crate::recognition) vad_results: Vec<VadResult>,
    pub(in crate::recognition) source_audio: Vec<f32>,
    pub(in crate::recognition) source_vad_results: Vec<VadResult>,
    pub(in crate::recognition) reason: SegmentCloseReason,
    pub(in crate::recognition) range: AudioRange,
    pub(in crate::recognition) created_at_frame: VadFrameIndex,
}

impl PendingAsrSegment {
    pub(in crate::recognition) fn kind(&self) -> AsrTaskKind {
        match self.reason {
            SegmentCloseReason::InterimChunkReached
            | SegmentCloseReason::InterimResultSilenceReached => AsrTaskKind::InterimDisplay,
            SegmentCloseReason::EndSilenceReached | SegmentCloseReason::SegmentMaxChunksReached => {
                AsrTaskKind::CompletionCheck
            }
        }
    }

    pub(in crate::recognition) fn turn_id(&self) -> TurnId {
        TurnId(self.previous_segment_id.unwrap_or(self.segment_id))
    }

    pub(in crate::recognition) fn first_segment_id(&self) -> SegmentId {
        SegmentId(self.previous_segment_id.unwrap_or(self.segment_id))
    }

    pub(in crate::recognition) fn last_segment_id(&self) -> SegmentId {
        SegmentId(self.segment_id)
    }

    fn is_contiguous_with(&self, next: &Self) -> bool {
        self.range.end_sample == next.range.start_sample
            && next.previous_segment_id == Some(self.segment_id)
            && self.last_segment_id() <= next.last_segment_id()
    }
}

pub(in crate::recognition) struct AsrRequestSegmentPlan {
    pub(in crate::recognition) kind: AsrTaskKind,
    segments: Vec<PendingAsrSegment>,
}

impl AsrRequestSegmentPlan {
    pub(in crate::recognition) fn target_turn_id(
        &self,
        config: &ParapperConfig,
        open_turn_id: Option<u64>,
        open_turn_accepts_root_segment: bool,
    ) -> u64 {
        let first = self
            .segments
            .first()
            .expect("ASR request plan requires at least one pending segment");
        if !config.can_connect_interim_after_completion() && first.previous_segment_id.is_none() {
            return first.segment_id;
        }
        if first.previous_segment_id.is_none() && !open_turn_accepts_root_segment {
            return first.segment_id;
        }
        open_turn_id.unwrap_or_else(|| first.turn_id().0)
    }

    #[cfg(test)]
    pub(in crate::recognition) fn audio(&self) -> Vec<f32> {
        let mut audio = Vec::new();
        for segment in &self.segments {
            audio.extend_from_slice(&segment.audio);
        }
        audio
    }

    pub(in crate::recognition) fn source_audio(&self) -> Vec<f32> {
        let mut audio = Vec::new();
        for segment in &self.segments {
            audio.extend_from_slice(&segment.source_audio);
        }
        audio
    }

    pub(in crate::recognition) fn first_reason(&self) -> SegmentCloseReason {
        self.first().reason
    }

    pub(in crate::recognition) fn range(&self) -> AudioRange {
        self.first().range.merge(self.last().range)
    }

    pub(in crate::recognition) fn into_request(
        self,
        config: &ParapperConfig,
        request_id: AsrRequestId,
        target_turn_id: u64,
        target_revision: u64,
        route_selection: RecognitionRouteSelection,
    ) -> AsrRequest {
        let first = self.first();
        let last = self.last();
        let close_reason = first.reason;
        let created_at_frame = first.created_at_frame;
        let target = AsrTarget::new(
            TurnId(target_turn_id),
            TurnRevision(target_revision),
            first.range.merge(last.range),
            Some(first.first_segment_id()),
            Some(last.last_segment_id()),
        );
        let mut audio = Vec::new();
        let mut vad_results = Vec::new();
        let mut source_audio = Vec::new();
        let mut source_vad_results = Vec::new();
        for segment in self.segments {
            audio.extend_from_slice(&segment.audio);
            vad_results.extend_from_slice(&segment.vad_results);
            source_audio.extend_from_slice(&segment.source_audio);
            source_vad_results.extend_from_slice(&segment.source_vad_results);
        }
        if self.kind == AsrTaskKind::InterimDisplay && !route_selection.route.model.is_nemotron() {
            ensure_asr_request_edge_silence(
                config,
                &mut audio,
                &mut vad_results,
                AsrRequestEdgePadding::LeadingAndTrailing,
            );
        }
        AsrRequest {
            request_id,
            kind: self.kind,
            target,
            route: route_selection.route,
            detected_language: route_selection.detected_language,
            audio,
            vad_results,
            source_audio,
            source_vad_results,
            close_reason: Some(close_reason),
            created_at_frame,
        }
    }

    fn first(&self) -> &PendingAsrSegment {
        self.segments
            .first()
            .expect("ASR request plan requires at least one pending segment")
    }

    fn last(&self) -> &PendingAsrSegment {
        self.segments
            .last()
            .expect("ASR request plan requires at least one pending segment")
    }
}

pub(in crate::recognition) fn drop_front_interim_segments_covered_by_completion(
    pending: &mut VecDeque<PendingAsrSegment>,
) {
    loop {
        let Some(front) = pending.front() else {
            break;
        };
        if front.kind() != AsrTaskKind::InterimDisplay {
            break;
        }
        let Some(covering_completion_index) = pending
            .iter()
            .skip(1)
            .position(|candidate| {
                candidate.kind() == AsrTaskKind::CompletionCheck
                    && candidate.turn_id() == front.turn_id()
                    && candidate.range.contains(front.range)
            })
            .map(|index_after_front| index_after_front + 1)
        else {
            break;
        };

        let covering_completion = pending
            .remove(covering_completion_index)
            .expect("covering completion should still be present");
        while pending.front().is_some_and(|candidate| {
            candidate.kind() == AsrTaskKind::InterimDisplay
                && candidate.turn_id() == covering_completion.turn_id()
                && covering_completion.range.contains(candidate.range)
        }) {
            pending.pop_front();
        }
        pending.push_front(covering_completion);
    }
}

pub(in crate::recognition) fn take_next_request_segment_plan(
    config: &ParapperConfig,
    pending: &mut VecDeque<PendingAsrSegment>,
) -> Option<AsrRequestSegmentPlan> {
    let first = pending.pop_front()?;
    let kind = first.kind();
    let mut segments = vec![first];

    match kind {
        AsrTaskKind::CompletionCheck if config.can_connect_interim_after_completion() => {
            take_following_interim_segments(pending, &mut segments);
        }
        AsrTaskKind::InterimDisplay => {
            take_following_interim_segments(pending, &mut segments);
        }
        AsrTaskKind::CompletionCheck | AsrTaskKind::Rerecognition => {}
    }

    Some(AsrRequestSegmentPlan { kind, segments })
}

fn take_following_interim_segments(
    pending: &mut VecDeque<PendingAsrSegment>,
    segments: &mut Vec<PendingAsrSegment>,
) {
    loop {
        let Some(next) = pending.front() else {
            break;
        };
        let Some(last) = segments.last() else {
            break;
        };
        if next.kind() != AsrTaskKind::InterimDisplay || !last.is_contiguous_with(next) {
            break;
        }
        let next = pending
            .pop_front()
            .expect("front pending segment should still exist");
        segments.push(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::TurnDetector,
        recognition::transcription::asr::task::{GlobalSampleIndex, VadFrameIndex},
    };

    #[test]
    fn request_plan_stops_at_non_contiguous_segment() {
        let mut pending = VecDeque::from([
            pending_segment(
                1,
                None,
                SegmentCloseReason::InterimResultSilenceReached,
                0..10,
            ),
            pending_segment(
                2,
                Some(99),
                SegmentCloseReason::InterimResultSilenceReached,
                10..20,
            ),
        ]);

        let plan = take_next_request_segment_plan(
            &parapper_config! {
                turn_detector: TurnDetector::Namo,
                ..ParapperConfig::default()
            },
            &mut pending,
        )
        .expect("first interim request should be planned");

        assert_eq!(plan.kind, AsrTaskKind::InterimDisplay);
        assert_eq!(plan.audio(), vec![1.0; 10]);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.front().map(|segment| segment.segment_id), Some(2));
    }

    #[test]
    fn covered_front_interim_segments_are_replaced_by_covering_completion() {
        let mut pending = VecDeque::from([
            pending_segment(
                1,
                None,
                SegmentCloseReason::InterimResultSilenceReached,
                0..10,
            ),
            pending_segment(
                2,
                Some(1),
                SegmentCloseReason::InterimResultSilenceReached,
                10..20,
            ),
            pending_segment(2, Some(1), SegmentCloseReason::EndSilenceReached, 0..20),
        ]);

        drop_front_interim_segments_covered_by_completion(&mut pending);
        let plan =
            take_next_request_segment_plan(&ParapperConfig::default(), &mut pending).unwrap();

        assert_eq!(plan.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(plan.audio(), vec![2.0; 20]);
        assert!(pending.is_empty());
    }

    #[test]
    fn turn_detector_controls_completion_and_following_interim_merge() {
        for (turn_detector, expected_audio_len, expected_remaining) in
            [(TurnDetector::Namo, 20, 0), (TurnDetector::Simple, 10, 1)]
        {
            let mut pending = VecDeque::from([
                pending_segment(1, None, SegmentCloseReason::EndSilenceReached, 0..10),
                pending_segment(
                    2,
                    Some(1),
                    SegmentCloseReason::InterimResultSilenceReached,
                    10..20,
                ),
            ]);

            let plan = take_next_request_segment_plan(
                &parapper_config! {
                    turn_detector: turn_detector,
                    ..ParapperConfig::default()
                },
                &mut pending,
            )
            .expect("completion request should be planned");

            assert_eq!(plan.kind, AsrTaskKind::CompletionCheck);
            assert_eq!(
                plan.audio().len(),
                expected_audio_len,
                "turn_detector={turn_detector:?}"
            );
            assert_eq!(
                pending.len(),
                expected_remaining,
                "turn_detector={turn_detector:?}"
            );
        }
    }

    fn pending_segment(
        segment_id: u64,
        previous_segment_id: Option<u64>,
        reason: SegmentCloseReason,
        range: std::ops::Range<u64>,
    ) -> PendingAsrSegment {
        let sample_value =
            f32::from(u16::try_from(segment_id).expect("test segment id should fit u16"));
        let audio = vec![
            sample_value;
            usize::try_from(range.end - range.start)
                .expect("test range should fit usize")
        ];
        let vad_results = vec![VadResult {
            probability: 0.9,
            is_speech: true,
        }];
        PendingAsrSegment {
            segment_id,
            previous_segment_id,
            source_audio: audio.clone(),
            source_vad_results: vad_results.clone(),
            audio,
            vad_results,
            reason,
            range: AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
            created_at_frame: VadFrameIndex(segment_id),
        }
    }
}
