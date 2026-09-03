use std::collections::VecDeque;

use crate::{SegmentCloseReason, SegmentId, TurnId, VadResult};

use super::{
    input::{AsrRequestEdgePadding, ensure_asr_request_edge_silence},
    route::RecognitionRouteSelection,
    task::{
        AsrRequest, AsrRequestId, AsrTarget, AsrTaskKind, AsrTurnTarget, AudioRange, TurnRevision,
        VadFrameIndex,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerConfig {
    pub can_connect_interim_after_completion: bool,
    pub vad_interval_ms: u32,
}

#[derive(Clone)]
pub struct PendingAsrSegment {
    pub segment_id: u64,
    pub previous_segment_id: Option<u64>,
    pub audio: Vec<f32>,
    pub vad_results: Vec<VadResult>,
    pub source_audio: Vec<f32>,
    pub source_vad_results: Vec<VadResult>,
    pub reason: SegmentCloseReason,
    pub range: AudioRange,
    pub created_at_frame: VadFrameIndex,
}

impl PendingAsrSegment {
    #[must_use]
    pub const fn kind(&self) -> AsrTaskKind {
        match self.reason {
            SegmentCloseReason::InterimChunkReached
            | SegmentCloseReason::InterimResultSilenceReached => AsrTaskKind::InterimDisplay,
            SegmentCloseReason::EndSilenceReached | SegmentCloseReason::SegmentMaxChunksReached => {
                AsrTaskKind::CompletionCheck
            }
        }
    }

    #[must_use]
    pub fn turn_id(&self) -> TurnId {
        TurnId(self.previous_segment_id.unwrap_or(self.segment_id))
    }

    #[must_use]
    pub fn first_segment_id(&self) -> SegmentId {
        SegmentId(self.previous_segment_id.unwrap_or(self.segment_id))
    }

    #[must_use]
    pub const fn last_segment_id(&self) -> SegmentId {
        SegmentId(self.segment_id)
    }

    fn is_contiguous_with(&self, next: &Self) -> bool {
        self.range.end_sample == next.range.start_sample
            && next.previous_segment_id == Some(self.segment_id)
            && self.last_segment_id() <= next.last_segment_id()
    }
}

pub struct AsrRequestSegmentPlan {
    pub kind: AsrTaskKind,
    segments: Vec<PendingAsrSegment>,
}

impl AsrRequestSegmentPlan {
    #[must_use]
    /// Resolves the target turn for this non-empty plan.
    ///
    /// # Panics
    ///
    /// Panics if the plan contains no pending segments.
    pub fn target_turn_id(
        &self,
        config: &PlannerConfig,
        open_turn_id: Option<u64>,
        open_turn_accepts_root_segment: bool,
    ) -> u64 {
        let first = self
            .segments
            .first()
            .expect("ASR request plan requires at least one pending segment");
        if !config.can_connect_interim_after_completion && first.previous_segment_id.is_none() {
            return first.segment_id;
        }
        if first.previous_segment_id.is_none() && !open_turn_accepts_root_segment {
            return first.segment_id;
        }
        open_turn_id.unwrap_or_else(|| first.turn_id().0)
    }

    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics when this plan contains no segments.
    pub fn target_for_request(
        &self,
        config: &PlannerConfig,
        open_turn_id: Option<u64>,
        open_turn_accepts_root_segment: bool,
    ) -> AsrTurnTarget {
        let first = self
            .segments
            .first()
            .expect("ASR request plan requires at least one pending segment");
        if first.previous_segment_id.is_none() && !open_turn_accepts_root_segment {
            return AsrTurnTarget::New;
        }
        if !config.can_connect_interim_after_completion && first.previous_segment_id.is_none() {
            return AsrTurnTarget::New;
        }
        open_turn_id.map_or(AsrTurnTarget::New, |turn_id| {
            AsrTurnTarget::Existing(TurnId(turn_id))
        })
    }

    #[must_use]
    pub fn audio(&self) -> Vec<f32> {
        self.segments
            .iter()
            .flat_map(|segment| segment.audio.iter().copied())
            .collect()
    }

    #[must_use]
    pub fn source_audio(&self) -> Vec<f32> {
        self.segments
            .iter()
            .flat_map(|segment| segment.source_audio.iter().copied())
            .collect()
    }

    #[must_use]
    pub fn first_reason(&self) -> SegmentCloseReason {
        self.first().reason
    }

    #[must_use]
    pub fn range(&self) -> AudioRange {
        self.first().range.merge(self.last().range)
    }

    #[must_use]
    pub fn into_request(
        self,
        config: &PlannerConfig,
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
                config.vad_interval_ms,
                &mut audio,
                &mut vad_results,
                AsrRequestEdgePadding::Both,
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

/// Removes queued interim work already covered by a completion request.
///
/// # Panics
///
/// Panics only if the queue changes between locating and removing the covering request.
pub fn drop_front_interim_segments_covered_by_completion(
    pending: &mut VecDeque<PendingAsrSegment>,
) {
    while let Some(front) = pending.front() {
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

pub fn take_next_request_segment_plan(
    config: &PlannerConfig,
    pending: &mut VecDeque<PendingAsrSegment>,
) -> Option<AsrRequestSegmentPlan> {
    let first = pending.pop_front()?;
    let kind = first.kind();
    let mut segments = vec![first];

    match kind {
        AsrTaskKind::CompletionCheck if config.can_connect_interim_after_completion => {
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
    while let Some(next) = pending.front() {
        let Some(last) = segments.last() else {
            break;
        };
        if next.kind() != AsrTaskKind::InterimDisplay || !last.is_contiguous_with(next) {
            break;
        }
        segments.push(
            pending
                .pop_front()
                .expect("front pending segment should still exist"),
        );
    }
}

#[cfg(test)]
mod tests {
    use parapper_models::asr::AsrModel;

    use super::*;
    use crate::transcription::{route::RecognitionRoute, task::GlobalSampleIndex};

    const CONNECTING: PlannerConfig = PlannerConfig {
        can_connect_interim_after_completion: true,
        vad_interval_ms: 32,
    };
    const SEPARATE: PlannerConfig = PlannerConfig {
        can_connect_interim_after_completion: false,
        vad_interval_ms: 32,
    };

    #[test]
    fn non_contiguous_segment_stays_queued_for_a_separate_asr_request() {
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

        let plan = take_next_request_segment_plan(&CONNECTING, &mut pending).unwrap();

        assert_eq!(plan.kind, AsrTaskKind::InterimDisplay);
        assert_eq!(plan.audio(), vec![1.0; 10]);
        assert_eq!(pending.front().map(|segment| segment.segment_id), Some(2));
    }

    #[test]
    fn covering_completion_replaces_all_covered_front_interims() {
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
        let plan = take_next_request_segment_plan(&SEPARATE, &mut pending).unwrap();

        assert_eq!(plan.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(plan.audio(), vec![2.0; 20]);
        assert!(pending.is_empty());
    }

    #[test]
    fn planner_materializes_source_identity_and_keeps_source_audio_unpadded() {
        let pending = pending_segment(1, None, SegmentCloseReason::EndSilenceReached, 0..10);
        let mut queue = VecDeque::from([pending]);
        let plan = take_next_request_segment_plan(&SEPARATE, &mut queue).unwrap();
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);

        let request = plan.into_request(
            &SEPARATE,
            AsrRequestId(4),
            7,
            2,
            RecognitionRouteSelection {
                route,
                detected_language: Some("ja".to_owned()),
            },
        );

        assert_eq!(request.target.turn_id, TurnId(7));
        assert_eq!(request.target.turn_revision, TurnRevision(2));
        assert_eq!(request.source_audio, vec![1.0; 10]);
        assert_eq!(
            request.close_reason,
            Some(SegmentCloseReason::EndSilenceReached)
        );
    }

    fn pending_segment(
        segment_id: u64,
        previous_segment_id: Option<u64>,
        reason: SegmentCloseReason,
        range: std::ops::Range<u64>,
    ) -> PendingAsrSegment {
        let value = f32::from(u16::try_from(segment_id).unwrap());
        let audio = vec![value; usize::try_from(range.end - range.start).unwrap()];
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
