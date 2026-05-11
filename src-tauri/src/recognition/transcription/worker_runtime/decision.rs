use std::time::Instant;

use crate::recognition::segment_builder::SegmentCloseReason;

use super::{AsrJobContext, emit_asr_warning};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnProgressDecision {
    Continue,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostSegmentAction {
    KeepOpenInterim,
    CloseFinal,
}

pub(crate) fn post_segment_action(
    reason: SegmentCloseReason,
    turn_decision: TurnProgressDecision,
) -> PostSegmentAction {
    match reason {
        SegmentCloseReason::InterimResultSilenceReached
        | SegmentCloseReason::SegmentMaxChunksReached => PostSegmentAction::KeepOpenInterim,
        SegmentCloseReason::EndSilenceReached => match turn_decision {
            TurnProgressDecision::Continue => PostSegmentAction::KeepOpenInterim,
            TurnProgressDecision::Complete => PostSegmentAction::CloseFinal,
        },
    }
}

pub(crate) fn refresh_open_turn_timeout_origin_if_activity_changed(
    open_turn_id: Option<u64>,
    open_turn_since: &mut Option<Instant>,
    open_turn_activity_epoch: &mut u64,
    current_activity_epoch: u64,
    now: Instant,
) -> bool {
    if open_turn_id.is_none() || *open_turn_activity_epoch == current_activity_epoch {
        return false;
    }
    *open_turn_activity_epoch = current_activity_epoch;
    *open_turn_since = Some(now);
    true
}

pub(super) fn namo_final_decision(context: &mut AsrJobContext<'_>, turn_id: u64) -> bool {
    let Some(turn) = context.turns.get(&turn_id) else {
        return false;
    };
    let draft = turn.draft();
    let Some(route) = draft.route else {
        return false;
    };
    match context.turn_detectors.decide(
        route.turn_detector_model,
        &draft.combined_text,
        context.config.namo_context_max_tokens,
    ) {
        Ok(decision) => {
            decision.is_end_of_turn
                && decision.confidence >= context.config.namo_turn_confidence_threshold
        }
        Err(err) => {
            emit_asr_warning(context.handle, &err);
            false
        }
    }
}
