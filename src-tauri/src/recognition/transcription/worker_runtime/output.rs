use crate::recognition::{
    route::RecognitionRoute,
    sli::route_without_language_detection,
    turn::{take_next_output_sequence, take_stale_turn_final_outputs},
};

use super::{AsrJobContext, rerecognize::rerecognize_full_turn_if_needed};

pub(super) fn emit_turn_output(context: &mut AsrJobContext<'_>, turn_id: u64, is_final: bool) {
    let Some(turn) = context.turns.get(&turn_id) else {
        return;
    };
    let draft = turn.draft();
    if draft.combined_text.is_empty() {
        return;
    }

    let route = draft
        .route
        .unwrap_or_else(|| route_without_language_detection(context.config, None));

    let output_sequence = take_next_output_sequence(context.next_output_sequence);
    let output = if is_final {
        let Some(turn) = context.turns.remove(&turn_id) else {
            return;
        };
        let Some(confirmed) =
            turn.into_draft()
                .confirm(context.turn_session_id, turn_id, output_sequence, route)
        else {
            return;
        };
        *context.turn_revisions.entry(turn_id).or_insert(0) += 1;
        confirmed.into_output()
    } else {
        let Some(output) =
            draft.interim_output(context.turn_session_id, turn_id, output_sequence, route)
        else {
            return;
        };
        output
    };
    emit_output_and_update_last_route(context, &output);
}

pub(super) fn emit_stale_turn_finals(context: &mut AsrJobContext<'_>, before_turn_id: u64) {
    let stale_turn_ids = context
        .turns
        .keys()
        .copied()
        .filter(|turn_id| *turn_id < before_turn_id)
        .collect::<Vec<_>>();
    for turn_id in stale_turn_ids {
        rerecognize_full_turn_if_needed(context, turn_id);
    }

    for output in take_stale_turn_final_outputs(
        context.config,
        context.turns,
        context.turn_revisions,
        context.turn_session_id,
        context.next_output_sequence,
        before_turn_id,
    ) {
        emit_output_and_update_last_route(context, &output);
    }
}

fn emit_output_and_update_last_route(
    context: &mut AsrJobContext<'_>,
    output: &crate::delivery::RecognizedTextOutput,
) {
    if output.meta.is_final() {
        *context.last_spoken_route = Some(RecognitionRoute::from_model(output.source_asr_model));
    }
    context.emit(output);
}
