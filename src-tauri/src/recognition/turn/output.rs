use std::collections::HashMap;

use crate::{
    config::ParapperConfig, delivery::RecognizedTextOutput, recognition::route::RecognitionRoute,
};

use super::Turn;

pub(crate) fn take_stale_turn_final_outputs(
    config: &ParapperConfig,
    turns: &mut HashMap<u64, Turn>,
    turn_revisions: &mut HashMap<u64, u64>,
    turn_session_id: u64,
    next_output_sequence: &mut u64,
    before_turn_id: u64,
) -> Vec<RecognizedTextOutput> {
    let mut stale_turn_ids = turns
        .keys()
        .copied()
        .filter(|turn_id| *turn_id < before_turn_id)
        .collect::<Vec<_>>();
    stale_turn_ids.sort_unstable();

    stale_turn_ids
        .into_iter()
        .filter_map(|turn_id| {
            let turn = turns.remove(&turn_id)?;
            let draft = turn.into_draft();
            let route = draft
                .route
                .unwrap_or_else(|| RecognitionRoute::from_language(config.asr_language));
            let output_sequence = take_next_output_sequence(next_output_sequence);
            let confirmed = draft.confirm(turn_session_id, turn_id, output_sequence, route)?;
            *turn_revisions.entry(turn_id).or_insert(0) += 1;
            Some(confirmed.into_output())
        })
        .collect()
}

pub(crate) fn turn_event_id(session_id: u64, turn_id: u64, revision: u64) -> String {
    format!("turn-{session_id}-{turn_id}-{revision}")
}

pub(crate) fn take_next_output_sequence(next_output_sequence: &mut u64) -> u64 {
    let output_sequence = *next_output_sequence;
    *next_output_sequence = next_output_sequence.saturating_add(1);
    output_sequence
}
