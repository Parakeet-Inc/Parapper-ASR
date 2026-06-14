pub(crate) fn turn_event_id(session_id: u64, turn_id: u64, revision: u64) -> String {
    format!("turn-{session_id}-{turn_id}-{revision}")
}

pub(crate) fn take_next_output_sequence(next_output_sequence: &mut u64) -> u64 {
    let output_sequence = *next_output_sequence;
    *next_output_sequence = next_output_sequence.saturating_add(1);
    output_sequence
}
