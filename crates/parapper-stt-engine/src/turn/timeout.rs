#[derive(Clone, Copy)]
pub struct Input {
    pub open_turn_id: Option<u64>,
    pub open_turn_activity_epoch: u64,
    pub segment_activity_epoch: u64,
    pub open_turn_since_tick: Option<u64>,
    pub next_runtime_tick: u64,
    pub timeout_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    NoOpenTurn,
    ResetTimeoutOrigin,
    Waiting,
    Timeout { turn_id: u64 },
}

#[must_use]
pub fn ticks(vad_interval_ms: u32, turn_check_silence_ms: u32) -> u64 {
    let vad_interval_ms = u64::from(vad_interval_ms).max(1);
    let timeout_ms = u64::from(turn_check_silence_ms).saturating_mul(2);
    timeout_ms.div_ceil(vad_interval_ms).max(1)
}

#[must_use]
pub fn action(input: Input) -> Action {
    let Some(turn_id) = input.open_turn_id else {
        return Action::NoOpenTurn;
    };
    if input.open_turn_activity_epoch != input.segment_activity_epoch {
        return Action::ResetTimeoutOrigin;
    }
    let Some(open_since_tick) = input.open_turn_since_tick else {
        return Action::Waiting;
    };
    if input.next_runtime_tick.saturating_sub(open_since_tick) < input.timeout_ticks {
        return Action::Waiting;
    }
    Action::Timeout { turn_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_segment_resets_the_open_turn_timeout_origin() {
        assert_eq!(
            action(Input {
                open_turn_id: Some(7),
                open_turn_activity_epoch: 1,
                segment_activity_epoch: 2,
                open_turn_since_tick: Some(10),
                next_runtime_tick: 100,
                timeout_ticks: 1,
            }),
            Action::ResetTimeoutOrigin
        );
    }

    #[test]
    fn unchanged_open_turn_times_out_only_after_the_full_threshold() {
        let waiting = Input {
            open_turn_id: Some(7),
            open_turn_activity_epoch: 1,
            segment_activity_epoch: 1,
            open_turn_since_tick: Some(10),
            next_runtime_tick: 11,
            timeout_ticks: 2,
        };
        assert_eq!(action(waiting), Action::Waiting);
        assert_eq!(
            action(Input {
                next_runtime_tick: 12,
                ..waiting
            }),
            Action::Timeout { turn_id: 7 }
        );
    }

    #[test]
    fn timeout_duration_is_twice_turn_check_silence_rounded_to_vad_ticks() {
        assert_eq!(ticks(32, 640), 40);
        assert_eq!(ticks(0, 0), 1);
    }
}
