use crate::config::ParapperConfig;

pub(in crate::recognition) fn ticks(config: &ParapperConfig) -> u64 {
    let vad_interval_ms = u64::from(config.segmentation.vad_interval_ms).max(1);
    let timeout_ms = u64::from(config.turn.check_silence_ms).saturating_mul(2);
    timeout_ms.div_ceil(vad_interval_ms).max(1)
}

#[derive(Clone, Copy)]
pub(in crate::recognition) struct Input {
    pub(in crate::recognition) open_turn_id: Option<u64>,
    pub(in crate::recognition) open_turn_activity_epoch: u64,
    pub(in crate::recognition) segment_activity_epoch: u64,
    pub(in crate::recognition) open_turn_since_tick: Option<u64>,
    pub(in crate::recognition) next_runtime_tick: u64,
    pub(in crate::recognition) timeout_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::recognition) enum Action {
    NoOpenTurn,
    ResetTimeoutOrigin,
    Waiting,
    Timeout { turn_id: u64 },
}

pub(in crate::recognition) fn action(input: Input) -> Action {
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
    fn action_resets_when_activity_epoch_changes() {
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
    fn action_waits_until_timeout_ticks_elapsed() {
        assert_eq!(
            action(Input {
                open_turn_id: Some(7),
                open_turn_activity_epoch: 1,
                segment_activity_epoch: 1,
                open_turn_since_tick: Some(10),
                next_runtime_tick: 11,
                timeout_ticks: 2,
            }),
            Action::Waiting
        );
        assert_eq!(
            action(Input {
                open_turn_id: Some(7),
                open_turn_activity_epoch: 1,
                segment_activity_epoch: 1,
                open_turn_since_tick: Some(10),
                next_runtime_tick: 12,
                timeout_ticks: 2,
            }),
            Action::Timeout { turn_id: 7 }
        );
    }
}
