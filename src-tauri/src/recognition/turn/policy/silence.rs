use crate::{config::TurnDetector, recognition::control::RerecognitionPurpose};

#[derive(Clone, Copy)]
pub(in crate::recognition) struct Input {
    pub(in crate::recognition) open_turn: OpenTurn,
    pub(in crate::recognition) previous_segment_id: u64,
    pub(in crate::recognition) asr_state: AsrState,
    pub(in crate::recognition) pending_interim: PendingInterim,
    pub(in crate::recognition) completion_strategy: CompletionStrategy,
}

#[derive(Clone, Copy)]
pub(in crate::recognition) enum OpenTurn {
    None,
    Missing,
    Present {
        turn_id: u64,
        latest_segment_id: Option<u64>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::recognition) enum AsrState {
    Idle,
    Busy,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::recognition) enum PendingInterim {
    None,
    Promotable,
}

#[derive(Clone, Copy)]
pub(in crate::recognition) enum CompletionStrategy {
    Namo,
    Morph,
    SimpleRerecognizeFull,
    CompleteWithoutGrammar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::recognition) enum Action {
    Ignore,
    WaitForBusyAsr,
    PromotePendingInterim,
    RefreshRouteThenDispatchRerecognition {
        turn_id: u64,
        purpose: RerecognitionPurpose,
        fallback_complete_without_grammar: bool,
    },
    CompleteWithoutGrammar {
        turn_id: u64,
    },
}

pub(in crate::recognition) fn action(input: Input) -> Action {
    let (turn_id, latest_segment_id) = match input.open_turn {
        OpenTurn::None => {
            if input.asr_state == AsrState::Busy {
                return Action::WaitForBusyAsr;
            }
            return if input.pending_interim == PendingInterim::Promotable {
                Action::PromotePendingInterim
            } else {
                Action::Ignore
            };
        }
        OpenTurn::Missing => return Action::Ignore,
        OpenTurn::Present {
            turn_id,
            latest_segment_id,
        } => (turn_id, latest_segment_id),
    };

    if latest_segment_id != Some(input.previous_segment_id) {
        return Action::Ignore;
    }
    if input.asr_state == AsrState::Busy {
        return Action::WaitForBusyAsr;
    }
    match input.completion_strategy {
        CompletionStrategy::Namo | CompletionStrategy::Morph => {
            Action::RefreshRouteThenDispatchRerecognition {
                turn_id,
                purpose: RerecognitionPurpose::GrammarAfterCompletion,
                fallback_complete_without_grammar: false,
            }
        }
        CompletionStrategy::SimpleRerecognizeFull => {
            Action::RefreshRouteThenDispatchRerecognition {
                turn_id,
                purpose: RerecognitionPurpose::SimpleTurnCheckFinal,
                fallback_complete_without_grammar: true,
            }
        }
        CompletionStrategy::CompleteWithoutGrammar => Action::CompleteWithoutGrammar { turn_id },
    }
}

pub(in crate::recognition) fn asr_state(is_busy: bool) -> AsrState {
    if is_busy {
        AsrState::Busy
    } else {
        AsrState::Idle
    }
}

pub(in crate::recognition) fn pending_interim(can_promote: bool) -> PendingInterim {
    if can_promote {
        PendingInterim::Promotable
    } else {
        PendingInterim::None
    }
}

pub(in crate::recognition) fn completion_strategy(
    turn_detector: TurnDetector,
    simple_rerecognize_full_on_complete: bool,
) -> CompletionStrategy {
    match turn_detector {
        TurnDetector::Namo => CompletionStrategy::Namo,
        TurnDetector::Morph => CompletionStrategy::Morph,
        TurnDetector::Simple if simple_rerecognize_full_on_complete => {
            CompletionStrategy::SimpleRerecognizeFull
        }
        TurnDetector::Simple => CompletionStrategy::CompleteWithoutGrammar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_waits_promotes_or_dispatches_by_open_turn_state() {
        assert_eq!(
            action(Input {
                open_turn: OpenTurn::None,
                previous_segment_id: 7,
                asr_state: AsrState::Busy,
                pending_interim: PendingInterim::Promotable,
                completion_strategy: CompletionStrategy::CompleteWithoutGrammar,
            }),
            Action::WaitForBusyAsr
        );
        assert_eq!(
            action(Input {
                open_turn: OpenTurn::None,
                previous_segment_id: 7,
                asr_state: AsrState::Idle,
                pending_interim: PendingInterim::Promotable,
                completion_strategy: CompletionStrategy::CompleteWithoutGrammar,
            }),
            Action::PromotePendingInterim
        );
        assert_eq!(
            action(Input {
                open_turn: OpenTurn::Present {
                    turn_id: 3,
                    latest_segment_id: Some(7),
                },
                previous_segment_id: 7,
                asr_state: AsrState::Idle,
                pending_interim: PendingInterim::None,
                completion_strategy: CompletionStrategy::Namo,
            }),
            Action::RefreshRouteThenDispatchRerecognition {
                turn_id: 3,
                purpose: RerecognitionPurpose::GrammarAfterCompletion,
                fallback_complete_without_grammar: false,
            }
        );
        assert_eq!(
            action(Input {
                open_turn: OpenTurn::Present {
                    turn_id: 3,
                    latest_segment_id: Some(7),
                },
                previous_segment_id: 7,
                asr_state: AsrState::Idle,
                pending_interim: PendingInterim::None,
                completion_strategy: CompletionStrategy::SimpleRerecognizeFull,
            }),
            Action::RefreshRouteThenDispatchRerecognition {
                turn_id: 3,
                purpose: RerecognitionPurpose::SimpleTurnCheckFinal,
                fallback_complete_without_grammar: true,
            }
        );
    }
}
