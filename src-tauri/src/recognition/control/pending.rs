#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) enum RerecognitionPurpose {
    GrammarAfterCompletion,
    SimpleTurnCheckFinal,
    TimeoutFinal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) struct PendingFinalization {
    // All blocked finalization paths now share the same outcome: finalize this turn
    // after older ASR/pending segment work can no longer update it.
    turn_id: u64,
}

impl PendingFinalization {
    pub(in crate::recognition) fn new(turn_id: u64) -> Self {
        Self { turn_id }
    }

    pub(in crate::recognition) fn turn_id(self) -> u64 {
        self.turn_id
    }
}

#[derive(Clone, Copy)]
pub(in crate::recognition) struct PendingTurnCheck {
    pub(in crate::recognition) previous_segment_id: u64,
    pub(in crate::recognition) activity_epoch: u64,
}
