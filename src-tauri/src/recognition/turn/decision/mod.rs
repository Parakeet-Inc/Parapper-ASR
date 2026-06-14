pub(crate) mod engine;
pub(crate) mod port;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TurnDecision {
    pub(crate) is_end_of_turn: bool,
    pub(crate) confidence: f32,
}

impl From<engine::NamoTurnDecision> for TurnDecision {
    fn from(decision: engine::NamoTurnDecision) -> Self {
        Self {
            is_end_of_turn: decision.is_end_of_turn,
            confidence: decision.confidence,
        }
    }
}
