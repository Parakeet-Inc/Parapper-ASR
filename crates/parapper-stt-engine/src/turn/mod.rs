//! Host-neutral Turn boundary and timeout policies.

mod boundary_flow;
mod domain;
mod flow;
mod transcript;

pub mod completion;
pub mod grammar;
pub mod namo;
pub mod silence;
pub mod timeout;

use serde::{Deserialize, Serialize};

pub use domain::{Turn, TurnConfirmed, TurnDraft};
pub use parapper_models::td::{GrammarBoundaryClass, TurnBoundaryCandidate};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum TurnDetector {
    #[default]
    Simple,
    Morph,
    Namo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorClass {
    Simple,
    Model(TurnDetectorModel),
}

impl TurnDetectorClass {
    #[must_use]
    pub const fn model(self) -> Option<TurnDetectorModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::Simple => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorModel {
    Namo,
}

impl TurnDetector {
    #[must_use]
    pub const fn class(self) -> TurnDetectorClass {
        match self {
            Self::Simple | Self::Morph => TurnDetectorClass::Simple,
            Self::Namo => TurnDetectorClass::Model(TurnDetectorModel::Namo),
        }
    }

    #[must_use]
    pub const fn uses_namo_model(self) -> bool {
        matches!(self, Self::Namo)
    }

    #[must_use]
    pub const fn uses_morph_boundary(self) -> bool {
        matches!(self, Self::Namo | Self::Morph)
    }

    #[must_use]
    pub const fn confirms_normal_end_with_namo(self) -> bool {
        matches!(self, Self::Namo)
    }

    #[must_use]
    pub const fn uses_deferred_turn_completion(self) -> bool {
        !matches!(self, Self::Simple)
    }

    #[must_use]
    pub const fn can_connect_interim_after_completion(self) -> bool {
        match self {
            Self::Simple => false,
            Self::Morph | Self::Namo => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerecognitionPurpose {
    GrammarAfterCompletion,
    SimpleTurnCheckFinal,
    TimeoutFinal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnDecision {
    pub is_end_of_turn: bool,
    pub confidence: f32,
}
