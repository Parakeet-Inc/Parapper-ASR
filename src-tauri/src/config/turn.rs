use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnDetector {
    Simple,
    Morph,
    Namo,
}

impl Default for TurnDetector {
    fn default() -> Self {
        Self::Simple
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorClass {
    Simple,
    Model(TurnDetectorModel),
}

#[cfg(test)]
impl TurnDetectorClass {
    pub fn model(self) -> Option<TurnDetectorModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::Simple => None,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetectorModel {
    Namo,
}

impl TurnDetector {
    #[cfg(test)]
    pub fn class(self) -> TurnDetectorClass {
        match self {
            Self::Simple | Self::Morph => TurnDetectorClass::Simple,
            Self::Namo => TurnDetectorClass::Model(TurnDetectorModel::Namo),
        }
    }

    pub fn uses_namo_model(self) -> bool {
        match self {
            Self::Namo => true,
            Self::Simple | Self::Morph => false,
        }
    }

    pub fn uses_morph_boundary(self) -> bool {
        matches!(self, Self::Namo | Self::Morph)
    }

    pub fn confirms_normal_end_with_namo(self) -> bool {
        matches!(self, Self::Namo)
    }

    pub fn uses_deferred_turn_completion(self) -> bool {
        !matches!(self, Self::Simple)
    }

    pub fn can_connect_interim_after_completion(self) -> bool {
        match self {
            // Simple は ASR 後に TD / grammar split で安全に戻せないので、
            // completion と interim を別々の ASR request として扱う。
            Self::Simple => false,
            // Namo / Morph は VAD 完了を確定境界にせず、
            // 後続 interim まで含めて TD / grammar boundary に判断させる。
            Self::Morph | Self::Namo => true,
        }
    }
}
