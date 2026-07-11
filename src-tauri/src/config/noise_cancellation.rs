use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NoiseCancellationModel {
    #[serde(rename = "ul_unas")]
    UlUnas,
}

impl Default for NoiseCancellationModel {
    fn default() -> Self {
        Self::UlUnas
    }
}
