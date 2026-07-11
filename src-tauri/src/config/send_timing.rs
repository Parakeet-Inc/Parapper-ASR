use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NeoSendTiming {
    Interim,
    Final,
}

impl Default for NeoSendTiming {
    fn default() -> Self {
        Self::Interim
    }
}
