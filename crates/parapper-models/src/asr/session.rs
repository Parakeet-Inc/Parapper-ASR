use serde::{Deserialize, Serialize};

/// Host-neutral identity for one streaming recognition scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamingSessionId {
    pub scope: u64,
    pub item: Option<u64>,
}

impl StreamingSessionId {
    #[must_use]
    pub const fn new(scope: u64, item: Option<u64>) -> Self {
        Self { scope, item }
    }
}
