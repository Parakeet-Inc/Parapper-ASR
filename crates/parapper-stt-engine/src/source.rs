use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable, application-defined identity for one logical recognition source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(transparent)]
pub struct SourceId(pub String);

impl SourceId {
    pub const LEGACY_SINGLE_SOURCE: &'static str = "legacy-single-source";

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn legacy_single_source() -> Self {
        Self(Self::LEGACY_SINGLE_SOURCE.to_owned())
    }
}

impl From<String> for SourceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SourceId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Runtime scope that makes source-local counters unambiguous across concurrently running sources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SourceSessionKey {
    pub turn_session_id: u64,
    pub source_id: SourceId,
}

impl SourceSessionKey {
    #[must_use]
    pub fn new(turn_session_id: u64, source_id: SourceId) -> Self {
        Self {
            turn_session_id,
            source_id,
        }
    }

    #[must_use]
    pub fn legacy_single_source(turn_session_id: u64) -> Self {
        Self::new(turn_session_id, SourceId::legacy_single_source())
    }
}

/// Immutable identity captured when a recognition source runtime starts.
///
/// `channel_index` is absent for the existing averaged-mono input path, where no
/// individual physical channel can truthfully be named.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceIdentitySnapshot {
    pub source_id: SourceId,
    pub speaker_label: String,
    pub capture_endpoint_id: String,
    pub channel_index: Option<u16>,
}

impl SourceIdentitySnapshot {
    #[must_use]
    pub fn new(
        source_id: SourceId,
        speaker_label: String,
        capture_endpoint_id: String,
        channel_index: Option<u16>,
    ) -> Self {
        Self {
            source_id,
            speaker_label,
            capture_endpoint_id,
            channel_index,
        }
    }

    #[must_use]
    pub fn legacy_single_source() -> Self {
        Self::new(
            SourceId::legacy_single_source(),
            "Legacy single input".to_owned(),
            "legacy-single-capture".to_owned(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_single_source_snapshot_does_not_claim_a_physical_channel() {
        let identity = SourceIdentitySnapshot::legacy_single_source();

        assert_eq!(identity.source_id.as_str(), SourceId::LEGACY_SINGLE_SOURCE);
        assert_eq!(identity.channel_index, None);
    }
}
