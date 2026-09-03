use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunnerManifestV1 {
    pub schema_version: u32,
    pub split_id: String,
    pub dataset: DatasetIdentity,
    pub normalization: NormalizationIdentity,
    pub audio_format: AudioFormatContract,
    pub samples: Vec<RunnerSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetIdentity {
    pub id: String,
    pub release: String,
    pub source_split: String,
    pub language: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationIdentity {
    pub id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormatContract {
    pub encoding: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunnerSampleV1 {
    pub utterance_id: String,
    pub audio: DerivedAudio,
    pub reference: ReferenceText,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedAudio {
    pub relative_path: String,
    pub sha256: String,
    pub duration_samples: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceText {
    pub raw: String,
    pub normalized: String,
}

#[derive(Debug)]
pub enum ManifestValidationError {
    Json(serde_json::Error),
    UnsupportedSchemaVersion {
        actual: u32,
    },
    EmptyField {
        field: &'static str,
    },
    UnsupportedAudioFormat {
        encoding: String,
        sample_rate_hz: u32,
        channels: u8,
    },
    EmptySamples,
    EmptyUtteranceId,
    DuplicateUtteranceId {
        utterance_id: String,
    },
    InvalidAudioPath {
        utterance_id: String,
        path: String,
    },
    InvalidAudioSha256 {
        utterance_id: String,
    },
    ZeroDuration {
        utterance_id: String,
    },
    EmptyReference {
        utterance_id: String,
    },
}

impl RunnerManifestV1 {
    /// Deserializes and semantically validates the runner-owned manifest subset.
    ///
    /// # Errors
    ///
    /// Returns JSON syntax/type errors or a typed preflight error when identity,
    /// canonical audio, path, hash, duration, or reference invariants fail.
    pub fn parse(bytes: &[u8]) -> Result<Self, ManifestValidationError> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(ManifestValidationError::Json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Checks the fields required before any model is constructed.
    ///
    /// # Errors
    ///
    /// Returns the first semantic contract violation in manifest order.
    pub fn validate(&self) -> Result<(), ManifestValidationError> {
        if self.schema_version != 1 {
            return Err(ManifestValidationError::UnsupportedSchemaVersion {
                actual: self.schema_version,
            });
        }

        for (field, value) in [
            ("split_id", self.split_id.as_str()),
            ("dataset.id", self.dataset.id.as_str()),
            ("dataset.release", self.dataset.release.as_str()),
            ("dataset.source_split", self.dataset.source_split.as_str()),
            ("dataset.language", self.dataset.language.as_str()),
            ("normalization.id", self.normalization.id.as_str()),
            ("normalization.version", self.normalization.version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ManifestValidationError::EmptyField { field });
            }
        }

        if self.audio_format.encoding != "pcm_s16le"
            || self.audio_format.sample_rate_hz != 16_000
            || self.audio_format.channels != 1
        {
            return Err(ManifestValidationError::UnsupportedAudioFormat {
                encoding: self.audio_format.encoding.clone(),
                sample_rate_hz: self.audio_format.sample_rate_hz,
                channels: self.audio_format.channels,
            });
        }

        if self.samples.is_empty() {
            return Err(ManifestValidationError::EmptySamples);
        }

        let mut utterance_ids = HashSet::with_capacity(self.samples.len());
        for sample in &self.samples {
            if sample.utterance_id.trim().is_empty() {
                return Err(ManifestValidationError::EmptyUtteranceId);
            }
            if !utterance_ids.insert(sample.utterance_id.as_str()) {
                return Err(ManifestValidationError::DuplicateUtteranceId {
                    utterance_id: sample.utterance_id.clone(),
                });
            }
            if !is_portable_relative_path(&sample.audio.relative_path) {
                return Err(ManifestValidationError::InvalidAudioPath {
                    utterance_id: sample.utterance_id.clone(),
                    path: sample.audio.relative_path.clone(),
                });
            }
            if !is_lowercase_sha256(&sample.audio.sha256) {
                return Err(ManifestValidationError::InvalidAudioSha256 {
                    utterance_id: sample.utterance_id.clone(),
                });
            }
            if sample.audio.duration_samples == 0 {
                return Err(ManifestValidationError::ZeroDuration {
                    utterance_id: sample.utterance_id.clone(),
                });
            }
            if sample.reference.raw.trim().is_empty() {
                return Err(ManifestValidationError::EmptyReference {
                    utterance_id: sample.utterance_id.clone(),
                });
            }
        }

        Ok(())
    }
}

fn is_portable_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains(['\\', ':'])
        && path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl fmt::Display for ManifestValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid manifest JSON: {error}"),
            Self::UnsupportedSchemaVersion { actual } => {
                write!(formatter, "unsupported schema version {actual}; expected 1")
            }
            Self::EmptyField { field } => write!(formatter, "manifest field {field} is empty"),
            Self::UnsupportedAudioFormat {
                encoding,
                sample_rate_hz,
                channels,
            } => write!(
                formatter,
                "unsupported audio format {encoding}/{sample_rate_hz}Hz/{channels}ch; expected pcm_s16le/16000Hz/1ch"
            ),
            Self::EmptySamples => write!(formatter, "manifest contains no samples"),
            Self::EmptyUtteranceId => write!(formatter, "utterance_id is empty"),
            Self::DuplicateUtteranceId { utterance_id } => {
                write!(formatter, "duplicate utterance_id: {utterance_id}")
            }
            Self::InvalidAudioPath { utterance_id, path } => write!(
                formatter,
                "invalid audio path for {utterance_id}: {path:?}; expected a portable relative path"
            ),
            Self::InvalidAudioSha256 { utterance_id } => {
                write!(formatter, "invalid audio SHA-256 for {utterance_id}")
            }
            Self::ZeroDuration { utterance_id } => write!(
                formatter,
                "duration_samples must be positive for {utterance_id}"
            ),
            Self::EmptyReference { utterance_id } => {
                write!(formatter, "reference text is empty for {utterance_id}")
            }
        }
    }
}

impl Error for ManifestValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ManifestValidationError, RunnerManifestV1};

    fn valid_manifest() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "split_id": "cv26-ja-dev-strict-1000-v1",
            "dataset": {
                "id": "common_voice_ja",
                "release": "26.0",
                "source_split": "dev",
                "language": "ja",
                "license": "CC0-1.0"
            },
            "normalization": {
                "id": "nvidia_ja_cer",
                "version": "1",
                "audit_note": "public reproducibility fixture"
            },
            "audio_format": {
                "encoding": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1
            },
            "samples": [
                {
                    "utterance_id": "cv26-ja-012345",
                    "source": {
                        "relative_path": "clips/common_voice_ja_012345.mp3",
                        "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    },
                    "audio": {
                        "relative_path": "wav/cv26-ja-012345.wav",
                        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "duration_samples": 48231
                    },
                    "reference": {
                        "raw": "今日は晴れです。",
                        "normalized": "今日は晴れです"
                    },
                    "speaker_id": "speaker-42",
                    "sentence_id": null,
                    "overlap_flags": [],
                    "diagnostic_tags": []
                }
            ]
        })
    }

    #[test]
    fn parser_accepts_issue_30_audit_fields_without_redefining_them() {
        let bytes = serde_json::to_vec(&valid_manifest()).unwrap();

        let manifest = RunnerManifestV1::parse(&bytes).unwrap();

        assert_eq!(manifest.split_id, "cv26-ja-dev-strict-1000-v1");
        assert_eq!(manifest.dataset.id, "common_voice_ja");
        assert_eq!(manifest.samples[0].utterance_id, "cv26-ja-012345");
        assert_eq!(
            manifest.samples[0].audio.relative_path,
            "wav/cv26-ja-012345.wav"
        );
    }

    #[test]
    fn manifest_preflight_rejects_identity_and_audio_contract_violations() {
        let cases = [
            (
                "/schema_version",
                serde_json::json!(2),
                "unsupported schema version",
            ),
            (
                "/audio_format/sample_rate_hz",
                serde_json::json!(48000),
                "unsupported audio format",
            ),
            (
                "/samples/0/audio/relative_path",
                serde_json::json!("../outside.wav"),
                "invalid audio path",
            ),
            (
                "/samples/0/audio/relative_path",
                serde_json::json!("wav\\sample.wav"),
                "invalid audio path",
            ),
            (
                "/samples/0/audio/sha256",
                serde_json::json!("ABC"),
                "invalid audio SHA-256",
            ),
            (
                "/samples/0/audio/duration_samples",
                serde_json::json!(0),
                "duration_samples must be positive",
            ),
            (
                "/samples/0/reference/raw",
                serde_json::json!(""),
                "reference text is empty",
            ),
        ];

        for (pointer, replacement, expected) in cases {
            let mut value = valid_manifest();
            *value.pointer_mut(pointer).unwrap() = replacement;
            let error = RunnerManifestV1::parse(&serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        }
    }

    #[test]
    fn duplicate_utterance_ids_fail_the_whole_manifest_before_inference() {
        let mut value = valid_manifest();
        let duplicate = value["samples"][0].clone();
        value["samples"].as_array_mut().unwrap().push(duplicate);

        let error = RunnerManifestV1::parse(&serde_json::to_vec(&value).unwrap()).unwrap_err();

        assert!(matches!(
            error,
            ManifestValidationError::DuplicateUtteranceId { ref utterance_id }
                if utterance_id == "cv26-ja-012345"
        ));
    }
}
