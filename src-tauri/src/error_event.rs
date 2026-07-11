use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParapperErrorType {
    AudioInput,
    Resampler,
    Vad,
    Asr,
    RecognitionBusy,
    ModelDownload,
    NeoHttp,
    OscQuery,
    FileSave,
    Config,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ErrorSeverity {
    Warning,
    Fatal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ParapperErrorPayload {
    pub error_type: ParapperErrorType,
    pub severity: ErrorSeverity,
    pub detail: Option<String>,
}

impl ParapperErrorPayload {
    #[must_use]
    pub fn new(
        error_type: ParapperErrorType,
        severity: ErrorSeverity,
        detail: Option<String>,
    ) -> Self {
        Self {
            error_type,
            severity,
            detail,
        }
    }
}

pub fn parapper_error_payload(
    error_type: ParapperErrorType,
    severity: ErrorSeverity,
    detail: String,
) -> ParapperErrorPayload {
    ParapperErrorPayload::new(error_type, severity, Some(detail))
}

pub fn emit_parapper_error(
    handle: &AppHandle,
    error_type: ParapperErrorType,
    severity: ErrorSeverity,
    detail: impl Into<Option<String>>,
) {
    let payload = ParapperErrorPayload::new(error_type, severity, detail.into());
    if let Err(err) = handle.emit("parapper://error", payload) {
        log::error!("failed to emit parapper error event: {err}");
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ErrorSeverity, ParapperErrorPayload, ParapperErrorType};

    #[test]
    fn parapper_error_payload_json_contract() {
        let payload = ParapperErrorPayload::new(
            ParapperErrorType::NeoHttp,
            ErrorSeverity::Warning,
            Some("connection failed".to_string()),
        );

        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            json!({
                "errorType": "NEO_HTTP",
                "severity": "warning",
                "detail": "connection failed"
            })
        );
    }
}
