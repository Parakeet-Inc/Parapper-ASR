use serde::Serialize;
use serde_json::{Map, Value};

pub(crate) const PROTOCOL_VERSION: u32 = 1;
pub(crate) const MAX_AUDIO_FRAME_BYTES: usize = 3_200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AudioFormat {
    pub(crate) encoding: String,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u8,
}

impl AudioFormat {
    #[cfg(test)]
    pub(crate) fn pcm_s16le_16khz_mono() -> Self {
        Self {
            encoding: "pcm_s16le".to_string(),
            sample_rate: 16_000,
            channels: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClientControl {
    Start {
        version: u32,
        session_id: String,
        audio: AudioFormat,
    },
    Stop {
        version: u32,
        session_id: String,
    },
    Cancel {
        version: u32,
        session_id: String,
    },
    Ping {
        version: u32,
        request_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ErrorCode {
    UnsupportedVersion,
    InvalidJson,
    InvalidState,
    SessionIdRequired,
    SessionIdMismatch,
    UnsupportedAudioEncoding,
    UnsupportedSampleRate,
    UnsupportedChannelCount,
    InvalidAudioFrame,
    AudioFrameTooLarge,
    AudioQueueOverrun,
    RecognitionBusy,
    ModelUnavailable,
    RecognitionFailed,
    DrainTimeout,
    ServerStopping,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtocolError {
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
    pub(crate) fatal: bool,
}

impl ProtocolError {
    fn fatal(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            fatal: true,
        }
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ProtocolError {}

pub(crate) fn parse_client_control(raw: &str) -> Result<ClientControl, ProtocolError> {
    let value: Value = serde_json::from_str(raw).map_err(|error| {
        ProtocolError::fatal(ErrorCode::InvalidJson, format!("invalid JSON: {error}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        ProtocolError::fatal(ErrorCode::InvalidJson, "control message must be an object")
    })?;
    let version = parse_version(object)?;
    let kind = required_string(object, "type", ErrorCode::InvalidJson)?;

    match kind.as_str() {
        "session.start" => {
            let session_id = required_string(object, "session_id", ErrorCode::SessionIdRequired)?;
            let audio = parse_audio_format(object)?;
            Ok(ClientControl::Start {
                version,
                session_id,
                audio,
            })
        }
        "session.stop" => Ok(ClientControl::Stop {
            version,
            session_id: required_string(object, "session_id", ErrorCode::SessionIdRequired)?,
        }),
        "session.cancel" => Ok(ClientControl::Cancel {
            version,
            session_id: required_string(object, "session_id", ErrorCode::SessionIdRequired)?,
        }),
        "ping" => Ok(ClientControl::Ping {
            version,
            request_id: required_string(object, "request_id", ErrorCode::InvalidJson)?,
        }),
        _ => Err(ProtocolError::fatal(
            ErrorCode::InvalidJson,
            format!("unsupported control message type: {kind}"),
        )),
    }
}

fn parse_version(object: &Map<String, Value>) -> Result<u32, ProtocolError> {
    let version = object
        .get("version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok());
    match version {
        Some(PROTOCOL_VERSION) => Ok(PROTOCOL_VERSION),
        _ => Err(ProtocolError::fatal(
            ErrorCode::UnsupportedVersion,
            "control message version must be 1",
        )),
    }
}

fn required_string(
    object: &Map<String, Value>,
    field: &str,
    code: ErrorCode,
) -> Result<String, ProtocolError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ProtocolError::fatal(code, format!("{field} is required")))
}

fn parse_audio_format(object: &Map<String, Value>) -> Result<AudioFormat, ProtocolError> {
    let audio = object
        .get("audio")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProtocolError::fatal(
                ErrorCode::UnsupportedAudioEncoding,
                "audio format is required",
            )
        })?;
    let encoding = required_string(audio, "encoding", ErrorCode::UnsupportedAudioEncoding)?;
    if encoding != "pcm_s16le" {
        return Err(ProtocolError::fatal(
            ErrorCode::UnsupportedAudioEncoding,
            "audio encoding must be pcm_s16le",
        ));
    }
    let sample_rate = audio
        .get("sample_rate")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            ProtocolError::fatal(ErrorCode::UnsupportedSampleRate, "sample_rate is required")
        })?;
    if sample_rate != 16_000 {
        return Err(ProtocolError::fatal(
            ErrorCode::UnsupportedSampleRate,
            "sample_rate must be 16000",
        ));
    }
    let channels = audio
        .get("channels")
        .and_then(Value::as_u64)
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| {
            ProtocolError::fatal(ErrorCode::UnsupportedChannelCount, "channels is required")
        })?;
    if channels != 1 {
        return Err(ProtocolError::fatal(
            ErrorCode::UnsupportedChannelCount,
            "channels must be 1",
        ));
    }
    Ok(AudioFormat {
        encoding,
        sample_rate,
        channels,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProtocolState {
    AwaitingStart,
    Active,
    Draining,
    Cancelled,
    Done,
    ProtocolError,
    #[cfg(test)]
    Disconnected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProtocolAction {
    Start {
        session_id: String,
        audio: AudioFormat,
    },
    Audio {
        byte_len: usize,
    },
    GracefulStop,
    Cancel,
    Pong {
        request_id: String,
    },
    #[cfg(test)]
    None,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionProtocol {
    state: ProtocolState,
    session_id: Option<String>,
}

impl SessionProtocol {
    pub(crate) fn new() -> Self {
        Self {
            state: ProtocolState::AwaitingStart,
            session_id: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> ProtocolState {
        self.state
    }

    pub(crate) fn on_text(&mut self, raw: &str) -> Result<ProtocolAction, ProtocolError> {
        let control = parse_client_control(raw).map_err(|error| self.fail(error))?;
        match control {
            ClientControl::Start {
                session_id, audio, ..
            } if self.state == ProtocolState::AwaitingStart => {
                self.session_id = Some(session_id.clone());
                self.state = ProtocolState::Active;
                Ok(ProtocolAction::Start { session_id, audio })
            }
            ClientControl::Stop { session_id, .. } if self.state == ProtocolState::Active => {
                self.require_session_id(&session_id)?;
                self.state = ProtocolState::Draining;
                Ok(ProtocolAction::GracefulStop)
            }
            ClientControl::Cancel { session_id, .. } if self.state == ProtocolState::Active => {
                self.require_session_id(&session_id)?;
                self.state = ProtocolState::Cancelled;
                Ok(ProtocolAction::Cancel)
            }
            ClientControl::Ping { request_id, .. }
                if matches!(
                    self.state,
                    ProtocolState::AwaitingStart | ProtocolState::Active | ProtocolState::Draining
                ) =>
            {
                Ok(ProtocolAction::Pong { request_id })
            }
            _ => Err(self.fail(ProtocolError::fatal(
                ErrorCode::InvalidState,
                "control message is not valid in the current session state",
            ))),
        }
    }

    pub(crate) fn on_binary(&mut self, byte_len: usize) -> Result<ProtocolAction, ProtocolError> {
        if self.state != ProtocolState::Active {
            return Err(self.fail(ProtocolError::fatal(
                ErrorCode::InvalidState,
                "binary audio is only accepted in an active session",
            )));
        }
        if byte_len == 0 || !byte_len.is_multiple_of(2) {
            return Err(self.fail(ProtocolError::fatal(
                ErrorCode::InvalidAudioFrame,
                "audio frame must contain a non-empty even number of bytes",
            )));
        }
        if byte_len > MAX_AUDIO_FRAME_BYTES {
            return Err(self.fail(ProtocolError::fatal(
                ErrorCode::AudioFrameTooLarge,
                "audio frame exceeds the 100 ms limit",
            )));
        }
        Ok(ProtocolAction::Audio { byte_len })
    }

    #[cfg(test)]
    pub(crate) fn on_disconnect(&mut self) -> ProtocolAction {
        let should_cancel = matches!(self.state, ProtocolState::Active | ProtocolState::Draining);
        self.state = ProtocolState::Disconnected;
        if should_cancel {
            ProtocolAction::Cancel
        } else {
            ProtocolAction::None
        }
    }

    #[cfg(test)]
    pub(crate) fn can_emit_turn_result(&self) -> bool {
        matches!(self.state, ProtocolState::Active | ProtocolState::Draining)
    }

    pub(crate) fn mark_done(&mut self) -> Result<(), ProtocolError> {
        if self.state != ProtocolState::Draining {
            return Err(self.fail(ProtocolError::fatal(
                ErrorCode::InvalidState,
                "session.done is only valid after graceful stop",
            )));
        }
        self.state = ProtocolState::Done;
        Ok(())
    }

    fn require_session_id(&mut self, actual: &str) -> Result<(), ProtocolError> {
        if self.session_id.as_deref() == Some(actual) {
            Ok(())
        } else {
            Err(self.fail(ProtocolError::fatal(
                ErrorCode::SessionIdMismatch,
                "control message session_id does not match the active session",
            )))
        }
    }

    fn fail(&mut self, error: ProtocolError) -> ProtocolError {
        self.state = ProtocolState::ProtocolError;
        error
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Capabilities {
    partial: bool,
    speech_started: bool,
    cancel: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ServerMessage {
    #[serde(rename = "session.ready")]
    SessionReady {
        version: u32,
        session_id: String,
        capabilities: Capabilities,
    },
    #[serde(rename = "speech.started")]
    SpeechStarted { version: u32, session_id: String },
    #[serde(rename = "turn.partial")]
    TurnPartial {
        version: u32,
        session_id: String,
        turn_session_id: u64,
        turn_id: u64,
        revision: u64,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        text: String,
        source_asr_model: String,
        source_language: String,
        detected_language: Option<String>,
        elapsed_ms: u64,
    },
    #[serde(rename = "turn.final")]
    TurnFinal {
        version: u32,
        session_id: String,
        turn_session_id: u64,
        turn_id: u64,
        revision: u64,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        text: String,
        source_asr_model: String,
        source_language: String,
        detected_language: Option<String>,
        audio_duration_ms: u64,
        elapsed_ms: u64,
    },
    #[serde(rename = "error")]
    Error {
        version: u32,
        session_id: Option<String>,
        code: ErrorCode,
        message: String,
        fatal: bool,
    },
    #[serde(rename = "session.done")]
    SessionDone { version: u32, session_id: String },
    #[serde(rename = "session.cancelled")]
    SessionCancelled { version: u32, session_id: String },
    #[serde(rename = "pong")]
    Pong { version: u32, request_id: String },
}

impl ServerMessage {
    pub(crate) fn ready(session_id: &str) -> Self {
        Self::SessionReady {
            version: PROTOCOL_VERSION,
            session_id: session_id.to_string(),
            capabilities: Capabilities {
                partial: true,
                speech_started: true,
                cancel: true,
            },
        }
    }

    pub(crate) fn error(
        session_id: Option<&str>,
        code: ErrorCode,
        message: impl Into<String>,
    ) -> Self {
        Self::Error {
            version: PROTOCOL_VERSION,
            session_id: session_id.map(str::to_string),
            code,
            message: message.into(),
            fatal: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{
        AudioFormat, ClientControl, ErrorCode, ProtocolAction, ProtocolState, ServerMessage,
        SessionProtocol, parse_client_control,
    };

    const START: &str = r#"{
        "version": 1,
        "type": "session.start",
        "session_id": "client-1",
        "audio": {
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1
        }
    }"#;

    fn active_session() -> SessionProtocol {
        let mut protocol = SessionProtocol::new();
        assert_eq!(
            protocol.on_text(START).unwrap(),
            ProtocolAction::Start {
                session_id: "client-1".to_string(),
                audio: AudioFormat::pcm_s16le_16khz_mono(),
            }
        );
        protocol
    }

    #[test]
    fn binary_audio_before_session_start_is_fatal_invalid_state() {
        let mut protocol = SessionProtocol::new();

        let error = protocol.on_binary(1_024).unwrap_err();

        assert_eq!(error.code, ErrorCode::InvalidState);
        assert!(error.fatal);
        assert_eq!(protocol.state(), ProtocolState::ProtocolError);
    }

    #[test]
    fn session_start_requires_a_non_empty_session_id() {
        for raw in [
            r#"{"version":1,"type":"session.start","audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":1}}"#,
            r#"{"version":1,"type":"session.start","session_id":"","audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":1}}"#,
        ] {
            let error = parse_client_control(raw).unwrap_err();
            assert_eq!(error.code, ErrorCode::SessionIdRequired, "raw={raw}");
        }
    }

    #[test]
    fn every_control_message_rejects_an_unsupported_version() {
        let cases = [
            r#"{"type":"ping","request_id":"ping-1"}"#,
            r#"{"version":2,"type":"session.start","session_id":"client-1","audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":1}}"#,
            r#"{"version":2,"type":"session.stop","session_id":"client-1"}"#,
            r#"{"version":2,"type":"session.cancel","session_id":"client-1"}"#,
            r#"{"version":2,"type":"ping","request_id":"ping-1"}"#,
        ];

        for raw in cases {
            let error = parse_client_control(raw).unwrap_err();
            assert_eq!(error.code, ErrorCode::UnsupportedVersion, "raw={raw}");
        }
    }

    #[test]
    fn duplicate_start_is_fatal_invalid_state() {
        let mut protocol = active_session();

        let error = protocol.on_text(START).unwrap_err();

        assert_eq!(error.code, ErrorCode::InvalidState);
        assert_eq!(protocol.state(), ProtocolState::ProtocolError);
    }

    #[test]
    fn stop_and_cancel_require_the_active_session_id() {
        for kind in ["session.stop", "session.cancel"] {
            let mut protocol = active_session();
            let raw = format!(r#"{{"version":1,"type":"{kind}","session_id":"other-client"}}"#);

            let error = protocol.on_text(&raw).unwrap_err();

            assert_eq!(error.code, ErrorCode::SessionIdMismatch, "kind={kind}");
            assert_eq!(protocol.state(), ProtocolState::ProtocolError);
        }
    }

    #[test]
    fn graceful_stop_enters_draining_and_rejects_late_audio() {
        let mut protocol = active_session();

        assert_eq!(
            protocol
                .on_text(r#"{"version":1,"type":"session.stop","session_id":"client-1"}"#,)
                .unwrap(),
            ProtocolAction::GracefulStop
        );
        assert_eq!(protocol.state(), ProtocolState::Draining);

        let error = protocol.on_binary(1_024).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidState);
    }

    #[test]
    fn cancel_and_disconnect_never_allow_a_turn_final() {
        let mut cancelled = active_session();
        assert_eq!(
            cancelled
                .on_text(r#"{"version":1,"type":"session.cancel","session_id":"client-1"}"#,)
                .unwrap(),
            ProtocolAction::Cancel
        );
        assert_eq!(cancelled.state(), ProtocolState::Cancelled);
        assert!(!cancelled.can_emit_turn_result());

        let mut disconnected = active_session();
        assert_eq!(disconnected.on_disconnect(), ProtocolAction::Cancel);
        assert_eq!(disconnected.state(), ProtocolState::Disconnected);
        assert!(!disconnected.can_emit_turn_result());
    }

    #[test]
    fn audio_frame_limits_are_checked_before_queueing() {
        let cases = [
            (0, ErrorCode::InvalidAudioFrame),
            (1, ErrorCode::InvalidAudioFrame),
            (3_202, ErrorCode::AudioFrameTooLarge),
        ];
        for (byte_len, expected) in cases {
            let mut protocol = active_session();
            let error = protocol.on_binary(byte_len).unwrap_err();
            assert_eq!(error.code, expected, "byte_len={byte_len}");
        }

        let mut protocol = active_session();
        assert_eq!(
            protocol.on_binary(3_200).unwrap(),
            ProtocolAction::Audio { byte_len: 3_200 }
        );
    }

    #[test]
    fn start_rejects_each_unsupported_audio_property_with_a_stable_code() {
        let cases = [
            (
                r#"{"version":1,"type":"session.start","session_id":"client-1","audio":{"encoding":"pcm_f32le","sample_rate":16000,"channels":1}}"#,
                ErrorCode::UnsupportedAudioEncoding,
            ),
            (
                r#"{"version":1,"type":"session.start","session_id":"client-1","audio":{"encoding":"pcm_s16le","sample_rate":48000,"channels":1}}"#,
                ErrorCode::UnsupportedSampleRate,
            ),
            (
                r#"{"version":1,"type":"session.start","session_id":"client-1","audio":{"encoding":"pcm_s16le","sample_rate":16000,"channels":2}}"#,
                ErrorCode::UnsupportedChannelCount,
            ),
        ];
        for (raw, expected) in cases {
            let error = parse_client_control(raw).unwrap_err();
            assert_eq!(error.code, expected, "raw={raw}");
        }
    }

    #[test]
    fn ping_is_available_before_during_and_while_draining_a_session() {
        let ping = r#"{"version":1,"type":"ping","request_id":"ping-1"}"#;
        let expected = ProtocolAction::Pong {
            request_id: "ping-1".to_string(),
        };

        let mut awaiting = SessionProtocol::new();
        assert_eq!(awaiting.on_text(ping).unwrap(), expected);

        let mut active = active_session();
        assert_eq!(active.on_text(ping).unwrap(), expected);

        active
            .on_text(r#"{"version":1,"type":"session.stop","session_id":"client-1"}"#)
            .unwrap();
        assert_eq!(active.on_text(ping).unwrap(), expected);
    }

    #[test]
    fn graceful_stop_allows_results_until_done_then_becomes_terminal() {
        let mut protocol = active_session();
        protocol
            .on_text(r#"{"version":1,"type":"session.stop","session_id":"client-1"}"#)
            .unwrap();
        assert!(protocol.can_emit_turn_result());

        protocol.mark_done().unwrap();

        assert_eq!(protocol.state(), ProtocolState::Done);
        assert!(!protocol.can_emit_turn_result());
    }

    #[test]
    fn client_start_fixture_matches_the_version_one_dto() {
        let raw = include_str!(
            "../../../documents/developer/protocol/fixtures/client-session-start-v1.json"
        );
        assert_eq!(
            parse_client_control(raw).unwrap(),
            ClientControl::Start {
                version: 1,
                session_id: "fixture-session".to_string(),
                audio: AudioFormat::pcm_s16le_16khz_mono(),
            }
        );
    }

    #[test]
    fn server_fixtures_match_the_exact_public_json_contract() {
        let ready = ServerMessage::ready("fixture-session");
        let ready_actual = serde_json::to_value(ready).unwrap();
        let ready_expected: Value = serde_json::from_str(include_str!(
            "../../../documents/developer/protocol/fixtures/server-session-ready-v1.json"
        ))
        .unwrap();
        assert_eq!(ready_actual, ready_expected);

        let error = ServerMessage::error(
            Some("fixture-session"),
            ErrorCode::RecognitionBusy,
            "another recognition session is active",
        );
        assert_eq!(
            serde_json::to_value(error).unwrap(),
            json!({
                "version": 1,
                "type": "error",
                "session_id": "fixture-session",
                "code": "recognition_busy",
                "message": "another recognition session is active",
                "fatal": true
            })
        );

        let final_message = ServerMessage::TurnFinal {
            version: 1,
            session_id: "fixture-session".to_string(),
            turn_session_id: 7,
            turn_id: 3,
            revision: 2,
            segment_id: 8,
            previous_segment_id: Some(7),
            text: "こんにちは。".to_string(),
            source_asr_model: "reazonspeech_k2_v2".to_string(),
            source_language: "ja".to_string(),
            detected_language: None,
            audio_duration_ms: 1_280,
            elapsed_ms: 96,
        };
        let final_expected: Value = serde_json::from_str(include_str!(
            "../../../documents/developer/protocol/fixtures/server-turn-final-v1.json"
        ))
        .unwrap();
        assert_eq!(serde_json::to_value(final_message).unwrap(), final_expected);
    }
}
