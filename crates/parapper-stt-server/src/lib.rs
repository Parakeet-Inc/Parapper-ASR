//! Tauri-free WebSocket transport for the Parapper STT engine.

mod backend;
pub mod protocol;
mod server;

pub use backend::{
    ActiveRecognitionSession, AudioInput, BackendStartError, InputSendError, RecognitionBackend,
    RecognitionBackendConfig, RecognitionStreamEvent, StartedRecognitionSession,
    TdtDagDecodingConfig, TdtDagDecodingConfigError,
};
pub use server::{StreamingRecognitionServer, StreamingRecognitionServerConfig};
