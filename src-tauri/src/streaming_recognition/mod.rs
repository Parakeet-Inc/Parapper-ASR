//! Product-independent streaming recognition protocol and server.
//!
//! Phase 0 defines only the wire contract and its state machine. The socket
//! The binary WebSocket server feeds the shared recognition pipeline.

mod backend;
pub(crate) mod protocol;
mod server;

pub(crate) use backend::NetworkOutputMode;
pub(crate) use server::{StreamingRecognitionServer, StreamingRecognitionServerConfig};
