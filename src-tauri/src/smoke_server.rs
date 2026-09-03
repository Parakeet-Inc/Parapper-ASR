//! Deterministic fake-backend `/ws/recognition` server for local/CI
//! protocol smoke tests, driven by an external protocol client.
//!
//! Only compiled with the `smoke-server` Cargo feature, which is off by
//! default. No ASR model is loaded and this module is never reachable from
//! the shipped application.

use std::net::SocketAddr;

use crate::recognition::StreamingRecognitionServer;

/// Owns a running smoke-test listener. Dropping it stops the listener.
pub struct SmokeServerHandle {
    server: StreamingRecognitionServer,
}

impl SmokeServerHandle {
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.server.local_addr()
    }
}

/// Starts a `/ws/recognition` listener backed by a deterministic fake
/// recognition backend: the first audio chunk in a session produces
/// `speech.started` plus one fixed `turn.partial`, and a graceful
/// `session.stop` produces one fixed `turn.final`.
///
/// # Errors
///
/// Returns an error when the loopback listener cannot bind or its worker thread cannot start.
pub fn start(bind_addr: SocketAddr, api_key: Option<String>) -> anyhow::Result<SmokeServerHandle> {
    let server = StreamingRecognitionServer::start_smoke(bind_addr, api_key)?;
    Ok(SmokeServerHandle { server })
}
