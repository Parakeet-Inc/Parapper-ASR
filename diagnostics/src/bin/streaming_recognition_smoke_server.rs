//! Standalone process that runs Parapper's `/ws/recognition` WebSocket
//! server against a deterministic fake recognition backend. Used only by
//! external protocol-client automated end-to-end tests; never shipped in
//! the real application.
//!
//! Usage: `cargo run -p parapper-diagnostics
//! --bin streaming_recognition_smoke_server -- [bind_addr] [api_key]`
//! (default bind_addr is `127.0.0.1:0`, i.e. an OS-assigned port).
//!
//! Prints `LISTENING <host>:<port>` once bound, then blocks until stdin
//! closes (EOF). The test harness terminates it by closing the pipe rather
//! than relying on OS signal delivery, which behaves inconsistently for
//! console subprocesses on Windows.

use std::io::{Read, Write};
use std::net::SocketAddr;

fn main() {
    let bind_addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:0".to_string())
        .parse()
        .expect("first argument must be a valid bind address, e.g. 127.0.0.1:0");
    let api_key = std::env::args().nth(2);

    let handle = app_lib::smoke_server::start(bind_addr, api_key)
        .expect("failed to start streaming recognition smoke server");

    println!("LISTENING {}", handle.local_addr());
    std::io::stdout().flush().expect("failed to flush stdout");

    let mut buf = [0u8; 1];
    let _ = std::io::stdin().read(&mut buf);
}
