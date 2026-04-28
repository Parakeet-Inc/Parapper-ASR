#[cfg(not(target_os = "macos"))]
mod osc;
mod transport;

#[cfg(not(target_os = "macos"))]
pub use osc::query_current_mute_state;
pub use transport::{
    NeoHttpTextTransport, TextTransport, detect_neo_http_port, neo_http_available,
};

#[cfg(target_os = "macos")]
pub fn query_current_mute_state() -> anyhow::Result<bool> {
    anyhow::bail!("OSCQuery support is disabled on macOS")
}
