#[cfg(not(target_os = "macos"))]
mod osc;
mod registry;
#[cfg(test)]
pub(crate) mod test_support;
mod transport;
mod ync;

#[cfg(not(target_os = "macos"))]
pub use osc::query_current_mute_state;
pub use transport::TextTransport;
pub use ync::{
    SpeechRequest, YncPluginClient, YncTextInputTransport, detect_ync_plugin_http_port,
    detect_ync_text_input_http_port, ync_text_input_http_available,
};

#[cfg(target_os = "macos")]
pub fn query_current_mute_state() -> anyhow::Result<bool> {
    anyhow::bail!("OSCQuery support is disabled on macOS")
}
