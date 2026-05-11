mod client;
mod discovery;
mod protocol;
mod speech;
#[cfg(test)]
mod tests;
mod text_input;
mod translation;

const DEFAULT_HOST: &str = "127.0.0.1";
const PLUGIN_COMMAND_PATH: &str = "/";
const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const SPEECH_HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

pub use client::YncPluginClient;
pub use discovery::detect_ync_plugin_http_port;
pub use speech::SpeechRequest;
pub use text_input::{
    YncTextInputTransport, detect_ync_text_input_http_port, ync_text_input_http_available,
};
