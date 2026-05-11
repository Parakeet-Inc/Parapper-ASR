use super::YncPluginClient;
use crate::connect::registry::detect_dword_value_u16;

pub fn detect_ync_plugin_http_port() -> Option<u16> {
    let port = detect_ync_plugin_http_port_from_registry()?;
    YncPluginClient::for_command(port)
        .and_then(|client| client.probe_plugin_port(port))
        .ok()
        .map(|()| port)
}

#[cfg(windows)]
fn detect_ync_plugin_http_port_from_registry() -> Option<u16> {
    detect_dword_value_u16(r"HKCU\Software\YukarinetteConnectorNeo\TransServer", "HTTP")
}

#[cfg(not(windows))]
fn detect_ync_plugin_http_port_from_registry() -> Option<u16> {
    None
}
