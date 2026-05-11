use std::thread::{self, JoinHandle};

use tauri::{AppHandle, Emitter};

use crate::{
    config::ParapperConfig,
    connect::query_current_mute_state,
    delivery::sinks::ui_event::emit_connection_state,
    recognition::events::{ConnectionTarget, OscMuteStateEvent},
};

pub(crate) fn spawn_mute_check_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Option<JoinHandle<bool>> {
    (config.vrc_osc_micmute && ParapperConfig::vrc_osc_supported())
        .then(|| spawn_vrchat_mute_check(handle.clone()))
}

pub(crate) fn is_vrchat_muted_before_send(
    handle: &AppHandle,
    mute_check: Option<JoinHandle<bool>>,
) -> bool {
    if let Some(mute_check) = mute_check {
        return if let Ok(is_muted) = mute_check.join() {
            is_muted
        } else {
            emit_osc_mute_state(handle, None);
            false
        };
    }

    query_vrchat_mute_state_with_cache(handle)
}

fn spawn_vrchat_mute_check(handle: AppHandle) -> JoinHandle<bool> {
    thread::spawn(move || query_vrchat_mute_state_with_cache(&handle))
}

fn query_vrchat_mute_state_with_cache(handle: &AppHandle) -> bool {
    if let Ok(is_muted) = query_current_mute_state() {
        emit_connection_state(handle, ConnectionTarget::Vrchat, true, None);
        emit_osc_mute_state(handle, Some(is_muted));
        is_muted
    } else {
        emit_connection_state(handle, ConnectionTarget::Vrchat, false, None);
        emit_osc_mute_state(handle, None);
        false
    }
}

fn emit_osc_mute_state(handle: &AppHandle, muted: Option<bool>) {
    let _ = handle.emit("parapper://osc-mute-state", OscMuteStateEvent { muted });
}
