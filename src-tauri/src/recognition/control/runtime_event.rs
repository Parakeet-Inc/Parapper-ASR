use tauri::{AppHandle, Emitter};

use crate::recognition::control::events::{MissingModelEvent, MissingModelKind};

pub(in crate::recognition) fn emit_missing_model_event(
    handle: &AppHandle,
    kind: MissingModelKind,
    reason: String,
) {
    let _ = handle.emit("parapper://asr-missing", MissingModelEvent { kind, reason });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    use tauri::Listener;

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn emit_missing_model_event_dispatches_kind_and_reason_to_ui_channel() {
        let handle = crate::recognition::control::tests::tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<MissingModelEvent>();
        let _event_id = handle.listen("parapper://asr-missing", move |event| {
            let payload = serde_json::from_str::<MissingModelEvent>(event.payload())
                .expect("missing model payload should decode");
            sender
                .send(payload)
                .expect("missing model event should be recorded");
        });

        emit_missing_model_event(
            &handle,
            MissingModelKind::TurnDetector,
            "missing model".to_string(),
        );

        let event = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("missing model event should be emitted");
        assert_eq!(event.kind, MissingModelKind::TurnDetector);
        assert_eq!(event.reason, "missing model");
    }
}
