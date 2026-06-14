use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        control::{
            engine_cache::build_language_id_engine, events::MissingModelKind,
            runtime_event::emit_missing_model_event,
        },
        transcription::route::language_id::LanguageDetector,
    },
};

pub(crate) fn build_id_detector(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Option<Box<dyn LanguageDetector>> {
    match build_language_id_engine(handle, config) {
        Ok(Some(engine)) => Some(Box::new(engine)),
        Ok(None) => None,
        Err(err) => {
            let reason = format!("Failed to initialize language identification: {err}");
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::LanguageId, reason);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use tauri::Listener;

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn build_id_detector_failure_emits_language_id_missing_event() {
        let handle = crate::recognition::control::tests::tauri_test_handle();
        let (sender, receiver) =
            mpsc::channel::<crate::recognition::control::events::MissingModelEvent>();
        let _event_id = handle.listen("parapper://asr-missing", move |event| {
            let payload = serde_json::from_str::<
                crate::recognition::control::events::MissingModelEvent,
            >(event.payload())
            .expect("missing model payload should decode");
            sender
                .send(payload)
                .expect("missing model event should be recorded");
        });
        let config = parapper_config! {
            multilingual_asr_enabled: true,
            model_dir: Some(missing_model_dir("language-id-detector-failure")),
            ..ParapperConfig::default()
        };

        let detector = build_id_detector(&handle, &config);

        assert!(
            detector.is_none(),
            "missing local language ID model should leave SLI unavailable"
        );
        let event = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("language ID initialization failure should emit missing model");
        assert_eq!(
            event.kind,
            crate::recognition::control::events::MissingModelKind::LanguageId
        );
        assert!(
            event
                .reason
                .contains("Failed to initialize language identification"),
            "unexpected missing model reason: {}",
            event.reason
        );
    }

    fn missing_model_dir(test_name: &str) -> String {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "parapper-missing-language-id-model-{test_name}-{}-{unique}",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
    }
}
