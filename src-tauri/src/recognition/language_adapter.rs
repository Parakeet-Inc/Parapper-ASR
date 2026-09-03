use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        asr_worker::emit_asr_warning,
        events::{MissingModelKind, emit_missing_model_event},
        model_factory::build_language_id_engine,
    },
};

pub(crate) use parapper_stt_engine::ports::{LanguageDetectionWarningSink, LanguageDetector};

pub(crate) struct AppLanguageDetector(parapper_models::asr::SpokenLanguageIdentificationEngine);

impl AppLanguageDetector {
    pub(crate) const fn new(
        engine: parapper_models::asr::SpokenLanguageIdentificationEngine,
    ) -> Self {
        Self(engine)
    }
}

impl LanguageDetector for AppLanguageDetector {
    fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> anyhow::Result<String> {
        self.0.detect(samples, candidates)
    }
}

pub(crate) fn build_id_detector(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Option<Box<dyn LanguageDetector>> {
    match build_language_id_engine(handle, config) {
        Ok(Some(engine)) => Some(Box::new(AppLanguageDetector::new(engine))),
        Ok(None) => None,
        Err(err) => {
            let reason = format!("Failed to initialize language identification: {err}");
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::LanguageId, reason);
            None
        }
    }
}

pub(in crate::recognition) trait LanguageIdRuntime:
    LanguageDetectionWarningSink
{
}

pub(crate) struct LanguageWarningAdapter(pub(crate) Box<dyn LanguageIdRuntime>);

impl LanguageDetectionWarningSink for LanguageWarningAdapter {
    fn emit_language_detection_warning(&self, error: &anyhow::Error) {
        self.0.emit_language_detection_warning(error);
    }
}

pub(crate) fn tauri_language_warning_runtime(handle: &AppHandle) -> Box<dyn LanguageIdRuntime> {
    Box::new(TauriLanguageIdRuntime {
        handle: handle.clone(),
    })
}

struct TauriLanguageIdRuntime {
    handle: AppHandle,
}

impl LanguageDetectionWarningSink for TauriLanguageIdRuntime {
    fn emit_language_detection_warning(&self, error: &anyhow::Error) {
        emit_asr_warning(&self.handle, error);
    }
}

impl LanguageIdRuntime for TauriLanguageIdRuntime {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use tauri::Listener;

    use crate::recognition::events::{MissingModelEvent, MissingModelKind};

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn build_id_detector_failure_emits_language_id_missing_event() {
        let handle = crate::recognition::tests::tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<MissingModelEvent>();
        let _event_id = handle.listen("parapper://asr-missing", move |event| {
            let payload = serde_json::from_str::<MissingModelEvent>(event.payload())
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
        assert_eq!(event.kind, MissingModelKind::LanguageId);
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
