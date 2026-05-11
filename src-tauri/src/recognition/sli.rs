use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        engines::SpokenLanguageIdentificationEngine,
        route::{
            RecognitionRoute, RecognitionRouteSelection, language_id_candidate_codes,
            route_for_detected_language,
        },
        transcription::worker_runtime::{
            MIN_LANGUAGE_ID_SAMPLES, emit_asr_warning, normalize_asr_input_audio,
        },
    },
};

pub(crate) trait LanguageDetector {
    fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> anyhow::Result<String>;
}

impl LanguageDetector for SpokenLanguageIdentificationEngine {
    fn detect(&mut self, samples: &[f32], candidates: Option<&[&str]>) -> anyhow::Result<String> {
        SpokenLanguageIdentificationEngine::detect(self, samples, candidates)
    }
}

pub(crate) struct SliContext<'a> {
    pub(crate) handle: &'a AppHandle,
    pub(crate) config: &'a ParapperConfig,
    pub(crate) language_id: Option<&'a mut dyn LanguageDetector>,
}

// Language identification is only for selecting the ASR route when
// multilingual ASR routing is enabled. Translation target selection must not
// depend on this result; delivery uses configured translation mappings.
pub(crate) fn detect_recognition_route(
    context: &mut SliContext<'_>,
    current_route: Option<RecognitionRoute>,
    full_audio: &[f32],
) -> RecognitionRouteSelection {
    let default_route = route_without_language_detection(context.config, current_route);
    let default_selection = |_reason| RecognitionRouteSelection {
        route: default_route,
        detected_language: None,
    };
    let default_selection_with_language = |_reason, detected_language| RecognitionRouteSelection {
        route: default_route,
        detected_language,
    };
    if !context.config.multilingual_asr_enabled {
        return default_selection("multilingual ASR routing is disabled");
    }
    if !should_detect_language(full_audio.len()) {
        return default_selection("audio is shorter than the language identification minimum");
    }

    let language_audio = normalize_asr_input_audio(context.config, full_audio);
    let Some(language_id) = context.language_id.as_deref_mut() else {
        return default_selection("language identification engine is unavailable");
    };
    let language_candidates = language_id_candidate_codes(context.config);
    match language_id.detect(language_audio.as_ref(), language_candidates.as_deref()) {
        Ok(language_code) if !language_code.is_empty() => {
            let detected_language = Some(language_code.clone());
            let Some(route) = route_for_detected_language(context.config, &language_code) else {
                return default_selection_with_language(
                    &format!("detected language {language_code} has no enabled ASR route"),
                    detected_language,
                );
            };
            RecognitionRouteSelection {
                route,
                detected_language,
            }
        }
        Ok(_) => default_selection("language identification returned an empty language"),
        Err(err) => {
            emit_asr_warning(context.handle, &err);
            default_selection("language identification failed")
        }
    }
}

pub(crate) fn route_without_language_detection(
    config: &ParapperConfig,
    current_route: Option<RecognitionRoute>,
) -> RecognitionRoute {
    if let Some(route) = current_route {
        return route;
    }
    RecognitionRoute::from_language(config.asr_language)
}

pub(crate) fn should_detect_language(sample_len: usize) -> bool {
    sample_len >= MIN_LANGUAGE_ID_SAMPLES
}

#[cfg(test)]
mod tests {
    use super::{
        LanguageDetector, SliContext, detect_recognition_route, route_without_language_detection,
        should_detect_language,
    };
    use crate::{
        config::{AsrLanguage, AsrModel, ParapperConfig},
        recognition::{
            route::RecognitionRoute, transcription::worker_runtime::MIN_LANGUAGE_ID_SAMPLES,
        },
    };
    use anyhow::Result;

    #[test]
    fn route_without_language_detection_prefers_current_turn_route() {
        let config = ParapperConfig {
            asr_language: AsrLanguage::Japanese,
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![AsrModel::ReazonSpeechK2V2],
            ..ParapperConfig::default()
        };
        let current_route = RecognitionRoute::from_language(AsrLanguage::English);

        assert_eq!(
            route_without_language_detection(&config, Some(current_route)),
            current_route
        );
    }

    #[test]
    fn route_without_language_detection_uses_configured_language_without_current_turn_route() {
        let config = ParapperConfig {
            asr_language: AsrLanguage::Japanese,
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![AsrModel::ReazonSpeechK2V2],
            ..ParapperConfig::default()
        };

        assert_eq!(
            route_without_language_detection(&config, None),
            RecognitionRoute::from_language(AsrLanguage::Japanese)
        );
    }

    #[test]
    fn language_detection_threshold_requires_minimum_samples() {
        assert!(!should_detect_language(
            MIN_LANGUAGE_ID_SAMPLES.saturating_sub(1)
        ));
        assert!(should_detect_language(MIN_LANGUAGE_ID_SAMPLES));
    }

    #[test]
    fn detect_recognition_route_uses_language_id_when_audio_is_long_enough() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            ..ParapperConfig::default()
        };
        let app = tauri::Builder::default()
            .any_thread()
            .build(tauri::generate_context!())
            .expect("test app should build");
        let mut language_id = ScriptedLanguageDetector {
            detected_language: "en".to_string(),
            calls: 0,
        };

        let selection = detect_recognition_route(
            &mut SliContext {
                handle: app.handle(),
                config: &config,
                language_id: Some(&mut language_id),
            },
            Some(RecognitionRoute::from_language(AsrLanguage::Japanese)),
            &vec![1.0; MIN_LANGUAGE_ID_SAMPLES],
        );

        assert_eq!(language_id.calls, 1);
        assert_eq!(selection.detected_language.as_deref(), Some("en"));
        assert_eq!(selection.route.model, AsrModel::NemoParakeetTdt0_6BV2Int8);
    }

    struct ScriptedLanguageDetector {
        detected_language: String,
        calls: usize,
    }

    impl LanguageDetector for ScriptedLanguageDetector {
        fn detect(&mut self, _samples: &[f32], _candidates: Option<&[&str]>) -> Result<String> {
            self.calls += 1;
            Ok(self.detected_language.clone())
        }
    }
}
