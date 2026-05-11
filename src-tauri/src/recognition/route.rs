use crate::{
    config::{AsrLanguage, AsrModel, ParapperConfig},
    model::NamoTurnDetectorModel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecognitionRoute {
    pub(crate) language: AsrLanguage,
    pub(crate) model: AsrModel,
    pub(crate) turn_detector_model: NamoTurnDetectorModel,
}

pub(crate) struct RecognitionRouteSelection {
    pub(crate) route: RecognitionRoute,
    pub(crate) detected_language: Option<String>,
}

impl RecognitionRoute {
    pub(crate) fn from_language(language: AsrLanguage) -> Self {
        let model = AsrModel::default_for_language(language);
        let turn_detector_model = NamoTurnDetectorModel::for_asr_language(language);
        Self {
            language,
            model,
            turn_detector_model,
        }
    }

    pub(crate) fn from_model(model: AsrModel) -> Self {
        let language = model.language();
        let turn_detector_model = NamoTurnDetectorModel::for_asr_language(language);
        Self {
            language,
            model,
            turn_detector_model,
        }
    }

    fn preferred_for_detected_language_code(language_code: &str) -> Self {
        match language_code {
            "ja" => Self::from_language(AsrLanguage::Japanese),
            "en" => Self::from_language(AsrLanguage::English),
            _ => Self::from_language(AsrLanguage::EuropeanMultilingual),
        }
    }
}

pub(crate) fn route_for_detected_language(
    config: &ParapperConfig,
    language_code: &str,
) -> Option<RecognitionRoute> {
    if !config.multilingual_asr_enabled {
        return None;
    }
    let language_code = canonical_language_code(language_code);
    let preferred_route = RecognitionRoute::preferred_for_detected_language_code(&language_code);
    if config.enabled_asr_models.contains(&preferred_route.model)
        && preferred_route
            .model
            .supported_language_codes()
            .contains(&language_code.as_str())
    {
        return Some(preferred_route);
    }

    config
        .enabled_asr_models
        .iter()
        .copied()
        .find(|model| {
            model
                .supported_language_codes()
                .contains(&language_code.as_str())
        })
        .map(RecognitionRoute::from_model)
}

pub(crate) fn language_id_candidate_codes(config: &ParapperConfig) -> Option<Vec<&'static str>> {
    if !config.multilingual_asr_enabled {
        return None;
    }

    let mut candidates = Vec::new();
    for language_code in config
        .enabled_asr_models
        .iter()
        .flat_map(|model| model.supported_language_codes())
    {
        if !candidates.contains(language_code) {
            candidates.push(*language_code);
        }
    }

    (!candidates.is_empty()).then_some(candidates)
}

fn canonical_language_code(language_code: &str) -> String {
    let normalized = language_code.trim().to_ascii_lowercase();
    normalized
        .split_once(['-', '_'])
        .map_or(normalized.as_str(), |(language, _)| language)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{language_id_candidate_codes, route_for_detected_language};
    use crate::config::{AsrModel, ParapperConfig, TurnDetector};

    #[test]
    fn sli_candidates_are_limited_to_enabled_japanese_and_english() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(language_id_candidate_codes(&config), Some(vec!["ja", "en"]));
    }

    #[test]
    fn sli_candidates_are_limited_to_enabled_asr_supported_languages() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
                AsrModel::NemoParakeetTdt0_6BV3Int8,
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            language_id_candidate_codes(&config),
            Some(vec![
                "ja", "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it",
                "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
            ])
        );
    }

    #[test]
    fn sli_route_uses_multilingual_asr_for_its_supported_languages() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![AsrModel::NemoParakeetTdt0_6BV3Int8],
            ..ParapperConfig::default()
        };

        assert_eq!(
            route_for_detected_language(&config, "de-DE").map(|route| route.model),
            Some(AsrModel::NemoParakeetTdt0_6BV3Int8)
        );
        assert_eq!(
            route_for_detected_language(&config, "en").map(|route| route.model),
            Some(AsrModel::NemoParakeetTdt0_6BV3Int8)
        );
        assert!(route_for_detected_language(&config, "ko").is_none());
    }

    #[test]
    fn sli_route_prefers_language_specific_asr_when_multiple_models_support_language() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![
                AsrModel::NemoParakeetTdt0_6BV2Int8,
                AsrModel::NemoParakeetTdt0_6BV3Int8,
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            route_for_detected_language(&config, "en").map(|route| route.model),
            Some(AsrModel::NemoParakeetTdt0_6BV2Int8)
        );
    }

    #[test]
    fn sli_route_uses_enabled_model_that_supports_detected_language() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![AsrModel::ReazonSpeechK2V2],
            ..ParapperConfig::default()
        };

        assert!(route_for_detected_language(&config, "en").is_none());
        assert_eq!(
            route_for_detected_language(&config, "ja").map(|route| route.model),
            Some(AsrModel::ReazonSpeechK2V2)
        );
    }

    #[test]
    fn sli_route_normalizes_language_region_suffixes() {
        let config = ParapperConfig {
            multilingual_asr_enabled: true,
            enabled_asr_models: vec![
                AsrModel::ReazonSpeechK2V2,
                AsrModel::NemoParakeetTdt0_6BV2Int8,
            ],
            ..ParapperConfig::default()
        };

        assert_eq!(
            route_for_detected_language(&config, "ja-JP").map(|route| route.model),
            Some(AsrModel::ReazonSpeechK2V2)
        );
        assert_eq!(
            route_for_detected_language(&config, "en_US").map(|route| route.model),
            Some(AsrModel::NemoParakeetTdt0_6BV2Int8)
        );
    }

    #[test]
    fn sli_candidates_and_routes_are_independent_of_turn_detector_mode() {
        for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
            let config = ParapperConfig {
                multilingual_asr_enabled: true,
                turn_detector,
                enabled_asr_models: vec![
                    AsrModel::ReazonSpeechK2V2,
                    AsrModel::NemoParakeetTdt0_6BV2Int8,
                ],
                ..ParapperConfig::default()
            };

            assert_eq!(
                language_id_candidate_codes(&config),
                Some(vec!["ja", "en"]),
                "turn_detector={turn_detector:?}"
            );
            assert_eq!(
                route_for_detected_language(&config, "ja").map(|route| route.model),
                Some(AsrModel::ReazonSpeechK2V2),
                "turn_detector={turn_detector:?}"
            );
            assert_eq!(
                route_for_detected_language(&config, "en").map(|route| route.model),
                Some(AsrModel::NemoParakeetTdt0_6BV2Int8),
                "turn_detector={turn_detector:?}"
            );
        }
    }
}
