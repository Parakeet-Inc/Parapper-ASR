use parapper_models::asr::{AsrLanguage, AsrModel};

use crate::{
    NamoTurnDetectorModel, SttEngineConfig,
    ports::{LanguageDetectionWarningSink, LanguageDetector},
};

use super::task::AsrTaskKind;
use crate::SegmentCloseReason;

const MIN_LANGUAGE_ID_SAMPLES: usize = parapper_models::SAMPLE_RATE_HZ as usize;
const NORMALIZED_ASR_INPUT_PEAK: f32 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecognitionRoute {
    pub language: AsrLanguage,
    pub model: AsrModel,
    pub turn_detector_model: NamoTurnDetectorModel,
}

pub struct RecognitionRouteSelection {
    pub route: RecognitionRoute,
    pub detected_language: Option<String>,
}

impl RecognitionRoute {
    #[must_use]
    pub fn from_language(language: AsrLanguage) -> Self {
        let model = AsrModel::default_for_language(language);
        let turn_detector_model = NamoTurnDetectorModel::for_asr_language(language);
        Self {
            language,
            model,
            turn_detector_model,
        }
    }

    #[must_use]
    pub fn from_model(model: AsrModel) -> Self {
        let language = model.language();
        let turn_detector_model = NamoTurnDetectorModel::for_asr_language(language);
        Self {
            language,
            model,
            turn_detector_model,
        }
    }

    #[must_use]
    pub fn preferred_for_detected_language_code(language_code: &str) -> Self {
        match language_code {
            "ja" => Self::from_language(AsrLanguage::Japanese),
            "en" => Self::from_language(AsrLanguage::English),
            _ => Self::from_language(AsrLanguage::EuropeanMultilingual),
        }
    }
}

#[must_use]
pub fn configured_split_route(
    config: &SttEngineConfig,
    kind: AsrTaskKind,
) -> Option<RecognitionRoute> {
    match kind {
        AsrTaskKind::InterimDisplay => config.asr.interim_model.map(RecognitionRoute::from_model),
        AsrTaskKind::CompletionCheck | AsrTaskKind::Rerecognition => None,
    }
}

#[must_use]
pub fn language_id_candidate_codes(config: &SttEngineConfig) -> Option<Vec<&'static str>> {
    if !config.asr.multilingual_enabled {
        return None;
    }
    let mut candidates = Vec::new();
    for language_code in config
        .asr
        .enabled_models
        .iter()
        .flat_map(|model| model.supported_language_codes())
    {
        if !candidates.contains(language_code) {
            candidates.push(*language_code);
        }
    }
    (!candidates.is_empty()).then_some(candidates)
}

#[must_use]
pub fn route_for_detected_language(
    config: &SttEngineConfig,
    language_code: &str,
) -> Option<RecognitionRoute> {
    if !config.asr.multilingual_enabled {
        return None;
    }
    let language_code = canonical_language_code(language_code);
    let preferred_route = RecognitionRoute::preferred_for_detected_language_code(&language_code);
    if config.asr.enabled_models.contains(&preferred_route.model)
        && preferred_route
            .model
            .supported_language_codes()
            .contains(&language_code.as_str())
    {
        return Some(preferred_route);
    }
    config
        .asr
        .enabled_models
        .iter()
        .copied()
        .find(|model| {
            model
                .supported_language_codes()
                .contains(&language_code.as_str())
        })
        .map(RecognitionRoute::from_model)
}

fn canonical_language_code(language_code: &str) -> String {
    let normalized = language_code.trim().to_ascii_lowercase();
    normalized
        .split_once(['-', '_'])
        .map_or(normalized.as_str(), |(language, _)| language)
        .to_string()
}

pub struct SliContext<'a> {
    pub config: &'a SttEngineConfig,
    pub warning_sink: Option<&'a dyn LanguageDetectionWarningSink>,
    pub language_id: Option<&'a mut (dyn LanguageDetector + 'a)>,
}

#[must_use]
pub fn detect_recognition_route(
    context: &mut SliContext<'_>,
    current_route: Option<RecognitionRoute>,
    full_audio: &[f32],
) -> RecognitionRouteSelection {
    let default_route = route_without_language_detection(context.config, current_route);
    let default_selection = || RecognitionRouteSelection {
        route: default_route,
        detected_language: None,
    };
    if !context.config.asr.multilingual_enabled || full_audio.len() < MIN_LANGUAGE_ID_SAMPLES {
        return default_selection();
    }
    let Some(language_id) = context.language_id.as_deref_mut() else {
        return default_selection();
    };
    let normalized_audio;
    let language_audio = if context.config.asr.normalize_input_audio {
        normalized_audio = normalize_audio(full_audio);
        normalized_audio.as_slice()
    } else {
        full_audio
    };
    let candidates = language_id_candidate_codes(context.config);
    match language_id.detect(language_audio, candidates.as_deref()) {
        Ok(language_code) if !language_code.is_empty() => {
            let Some(route) = route_for_detected_language(context.config, &language_code) else {
                return default_selection();
            };
            RecognitionRouteSelection {
                route,
                detected_language: Some(language_code),
            }
        }
        Ok(_) => default_selection(),
        Err(error) => {
            if let Some(warning_sink) = context.warning_sink {
                warning_sink.emit_language_detection_warning(&error);
            }
            default_selection()
        }
    }
}

#[must_use]
pub fn route_without_language_detection(
    config: &SttEngineConfig,
    current_route: Option<RecognitionRoute>,
) -> RecognitionRoute {
    current_route.unwrap_or_else(|| RecognitionRoute::from_model(config.asr.model))
}

fn normalize_audio(audio: &[f32]) -> Vec<f32> {
    let peak = audio
        .iter()
        .copied()
        .filter(|sample| sample.is_finite())
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if peak <= f32::EPSILON {
        return audio.to_vec();
    }
    let gain = NORMALIZED_ASR_INPUT_PEAK / peak;
    audio
        .iter()
        .map(|sample| {
            if sample.is_finite() {
                sample * gain
            } else {
                0.0
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
pub struct AsrRouteInput<'a> {
    pub config: &'a SttEngineConfig,
    pub warning_sink: Option<&'a dyn LanguageDetectionWarningSink>,
    pub kind: AsrTaskKind,
    pub close_reason: SegmentCloseReason,
    pub current_route: Option<RecognitionRoute>,
    pub fallback_route: RecognitionRoute,
    pub draft_audio: Option<&'a [f32]>,
    pub request_audio: &'a [f32],
}

#[must_use]
pub fn select_asr_route<'a>(
    input: AsrRouteInput<'a>,
    language_id: Option<&'a mut (dyn LanguageDetector + 'a)>,
) -> RecognitionRouteSelection {
    let split_route = configured_split_route(input.config, input.kind);
    let current_route = usable_current_route(input.current_route, input.kind);
    let default_selection = || RecognitionRouteSelection {
        route: split_route
            .or(current_route)
            .unwrap_or(input.fallback_route),
        detected_language: None,
    };
    if split_route.is_some()
        || input.kind != AsrTaskKind::CompletionCheck
        || input.close_reason != SegmentCloseReason::EndSilenceReached
        || input.warning_sink.is_none()
    {
        return default_selection();
    }
    let full_audio = joined_audio(input.draft_audio, input.request_audio);
    detect_recognition_route(
        &mut SliContext {
            config: input.config,
            warning_sink: input.warning_sink,
            language_id,
        },
        current_route,
        &full_audio,
    )
}

#[derive(Clone, Copy)]
pub struct TurnRouteInput<'a> {
    pub config: &'a SttEngineConfig,
    pub warning_sink: Option<&'a dyn LanguageDetectionWarningSink>,
    pub current_route: Option<RecognitionRoute>,
    pub full_audio: &'a [f32],
}

#[must_use]
pub fn refresh_turn_route<'a>(
    input: TurnRouteInput<'a>,
    language_id: Option<&'a mut (dyn LanguageDetector + 'a)>,
) -> Option<RecognitionRouteSelection> {
    if input.full_audio.is_empty() || input.warning_sink.is_none() {
        return None;
    }
    Some(detect_recognition_route(
        &mut SliContext {
            config: input.config,
            warning_sink: input.warning_sink,
            language_id,
        },
        input.current_route,
        input.full_audio,
    ))
}

fn usable_current_route(
    current_route: Option<RecognitionRoute>,
    kind: AsrTaskKind,
) -> Option<RecognitionRoute> {
    let current_route = current_route?;
    if kind != AsrTaskKind::InterimDisplay && current_route.model.is_interim_only() {
        return None;
    }
    Some(current_route)
}

fn joined_audio(draft_audio: Option<&[f32]>, request_audio: &[f32]) -> Vec<f32> {
    let Some(draft_audio) = draft_audio.filter(|audio| !audio.is_empty()) else {
        return request_audio.to_vec();
    };
    let mut audio = Vec::with_capacity(draft_audio.len() + request_audio.len());
    audio.extend_from_slice(draft_audio);
    audio.extend_from_slice(request_audio);
    audio
}

#[cfg(test)]
mod routing_tests {
    use super::*;

    #[test]
    fn region_suffix_is_normalized_before_route_selection() {
        let mut config = SttEngineConfig::default();
        config.asr.multilingual_enabled = true;
        config.asr.enabled_models = vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ];

        assert_eq!(
            route_for_detected_language(&config, "en_US").map(|route| route.model),
            Some(AsrModel::NemoParakeetTdt0_6BV2Int8)
        );
    }
}
