use std::{collections::HashMap, sync::OnceLock};

use anyhow::Result;
use parapper_models::td::{
    morph::{JapaneseMorphAnalyzer, candidates_for_transcript},
    namo::{NamoTokenizerKind, NamoTurnDecision, NamoTurnDetectorEngine},
};
use parapper_stt_engine::{
    ports::TranscriptBoundaryDetector, transcription::route::RecognitionRoute, turn::TurnDecision,
};
use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    model::{NamoTurnDetectorModel, models_root, namo_turn_detector_model_dir_from_root},
    recognition::events::{MissingModelKind, emit_missing_model_event},
};

#[cfg(test)]
pub(crate) use parapper_stt_engine::ports::TurnDecisionRunner;

#[derive(Default)]
pub(crate) struct NamoTurnDetectorCache {
    engines: HashMap<NamoTurnDetectorModel, Box<dyn CachedNamoTurnDetector>>,
}

impl NamoTurnDetectorCache {
    pub(crate) fn preload_required(
        &mut self,
        handle: &AppHandle,
        config: &ParapperConfig,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for model in namo_turn_detector_models_for_config(config) {
            if let Err(err) = self.ensure(handle, model) {
                errors.push(format!("Failed to preload {model:?} turn detector: {err}"));
            }
        }
        errors
    }

    fn ensure(&mut self, handle: &AppHandle, model: NamoTurnDetectorModel) -> Result<()> {
        if self.engines.contains_key(&model) {
            return Ok(());
        }
        let root = models_root(handle)?;
        let model_dir = namo_turn_detector_model_dir_from_root(&root, model);
        let tokenizer_kind = match model {
            NamoTurnDetectorModel::Japanese => NamoTokenizerKind::Character,
            NamoTurnDetectorModel::English | NamoTurnDetectorModel::Multilingual => {
                NamoTokenizerKind::TokenizerJson
            }
        };
        let engine = NamoTurnDetectorEngine::new(&model_dir, tokenizer_kind)?;
        self.engines.insert(model, Box::new(engine));
        Ok(())
    }

    pub(crate) fn decide(
        &mut self,
        model: NamoTurnDetectorModel,
        text: &str,
        max_context_tokens: u32,
    ) -> Result<NamoTurnDecision> {
        let engine = self
            .engines
            .get_mut(&model)
            .ok_or_else(|| anyhow::anyhow!("{model:?} turn detector was not preloaded"))?;
        engine.decide(text, max_context_tokens)
    }

    #[cfg(test)]
    pub(crate) fn insert_engine_for_test(
        &mut self,
        model: NamoTurnDetectorModel,
        engine: Box<dyn CachedNamoTurnDetector>,
    ) {
        self.engines.insert(model, engine);
    }
}

pub(crate) trait CachedNamoTurnDetector: Send {
    fn decide(&mut self, text: &str, max_context_tokens: u32) -> Result<NamoTurnDecision>;
}

impl CachedNamoTurnDetector for NamoTurnDetectorEngine {
    fn decide(&mut self, text: &str, max_context_tokens: u32) -> Result<NamoTurnDecision> {
        NamoTurnDetectorEngine::decide(self, text, max_context_tokens)
    }
}

fn namo_turn_detector_models_for_config(config: &ParapperConfig) -> Vec<NamoTurnDetectorModel> {
    config
        .required_namo_turn_detector_languages()
        .into_iter()
        .map(NamoTurnDetectorModel::for_asr_language)
        .collect()
}

#[cfg(test)]
pub(crate) struct NoopTurnDecisionRunner;

#[cfg(test)]
impl parapper_stt_engine::ports::TurnDecisionRunner for NoopTurnDecisionRunner {
    fn decide(
        &mut self,
        _route: RecognitionRoute,
        _text: &str,
        _max_context_tokens: u32,
    ) -> Result<TurnDecision> {
        Ok(TurnDecision {
            is_end_of_turn: false,
            confidence: 0.0,
        })
    }
}

pub(crate) struct EngineTurnDecisionRunner {
    turn_detectors: NamoTurnDetectorCache,
}

impl EngineTurnDecisionRunner {
    pub(crate) fn new(handle: &AppHandle, config: &ParapperConfig) -> Self {
        let mut turn_detectors = NamoTurnDetectorCache::default();
        for reason in turn_detectors.preload_required(handle, config) {
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::TurnDetector, reason);
        }
        Self { turn_detectors }
    }

    #[cfg(test)]
    fn from_turn_detectors_for_test(turn_detectors: NamoTurnDetectorCache) -> Self {
        Self { turn_detectors }
    }
}

impl parapper_stt_engine::ports::TurnDecisionRunner for EngineTurnDecisionRunner {
    fn decide(
        &mut self,
        route: RecognitionRoute,
        text: &str,
        max_context_tokens: u32,
    ) -> Result<TurnDecision> {
        self.turn_detectors
            .decide(route.turn_detector_model, text, max_context_tokens)
            .map(|decision| TurnDecision {
                is_end_of_turn: decision.is_end_of_turn,
                confidence: decision.confidence,
            })
    }
}

pub(crate) struct AppTranscriptBoundaryDetector {
    japanese_morph: Option<JapaneseMorphAnalyzer>,
}

impl AppTranscriptBoundaryDetector {
    pub(crate) fn new(_handle: &AppHandle, config: &ParapperConfig) -> Self {
        let japanese_morph = config
            .requires_japanese_morph_analyzer()
            .then(load_japanese_morph_analyzer)
            .flatten();
        Self { japanese_morph }
    }

    #[cfg(test)]
    pub(crate) const fn without_morph() -> Self {
        Self {
            japanese_morph: None,
        }
    }
}

impl TranscriptBoundaryDetector for AppTranscriptBoundaryDetector {
    fn candidates_for_transcript(
        &self,
        language: parapper_models::asr::AsrLanguage,
        transcript: &parapper_models::asr::AsrTranscript,
        audio: &[f32],
        vad_results: &[parapper_models::vad::VadResult],
    ) -> Vec<parapper_models::td::TurnBoundaryCandidate> {
        candidates_for_transcript(
            language,
            transcript,
            audio,
            vad_results,
            self.japanese_morph.as_ref(),
        )
    }
}

const EMBEDDED_JAPANESE_MORPH_DICTIONARY: &[u8] =
    include_bytes!("../../resources/morph/system.dic.zst");
static JAPANESE_MORPH_ANALYZER: OnceLock<Result<JapaneseMorphAnalyzer, String>> = OnceLock::new();

pub(crate) fn load_japanese_morph_analyzer() -> Option<JapaneseMorphAnalyzer> {
    match JAPANESE_MORPH_ANALYZER.get_or_init(|| {
        let decoder = zstd::Decoder::new(EMBEDDED_JAPANESE_MORPH_DICTIONARY).map_err(|error| {
            format!("Failed to decode embedded Japanese morph dictionary: {error}")
        })?;
        JapaneseMorphAnalyzer::from_dictionary_reader(decoder)
            .map_err(|error| format!("Failed to load embedded Japanese morph dictionary: {error}"))
    }) {
        Ok(analyzer) => Some(analyzer.clone()),
        Err(error) => {
            log::warn!("{error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::{
        config::{AsrLanguage, AsrModel, TurnDetector},
        model::NamoTurnDetectorModel,
    };

    #[test]
    fn embedded_morph_dictionary_loads_and_finds_a_japanese_terminal_boundary() {
        let analyzer = load_japanese_morph_analyzer()
            .expect("the built-in compressed UniDic dictionary should load");
        let text = "東京駅へ行きます";
        let transcript = parapper_models::asr::AsrTranscript::from_parts(
            text,
            vec![text.to_owned()],
            Some(&[0.0]),
            Some(&[1.0]),
        );

        let candidates = candidates_for_transcript(
            AsrLanguage::Japanese,
            &transcript,
            &vec![0.0; 16_000],
            &[],
            Some(&analyzer),
        );

        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.char_end == text.chars().count()),
            "the embedded production dictionary must recognize the terminal predicate: {candidates:?}"
        );
    }

    #[test]
    fn engine_turn_decision_runner_delegates_route_text_and_context_to_cached_detector() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut turn_detectors = NamoTurnDetectorCache::default();
        turn_detectors.insert_engine_for_test(
            NamoTurnDetectorModel::Japanese,
            Box::new(RecordingTurnDetector {
                captured: captured.clone(),
                decision: NamoTurnDecision {
                    is_end_of_turn: true,
                    confidence: 0.91,
                },
            }),
        );
        let mut runner = EngineTurnDecisionRunner::from_turn_detectors_for_test(turn_detectors);

        let decision = runner
            .decide(
                RecognitionRoute::from_language(AsrLanguage::Japanese),
                "ここで終わります",
                128,
            )
            .expect("cached detector should return its scripted decision");

        assert_eq!(
            decision,
            TurnDecision {
                is_end_of_turn: true,
                confidence: 0.91,
            }
        );
        assert_eq!(
            *captured
                .lock()
                .expect("captured turn detector calls should be readable"),
            vec![("ここで終わります".to_string(), 128)]
        );
    }

    #[test]
    fn engine_turn_decision_runner_returns_error_when_route_detector_was_not_preloaded() {
        let mut runner = EngineTurnDecisionRunner::from_turn_detectors_for_test(
            NamoTurnDetectorCache::default(),
        );

        let err = runner
            .decide(
                RecognitionRoute::from_language(AsrLanguage::Japanese),
                "未ロード",
                64,
            )
            .expect_err("missing cached detector should be reported as a decision error");

        assert!(
            err.to_string().contains("turn detector was not preloaded"),
            "unexpected missing-detector error: {err}"
        );
    }

    #[test]
    fn required_models_follow_multilingual_and_turn_detector_matrix() {
        for turn_detector in [
            TurnDetector::Simple,
            TurnDetector::Namo,
            TurnDetector::Morph,
        ] {
            for multilingual_asr_enabled in [false, true] {
                let config = parapper_config! {
                    multilingual_asr_enabled: multilingual_asr_enabled,
                    turn_detector: turn_detector,
                    enabled_asr_models: vec![
                        AsrModel::ReazonSpeechK2V2,
                        AsrModel::NemoParakeetTdt0_6BV2Int8,
                    ],
                    ..ParapperConfig::default()
                };
                let expected = if config.uses_namo_turn_detector() && multilingual_asr_enabled {
                    vec![
                        NamoTurnDetectorModel::Japanese,
                        NamoTurnDetectorModel::English,
                    ]
                } else if config.uses_namo_turn_detector() {
                    vec![NamoTurnDetectorModel::Japanese]
                } else {
                    Vec::new()
                };
                assert_eq!(
                    namo_turn_detector_models_for_config(&config),
                    expected,
                    "turn_detector={turn_detector:?}, multilingual={multilingual_asr_enabled}"
                );
            }
        }
    }

    #[test]
    fn non_multilingual_namo_model_follows_selected_asr_model() {
        let config = parapper_config! {
            asr_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            turn_detector: TurnDetector::Namo,
            multilingual_asr_enabled: false,
            ..ParapperConfig::default()
        };

        assert_eq!(
            namo_turn_detector_models_for_config(&config),
            vec![NamoTurnDetectorModel::English]
        );
    }

    struct RecordingTurnDetector {
        captured: Arc<Mutex<Vec<(String, u32)>>>,
        decision: NamoTurnDecision,
    }

    impl CachedNamoTurnDetector for RecordingTurnDetector {
        fn decide(&mut self, text: &str, max_context_tokens: u32) -> Result<NamoTurnDecision> {
            self.captured
                .lock()
                .expect("captured turn detector calls should be writable")
                .push((text.to_string(), max_context_tokens));
            Ok(self.decision)
        }
    }
}
