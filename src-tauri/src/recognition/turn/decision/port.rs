use anyhow::Result;
use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        control::{
            engine_cache::NamoTurnDetectorCache, events::MissingModelKind,
            runtime_event::emit_missing_model_event,
        },
        transcription::route::RecognitionRoute,
        turn::decision::TurnDecision,
    },
};

pub(crate) trait TurnDecisionRunner {
    fn update_config(&mut self, _config: &ParapperConfig) {}
    fn decide(
        &mut self,
        route: RecognitionRoute,
        text: &str,
        max_context_tokens: u32,
    ) -> Result<TurnDecision>;
}

#[cfg(test)]
pub(crate) struct NoopTurnDecisionRunner;

#[cfg(test)]
impl TurnDecisionRunner for NoopTurnDecisionRunner {
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
    handle: Option<AppHandle>,
    turn_detectors: NamoTurnDetectorCache,
}

impl EngineTurnDecisionRunner {
    pub(crate) fn new(handle: &AppHandle, config: &ParapperConfig) -> Self {
        let mut turn_detectors = NamoTurnDetectorCache::default();
        for reason in turn_detectors.preload_required(handle, config) {
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::TurnDetector, reason);
        }
        Self {
            handle: Some(handle.clone()),
            turn_detectors,
        }
    }

    #[cfg(test)]
    fn from_turn_detectors_for_test(turn_detectors: NamoTurnDetectorCache) -> Self {
        Self {
            handle: None,
            turn_detectors,
        }
    }
}

impl TurnDecisionRunner for EngineTurnDecisionRunner {
    fn update_config(&mut self, config: &ParapperConfig) {
        let Some(handle) = self.handle.as_ref() else {
            return;
        };
        for reason in self.turn_detectors.preload_required(handle, config) {
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::TurnDetector, reason);
        }
    }

    fn decide(
        &mut self,
        route: RecognitionRoute,
        text: &str,
        max_context_tokens: u32,
    ) -> Result<TurnDecision> {
        self.turn_detectors
            .decide(route.turn_detector_model, text, max_context_tokens)
            .map(TurnDecision::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::{
        config::AsrLanguage, model::NamoTurnDetectorModel,
        recognition::control::engine_cache::CachedNamoTurnDetector,
    };

    #[test]
    fn engine_turn_decision_runner_delegates_route_text_and_context_to_cached_detector() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut turn_detectors = NamoTurnDetectorCache::default();
        turn_detectors.insert_engine_for_test(
            NamoTurnDetectorModel::Japanese,
            Box::new(RecordingTurnDetector {
                captured: captured.clone(),
                decision: crate::recognition::turn::decision::engine::NamoTurnDecision {
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

    struct RecordingTurnDetector {
        captured: Arc<Mutex<Vec<(String, u32)>>>,
        decision: crate::recognition::turn::decision::engine::NamoTurnDecision,
    }

    impl CachedNamoTurnDetector for RecordingTurnDetector {
        fn decide(
            &mut self,
            text: &str,
            max_context_tokens: u32,
        ) -> Result<crate::recognition::turn::decision::engine::NamoTurnDecision> {
            self.captured
                .lock()
                .expect("captured turn detector calls should be writable")
                .push((text.to_string(), max_context_tokens));
            Ok(self.decision)
        }
    }
}
