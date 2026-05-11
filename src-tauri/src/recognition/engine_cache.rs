use std::collections::HashMap;

use anyhow::Result;
use tauri::AppHandle;

use super::route::RecognitionRoute;
use crate::{
    config::{AsrModel, ParapperConfig},
    model::{
        NamoTurnDetectorModel, asr_model_dir_for, language_id_model_dir, models_root,
        namo_turn_detector_model_dir_from_root,
    },
    recognition::engines::{
        AsrEngine, NamoTokenizerKind, NamoTurnDecision, NamoTurnDetectorEngine,
        SherpaOnnxAsrEngine, SpokenLanguageIdentificationEngine,
    },
};

#[derive(Default)]
pub(crate) struct AsrEngineCache {
    engines: HashMap<AsrModel, Box<dyn AsrEngine>>,
}

impl AsrEngineCache {
    pub(crate) fn preload_required(
        &mut self,
        handle: &AppHandle,
        config: &ParapperConfig,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for model in config.required_asr_models() {
            if let Err(err) = self.ensure(handle, config, model) {
                errors.push(format!("Failed to preload {model:?} ASR engine: {err}"));
            }
        }
        errors
    }

    fn ensure(
        &mut self,
        handle: &AppHandle,
        config: &ParapperConfig,
        model: AsrModel,
    ) -> Result<()> {
        if self.engines.contains_key(&model) {
            return Ok(());
        }
        let model_dir = asr_model_dir_for(handle, config, model)?;
        let precision = config.asr_precision_for(model);
        let engine =
            SherpaOnnxAsrEngine::new(&model_dir, model, precision, config.asr_num_threads)?;
        self.engines.insert(model, Box::new(engine));
        Ok(())
    }

    pub(crate) fn transcribe(
        &mut self,
        route: RecognitionRoute,
        samples: &[f32],
    ) -> Result<String> {
        let engine = self
            .engines
            .get_mut(&route.model)
            .ok_or_else(|| anyhow::anyhow!("{:?} ASR engine was not preloaded", route.model))?;
        engine.transcribe(samples)
    }

    #[cfg(test)]
    pub(crate) fn insert_engine_for_test(&mut self, model: AsrModel, engine: Box<dyn AsrEngine>) {
        self.engines.insert(model, engine);
    }
}

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
        let model_dir = namo_turn_detector_model_dir_for(handle, model)?;
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

pub(crate) fn build_language_id_engine(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<Option<SpokenLanguageIdentificationEngine>> {
    if !config.multilingual_asr_enabled {
        return Ok(None);
    }
    let model_dir = language_id_model_dir(handle)?;
    SpokenLanguageIdentificationEngine::new(&model_dir, config.asr_num_threads.max(1)).map(Some)
}

pub(crate) fn namo_turn_detector_models_for_config(
    config: &ParapperConfig,
) -> Vec<NamoTurnDetectorModel> {
    config
        .required_namo_turn_detector_languages()
        .into_iter()
        .map(NamoTurnDetectorModel::for_asr_language)
        .collect()
}

fn namo_turn_detector_model_dir_for(
    handle: &AppHandle,
    model: NamoTurnDetectorModel,
) -> Result<std::path::PathBuf> {
    let root = models_root(handle)?;
    Ok(namo_turn_detector_model_dir_from_root(&root, model))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Result;

    use super::{AsrEngineCache, namo_turn_detector_models_for_config};
    use crate::{
        config::{AsrLanguage, AsrModel, ParapperConfig, TurnDetector},
        model::NamoTurnDetectorModel,
        recognition::{engines::AsrEngine, route::RecognitionRoute},
    };

    #[test]
    fn required_models_follow_multilingual_and_turn_detector_matrix() {
        for turn_detector in [TurnDetector::Simple, TurnDetector::Namo] {
            for multilingual_asr_enabled in [false, true] {
                let config = ParapperConfig {
                    multilingual_asr_enabled,
                    turn_detector,
                    enabled_asr_models: vec![
                        AsrModel::ReazonSpeechK2V2,
                        AsrModel::NemoParakeetTdt0_6BV2Int8,
                    ],
                    ..ParapperConfig::default()
                };

                let expected_asr = if multilingual_asr_enabled {
                    vec![
                        AsrModel::ReazonSpeechK2V2,
                        AsrModel::NemoParakeetTdt0_6BV2Int8,
                    ]
                } else {
                    vec![config.asr_model]
                };
                assert_eq!(config.required_asr_models(), expected_asr);

                let expected_turn_detectors =
                    if config.uses_namo_turn_detector() && multilingual_asr_enabled {
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
                    expected_turn_detectors,
                    "turn_detector={turn_detector:?}, multilingual={multilingual_asr_enabled}"
                );
            }
        }
    }

    #[test]
    fn non_multilingual_namo_model_follows_selected_asr_model() {
        let config = ParapperConfig {
            asr_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            turn_detector: TurnDetector::Namo,
            multilingual_asr_enabled: false,
            ..ParapperConfig::default()
        };

        assert_eq!(config.required_asr_models(), vec![config.asr_model]);
        assert_eq!(
            namo_turn_detector_models_for_config(&config),
            vec![NamoTurnDetectorModel::English]
        );
    }

    #[test]
    fn mock_asr_engine_transcribes_supplied_audio() {
        let captured_audio = Arc::new(Mutex::new(Vec::new()));
        let mut cache = AsrEngineCache::default();
        cache.insert_engine_for_test(
            AsrModel::ReazonSpeechK2V2,
            Box::new(MockAsrEngine {
                text: "モック文字起こし".to_string(),
                captured_audio: captured_audio.clone(),
            }),
        );
        let audio = vec![0.0, 0.25, -0.5, 0.75];

        let text = cache
            .transcribe(
                RecognitionRoute::from_language(AsrLanguage::Japanese),
                &audio,
            )
            .unwrap();

        assert_eq!(text, "モック文字起こし");
        assert_eq!(*captured_audio.lock().unwrap(), audio);
    }

    struct MockAsrEngine {
        text: String,
        captured_audio: Arc<Mutex<Vec<f32>>>,
    }

    impl AsrEngine for MockAsrEngine {
        fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
            *self.captured_audio.lock().unwrap() = samples.to_vec();
            Ok(self.text.clone())
        }
    }
}
