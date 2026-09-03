use anyhow::Result;
use parapper_models::asr::{AsrEngine, backend::HotwordEntry};
use parapper_stt_engine::AsrModelRegistry;
use tauri::AppHandle;

use crate::{
    config::{AsrMode, AsrModel, AsrPrecision, ParapperConfig},
    model::{asr_model_dir_for, language_id_model_dir},
};

pub(crate) fn load_asr_models(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> (AsrModelRegistry, Vec<String>) {
    let mut registry = AsrModelRegistry::default();
    let mut errors = Vec::new();
    for model in config.required_asr_models() {
        let load = (|| {
            let model_dir = asr_model_dir_for(handle, config, model)?;
            let precision = config.asr_precision_for(model);
            let hotwords = hotword_entries_for(config, model);
            build_asr_engine(
                &model_dir,
                model,
                precision,
                config.effective_asr_num_threads(),
                decoding_strategy_for(config, model),
                &hotwords,
            )
        })();
        match load {
            Ok(engine) => {
                if let Err(error) = registry.insert(model, engine) {
                    errors.push(format!("Failed to register {model:?} ASR engine: {error}"));
                }
            }
            Err(error) => errors.push(format!("Failed to preload {model:?} ASR engine: {error}")),
        }
    }
    (registry, errors)
}

#[cfg(any(not(test), feature = "real-asr-tests"))]
fn build_asr_engine(
    model_dir: &std::path::Path,
    model: AsrModel,
    precision: AsrPrecision,
    num_threads: i32,
    decoding: parapper_models::asr::backend::AsrDecodingStrategy,
    hotwords: &[HotwordEntry],
) -> Result<Box<dyn AsrEngine>> {
    parapper_models::asr::backend::build_engine_with_decoding_and_hotwords(
        model_dir,
        model,
        precision,
        num_threads,
        decoding,
        hotwords,
    )
}

#[cfg(all(test, not(feature = "real-asr-tests")))]
fn build_asr_engine(
    _model_dir: &std::path::Path,
    _model: AsrModel,
    _precision: AsrPrecision,
    _num_threads: i32,
    _decoding: parapper_models::asr::backend::AsrDecodingStrategy,
    _hotwords: &[HotwordEntry],
) -> Result<Box<dyn AsrEngine>> {
    Err(anyhow::anyhow!(
        "native ASR models are unavailable in unit tests"
    ))
}

fn hotword_entries_for(config: &ParapperConfig, model: AsrModel) -> Vec<HotwordEntry> {
    if model != config.asr.model || !config.hotwords_enabled() {
        return Vec::new();
    }
    config
        .asr
        .hotwords
        .iter()
        .map(|entry| HotwordEntry {
            surface: entry.surface.clone(),
            readings: entry.readings.clone(),
            phrase_score: entry.score,
        })
        .collect()
}

fn decoding_strategy_for(
    config: &ParapperConfig,
    model: AsrModel,
) -> parapper_models::asr::backend::AsrDecodingStrategy {
    use parapper_models::asr::backend::{
        AsrDecodingStrategy, PARAKEET_JA_PRODUCTION_BEAM_SIZE,
        PARAKEET_JA_PRODUCTION_CTC_GATE_THRESHOLD, ParakeetJaTdtDagConfig,
        REAZON_PRODUCTION_BEAM_SIZE, REAZON_PRODUCTION_RETAINED_CANDIDATES,
    };

    if model != config.asr.model {
        return AsrDecodingStrategy::Greedy;
    }

    match (model, config.asr.mode) {
        (AsrModel::ReazonSpeechK2V2, AsrMode::Accurate) => {
            AsrDecodingStrategy::ReazonOneSpliceRerank {
                beam_size: REAZON_PRODUCTION_BEAM_SIZE,
                retained_candidates: REAZON_PRODUCTION_RETAINED_CANDIDATES,
            }
        }
        (AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, AsrMode::Fast) => {
            AsrDecodingStrategy::ParakeetJaCtcGreedy
        }
        (AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, AsrMode::Accurate) => {
            AsrDecodingStrategy::ParakeetJaTdtVariableDag(ParakeetJaTdtDagConfig {
                beam_size: PARAKEET_JA_PRODUCTION_BEAM_SIZE,
                ctc_gate_threshold: config
                    .hotwords_enabled()
                    .then_some(PARAKEET_JA_PRODUCTION_CTC_GATE_THRESHOLD),
            })
        }
        _ => AsrDecodingStrategy::Greedy,
    }
}

pub(crate) fn build_language_id_engine(
    handle: &AppHandle,
    config: &ParapperConfig,
) -> Result<Option<parapper_models::asr::SpokenLanguageIdentificationEngine>> {
    if !config.asr.multilingual_enabled {
        return Ok(None);
    }
    let model_dir = language_id_model_dir(handle)?;
    parapper_models::asr::SpokenLanguageIdentificationEngine::new(
        &model_dir,
        config.effective_asr_num_threads(),
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use parapper_models::asr::backend::AsrDecodingStrategy;

    use super::{decoding_strategy_for, hotword_entries_for};
    use crate::config::{AsrHotword, AsrMode, AsrModel, ParapperConfig};

    #[test]
    fn fixed_mode_maps_each_primary_model_to_its_product_decoder() {
        let mut config = ParapperConfig::default();
        let parakeet = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;

        assert_eq!(
            [
                decoding_strategy_for(&config, AsrModel::ReazonSpeechK2V2),
                decoding_strategy_for(&config, parakeet),
            ],
            [AsrDecodingStrategy::Greedy, AsrDecodingStrategy::Greedy]
        );

        config.asr.mode = AsrMode::Accurate;

        assert_eq!(
            [
                decoding_strategy_for(&config, AsrModel::ReazonSpeechK2V2),
                decoding_strategy_for(&config, parakeet),
            ],
            [
                AsrDecodingStrategy::ReazonOneSpliceRerank {
                    beam_size: 4,
                    retained_candidates: 2,
                },
                AsrDecodingStrategy::Greedy,
            ]
        );

        config.asr.model = parakeet;
        assert_eq!(
            decoding_strategy_for(&config, AsrModel::ReazonSpeechK2V2),
            AsrDecodingStrategy::Greedy,
            "a hidden Reazon engine must not retain beam search after selecting another primary model"
        );
        assert_eq!(
            decoding_strategy_for(&config, parakeet),
            AsrDecodingStrategy::ParakeetJaTdtVariableDag(
                parapper_models::asr::backend::ParakeetJaTdtDagConfig {
                    beam_size: 2,
                    ctc_gate_threshold: None,
                }
            )
        );

        config.asr.mode = AsrMode::Fast;
        assert_eq!(
            decoding_strategy_for(&config, parakeet),
            AsrDecodingStrategy::ParakeetJaCtcGreedy
        );
    }

    #[test]
    fn hotwords_are_passed_only_to_selected_accurate_engine_and_enable_parakeet_gate() {
        let mut config = ParapperConfig::default();
        config.asr.hotwords = vec![AsrHotword {
            surface: "斎藤".to_string(),
            readings: vec!["さいとう".to_string()],
            score: Some(1.25),
        }];

        assert!(
            hotword_entries_for(&config, AsrModel::ReazonSpeechK2V2).is_empty(),
            "a saved list must not change greedy recognition"
        );

        config.asr.mode = AsrMode::Accurate;
        config.asr.hotwords_enabled = true;
        let entries = hotword_entries_for(&config, AsrModel::ReazonSpeechK2V2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].surface, "斎藤");
        assert_eq!(entries[0].readings, ["さいとう"]);
        assert_eq!(entries[0].phrase_score, Some(1.25));

        assert!(
            hotword_entries_for(&config, AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8).is_empty(),
            "the same registry load must not pass Reazon hotwords to another family"
        );

        config.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        assert!(
            hotword_entries_for(&config, AsrModel::ReazonSpeechK2V2).is_empty(),
            "a hidden Reazon engine must not receive primary-model hotwords"
        );
        assert_eq!(
            hotword_entries_for(&config, AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8).len(),
            1
        );
        assert_eq!(
            decoding_strategy_for(&config, AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8),
            AsrDecodingStrategy::ParakeetJaTdtVariableDag(
                parapper_models::asr::backend::ParakeetJaTdtDagConfig {
                    beam_size: 2,
                    ctc_gate_threshold: Some(-5.0),
                }
            )
        );
    }
}
