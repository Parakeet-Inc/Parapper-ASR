use crate::config::ParapperConfig;

pub(crate) fn stt_engine_config(config: &ParapperConfig) -> parapper_stt_engine::SttEngineConfig {
    parapper_stt_engine::SttEngineConfig {
        asr: parapper_stt_engine::SttAsrConfig {
            language: config.asr.language,
            model: config.asr.model,
            interim_model: config.asr.interim_model,
            normalize_input_audio: config.asr.normalize_input_audio,
            multilingual_enabled: config.asr.multilingual_enabled,
            enabled_models: config.asr.enabled_models.clone(),
        },
        segmentation: parapper_stt_engine::SttSegmentationConfig {
            vad_interval_ms: config.segmentation.vad_interval_ms,
            segment_start_speech_ms: config.segmentation.segment_start_speech_ms,
        },
        turn: parapper_stt_engine::SttTurnConfig {
            detector: config.turn.detector,
            interim_result_enabled: config.turn.interim_result_enabled,
            interim_result_silence_ms: config.turn.interim_result_silence_ms,
            check_silence_ms: config.turn.check_silence_ms,
            namo_confidence_threshold: config.turn.namo_confidence_threshold,
            namo_context_max_tokens: config.turn.namo_context_max_tokens,
            rerecognize_full_on_complete: config.turn.rerecognize_full_on_complete,
        },
    }
}

pub(crate) fn stt_runtime_parameters(
    config: &ParapperConfig,
) -> parapper_stt_engine::SttRuntimeParameters {
    stt_engine_config(config).runtime_parameters()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AsrModel, TurnDetector};

    #[test]
    fn persisted_app_config_maps_only_stt_fields_into_engine_config() {
        let mut app = ParapperConfig::default();
        app.asr.model = AsrModel::NemoParakeetTdt0_6BV2Int8;
        app.turn.detector = TurnDetector::Namo;
        app.turn.check_silence_ms = 777;
        app.translation.enabled = true;
        app.models.dir = Some("host-owned-model-directory".to_string());

        let engine = stt_engine_config(&app);

        assert_eq!(engine.asr.model, AsrModel::NemoParakeetTdt0_6BV2Int8);
        assert_eq!(engine.turn.detector, TurnDetector::Namo);
        assert_eq!(engine.turn.check_silence_ms, 777);
    }

    #[test]
    fn runtime_parameter_mapping_excludes_session_fixed_stt_fields() {
        let app = ParapperConfig::default();
        let mut changed = app.clone();
        changed.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        changed.turn.detector = TurnDetector::Namo;

        assert_eq!(
            stt_runtime_parameters(&changed),
            stt_runtime_parameters(&app)
        );
    }
}
