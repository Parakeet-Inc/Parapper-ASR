use parapper_models::asr::{AsrLanguage, AsrModel};

use crate::{RecognitionConfig, transcription::planner::PlannerConfig, turn::TurnDetector};

/// STT orchestration settings independent from an application's persisted config shape.
///
/// Hosts translate their own configuration into this value. File layout, model paths,
/// transport settings, translation, and speech synthesis deliberately do not cross the
/// engine boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct SttEngineConfig {
    pub asr: SttAsrConfig,
    pub segmentation: SttSegmentationConfig,
    pub turn: SttTurnConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttAsrConfig {
    pub language: AsrLanguage,
    pub model: AsrModel,
    pub interim_model: Option<AsrModel>,
    pub normalize_input_audio: bool,
    pub multilingual_enabled: bool,
    pub enabled_models: Vec<AsrModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttSegmentationConfig {
    pub vad_interval_ms: u32,
    pub segment_start_speech_ms: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttTurnConfig {
    pub detector: TurnDetector,
    pub interim_result_enabled: bool,
    pub interim_result_silence_ms: u32,
    pub check_silence_ms: u32,
    pub namo_confidence_threshold: f32,
    pub namo_context_max_tokens: u32,
    pub rerecognize_full_on_complete: bool,
}

/// Parameters that may change while an STT session is running.
///
/// Session constructors, model routing, and pipeline topology are deliberately
/// absent so callers cannot request a partial runtime rebuild through this API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SttRuntimeParameters {
    pub normalize_input_audio: bool,
    pub segment_start_speech_ms: u32,
    pub interim_result_silence_ms: u32,
    pub check_silence_ms: u32,
    pub namo_confidence_threshold: f32,
    pub namo_context_max_tokens: u32,
    pub rerecognize_full_on_complete: bool,
}

impl Default for SttEngineConfig {
    fn default() -> Self {
        Self {
            asr: SttAsrConfig {
                language: AsrLanguage::Japanese,
                model: AsrModel::ReazonSpeechK2V2,
                interim_model: None,
                normalize_input_audio: true,
                multilingual_enabled: false,
                enabled_models: vec![AsrModel::ReazonSpeechK2V2],
            },
            segmentation: SttSegmentationConfig {
                vad_interval_ms: 32,
                segment_start_speech_ms: 96,
            },
            turn: SttTurnConfig {
                detector: TurnDetector::Simple,
                interim_result_enabled: true,
                interim_result_silence_ms: 96,
                check_silence_ms: 320,
                namo_confidence_threshold: 0.8,
                namo_context_max_tokens: 256,
                rerecognize_full_on_complete: false,
            },
        }
    }
}

impl SttEngineConfig {
    #[must_use]
    pub const fn runtime_parameters(&self) -> SttRuntimeParameters {
        SttRuntimeParameters {
            normalize_input_audio: self.asr.normalize_input_audio,
            segment_start_speech_ms: self.segmentation.segment_start_speech_ms,
            interim_result_silence_ms: self.turn.interim_result_silence_ms,
            check_silence_ms: self.turn.check_silence_ms,
            namo_confidence_threshold: self.turn.namo_confidence_threshold,
            namo_context_max_tokens: self.turn.namo_context_max_tokens,
            rerecognize_full_on_complete: self.turn.rerecognize_full_on_complete,
        }
    }

    pub(crate) fn apply_runtime_parameters(&mut self, parameters: SttRuntimeParameters) {
        self.asr.normalize_input_audio = parameters.normalize_input_audio;
        self.segmentation.segment_start_speech_ms = parameters.segment_start_speech_ms;
        self.turn.interim_result_silence_ms = parameters.interim_result_silence_ms;
        self.turn.check_silence_ms = parameters.check_silence_ms;
        self.turn.namo_confidence_threshold = parameters.namo_confidence_threshold;
        self.turn.namo_context_max_tokens = parameters.namo_context_max_tokens;
        self.turn.rerecognize_full_on_complete = parameters.rerecognize_full_on_complete;
    }

    #[must_use]
    pub fn recognition_config(&self) -> RecognitionConfig {
        RecognitionConfig {
            vad_interval_ms: self.segmentation.vad_interval_ms,
            segment_start_speech_ms: self.segmentation.segment_start_speech_ms,
            interim_result_enabled: self.turn.interim_result_enabled,
            interim_result_silence_ms: self.turn.interim_result_silence_ms,
            turn_check_silence_ms: self.turn.check_silence_ms,
            streaming_interim_asr_enabled: self.streaming_interim_asr_enabled(),
        }
    }

    #[must_use]
    pub const fn planner_config(&self) -> PlannerConfig {
        PlannerConfig {
            can_connect_interim_after_completion: self
                .turn
                .detector
                .can_connect_interim_after_completion(),
            vad_interval_ms: self.segmentation.vad_interval_ms,
        }
    }

    #[must_use]
    pub fn streaming_interim_asr_enabled(&self) -> bool {
        self.turn.interim_result_enabled
            && self
                .asr
                .interim_model
                .unwrap_or(self.asr.model)
                .is_nemotron()
    }

    #[must_use]
    pub const fn uses_namo_turn_detector(&self) -> bool {
        self.turn.detector.uses_namo_model()
    }

    #[must_use]
    pub const fn confirms_normal_end_with_namo(&self) -> bool {
        self.turn.detector.confirms_normal_end_with_namo()
    }

    #[must_use]
    pub const fn uses_deferred_turn_completion(&self) -> bool {
        self.turn.detector.uses_deferred_turn_completion()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_config_contains_only_stt_runtime_decisions() {
        let config = SttEngineConfig::default();

        assert_eq!(config.recognition_config().vad_interval_ms, 32);
        assert_eq!(config.planner_config().vad_interval_ms, 32);
        assert!(!config.streaming_interim_asr_enabled());
    }

    #[test]
    fn runtime_parameters_exclude_session_and_route_settings() {
        let config = SttEngineConfig::default();
        let mut updated = config.clone();
        updated.asr.model = AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8;
        updated.asr.multilingual_enabled = true;
        updated.turn.detector = TurnDetector::Namo;
        updated.turn.interim_result_enabled = false;

        assert_eq!(updated.runtime_parameters(), config.runtime_parameters());
    }
}
