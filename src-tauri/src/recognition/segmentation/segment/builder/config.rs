use crate::config::ParapperConfig;

const MAX_PHRASE_MILLIS: u32 = 25_000;

#[derive(Debug, Clone)]
pub(super) struct SegmentBuilderConfig {
    pub(super) segment_start_threshold: u32,
    pub(super) interim_result_threshold: Option<u32>,
    pub(super) turn_check_threshold: u32,
    pub(super) max_chunks: u32,
    pub(super) pre_speech_max_chunks: usize,
}

impl SegmentBuilderConfig {
    pub(super) fn from_config(config: &ParapperConfig) -> Self {
        let streaming_interim_asr_enabled = config.turn.interim_result_enabled
            && config
                .asr
                .interim_model
                .unwrap_or(config.asr.model)
                .is_nemotron();
        Self {
            segment_start_threshold: chunks_for_millis(
                config.segmentation.segment_start_speech_ms,
                config.segmentation.vad_interval_ms,
            ),
            interim_result_threshold: (config.turn.interim_result_enabled
                && !streaming_interim_asr_enabled)
                .then(|| {
                    chunks_for_millis(
                        config.turn.interim_result_silence_ms,
                        config.segmentation.vad_interval_ms,
                    )
                }),
            turn_check_threshold: chunks_for_millis(
                config.turn.check_silence_ms,
                config.segmentation.vad_interval_ms,
            ),
            max_chunks: max_chunks_for_interval(config.segmentation.vad_interval_ms),
            pre_speech_max_chunks: pre_speech_max_chunks(config),
        }
    }
}

fn chunks_for_millis(threshold_ms: u32, interval_ms: u32) -> u32 {
    threshold_ms.div_ceil(interval_ms.max(1)).max(1)
}

fn max_chunks_for_interval(interval_ms: u32) -> u32 {
    MAX_PHRASE_MILLIS.div_ceil(interval_ms.max(1)).max(1)
}

fn pre_speech_max_chunks(config: &ParapperConfig) -> usize {
    chunks_for_millis(
        config.turn.check_silence_ms,
        config.segmentation.vad_interval_ms,
    ) as usize
}
