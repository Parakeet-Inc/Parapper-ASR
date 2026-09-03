mod buffer;
mod builder;
mod config;

pub use builder::{SegmentBuilder, SegmentBuilderEvent, SegmentCloseReason};

use serde::{Deserialize, Serialize};

pub use parapper_models::vad::VadResult;

/// Host-neutral settings consumed by the recognition segment state machine.
///
/// Model selection remains an ASR-engine concern. The host resolves that
/// selection to `streaming_interim_asr_enabled` before constructing this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecognitionConfig {
    pub vad_interval_ms: u32,
    pub segment_start_speech_ms: u32,
    pub interim_result_enabled: bool,
    pub interim_result_silence_ms: u32,
    pub turn_check_silence_ms: u32,
    pub streaming_interim_asr_enabled: bool,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            vad_interval_ms: 32,
            segment_start_speech_ms: 64,
            interim_result_enabled: true,
            interim_result_silence_ms: 320,
            turn_check_silence_ms: 640,
            streaming_interim_asr_enabled: false,
        }
    }
}

/// Tauri-free recognition entry point for VAD-classified audio frames.
///
/// VAD execution and audio-device ownership remain host ports. This engine owns
/// the Segment lifecycle and is shared by the desktop host and headless tests.
pub struct RecognitionSegmentEngine {
    segment_builder: SegmentBuilder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecognitionFrame {
    pub samples_len: usize,
    pub events: Vec<SegmentBuilderEvent>,
}

impl RecognitionSegmentEngine {
    #[must_use]
    pub fn new(config: &RecognitionConfig) -> Self {
        Self {
            segment_builder: SegmentBuilder::new(config),
        }
    }

    pub fn update_config(&mut self, config: &RecognitionConfig) {
        self.segment_builder.update_config(config);
    }

    pub fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult) -> RecognitionFrame {
        RecognitionFrame {
            samples_len: samples.len(),
            events: self.segment_builder.push(samples, vad_result),
        }
    }

    pub fn flush(&mut self) -> RecognitionFrame {
        RecognitionFrame {
            samples_len: 0,
            events: self.segment_builder.flush(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RecognitionConfig, RecognitionSegmentEngine, SegmentBuilder, SegmentBuilderEvent,
        SegmentCloseReason, VadResult,
    };

    fn vad(is_speech: bool) -> VadResult {
        VadResult {
            probability: if is_speech { 0.9 } else { 0.0 },
            is_speech,
        }
    }

    #[test]
    fn vad_frames_drive_the_complete_started_extended_and_closed_event_sequence() {
        let config = RecognitionConfig {
            vad_interval_ms: 32,
            segment_start_speech_ms: 1,
            turn_check_silence_ms: 64,
            ..RecognitionConfig::default()
        };
        let mut builder = SegmentBuilder::new(&config);

        assert_eq!(
            builder.push(&[1.0], vad(true)),
            [SegmentBuilderEvent::SegmentStarted {
                segment_id: 1,
                previous_segment_id: None,
                audio_so_far: vec![1.0],
                vad_results: vec![vad(true)],
            }]
        );
        assert_eq!(
            builder.push(&[2.0], vad(true)),
            [SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![2.0],
                vad_result: vad(true),
            }]
        );
        assert!(matches!(
            builder.push(&[0.0], vad(false)).as_slice(),
            [SegmentBuilderEvent::SegmentExtended { .. }]
        ));
        assert_eq!(
            builder.push(&[0.0], vad(false)),
            [
                SegmentBuilderEvent::SegmentExtended {
                    segment_id: 1,
                    previous_segment_id: None,
                    new_audio: vec![0.0],
                    vad_result: vad(false),
                },
                SegmentBuilderEvent::SegmentClosed {
                    segment_id: 1,
                    previous_segment_id: None,
                    full_audio: vec![1.0, 2.0, 0.0, 0.0],
                    vad_results: vec![vad(true), vad(true), vad(false), vad(false)],
                    source_audio: vec![1.0, 2.0, 0.0, 0.0],
                    source_vad_results: vec![vad(true), vad(true), vad(false), vad(false)],
                    reason: SegmentCloseReason::EndSilenceReached,
                },
            ]
        );
    }

    #[test]
    fn runtime_parameter_update_changes_threshold_without_recreating_builder() {
        let initial = RecognitionConfig {
            segment_start_speech_ms: 1,
            turn_check_silence_ms: 128,
            ..RecognitionConfig::default()
        };
        let next = RecognitionConfig {
            turn_check_silence_ms: 32,
            ..initial.clone()
        };
        let mut builder = SegmentBuilder::new(&initial);
        assert!(matches!(
            builder.push(&[1.0], vad(true)).as_slice(),
            [SegmentBuilderEvent::SegmentStarted { .. }]
        ));

        builder.update_config(&next);

        assert!(matches!(
            builder.push(&[0.0], vad(false)).as_slice(),
            [
                SegmentBuilderEvent::SegmentExtended { .. },
                SegmentBuilderEvent::SegmentClosed {
                    reason: SegmentCloseReason::EndSilenceReached,
                    ..
                }
            ]
        ));
    }

    #[test]
    fn desktop_and_headless_hosts_can_drive_the_same_segment_engine_contract() {
        let config = RecognitionConfig {
            segment_start_speech_ms: 1,
            ..RecognitionConfig::default()
        };
        let mut engine = RecognitionSegmentEngine::new(&config);

        let frame = engine.push_vad_frame(&[0.25, 0.5], vad(true));

        assert_eq!(frame.samples_len, 2);
        assert!(matches!(
            frame.events.as_slice(),
            [SegmentBuilderEvent::SegmentStarted {
                segment_id: 1,
                audio_so_far,
                ..
            }] if audio_so_far == &[0.25, 0.5]
        ));
    }
}
