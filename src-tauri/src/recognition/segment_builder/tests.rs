use super::{SegmentBuilder, SegmentBuilderEvent, SegmentCloseReason};
use crate::{config::ParapperConfig, recognition::engines::VadResult};

const CHUNK_MS: u32 = 32;
const START_IMMEDIATELY_MS: u32 = 1;
const TWO_CHUNKS_MS: u32 = CHUNK_MS * 2;
const THREE_CHUNKS_MS: u32 = CHUNK_MS * 3;
const FOUR_CHUNKS_MS: u32 = CHUNK_MS * 4;
const TEN_CHUNKS_MS: u32 = CHUNK_MS * 10;

#[test]
fn segment_builder_emits_started_extended_and_closed() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: TWO_CHUNKS_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    let events = segment_builder.push(&[1.0], speech_vad());
    assert_eq!(
        events,
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 1,
            previous_segment_id: None,
            audio_so_far: vec![1.0],
            vad_results: vads(&[true]),
        }]
    );

    let events = segment_builder.push(&[2.0], speech_vad());
    assert_eq!(
        events,
        vec![SegmentBuilderEvent::SegmentExtended {
            segment_id: 1,
            previous_segment_id: None,
            new_audio: vec![2.0],
            vad_result: speech_vad(),
        }]
    );

    assert_eq!(
        segment_builder.push(&[0.0], silence_vad()),
        vec![SegmentBuilderEvent::SegmentExtended {
            segment_id: 1,
            previous_segment_id: None,
            new_audio: vec![0.0],
            vad_result: silence_vad(),
        }]
    );
    assert_eq!(
        segment_builder.push(&[0.0], silence_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![0.0],
                vad_result: silence_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![1.0, 2.0, 0.0, 0.0],
                vad_results: vads(&[true, true, false, false]),
                reason: SegmentCloseReason::EndSilenceReached
            }
        ]
    );
}

#[test]
fn segment_builder_keeps_initial_silence_as_pre_speech_audio() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: THREE_CHUNKS_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(segment_builder.push(&[10.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[20.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[30.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[40.0], silence_vad()).is_empty());

    let events = segment_builder.push(&[1.0], speech_vad());
    assert_eq!(
        events,
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 1,
            previous_segment_id: None,
            audio_so_far: vec![20.0, 30.0, 40.0, 1.0],
            vad_results: vads(&[false, false, false, true]),
        }]
    );
}

#[test]
fn segment_builder_waits_for_segment_start_speech_ms_before_starting() {
    let config = ParapperConfig {
        segment_start_speech_ms: TWO_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(segment_builder.push(&[1.0], speech_vad()).is_empty());
    assert_eq!(
        segment_builder.push(&[2.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 1,
            previous_segment_id: None,
            audio_so_far: vec![1.0, 2.0],
            vad_results: vads(&[true, true]),
        }]
    );
}

#[test]
fn segment_builder_closes_with_max_chunks_reason_when_audio_too_long() {
    let config = ParapperConfig {
        vad_interval_ms: 25_000,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert_eq!(
        segment_builder.push(&[1.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 1,
            previous_segment_id: None,
            audio_so_far: vec![1.0],
            vad_results: vads(&[true]),
        }]
    );
    assert_eq!(
        segment_builder.push(&[2.0], speech_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![2.0],
                vad_result: speech_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![1.0, 2.0],
                vad_results: vads(&[true, true]),
                reason: SegmentCloseReason::SegmentMaxChunksReached
            }
        ]
    );
}

#[test]
fn segment_after_max_chunks_points_to_previous_segment() {
    let config = ParapperConfig {
        vad_interval_ms: 25_000,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    let _ = segment_builder.push(&[1.0], speech_vad());
    assert!(matches!(
        segment_builder.push(&[2.0], speech_vad()).as_slice(),
        [
            SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                reason: SegmentCloseReason::SegmentMaxChunksReached,
                ..
            }
        ]
    ));

    assert_eq!(
        segment_builder.push(&[3.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 2,
            previous_segment_id: Some(1),
            audio_so_far: vec![3.0],
            vad_results: vads(&[true]),
        }]
    );
}

#[test]
fn pending_speech_chunks_is_cleared_when_silence_breaks_run() {
    let config = ParapperConfig {
        segment_start_speech_ms: TWO_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(segment_builder.push(&[1.0], speech_vad()).is_empty());
    assert!(segment_builder.push(&[0.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[2.0], speech_vad()).is_empty());
    assert_eq!(
        segment_builder.push(&[3.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 1,
            previous_segment_id: None,
            audio_so_far: vec![0.0, 2.0, 3.0],
            vad_results: vads(&[false, true, true]),
        }]
    );
}

#[test]
fn silence_timeout_reuses_trailing_silence_as_next_pre_speech() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: THREE_CHUNKS_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(segment_builder.push(&[10.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[20.0], silence_vad()).is_empty());
    assert!(matches!(
        segment_builder.push(&[1.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentStarted { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[2.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[3.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[30.0], silence_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[40.0], silence_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert_eq!(
        segment_builder.push(&[50.0], silence_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![50.0],
                vad_result: silence_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![10.0, 20.0, 1.0, 2.0, 3.0, 30.0, 40.0, 50.0],
                vad_results: vads(&[false, false, true, true, true, false, false, false]),
                reason: SegmentCloseReason::EndSilenceReached
            }
        ]
    );

    assert_eq!(
        segment_builder.push(&[4.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 2,
            previous_segment_id: None,
            audio_so_far: vec![30.0, 40.0, 50.0, 4.0],
            vad_results: vads(&[false, false, false, true]),
        }]
    );
}

#[test]
fn interim_result_silence_emits_interim_segment_and_keeps_turn_link() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        interim_result_enabled: true,
        interim_result_silence_ms: THREE_CHUNKS_MS,
        turn_check_silence_ms: TEN_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(matches!(
        segment_builder.push(&[1.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentStarted { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[2.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[10.0], silence_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[20.0], silence_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
    ));
    assert_eq!(
        segment_builder.push(&[30.0], silence_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![30.0],
                vad_result: silence_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![1.0, 2.0, 10.0, 20.0, 30.0],
                vad_results: vads(&[true, true, false, false, false]),
                reason: SegmentCloseReason::InterimResultSilenceReached
            }
        ]
    );

    assert_eq!(
        segment_builder.push(&[3.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 2,
            previous_segment_id: Some(1),
            audio_so_far: vec![10.0, 20.0, 30.0, 3.0],
            vad_results: vads(&[false, false, false, true]),
        }]
    );
}

#[test]
fn speech_after_interim_result_continues_from_previous_segment_before_turn_check() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        interim_result_enabled: true,
        interim_result_silence_ms: THREE_CHUNKS_MS,
        turn_check_silence_ms: TEN_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    let _ = segment_builder.push(&[1.0], speech_vad());
    let _ = segment_builder.push(&[2.0], speech_vad());
    let _ = segment_builder.push(&[10.0], silence_vad());
    let _ = segment_builder.push(&[20.0], silence_vad());
    assert!(matches!(
        segment_builder.push(&[30.0], silence_vad()).as_slice(),
        [
            SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                reason: SegmentCloseReason::InterimResultSilenceReached,
                ..
            }
        ]
    ));

    assert!(segment_builder.push(&[40.0], silence_vad()).is_empty());
    assert!(segment_builder.push(&[50.0], silence_vad()).is_empty());
    assert_eq!(
        segment_builder.push(&[3.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 2,
            previous_segment_id: Some(1),
            audio_so_far: vec![10.0, 20.0, 30.0, 40.0, 50.0, 3.0],
            vad_results: vads(&[false, false, false, false, false, true]),
        }]
    );
}

#[test]
fn continued_silence_after_interim_result_reaches_turn_check_without_another_interim_result() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        interim_result_enabled: true,
        interim_result_silence_ms: THREE_CHUNKS_MS,
        turn_check_silence_ms: TEN_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    let _ = segment_builder.push(&[1.0], speech_vad());
    let _ = segment_builder.push(&[2.0], speech_vad());
    let _ = segment_builder.push(&[10.0], silence_vad());
    let _ = segment_builder.push(&[20.0], silence_vad());
    assert!(matches!(
        segment_builder.push(&[30.0], silence_vad()).as_slice(),
        [
            SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                reason: SegmentCloseReason::InterimResultSilenceReached,
                ..
            }
        ]
    ));

    for sample in [40.0, 50.0, 60.0, 70.0, 80.0, 90.0] {
        assert!(
            segment_builder.push(&[sample], silence_vad()).is_empty(),
            "continued silence before turn_check_silence_ms must not create another interim ASR"
        );
    }
    assert_eq!(
        segment_builder.push(&[100.0], silence_vad()),
        vec![SegmentBuilderEvent::TurnCheckSilenceReached {
            previous_segment_id: 1
        }]
    );
}

#[test]
fn interim_result_disabled_does_not_close_segment_before_turn_check_silence() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        interim_result_enabled: false,
        interim_result_silence_ms: THREE_CHUNKS_MS,
        turn_check_silence_ms: TEN_CHUNKS_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    let _ = segment_builder.push(&[1.0], speech_vad());
    let _ = segment_builder.push(&[2.0], speech_vad());
    let _ = segment_builder.push(&[10.0], silence_vad());
    let _ = segment_builder.push(&[20.0], silence_vad());

    assert_eq!(
        segment_builder.push(&[30.0], silence_vad()),
        vec![SegmentBuilderEvent::SegmentExtended {
            segment_id: 1,
            previous_segment_id: None,
            new_audio: vec![30.0],
            vad_result: silence_vad(),
        }],
        "disabled interim result must not run ASR at interim_result_silence_ms"
    );

    for sample in [40.0, 50.0, 60.0, 70.0, 80.0, 90.0] {
        assert!(matches!(
            segment_builder.push(&[sample], silence_vad()).as_slice(),
            [SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. }]
        ));
    }
    assert_eq!(
        segment_builder.push(&[100.0], silence_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![100.0],
                vad_result: silence_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![
                    1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
                ],
                vad_results: vads(&[
                    true, true, false, false, false, false, false, false, false, false, false,
                    false
                ]),
                reason: SegmentCloseReason::EndSilenceReached
            }
        ]
    );
}

#[test]
fn update_config_shortens_turn_check_threshold_during_active() {
    let initial_config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: FOUR_CHUNKS_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let next_config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&initial_config);

    assert!(matches!(
        segment_builder.push(&[1.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentStarted { segment_id: 1, .. }]
    ));

    segment_builder.update_config(&next_config);

    assert_eq!(
        segment_builder.push(&[0.0], silence_vad()),
        vec![
            SegmentBuilderEvent::SegmentExtended {
                segment_id: 1,
                previous_segment_id: None,
                new_audio: vec![0.0],
                vad_result: silence_vad(),
            },
            SegmentBuilderEvent::SegmentClosed {
                segment_id: 1,
                previous_segment_id: None,
                full_audio: vec![1.0, 0.0],
                vad_results: vads(&[true, false]),
                reason: SegmentCloseReason::EndSilenceReached
            }
        ]
    );
}

#[test]
fn next_segment_starts_with_incremented_id_after_close() {
    let config = ParapperConfig {
        vad_interval_ms: CHUNK_MS,
        turn_check_silence_ms: CHUNK_MS,
        segment_start_speech_ms: START_IMMEDIATELY_MS,
        ..ParapperConfig::default()
    };
    let mut segment_builder = SegmentBuilder::new(&config);

    assert!(matches!(
        segment_builder.push(&[1.0], speech_vad()).as_slice(),
        [SegmentBuilderEvent::SegmentStarted { segment_id: 1, .. }]
    ));
    assert!(matches!(
        segment_builder.push(&[0.0], silence_vad()).as_slice(),
        [
            SegmentBuilderEvent::SegmentExtended { segment_id: 1, .. },
            SegmentBuilderEvent::SegmentClosed { segment_id: 1, .. }
        ]
    ));
    assert_eq!(
        segment_builder.push(&[2.0], speech_vad()),
        vec![SegmentBuilderEvent::SegmentStarted {
            segment_id: 2,
            previous_segment_id: None,
            audio_so_far: vec![0.0, 2.0],
            vad_results: vads(&[false, true]),
        }]
    );
}

fn vads(pattern: &[bool]) -> Vec<VadResult> {
    pattern
        .iter()
        .map(|is_speech| {
            if *is_speech {
                speech_vad()
            } else {
                silence_vad()
            }
        })
        .collect()
}

fn speech_vad() -> VadResult {
    VadResult {
        probability: 0.9,
        is_speech: true,
    }
}

fn silence_vad() -> VadResult {
    VadResult {
        probability: 0.0,
        is_speech: false,
    }
}
