use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use tauri::{AppHandle, Listener};

use super::{
    AsrJobContext, MIN_LANGUAGE_ID_SAMPLES, handle_asr_job, handle_tick_at,
    normalize_asr_input_audio,
};
use crate::{
    config::{AsrModel, ParapperConfig, TurnDetector},
    model::NamoTurnDetectorModel,
    recognition::{
        engine_cache::{AsrEngineCache, CachedNamoTurnDetector, NamoTurnDetectorCache},
        engines::{AsrEngine, NamoTurnDecision},
        events::RecognizedTextEvent,
        route::RecognitionRoute,
        segment_builder::SegmentCloseReason,
        transcription::job::AsrJob,
        turn::Turn,
    },
};

#[test]
fn normalize_asr_input_audio_scales_peak_to_target() {
    let config = ParapperConfig {
        asr_normalize_input_audio: true,
        ..ParapperConfig::default()
    };
    let normalized = normalize_asr_input_audio(&config, &[0.0, 0.5, -0.25]);

    assert!(matches!(normalized, Cow::Owned(_)));
    assert!((normalized[1] - 0.95).abs() < 0.0001);
    assert!((normalized[2] + 0.475).abs() < 0.0001);
}

#[test]
fn normalize_asr_input_audio_keeps_audio_when_disabled() {
    let config = ParapperConfig {
        asr_normalize_input_audio: false,
        ..ParapperConfig::default()
    };
    let audio = [0.0, 0.5, -0.25];
    let normalized = normalize_asr_input_audio(&config, &audio);

    assert!(matches!(normalized, Cow::Borrowed(_)));
    assert_eq!(normalized.as_ref(), audio);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn worker_turn_check_silence_uses_namo_continue_after_interim_segment() {
    let mut worker = WorkerHarness::new(
        vec![NamoTurnDecision {
            is_end_of_turn: false,
            confidence: 0.99,
        }],
        vec!["まだ続く"],
    );

    worker.handle(AsrJob::SegmentClosed {
        segment_id: 1,
        previous_segment_id: None,
        full_audio: vec![1.0],
        reason: SegmentCloseReason::InterimResultSilenceReached,
    });
    let interim = worker.recv_output();
    worker.handle(AsrJob::TurnCheckSilenceReached {
        previous_segment_id: 1,
    });

    assert!(!interim.is_final);
    assert_eq!(interim.text, "まだ続く...");
    assert_eq!(interim.source.turn_id, 1);
    assert_eq!(
        worker.turn_detector_texts(),
        vec!["まだ続く".to_string()],
        "turn-check silence must ask TD about the current TurnDraft text"
    );
    worker.assert_no_output();
    assert_eq!(worker.open_turn_id, Some(1));
    assert!(worker.turns.contains_key(&1));
}

#[test]
#[cfg(not(target_os = "macos"))]
fn worker_turn_check_silence_uses_namo_complete_after_interim_segment() {
    let mut worker = WorkerHarness::new(
        vec![NamoTurnDecision {
            is_end_of_turn: true,
            confidence: 0.99,
        }],
        vec!["ここで終わり"],
    );

    worker.handle(AsrJob::SegmentClosed {
        segment_id: 1,
        previous_segment_id: None,
        full_audio: vec![1.0],
        reason: SegmentCloseReason::InterimResultSilenceReached,
    });
    let interim = worker.recv_output();
    worker.handle(AsrJob::TurnCheckSilenceReached {
        previous_segment_id: 1,
    });
    let final_output = worker.recv_output();

    assert!(!interim.is_final);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "ここで終わり。");
    assert_eq!(final_output.source.turn_id, 1);
    assert_eq!(
        worker.turn_detector_texts(),
        vec!["ここで終わり".to_string()],
        "TD Complete must be an observed decision, not the default fallback"
    );
    assert!(worker.open_turn_id.is_none());
    assert!(!worker.turns.contains_key(&1));
}

#[test]
#[cfg(not(target_os = "macos"))]
fn worker_td_continue_without_activity_times_out_open_turn() {
    let mut worker = WorkerHarness::new(
        vec![NamoTurnDecision {
            is_end_of_turn: false,
            confidence: 0.99,
        }],
        vec!["まだ続く"],
    );

    worker.handle(AsrJob::SegmentClosed {
        segment_id: 1,
        previous_segment_id: None,
        full_audio: vec![1.0],
        reason: SegmentCloseReason::EndSilenceReached,
    });
    let interim = worker.recv_output();
    worker.tick_after_open_turn_timeout();
    let timeout_final = worker.recv_output();

    assert!(!interim.is_final);
    assert_eq!(interim.text, "まだ続く...");
    assert_eq!(worker.turn_detector_texts(), vec!["まだ続く".to_string()]);
    assert!(timeout_final.is_final);
    assert_eq!(timeout_final.text, "まだ続く。");
    assert_eq!(timeout_final.source.turn_id, 1);
    assert!(worker.open_turn_id.is_none());
    assert!(!worker.turns.contains_key(&1));
}

#[test]
#[cfg(not(target_os = "macos"))]
fn worker_td_continue_activity_side_channel_prevents_timeout_before_next_segment() {
    let mut worker = WorkerHarness::new(
        vec![
            NamoTurnDecision {
                is_end_of_turn: false,
                confidence: 0.99,
            },
            NamoTurnDecision {
                is_end_of_turn: true,
                confidence: 0.99,
            },
        ],
        vec!["これは", "続きです"],
    );

    worker.handle(AsrJob::SegmentClosed {
        segment_id: 1,
        previous_segment_id: None,
        full_audio: vec![1.0],
        reason: SegmentCloseReason::EndSilenceReached,
    });
    let interim = worker.recv_output();
    worker.send_activity();
    worker.tick_after_open_turn_timeout();
    worker.assert_no_output();
    worker.handle(AsrJob::SegmentClosed {
        segment_id: 2,
        previous_segment_id: None,
        full_audio: vec![2.0],
        reason: SegmentCloseReason::EndSilenceReached,
    });
    let final_output = worker.recv_output();

    assert!(!interim.is_final);
    assert_eq!(interim.text, "これは...");
    assert_eq!(worker.open_turn_id, None);
    assert!(final_output.is_final);
    assert_eq!(final_output.text, "これは続きです。");
    assert_eq!(final_output.source.turn_id, 1);
    assert_eq!(final_output.source.segment_id, 2);
    assert_eq!(
        worker.turn_detector_texts(),
        vec!["これは".to_string(), "これは続きです".to_string()],
        "the second SegmentClosed should still belong to the open turn"
    );
}

#[test]
#[cfg(not(target_os = "macos"))]
fn worker_interim_segment_uses_last_spoken_route_without_sli() {
    let mut worker = WorkerHarness::new(Vec::new(), vec!["これは使われない"]);
    worker.config.multilingual_asr_enabled = true;
    worker.config.enabled_asr_models = vec![
        AsrModel::ReazonSpeechK2V2,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
    ];
    worker.last_spoken_route = Some(RecognitionRoute::from_model(
        AsrModel::NemoParakeetTdt0_6BV2Int8,
    ));
    worker.asr.insert_engine_for_test(
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        Box::new(ScriptedAsrEngine {
            texts: VecDeque::from(["hello".to_string()]),
        }),
    );

    worker.handle(AsrJob::SegmentClosed {
        segment_id: 1,
        previous_segment_id: None,
        full_audio: vec![1.0; MIN_LANGUAGE_ID_SAMPLES],
        reason: SegmentCloseReason::InterimResultSilenceReached,
    });
    let interim = worker.recv_output();

    assert!(!interim.is_final);
    assert_eq!(interim.text, "hello...");
}

struct WorkerHarness {
    _app: tauri::App,
    handle: AppHandle,
    config: ParapperConfig,
    asr: AsrEngineCache,
    turn_detectors: NamoTurnDetectorCache,
    turns: HashMap<u64, Turn>,
    turn_revisions: HashMap<u64, u64>,
    open_turn_id: Option<u64>,
    open_turn_since: Option<Instant>,
    open_turn_activity_epoch: u64,
    last_spoken_route: Option<RecognitionRoute>,
    segment_activity_epoch: AtomicU64,
    next_output_sequence: u64,
    output_receiver: Receiver<RecognizedTextEvent>,
    turn_detector_texts: Arc<Mutex<Vec<String>>>,
}

impl WorkerHarness {
    fn new(turn_decisions: Vec<NamoTurnDecision>, asr_texts: Vec<&str>) -> Self {
        let app = tauri::Builder::default()
            .any_thread()
            .build(tauri::generate_context!())
            .expect("test app should build");
        let handle = app.handle().clone();
        let (output_sender, output_receiver) = mpsc::channel();
        let _event_id = handle.listen("parapper://recognized-text", move |event| {
            let output = serde_json::from_str::<RecognizedTextEvent>(event.payload())
                .expect("recognized-text event should be valid JSON");
            output_sender
                .send(output)
                .expect("recognized-text event should be recorded");
        });

        let config = ParapperConfig {
            neo_http_enabled: false,
            turn_detector: TurnDetector::Namo,
            ..ParapperConfig::default()
        };
        let mut asr = AsrEngineCache::default();
        asr.insert_engine_for_test(
            AsrModel::ReazonSpeechK2V2,
            Box::new(ScriptedAsrEngine {
                texts: asr_texts
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<VecDeque<_>>(),
            }),
        );

        let turn_detector_texts = Arc::new(Mutex::new(Vec::new()));
        let mut turn_detectors = NamoTurnDetectorCache::default();
        turn_detectors.insert_engine_for_test(
            NamoTurnDetectorModel::Japanese,
            Box::new(ScriptedTurnDetector {
                decisions: turn_decisions.into(),
                texts: turn_detector_texts.clone(),
            }),
        );

        Self {
            _app: app,
            handle,
            config,
            asr,
            turn_detectors,
            turns: HashMap::new(),
            turn_revisions: HashMap::new(),
            open_turn_id: None,
            open_turn_since: None,
            open_turn_activity_epoch: 0,
            last_spoken_route: None,
            segment_activity_epoch: AtomicU64::new(0),
            next_output_sequence: 1,
            output_receiver,
            turn_detector_texts,
        }
    }

    fn handle(&mut self, job: AsrJob) {
        handle_asr_job(
            AsrJobContext {
                handle: &self.handle,
                config: &self.config,
                asr: &mut self.asr,
                language_id: None,
                turn_detectors: &mut self.turn_detectors,
                turns: &mut self.turns,
                turn_revisions: &mut self.turn_revisions,
                open_turn_id: &mut self.open_turn_id,
                open_turn_since: &mut self.open_turn_since,
                open_turn_activity_epoch: &mut self.open_turn_activity_epoch,
                last_spoken_route: &mut self.last_spoken_route,
                segment_activity_epoch: &self.segment_activity_epoch,
                turn_session_id: 1,
                next_output_sequence: &mut self.next_output_sequence,
            },
            job,
        );
    }

    fn send_activity(&self) {
        self.segment_activity_epoch.fetch_add(1, Ordering::Release);
    }

    fn tick_after_open_turn_timeout(&mut self) {
        let open_since = self
            .open_turn_since
            .expect("tick after timeout requires an open turn");
        let timeout = Duration::from_millis(u64::from(self.config.turn_check_silence_ms) * 2);
        self.tick_at(open_since + timeout + Duration::from_millis(1));
    }

    fn tick_at(&mut self, now: Instant) {
        handle_tick_at(
            &mut AsrJobContext {
                handle: &self.handle,
                config: &self.config,
                asr: &mut self.asr,
                language_id: None,
                turn_detectors: &mut self.turn_detectors,
                turns: &mut self.turns,
                turn_revisions: &mut self.turn_revisions,
                open_turn_id: &mut self.open_turn_id,
                open_turn_since: &mut self.open_turn_since,
                open_turn_activity_epoch: &mut self.open_turn_activity_epoch,
                last_spoken_route: &mut self.last_spoken_route,
                segment_activity_epoch: &self.segment_activity_epoch,
                turn_session_id: 1,
                next_output_sequence: &mut self.next_output_sequence,
            },
            now,
        );
    }

    fn recv_output(&self) -> RecognizedTextEvent {
        self.output_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("recognized-text output should be emitted")
    }

    fn assert_no_output(&self) {
        assert!(
            self.output_receiver
                .recv_timeout(Duration::from_millis(100))
                .is_err(),
            "no recognized-text output should be emitted"
        );
    }

    fn turn_detector_texts(&self) -> Vec<String> {
        self.turn_detector_texts
            .lock()
            .expect("turn detector texts should be readable")
            .clone()
    }
}

struct ScriptedAsrEngine {
    texts: VecDeque<String>,
}

impl AsrEngine for ScriptedAsrEngine {
    fn transcribe(&mut self, _samples: &[f32]) -> Result<String> {
        self.texts
            .pop_front()
            .ok_or_else(|| anyhow!("scripted ASR text was exhausted"))
    }
}

struct ScriptedTurnDetector {
    decisions: VecDeque<NamoTurnDecision>,
    texts: Arc<Mutex<Vec<String>>>,
}

impl CachedNamoTurnDetector for ScriptedTurnDetector {
    fn decide(&mut self, text: &str, _max_context_tokens: u32) -> Result<NamoTurnDecision> {
        self.texts
            .lock()
            .expect("turn detector texts should be writable")
            .push(text.to_string());
        self.decisions
            .pop_front()
            .ok_or_else(|| anyhow!("scripted TD decision was exhausted"))
    }
}
