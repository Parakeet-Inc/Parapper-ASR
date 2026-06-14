use super::*;

#[cfg(feature = "real-asr-tests")]
use std::{
    collections::VecDeque,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[cfg(feature = "real-asr-tests")]
use crate::{audio::ASR_SAMPLE_RATE, recognition::segmentation::vad::engine::VadEngine as _};

#[cfg(not(feature = "real-asr-tests"))]
#[test]
#[ignore = "requires --features real-asr-tests plus local FLEURS-R, VAD, ASR, SLI, and Namo model files"]
fn fleurs_single_language_td_model_timing_matrix_system_vad_asr_report() {
    print_real_asr_test_skip("fleurs_single_language_td_model_timing_matrix_system_vad_asr_report");
}

#[cfg(not(feature = "real-asr-tests"))]
#[test]
#[ignore = "requires --features real-asr-tests plus local FLEURS-R, VAD, ASR, SLI, and Namo model files"]
fn fleurs_ja_en_language_switch_td_timing_matrix_system_vad_asr_report() {
    print_real_asr_test_skip("fleurs_ja_en_language_switch_td_timing_matrix_system_vad_asr_report");
}

#[cfg(not(feature = "real-asr-tests"))]
fn print_real_asr_test_skip(test_name: &str) {
    println!(
        "skipped: run with `cargo test -p parapper --features real-asr-tests {test_name} -- --ignored --nocapture`"
    );
}

#[cfg(feature = "real-asr-tests")]
#[test]
#[ignore = "loads real VAD/ASR/TD models and reads local FLEURS-R corpus; runs 2 TD x 4 ASR models x 27 timing cases"]
fn fleurs_single_language_td_model_timing_matrix_system_vad_asr_report() {
    let env = FleursSystemEnv::new();
    let mut summaries = Vec::new();

    for scenario in SINGLE_LANGUAGE_SCENARIOS {
        let parts = read_short_fleurs_dev_parts(scenario.locale, FLEURS_SYSTEM_PART_COUNT);
        assert_eq!(parts.len(), FLEURS_SYSTEM_PART_COUNT);

        for turn_detector in SYSTEM_TURN_DETECTORS {
            for case in timing_matrix_cases() {
                summaries.push(run_fleurs_matrix_case(
                    &env,
                    FleursMatrixRunSpec {
                        scenario: scenario.name.to_string(),
                        parts: &parts,
                        turn_detector: *turn_detector,
                        primary_model: scenario.model,
                        enabled_models: vec![scenario.model],
                        multilingual: false,
                        case,
                    },
                ));
            }
        }
    }

    assert_interim_variants_keep_rerecognition_input(&summaries);
    print_fleurs_matrix_report("single-language", &summaries);
}

#[cfg(feature = "real-asr-tests")]
#[test]
#[ignore = "loads real VAD/ASR/SLI/TD models and reads local FLEURS-R corpus; runs 2 TD x 1 ASR-set x 27 timing cases"]
fn fleurs_ja_en_language_switch_td_timing_matrix_system_vad_asr_report() {
    let env = FleursSystemEnv::new();
    let ja_parts = read_short_fleurs_dev_parts("ja_jp", 2);
    let en_parts = read_short_fleurs_dev_parts("en_us", 1);
    assert_eq!(ja_parts.len(), 2);
    assert_eq!(en_parts.len(), 1);
    let parts = vec![
        into_indexed_part(ja_parts, 0),
        into_indexed_part(en_parts, 0),
        into_indexed_part(read_short_fleurs_dev_parts("ja_jp", 2), 1),
    ];
    let mut summaries = Vec::new();

    for turn_detector in SYSTEM_TURN_DETECTORS {
        for case in timing_matrix_cases() {
            let summary = run_fleurs_matrix_case(
                &env,
                FleursMatrixRunSpec {
                    scenario: "ja_en_reazonspeech_k2_v2_plus_parakeet_tdt_v2_en".to_string(),
                    parts: &parts,
                    turn_detector: *turn_detector,
                    primary_model: AsrModel::ReazonSpeechK2V2,
                    enabled_models: vec![
                        AsrModel::ReazonSpeechK2V2,
                        AsrModel::NemoParakeetTdt0_6BV2Int8,
                    ],
                    multilingual: true,
                    case,
                },
            );
            assert!(
                summary.route_models.contains(&AsrModel::ReazonSpeechK2V2),
                "{} {} should route at least one request to ReazonSpeech",
                summary.scenario,
                summary.case.label()
            );
            assert!(
                summary
                    .route_models
                    .contains(&AsrModel::NemoParakeetTdt0_6BV2Int8),
                "{} {} should route at least one request to English Parakeet",
                summary.scenario,
                summary.case.label()
            );
            summaries.push(summary);
        }
    }

    assert_interim_variants_keep_rerecognition_input(&summaries);
    print_fleurs_matrix_report("ja-en-language-switch", &summaries);
}

#[cfg(feature = "real-asr-tests")]
#[test]
fn fleurs_matrix_report_persistence_writes_timestamped_and_latest_reports() {
    let text_report = "FLEURS persistence smoke report\n";
    let record = AsrRequestRecord {
        sequence: 1,
        request_id: 42,
        kind: AsrTaskKind::Rerecognition,
        turn_id: 7,
        revision: 3,
        first_segment_id: Some(11),
        last_segment_id: Some(12),
        route: RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        detected_language: Some("ja".to_string()),
        close_reason: Some(SegmentCloseReason::EndSilenceReached),
        range_start: 0,
        range_end: 3200,
        audio_len: 3200,
        source_audio_len: 3000,
        vad_frames: 10,
        source_vad_frames: 9,
        audio_hash: 0xabcd,
        elapsed_millis: 123,
        status: "ok".to_string(),
        text: "保存テスト".to_string(),
    };
    let summaries = vec![FleursMatrixSummary {
        scenario: "persistence_smoke_scenario".to_string(),
        turn_detector: SystemTurnDetector::Namo,
        primary_model: AsrModel::ReazonSpeechK2V2,
        enabled_models: vec![AsrModel::ReazonSpeechK2V2],
        multilingual: false,
        case: FleursTimingMatrixCase {
            interim_silence_ms: Some(64),
            turn_check_silence_ms: 512,
            join_gap_ms: 128,
        },
        input_parts: vec![FleursInputPartSummary {
            locale: "ja_jp".to_string(),
            wav_path: PathBuf::from("fleurs-r/ja_jp/dev/dev/sample.wav"),
            samples: 1600,
        }],
        input_samples: 6400,
        vad_frames: 20,
        speech_frames: 10,
        max_vad_probability: 0.9,
        asr_request_count: 1,
        output_count: 1,
        final_output_count: 1,
        final_text: "保存テスト".to_string(),
        final_phrase_samples: 3000,
        final_rerecognition: Some(record.clone()),
        rerecognitions: vec![record],
        route_models: vec![AsrModel::ReazonSpeechK2V2],
    }];
    let saved_paths = save_fleurs_matrix_report("persistence-smoke", &summaries, text_report);

    assert!(
        saved_paths.text.is_file(),
        "timestamped text report should be saved: {}",
        saved_paths.text.display()
    );
    assert!(
        saved_paths.json.is_file(),
        "timestamped JSON report should be saved: {}",
        saved_paths.json.display()
    );
    assert_eq!(
        fs::read_to_string(&saved_paths.latest_text)
            .expect("latest text report should be readable"),
        text_report
    );

    let latest_json = fs::read_to_string(&saved_paths.latest_json)
        .expect("latest JSON report should be readable");
    let latest_json: serde_json::Value =
        serde_json::from_str(&latest_json).expect("latest JSON report should parse");
    assert_eq!(latest_json["schema_version"].as_u64(), Some(1));
    assert_eq!(latest_json["label"].as_str(), Some("persistence-smoke"));
    assert_eq!(latest_json["scenario_count"].as_u64(), Some(1));
    assert_eq!(
        latest_json["summaries"][0]["comparison_key"].as_str(),
        Some(
            "scenario=persistence_smoke_scenario|td=Namo|primary=ReazonSpeechK2V2|enabled=ReazonSpeechK2V2|multilingual=false|interim=64|turn_check=512|join_gap=128"
        )
    );
    assert_eq!(
        latest_json["summaries"][0]["rerecognitions"][0]["audio_hash_hex"].as_str(),
        Some("000000000000abcd")
    );
}

#[cfg(feature = "real-asr-tests")]
const FLEURS_SYSTEM_PART_COUNT: usize = 3;
#[cfg(feature = "real-asr-tests")]
const FLEURS_SYSTEM_EDGE_SILENCE_MS: u32 = 320;
#[cfg(feature = "real-asr-tests")]
const FLEURS_SYSTEM_FINAL_FLUSH_MS: u32 = 1280;
#[cfg(feature = "real-asr-tests")]
const FLEURS_SYSTEM_SEGMENT_START_MS: u32 = 128;
#[cfg(feature = "real-asr-tests")]
const FLEURS_SYSTEM_VAD_INTERVAL_MS: u32 = 32;

#[cfg(feature = "real-asr-tests")]
const SYSTEM_TURN_DETECTORS: &[SystemTurnDetector] =
    &[SystemTurnDetector::Simple, SystemTurnDetector::Namo];

#[cfg(feature = "real-asr-tests")]
const SINGLE_LANGUAGE_SCENARIOS: &[SingleLanguageScenario] = &[
    SingleLanguageScenario {
        name: "ja_reazonspeech_k2_v2",
        locale: "ja_jp",
        model: AsrModel::ReazonSpeechK2V2,
    },
    SingleLanguageScenario {
        name: "ja_parakeet_tdt_ctc_0_6b",
        locale: "ja_jp",
        model: AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
    },
    SingleLanguageScenario {
        name: "en_parakeet_tdt_0_6b_v2",
        locale: "en_us",
        model: AsrModel::NemoParakeetTdt0_6BV2Int8,
    },
    SingleLanguageScenario {
        name: "en_parakeet_tdt_0_6b_v3",
        locale: "en_us",
        model: AsrModel::NemoParakeetTdt0_6BV3Int8,
    },
];

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Copy)]
struct SingleLanguageScenario {
    name: &'static str,
    locale: &'static str,
    model: AsrModel,
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SystemTurnDetector {
    Simple,
    Namo,
}

#[cfg(feature = "real-asr-tests")]
impl SystemTurnDetector {
    fn config(self) -> TurnDetector {
        match self {
            Self::Simple => TurnDetector::Simple,
            Self::Namo => TurnDetector::Namo,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Namo => "Namo",
        }
    }
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_field_names,
    reason = "all timing matrix values are milliseconds and the suffix keeps printed parameters unambiguous"
)]
struct FleursTimingMatrixCase {
    interim_silence_ms: Option<u32>,
    turn_check_silence_ms: u32,
    join_gap_ms: u32,
}

#[cfg(feature = "real-asr-tests")]
impl FleursTimingMatrixCase {
    fn label(self) -> String {
        format!(
            "interim={:?}ms turn_check={}ms join_gap={}ms",
            self.interim_silence_ms, self.turn_check_silence_ms, self.join_gap_ms
        )
    }

    fn comparable_key(self) -> (u32, u32) {
        (self.turn_check_silence_ms, self.join_gap_ms)
    }
}

#[cfg(feature = "real-asr-tests")]
fn timing_matrix_cases() -> Vec<FleursTimingMatrixCase> {
    let mut cases = Vec::new();
    for interim_silence_ms in [None, Some(64), Some(128)] {
        for turn_check_silence_ms in [128, 256, 512] {
            for join_gap_ms in [64, 128, 256] {
                cases.push(FleursTimingMatrixCase {
                    interim_silence_ms,
                    turn_check_silence_ms,
                    join_gap_ms,
                });
            }
        }
    }
    cases
}

#[cfg(feature = "real-asr-tests")]
struct FleursSystemEnv {
    handle: tauri::AppHandle,
    models_root: std::path::PathBuf,
    vad_model: std::path::PathBuf,
}

#[cfg(feature = "real-asr-tests")]
impl FleursSystemEnv {
    fn new() -> Self {
        use tauri::Manager as _;

        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::generate_context!())
            .expect("Tauri production context should build");
        let handle = app.handle().clone();
        let models_root = diagnostic_models_root();
        let vad_model = models_root.join("silero_vad_v6").join("silero_vad.onnx");
        assert!(
            vad_model.is_file(),
            "VAD model should exist: {}",
            vad_model.display()
        );
        Self {
            handle,
            models_root,
            vad_model,
        }
    }
}

#[cfg(feature = "real-asr-tests")]
struct FleursMatrixRunSpec<'a> {
    scenario: String,
    parts: &'a [FleursPart],
    turn_detector: SystemTurnDetector,
    primary_model: AsrModel,
    enabled_models: Vec<AsrModel>,
    multilingual: bool,
    case: FleursTimingMatrixCase,
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Debug)]
struct AsrRequestRecord {
    sequence: usize,
    request_id: u64,
    kind: AsrTaskKind,
    turn_id: u64,
    revision: u64,
    first_segment_id: Option<u64>,
    last_segment_id: Option<u64>,
    route: RecognitionRoute,
    detected_language: Option<String>,
    close_reason: Option<SegmentCloseReason>,
    range_start: u64,
    range_end: u64,
    audio_len: usize,
    source_audio_len: usize,
    vad_frames: usize,
    source_vad_frames: usize,
    audio_hash: u64,
    elapsed_millis: u128,
    status: String,
    text: String,
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Debug)]
struct FleursInputPartSummary {
    locale: String,
    wav_path: PathBuf,
    samples: usize,
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Debug)]
struct FleursMatrixSummary {
    scenario: String,
    turn_detector: SystemTurnDetector,
    primary_model: AsrModel,
    enabled_models: Vec<AsrModel>,
    multilingual: bool,
    case: FleursTimingMatrixCase,
    input_parts: Vec<FleursInputPartSummary>,
    input_samples: usize,
    vad_frames: usize,
    speech_frames: usize,
    max_vad_probability: f32,
    asr_request_count: usize,
    output_count: usize,
    final_output_count: usize,
    final_text: String,
    final_phrase_samples: usize,
    final_rerecognition: Option<AsrRequestRecord>,
    rerecognitions: Vec<AsrRequestRecord>,
    route_models: Vec<AsrModel>,
}

#[cfg(feature = "real-asr-tests")]
struct ReportingEngineAsrRunner {
    handle: tauri::AppHandle,
    config: ParapperConfig,
    asr: AsrEngineCache,
    completed: VecDeque<AsrResult>,
    records: Arc<Mutex<Vec<AsrRequestRecord>>>,
}

#[cfg(feature = "real-asr-tests")]
impl ReportingEngineAsrRunner {
    fn new(
        handle: tauri::AppHandle,
        config: &ParapperConfig,
        records: Arc<Mutex<Vec<AsrRequestRecord>>>,
    ) -> Self {
        let mut asr = AsrEngineCache::default();
        let errors = asr.preload_required(&handle, config);
        assert!(
            errors.is_empty(),
            "ASR models should preload for system test: {}",
            errors.join("; ")
        );
        Self {
            handle,
            config: config.clone(),
            asr,
            completed: VecDeque::new(),
            records,
        }
    }
}

#[cfg(feature = "real-asr-tests")]
impl AsrRequestRunner for ReportingEngineAsrRunner {
    fn update_config(&mut self, config: &ParapperConfig) {
        self.config = config.clone();
        let errors = self.asr.preload_required(&self.handle, &self.config);
        assert!(
            errors.is_empty(),
            "ASR models should preload after config update: {}",
            errors.join("; ")
        );
    }

    fn submit(&mut self, request: AsrRequest) -> bool {
        let sequence = self
            .records
            .lock()
            .expect("ASR records should be readable")
            .len()
            + 1;
        let result = run_engine_asr_request(&self.handle, &self.config, &mut self.asr, &request);
        let (status, text) = match &result.status {
            AsrResultStatus::Ok(transcript) => ("ok".to_string(), transcript.text.clone()),
            AsrResultStatus::Failed(reason) => ("failed".to_string(), reason.clone()),
        };
        let record = AsrRequestRecord {
            sequence,
            request_id: request.request_id.0,
            kind: request.kind,
            turn_id: request.target.turn_id.0,
            revision: request.target.turn_revision.0,
            first_segment_id: request
                .target
                .first_segment_id
                .map(|segment_id| segment_id.0),
            last_segment_id: request
                .target
                .last_segment_id
                .map(|segment_id| segment_id.0),
            route: request.route,
            detected_language: request.detected_language.clone(),
            close_reason: request.close_reason,
            range_start: request.target.range.start_sample.0,
            range_end: request.target.range.end_sample.0,
            audio_len: request.audio.len(),
            source_audio_len: request.source_audio.len(),
            vad_frames: request.vad_results.len(),
            source_vad_frames: request.source_vad_results.len(),
            audio_hash: audio_hash(&request.audio),
            elapsed_millis: result.elapsed_millis,
            status,
            text,
        };
        print_asr_request_record(&record);
        self.records
            .lock()
            .expect("ASR records should be writable")
            .push(record);
        self.completed.push_back(result);
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.completed.pop_front()
    }
}

#[cfg(feature = "real-asr-tests")]
#[expect(
    clippy::too_many_lines,
    reason = "system scenario keeps setup, sequential prints, assertions, and summary capture together"
)]
fn run_fleurs_matrix_case(
    env: &FleursSystemEnv,
    spec: FleursMatrixRunSpec<'_>,
) -> FleursMatrixSummary {
    let model_dir = env
        .models_root
        .join(crate::model::catalog::asr_model_dir_name(
            spec.primary_model,
        ));
    assert!(
        model_dir.is_dir(),
        "ASR model directory should exist: {}",
        model_dir.display()
    );

    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(spec.turn_detector.config())
        .asr_model(spec.primary_model)
        .enabled_asr_models(spec.enabled_models.clone())
        .multilingual(spec.multilingual)
        .vad_interval_ms(FLEURS_SYSTEM_VAD_INTERVAL_MS)
        .segment_start_speech_ms(FLEURS_SYSTEM_SEGMENT_START_MS)
        .turn_check_silence_ms(spec.case.turn_check_silence_ms)
        .interim_display(spec.case.interim_silence_ms.is_some())
        .rerecognize_full_on_complete(true);
    if let Some(interim_silence_ms) = spec.case.interim_silence_ms {
        builder = builder.interim_result_silence_ms(interim_silence_ms);
    }
    if spec.multilingual {
        builder = builder.language_id_runtime();
    }
    {
        let config = builder.config_mut();
        config.asr.precision = spec.primary_model.default_precision();
        config.models.dir = Some(model_dir.display().to_string());
    }

    let records = Arc::new(Mutex::new(Vec::new()));
    let runner_config = builder.config_mut().clone();
    builder = builder.asr_runner(Box::new(ReportingEngineAsrRunner::new(
        env.handle.clone(),
        &runner_config,
        records.clone(),
    )));
    if spec.turn_detector == SystemTurnDetector::Namo {
        builder = builder.turn_decision_runner(Box::new(
            crate::recognition::control::EngineTurnDecisionRunner::new(&env.handle, &runner_config),
        ));
    }
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    runtime.update_config(&config);

    let input_audio = build_padded_fleurs_sequence(spec.parts, spec.case.join_gap_ms);
    println!();
    println!(
        "===== FLEURS matrix scenario={} td={} primary_model={:?} enabled_models={:?} case={} =====",
        spec.scenario,
        spec.turn_detector.label(),
        spec.primary_model,
        spec.enabled_models,
        spec.case.label()
    );
    println!(
        "params: multilingual={} vad_interval={}ms segment_start={}ms interim={:?}ms turn_check={}ms sentence_edge_silence={}ms join_gap={}ms final_flush={}ms input={} samples ({:.2}s)",
        spec.multilingual,
        config.segmentation.vad_interval_ms,
        config.segmentation.segment_start_speech_ms,
        spec.case.interim_silence_ms,
        spec.case.turn_check_silence_ms,
        FLEURS_SYSTEM_EDGE_SILENCE_MS,
        spec.case.join_gap_ms,
        FLEURS_SYSTEM_FINAL_FLUSH_MS,
        input_audio.len(),
        seconds(input_audio.len())
    );
    for (index, part) in spec.parts.iter().enumerate() {
        println!(
            "part[{index}]: locale={} samples={} ({:.2}s) path={}",
            part.locale,
            part.samples.len(),
            seconds(part.samples.len()),
            part.wav_path.display()
        );
    }

    let vad_summary =
        replay_audio_through_real_vad(&mut runtime, &config, &env.vad_model, &input_audio);
    drain_runtime(&mut runtime, 1000);
    assert!(
        runtime.turn_store.open_turn_id.is_none(),
        "{} {} should not leave an open turn",
        spec.scenario,
        spec.case.label()
    );

    let records = records
        .lock()
        .expect("ASR records should be readable")
        .clone();
    assert!(
        records.iter().all(|record| record.status == "ok"),
        "{} {} should not produce failed ASR records: {:?}",
        spec.scenario,
        spec.case.label(),
        records
            .iter()
            .filter(|record| record.status != "ok")
            .collect::<Vec<_>>()
    );
    let rerecognitions = records
        .iter()
        .filter(|record| record.kind == AsrTaskKind::Rerecognition)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        !rerecognitions.is_empty(),
        "{} {} should dispatch final rerecognition",
        spec.scenario,
        spec.case.label()
    );
    let final_rerecognition = rerecognitions.last().cloned();
    let route_models = unique_route_models(&records);

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    let final_output_count = outputs.iter().filter(|output| output.is_final).count();
    let final_text = outputs
        .iter()
        .filter(|output| output.is_final)
        .map(|output| output.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    let final_phrase_samples = outputs
        .iter()
        .filter(|output| output.is_final)
        .map(|output| output.phrase.len())
        .sum::<usize>();
    assert!(
        final_output_count > 0,
        "{} {} should emit at least one final GUI phrase",
        spec.scenario,
        spec.case.label()
    );
    assert_final_outputs_are_sane(&outputs, &spec.scenario, spec.case, spec.turn_detector);
    println!(
        "outputs: total={} final={} final_phrase_total={} samples ({:.2}s) final_text={}",
        outputs.len(),
        final_output_count,
        final_phrase_samples,
        seconds(final_phrase_samples),
        one_line(&final_text)
    );

    FleursMatrixSummary {
        scenario: spec.scenario,
        turn_detector: spec.turn_detector,
        primary_model: spec.primary_model,
        enabled_models: spec.enabled_models,
        multilingual: spec.multilingual,
        case: spec.case,
        input_parts: spec
            .parts
            .iter()
            .map(|part| FleursInputPartSummary {
                locale: part.locale.clone(),
                wav_path: part.wav_path.clone(),
                samples: part.samples.len(),
            })
            .collect(),
        input_samples: input_audio.len(),
        vad_frames: vad_summary.frames,
        speech_frames: vad_summary.speech_frames,
        max_vad_probability: vad_summary.max_probability,
        asr_request_count: records.len(),
        output_count: outputs.len(),
        final_output_count,
        final_text,
        final_phrase_samples,
        final_rerecognition,
        rerecognitions,
        route_models,
    }
}

#[cfg(feature = "real-asr-tests")]
fn into_indexed_part(mut parts: Vec<FleursPart>, index: usize) -> FleursPart {
    assert!(
        index < parts.len(),
        "requested FLEURS part index should exist"
    );
    parts.remove(index)
}

#[cfg(feature = "real-asr-tests")]
fn build_padded_fleurs_sequence(parts: &[FleursPart], join_gap_ms: u32) -> Vec<f32> {
    let edge_silence = vec![0.0; frames_for_millis(ASR_SAMPLE_RATE, FLEURS_SYSTEM_EDGE_SILENCE_MS)];
    let join_gap = vec![0.0; frames_for_millis(ASR_SAMPLE_RATE, join_gap_ms)];
    let final_flush = frames_for_millis(ASR_SAMPLE_RATE, FLEURS_SYSTEM_FINAL_FLUSH_MS);
    let mut audio = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        assert_eq!(part.sample_rate, ASR_SAMPLE_RATE);
        if index > 0 {
            audio.extend_from_slice(&join_gap);
        }
        audio.extend_from_slice(&edge_silence);
        audio.extend_from_slice(&part.samples);
        audio.extend_from_slice(&edge_silence);
    }
    audio.resize(audio.len() + final_flush, 0.0);
    audio
}

#[cfg(feature = "real-asr-tests")]
#[derive(Clone, Copy)]
struct VadReplaySummary {
    frames: usize,
    speech_frames: usize,
    max_probability: f32,
}

#[cfg(feature = "real-asr-tests")]
fn replay_audio_through_real_vad(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    vad_model: &std::path::Path,
    audio: &[f32],
) -> VadReplaySummary {
    let mut vad = crate::recognition::segmentation::vad::engine::OnnxRuntimeSileroVadEngine::new(
        vad_model,
        config.segmentation.vad_threshold,
    )
    .expect("real VAD engine should initialize");
    let chunk_len = frames_for_millis(ASR_SAMPLE_RATE, config.segmentation.vad_interval_ms);
    let mut summary = VadReplaySummary {
        frames: 0,
        speech_frames: 0,
        max_probability: 0.0,
    };

    for chunk in audio.chunks(chunk_len) {
        let mut frame = vec![0.0; chunk_len];
        frame[..chunk.len()].copy_from_slice(chunk);
        let vad_result = vad.process(&frame).expect("real VAD should process frame");
        summary.frames += 1;
        summary.speech_frames += usize::from(vad_result.is_speech);
        summary.max_probability = summary.max_probability.max(vad_result.probability);
        runtime.push_vad_frame(&frame, vad_result);
        runtime.step();
    }
    println!(
        "vad: frames={} speech_frames={} silence_frames={} max_probability={:.4}",
        summary.frames,
        summary.speech_frames,
        summary.frames.saturating_sub(summary.speech_frames),
        summary.max_probability
    );
    summary
}

#[cfg(feature = "real-asr-tests")]
fn drain_runtime(runtime: &mut RecognitionDriver, max_steps: usize) {
    for _ in 0..max_steps {
        if runtime.requests.in_flight_request.is_none()
            && runtime.pending.asr_segments.is_empty()
            && runtime.pending.turn_check.is_none()
            && runtime.pending.finalization.is_none()
            && runtime.requests.pending_rerecognition_purpose.is_none()
        {
            return;
        }
        runtime.step();
    }
    panic!("runtime did not drain after {max_steps} steps");
}

#[cfg(feature = "real-asr-tests")]
fn assert_final_outputs_are_sane(
    outputs: &[PhraseOutputSnapshot],
    scenario: &str,
    case: FleursTimingMatrixCase,
    turn_detector: SystemTurnDetector,
) {
    let final_outputs = outputs
        .iter()
        .filter(|output| output.is_final)
        .collect::<Vec<_>>();
    assert!(
        !final_outputs.is_empty(),
        "scenario={} td={} case={} should emit final GUI phrase outputs",
        scenario,
        turn_detector.label(),
        case.label()
    );
    for output in final_outputs {
        assert!(
            !output.phrase.is_empty(),
            "scenario={} td={} case={} final output turn={} segment={} should preserve saved source audio",
            scenario,
            turn_detector.label(),
            case.label(),
            output.turn_id,
            output.segment_id
        );
    }
}

#[cfg(feature = "real-asr-tests")]
fn assert_interim_variants_keep_rerecognition_input(summaries: &[FleursMatrixSummary]) {
    for baseline in summaries
        .iter()
        .filter(|summary| summary.case.interim_silence_ms.is_none())
    {
        for interim_silence_ms in [64, 128] {
            let matching = summaries
                .iter()
                .find(|summary| {
                    summary.scenario == baseline.scenario
                        && summary.turn_detector == baseline.turn_detector
                        && summary.primary_model == baseline.primary_model
                        && summary.enabled_models == baseline.enabled_models
                        && summary.case.interim_silence_ms == Some(interim_silence_ms)
                        && summary.case.comparable_key() == baseline.case.comparable_key()
                })
                .unwrap_or_else(|| {
                    panic!(
                        "missing interim variant for scenario={} td={} model={:?} interim={}ms turn_check={}ms gap={}ms",
                        baseline.scenario,
                        baseline.turn_detector.label(),
                        baseline.primary_model,
                        interim_silence_ms,
                        baseline.case.turn_check_silence_ms,
                        baseline.case.join_gap_ms
                    )
                });
            assert_eq!(
                baseline.rerecognitions.len(),
                matching.rerecognitions.len(),
                "rerecognition request count changed with interim={}ms for scenario={} td={} model={:?} case={}",
                interim_silence_ms,
                baseline.scenario,
                baseline.turn_detector.label(),
                baseline.primary_model,
                baseline.case.label()
            );
            for (index, (without_interim, with_interim)) in baseline
                .rerecognitions
                .iter()
                .zip(matching.rerecognitions.iter())
                .enumerate()
            {
                assert_eq!(
                    without_interim.audio_len,
                    with_interim.audio_len,
                    "rerecognition[{index}] audio length changed with interim={}ms for scenario={} td={} model={:?} case={}",
                    interim_silence_ms,
                    baseline.scenario,
                    baseline.turn_detector.label(),
                    baseline.primary_model,
                    baseline.case.label()
                );
                assert_eq!(
                    without_interim.audio_hash,
                    with_interim.audio_hash,
                    "rerecognition[{index}] audio samples changed with interim={}ms for scenario={} td={} model={:?} case={}",
                    interim_silence_ms,
                    baseline.scenario,
                    baseline.turn_detector.label(),
                    baseline.primary_model,
                    baseline.case.label()
                );
            }
        }
    }
}

#[cfg(feature = "real-asr-tests")]
fn unique_route_models(records: &[AsrRequestRecord]) -> Vec<AsrModel> {
    let mut models = Vec::new();
    for record in records {
        if !models.contains(&record.route.model) {
            models.push(record.route.model);
        }
    }
    models
}

#[cfg(feature = "real-asr-tests")]
fn print_asr_request_record(record: &AsrRequestRecord) {
    println!(
        "asr[{seq}]: id={id} kind={kind:?} turn={turn} rev={rev} segments={first:?}->{last:?} route={language:?}/{model:?} detected={detected:?} close={close:?} range={start}..{end} audio={audio} ({audio_sec:.2}s) source={source} vad={vad}/{source_vad} hash={hash:016x} elapsed={elapsed}ms status={status} text={text}",
        seq = record.sequence,
        id = record.request_id,
        kind = record.kind,
        turn = record.turn_id,
        rev = record.revision,
        first = record.first_segment_id,
        last = record.last_segment_id,
        language = record.route.language,
        model = record.route.model,
        detected = record.detected_language,
        close = record.close_reason,
        start = record.range_start,
        end = record.range_end,
        audio = record.audio_len,
        audio_sec = seconds(record.audio_len),
        source = record.source_audio_len,
        vad = record.vad_frames,
        source_vad = record.source_vad_frames,
        hash = record.audio_hash,
        elapsed = record.elapsed_millis,
        status = record.status,
        text = one_line(&record.text)
    );
}

#[cfg(feature = "real-asr-tests")]
fn print_fleurs_matrix_report(label: &str, summaries: &[FleursMatrixSummary]) {
    let report = format_fleurs_matrix_report(label, summaries);
    print!("{report}");
    let saved_paths = save_fleurs_matrix_report(label, summaries, &report);
    println!(
        "report_saved: text={} json={} latest_text={} latest_json={}",
        saved_paths.text.display(),
        saved_paths.json.display(),
        saved_paths.latest_text.display(),
        saved_paths.latest_json.display()
    );
}

#[cfg(feature = "real-asr-tests")]
fn format_fleurs_matrix_report(label: &str, summaries: &[FleursMatrixSummary]) -> String {
    use std::fmt::Write as _;

    let mut report = String::new();
    report.push('\n');
    writeln!(
        &mut report,
        "===== FLEURS {label} VAD+ASR matrix report ====="
    )
    .expect("writing FLEURS text report header should not fail");
    writeln!(&mut report, "scenario_count={}", summaries.len())
        .expect("writing FLEURS text report count should not fail");
    for summary in summaries {
        let final_rerecognition = summary.final_rerecognition.as_ref().map_or_else(
            || "none".to_string(),
            |record| {
                format!(
                    "{} samples ({:.2}s) hash={:016x}",
                    record.audio_len,
                    seconds(record.audio_len),
                    record.audio_hash
                )
            },
        );
        writeln!(
            &mut report,
            "report: scenario={} td={} primary_model={:?} enabled_models={:?} case={} input={} ({:.2}s) vad={}/{} max_vad={:.4} asr_requests={} rerecognitions={} routes={:?} outputs={}/{} final_phrase={} ({:.2}s) final_rerecognition={} final_text={}",
            summary.scenario,
            summary.turn_detector.label(),
            summary.primary_model,
            summary.enabled_models,
            summary.case.label(),
            summary.input_samples,
            seconds(summary.input_samples),
            summary.speech_frames,
            summary.vad_frames,
            summary.max_vad_probability,
            summary.asr_request_count,
            summary.rerecognitions.len(),
            summary.route_models,
            summary.final_output_count,
            summary.output_count,
            summary.final_phrase_samples,
            seconds(summary.final_phrase_samples),
            final_rerecognition,
            one_line(&summary.final_text)
        )
        .expect("writing FLEURS text report row should not fail");
    }
    report
}

#[cfg(feature = "real-asr-tests")]
struct SavedFleursReportPaths {
    text: PathBuf,
    json: PathBuf,
    latest_text: PathBuf,
    latest_json: PathBuf,
}

#[cfg(feature = "real-asr-tests")]
fn save_fleurs_matrix_report(
    label: &str,
    summaries: &[FleursMatrixSummary],
    text_report: &str,
) -> SavedFleursReportPaths {
    let generated_at = chrono::Utc::now();
    let run_id = generated_at.format("%Y%m%dT%H%M%S%3fZ").to_string();
    let dir = fleurs_report_dir(label);
    fs::create_dir_all(&dir)
        .unwrap_or_else(|err| panic!("failed to create report directory {}: {err}", dir.display()));

    let text_path = dir.join(format!("{run_id}.txt"));
    let json_path = dir.join(format!("{run_id}.json"));
    let latest_text_path = dir.join("latest.txt");
    let latest_json_path = dir.join("latest.json");
    let json_report = FleursMatrixJsonReport::from_summaries(
        label,
        generated_at.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        summaries,
    );
    let json = serde_json::to_string_pretty(&json_report)
        .expect("FLEURS matrix JSON report should serialize");

    write_report_file(&text_path, text_report.as_bytes());
    write_report_file(&json_path, json.as_bytes());
    write_report_file(&latest_text_path, text_report.as_bytes());
    write_report_file(&latest_json_path, json.as_bytes());

    SavedFleursReportPaths {
        text: text_path,
        json: json_path,
        latest_text: latest_text_path,
        latest_json: latest_json_path,
    }
}

#[cfg(feature = "real-asr-tests")]
fn write_report_file(path: &std::path::Path, bytes: &[u8]) {
    fs::write(path, bytes)
        .unwrap_or_else(|err| panic!("failed to write report {}: {err}", path.display()));
}

#[cfg(feature = "real-asr-tests")]
fn fleurs_report_dir(label: &str) -> PathBuf {
    workspace_target_dir()
        .join("parapper-system-reports")
        .join("fleurs")
        .join(sanitize_path_component(label))
}

#[cfg(feature = "real-asr-tests")]
fn workspace_target_dir() -> PathBuf {
    if let Some(target_dir) = std::env::var_os("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri should have a workspace parent")
        .join("target")
}

#[cfg(feature = "real-asr-tests")]
fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "real-asr-tests")]
#[derive(serde::Serialize)]
struct FleursMatrixJsonReport {
    schema_version: u32,
    label: String,
    generated_at_utc: String,
    package_name: &'static str,
    package_version: &'static str,
    sample_rate: u32,
    edge_silence_ms: u32,
    final_flush_ms: u32,
    segment_start_ms: u32,
    vad_interval_ms: u32,
    scenario_count: usize,
    summaries: Vec<FleursMatrixJsonSummary>,
}

#[cfg(feature = "real-asr-tests")]
impl FleursMatrixJsonReport {
    fn from_summaries(
        label: &str,
        generated_at_utc: String,
        summaries: &[FleursMatrixSummary],
    ) -> Self {
        Self {
            schema_version: 1,
            label: label.to_string(),
            generated_at_utc,
            package_name: env!("CARGO_PKG_NAME"),
            package_version: env!("CARGO_PKG_VERSION"),
            sample_rate: ASR_SAMPLE_RATE,
            edge_silence_ms: FLEURS_SYSTEM_EDGE_SILENCE_MS,
            final_flush_ms: FLEURS_SYSTEM_FINAL_FLUSH_MS,
            segment_start_ms: FLEURS_SYSTEM_SEGMENT_START_MS,
            vad_interval_ms: FLEURS_SYSTEM_VAD_INTERVAL_MS,
            scenario_count: summaries.len(),
            summaries: summaries
                .iter()
                .map(FleursMatrixJsonSummary::from_summary)
                .collect(),
        }
    }
}

#[cfg(feature = "real-asr-tests")]
#[derive(serde::Serialize)]
struct FleursMatrixJsonSummary {
    comparison_key: String,
    scenario: String,
    turn_detector: String,
    primary_model: AsrModel,
    enabled_models: Vec<AsrModel>,
    multilingual: bool,
    interim_silence_ms: Option<u32>,
    turn_check_silence_ms: u32,
    join_gap_ms: u32,
    input_parts: Vec<FleursInputPartJsonSummary>,
    input_samples: usize,
    input_seconds: f64,
    vad_frames: usize,
    speech_frames: usize,
    max_vad_probability: f32,
    asr_request_count: usize,
    output_count: usize,
    final_output_count: usize,
    final_text: String,
    final_phrase_samples: usize,
    final_phrase_seconds: f64,
    route_models: Vec<AsrModel>,
    final_rerecognition: Option<AsrRequestJsonRecord>,
    rerecognitions: Vec<AsrRequestJsonRecord>,
}

#[cfg(feature = "real-asr-tests")]
impl FleursMatrixJsonSummary {
    fn from_summary(summary: &FleursMatrixSummary) -> Self {
        Self {
            comparison_key: summary_comparison_key(summary),
            scenario: summary.scenario.clone(),
            turn_detector: summary.turn_detector.label().to_string(),
            primary_model: summary.primary_model,
            enabled_models: summary.enabled_models.clone(),
            multilingual: summary.multilingual,
            interim_silence_ms: summary.case.interim_silence_ms,
            turn_check_silence_ms: summary.case.turn_check_silence_ms,
            join_gap_ms: summary.case.join_gap_ms,
            input_parts: summary
                .input_parts
                .iter()
                .map(FleursInputPartJsonSummary::from_summary)
                .collect(),
            input_samples: summary.input_samples,
            input_seconds: seconds(summary.input_samples),
            vad_frames: summary.vad_frames,
            speech_frames: summary.speech_frames,
            max_vad_probability: summary.max_vad_probability,
            asr_request_count: summary.asr_request_count,
            output_count: summary.output_count,
            final_output_count: summary.final_output_count,
            final_text: summary.final_text.clone(),
            final_phrase_samples: summary.final_phrase_samples,
            final_phrase_seconds: seconds(summary.final_phrase_samples),
            route_models: summary.route_models.clone(),
            final_rerecognition: summary
                .final_rerecognition
                .as_ref()
                .map(AsrRequestJsonRecord::from_record),
            rerecognitions: summary
                .rerecognitions
                .iter()
                .map(AsrRequestJsonRecord::from_record)
                .collect(),
        }
    }
}

#[cfg(feature = "real-asr-tests")]
#[derive(serde::Serialize)]
struct FleursInputPartJsonSummary {
    locale: String,
    wav_path: String,
    samples: usize,
    seconds: f64,
}

#[cfg(feature = "real-asr-tests")]
impl FleursInputPartJsonSummary {
    fn from_summary(summary: &FleursInputPartSummary) -> Self {
        Self {
            locale: summary.locale.clone(),
            wav_path: summary.wav_path.display().to_string(),
            samples: summary.samples,
            seconds: seconds(summary.samples),
        }
    }
}

#[cfg(feature = "real-asr-tests")]
#[derive(serde::Serialize)]
struct AsrRequestJsonRecord {
    sequence: usize,
    request_id: u64,
    kind: String,
    turn_id: u64,
    revision: u64,
    first_segment_id: Option<u64>,
    last_segment_id: Option<u64>,
    route_language: AsrLanguage,
    route_model: AsrModel,
    detected_language: Option<String>,
    close_reason: Option<String>,
    range_start: u64,
    range_end: u64,
    audio_len: usize,
    audio_seconds: f64,
    source_audio_len: usize,
    source_audio_seconds: f64,
    vad_frames: usize,
    source_vad_frames: usize,
    audio_hash: u64,
    audio_hash_hex: String,
    elapsed_millis: u128,
    status: String,
    text: String,
}

#[cfg(feature = "real-asr-tests")]
impl AsrRequestJsonRecord {
    fn from_record(record: &AsrRequestRecord) -> Self {
        Self {
            sequence: record.sequence,
            request_id: record.request_id,
            kind: format!("{:?}", record.kind),
            turn_id: record.turn_id,
            revision: record.revision,
            first_segment_id: record.first_segment_id,
            last_segment_id: record.last_segment_id,
            route_language: record.route.language,
            route_model: record.route.model,
            detected_language: record.detected_language.clone(),
            close_reason: record.close_reason.map(|reason| format!("{reason:?}")),
            range_start: record.range_start,
            range_end: record.range_end,
            audio_len: record.audio_len,
            audio_seconds: seconds(record.audio_len),
            source_audio_len: record.source_audio_len,
            source_audio_seconds: seconds(record.source_audio_len),
            vad_frames: record.vad_frames,
            source_vad_frames: record.source_vad_frames,
            audio_hash: record.audio_hash,
            audio_hash_hex: format!("{:016x}", record.audio_hash),
            elapsed_millis: record.elapsed_millis,
            status: record.status.clone(),
            text: record.text.clone(),
        }
    }
}

#[cfg(feature = "real-asr-tests")]
fn summary_comparison_key(summary: &FleursMatrixSummary) -> String {
    format!(
        "scenario={}|td={}|primary={:?}|enabled={}|multilingual={}|interim={}|turn_check={}|join_gap={}",
        summary.scenario,
        summary.turn_detector.label(),
        summary.primary_model,
        model_list_key(&summary.enabled_models),
        summary.multilingual,
        summary
            .case
            .interim_silence_ms
            .map_or_else(|| "none".to_string(), |ms| ms.to_string()),
        summary.case.turn_check_silence_ms,
        summary.case.join_gap_ms
    )
}

#[cfg(feature = "real-asr-tests")]
fn model_list_key(models: &[AsrModel]) -> String {
    models
        .iter()
        .map(|model| format!("{model:?}"))
        .collect::<Vec<_>>()
        .join("+")
}

#[cfg(feature = "real-asr-tests")]
fn audio_hash(samples: &[f32]) -> u64 {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    let mut hasher = DefaultHasher::new();
    samples.len().hash(&mut hasher);
    for sample in samples {
        sample.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(feature = "real-asr-tests")]
fn one_line(text: &str) -> String {
    text.replace(['\r', '\n'], " ")
}

#[cfg(feature = "real-asr-tests")]
#[expect(
    clippy::cast_precision_loss,
    reason = "system report prints human-readable seconds from sample counts"
)]
fn seconds(samples: usize) -> f64 {
    samples as f64 / f64::from(ASR_SAMPLE_RATE)
}
