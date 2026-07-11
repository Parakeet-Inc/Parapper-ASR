use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Result, anyhow};

use super::{
    AsrRequestRunner, LanguageIdRuntime, PendingAsrSegment, PendingTurnCheck, RecognitionDriver,
    RecognitionDriverHandle, RecognitionSession, RecognitionShutdownResult, RerecognitionPurpose,
    SegmentCloseReason, TurnDecisionRunner, TurnOutputSink, replay_vad_frames_for_runtime,
    run_engine_asr_request,
};
use crate::{
    config::{AsrLanguage, AsrModel, AsrPrecision, ParapperConfig, TurnDetector},
    delivery::RecognizedTextOutput,
    recognition::{
        control::engine_cache::AsrEngineCache,
        segmentation::vad::engine::VadResult,
        transcription::{
            asr::{
                engine::{AsrEngine, AsrTranscript},
                input::MIN_LANGUAGE_ID_SAMPLES,
                task::{
                    AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AsrStreamingSessionKey,
                    AsrTarget, AsrTaskKind, AudioRange, GlobalSampleIndex, SegmentId, TurnId,
                    TurnRevision, VadFrameIndex,
                },
            },
            route::{
                RecognitionRoute,
                language_id::{LanguageDetectionWarningSink, LanguageDetector},
            },
        },
        turn::{GrammarBoundaryClass, Turn, TurnBoundaryCandidate, decision::TurnDecision},
    },
};

#[derive(Debug, PartialEq)]
enum RuntimeCall {
    UpdateConfig,
    VadFrame { sample: f32, is_speech: bool },
    Step,
}

#[derive(Default)]
struct RecordingRecognitionSession {
    calls: Vec<RuntimeCall>,
    shutdown_called: bool,
}

struct ScriptedAsrRunner {
    transcripts: VecDeque<AsrTranscript>,
    completed: VecDeque<AsrResult>,
}

struct ScriptedTurnDecisionRunner {
    decisions: VecDeque<TurnDecision>,
    texts: Arc<Mutex<Vec<String>>>,
}

struct ScriptedLanguageDetector {
    detected_languages: VecDeque<String>,
    call_audio_lens: Arc<Mutex<Vec<usize>>>,
}

struct TestLanguageIdRuntime;

#[derive(Clone)]
struct ManualAsrHandle {
    submitted: Arc<Mutex<VecDeque<AsrRequest>>>,
    completed: Arc<Mutex<VecDeque<AsrResult>>>,
    streaming_reset_count: Arc<Mutex<u32>>,
}

struct ManualAsrRunner {
    submitted: Arc<Mutex<VecDeque<AsrRequest>>>,
    completed: Arc<Mutex<VecDeque<AsrResult>>>,
    streaming_reset_count: Arc<Mutex<u32>>,
}

impl ManualAsrHandle {
    fn submitted_requests(&self) -> Vec<AsrRequest> {
        self.submitted
            .lock()
            .expect("submitted ASR requests should be readable")
            .iter()
            .cloned()
            .collect()
    }

    fn complete_next_with_text(&self, text: &str) {
        let request = self
            .submitted
            .lock()
            .expect("submitted ASR requests should be writable")
            .pop_front()
            .expect("an ASR request should be waiting for manual completion");
        self.completed
            .lock()
            .expect("completed ASR results should be writable")
            .push_back(AsrResult {
                request_id: request.request_id,
                kind: request.kind,
                target: request.target,
                route: request.route,
                status: AsrResultStatus::Ok(AsrTranscript::from_text(text)),
                completed_at_frame: VadFrameIndex(0),
                elapsed_millis: 0,
            });
    }

    fn complete_request_with_text(&self, request: &AsrRequest, text: &str) {
        self.complete_request_with_text_elapsed(request, text, 0);
    }

    fn complete_request_with_text_elapsed(
        &self,
        request: &AsrRequest,
        text: &str,
        elapsed_millis: u128,
    ) {
        self.completed
            .lock()
            .expect("completed ASR results should be writable")
            .push_back(AsrResult {
                request_id: request.request_id,
                kind: request.kind,
                target: request.target.clone(),
                route: request.route,
                status: AsrResultStatus::Ok(AsrTranscript::from_text(text)),
                completed_at_frame: VadFrameIndex(0),
                elapsed_millis,
            });
    }

    fn fail_request(&self, request: &AsrRequest) {
        self.completed
            .lock()
            .expect("completed ASR results should be writable")
            .push_back(AsrResult {
                request_id: request.request_id,
                kind: request.kind,
                target: request.target.clone(),
                route: request.route,
                status: AsrResultStatus::Failed("scripted ASR failure".to_string()),
                completed_at_frame: VadFrameIndex(0),
                elapsed_millis: 0,
            });
    }

    fn push_completed_result(&self, result: AsrResult) {
        self.completed
            .lock()
            .expect("completed ASR results should be writable")
            .push_back(result);
    }

    fn streaming_reset_count(&self) -> u32 {
        *self
            .streaming_reset_count
            .lock()
            .expect("streaming reset count should be readable")
    }
}

impl ManualAsrRunner {
    fn new() -> (Self, ManualAsrHandle) {
        let submitted = Arc::new(Mutex::new(VecDeque::new()));
        let completed = Arc::new(Mutex::new(VecDeque::new()));
        let streaming_reset_count = Arc::new(Mutex::new(0));
        (
            Self {
                submitted: submitted.clone(),
                completed: completed.clone(),
                streaming_reset_count: streaming_reset_count.clone(),
            },
            ManualAsrHandle {
                submitted,
                completed,
                streaming_reset_count,
            },
        )
    }
}

impl AsrRequestRunner for ManualAsrRunner {
    fn reset_streaming_sessions(&mut self) {
        *self
            .streaming_reset_count
            .lock()
            .expect("streaming reset count should be writable") += 1;
    }

    fn submit(&mut self, request: AsrRequest) -> bool {
        self.submitted
            .lock()
            .expect("submitted ASR requests should be writable")
            .push_back(request);
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.completed
            .lock()
            .expect("completed ASR results should be writable")
            .pop_front()
    }
}

impl ScriptedTurnDecisionRunner {
    fn new(decisions: Vec<TurnDecision>, texts: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            decisions: decisions.into(),
            texts,
        }
    }
}

impl TurnDecisionRunner for ScriptedTurnDecisionRunner {
    fn decide(
        &mut self,
        _route: RecognitionRoute,
        text: &str,
        _max_context_tokens: u32,
    ) -> Result<TurnDecision> {
        self.texts
            .lock()
            .expect("turn decision texts should be writable")
            .push(text.to_string());
        self.decisions
            .pop_front()
            .ok_or_else(|| anyhow!("scripted turn decision was exhausted"))
    }
}

impl LanguageDetector for ScriptedLanguageDetector {
    fn detect(&mut self, samples: &[f32], _candidates: Option<&[&str]>) -> Result<String> {
        self.call_audio_lens
            .lock()
            .expect("SLI call lengths should be writable")
            .push(samples.len());
        self.detected_languages
            .pop_front()
            .ok_or_else(|| anyhow!("scripted language detector was exhausted"))
    }
}

impl LanguageDetectionWarningSink for TestLanguageIdRuntime {
    fn emit_language_detection_warning(&self, _err: &anyhow::Error) {}
}

impl LanguageIdRuntime for TestLanguageIdRuntime {
    fn build_language_id(&self, _config: &ParapperConfig) -> Option<Box<dyn LanguageDetector>> {
        None
    }
}

impl ScriptedAsrRunner {
    fn from_texts(texts: Vec<&str>) -> Self {
        Self {
            transcripts: texts.into_iter().map(AsrTranscript::from_text).collect(),
            completed: VecDeque::new(),
        }
    }

    fn from_transcripts(transcripts: Vec<AsrTranscript>) -> Self {
        Self {
            transcripts: transcripts.into(),
            completed: VecDeque::new(),
        }
    }
}

impl AsrRequestRunner for ScriptedAsrRunner {
    fn submit(&mut self, request: AsrRequest) -> bool {
        let transcript = self
            .transcripts
            .pop_front()
            .expect("scripted ASR transcript should be available");
        self.completed.push_back(AsrResult {
            request_id: request.request_id,
            kind: request.kind,
            target: request.target,
            route: request.route,
            status: AsrResultStatus::Ok(transcript),
            completed_at_frame: VadFrameIndex(0),
            elapsed_millis: 0,
        });
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.completed.pop_front()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct OutputSnapshot {
    text: String,
    is_final: bool,
    turn_id: u64,
    segment_id: u64,
}

impl From<&RecognizedTextOutput> for OutputSnapshot {
    fn from(output: &RecognizedTextOutput) -> Self {
        Self {
            text: output.text.clone(),
            is_final: output.meta.is_final(),
            turn_id: output.meta.source().turn_id,
            segment_id: output.meta.source().segment_id,
        }
    }
}

fn output_snapshot(
    text: impl Into<String>,
    is_final: bool,
    turn_id: u64,
    segment_id: u64,
) -> OutputSnapshot {
    OutputSnapshot {
        text: text.into(),
        is_final,
        turn_id,
        segment_id,
    }
}

struct RecordingOutputSink {
    outputs: Arc<Mutex<Vec<OutputSnapshot>>>,
}

impl TurnOutputSink for RecordingOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        self.outputs
            .lock()
            .expect("outputs should be writable")
            .push(OutputSnapshot::from(&output));
    }
}

#[derive(Debug, PartialEq)]
struct PhraseOutputSnapshot {
    id: String,
    text: String,
    is_final: bool,
    source_asr_model: AsrModel,
    source_language: AsrLanguage,
    detected_language: Option<String>,
    turn_session_id: u64,
    turn_id: u64,
    segment_id: u64,
    output_sequence: u64,
    phrase: Vec<f32>,
    elapsed_millis: u128,
}

impl From<&RecognizedTextOutput> for PhraseOutputSnapshot {
    fn from(output: &RecognizedTextOutput) -> Self {
        Self {
            id: output.meta.id.clone(),
            text: output.text.clone(),
            is_final: output.meta.is_final(),
            source_asr_model: output.source_asr_model,
            source_language: output.source_language,
            detected_language: output.detected_language.clone(),
            turn_session_id: output.meta.source().turn_session_id,
            turn_id: output.meta.source().turn_id,
            segment_id: output.meta.source().segment_id,
            output_sequence: output.meta.source().output_sequence,
            phrase: output.phrase.to_vec(),
            elapsed_millis: output.elapsed_millis,
        }
    }
}

struct RecordingPhraseOutputSink {
    outputs: Arc<Mutex<Vec<PhraseOutputSnapshot>>>,
}

impl TurnOutputSink for RecordingPhraseOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        self.outputs
            .lock()
            .expect("phrase outputs should be writable")
            .push(PhraseOutputSnapshot::from(&output));
    }
}

impl RecognitionDriverHandle for RecordingRecognitionSession {
    fn update_config(&mut self, _config: &ParapperConfig) {
        self.calls.push(RuntimeCall::UpdateConfig);
    }

    fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult) {
        self.calls.push(RuntimeCall::VadFrame {
            sample: samples[0],
            is_speech: vad_result.is_speech,
        });
    }

    fn step(&mut self) {
        self.calls.push(RuntimeCall::Step);
    }

    fn shutdown(&mut self) -> RecognitionShutdownResult {
        self.shutdown_called = true;
        RecognitionShutdownResult::Completed
    }
}

struct ScriptedAsrEngine {
    transcripts: VecDeque<AsrTranscript>,
    call_audio_lens: Arc<Mutex<Vec<usize>>>,
}

impl AsrEngine for ScriptedAsrEngine {
    fn transcribe(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        self.call_audio_lens
            .lock()
            .expect("ASR call lengths should be writable")
            .push(samples.len());
        self.transcripts
            .pop_front()
            .ok_or_else(|| anyhow!("scripted ASR engine was exhausted"))
    }
}

fn vad(is_speech: bool) -> VadResult {
    VadResult {
        probability: if is_speech { 0.9 } else { 0.1 },
        is_speech,
    }
}

fn fixed_vad_frame(sample: f32, len: usize, is_speech: bool) -> (Vec<f32>, VadResult) {
    (vec![sample; len], vad(is_speech))
}

struct RuntimeStateBuilder<'a> {
    runtime: &'a mut RecognitionSession,
}

fn runtime_state(runtime: &mut RecognitionSession) -> RuntimeStateBuilder<'_> {
    RuntimeStateBuilder { runtime }
}

impl RuntimeStateBuilder<'_> {
    fn pending_segment(
        self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        reason: SegmentCloseReason,
        range: std::ops::Range<u64>,
    ) -> Self {
        self.runtime.pending.asr_segments.push_back(pending_segment(
            segment_id,
            previous_segment_id,
            reason,
            range,
        ));
        self
    }

    fn turn(self, turn_id: u64, turn: Turn) -> Self {
        self.runtime.turn_store.turns.insert(turn_id, turn);
        self
    }

    fn turn_audio_range(self, turn_id: u64, range: std::ops::Range<u64>) -> Self {
        self.runtime.turn_store.audio_ranges.insert(
            turn_id,
            AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
        );
        self
    }

    fn open_turn(self, turn_id: u64) -> Self {
        self.runtime.turn_store.open_turn_id = Some(turn_id);
        self.runtime.turn_store.open_turn_accepts_root_segment = true;
        self
    }

    fn open_turn_since(self, turn_id: u64, since_tick: u64) -> Self {
        let activity_epoch = self.runtime.activity.segment_activity_epoch;
        self.runtime.turn_store.open_turn_id = Some(turn_id);
        self.runtime.turn_store.open_turn_accepts_root_segment = true;
        self.runtime.activity.open_turn_since_tick = Some(since_tick);
        self.runtime.activity.open_turn_activity_epoch = activity_epoch;
        self
    }

    fn pending_turn_check(self, previous_segment_id: u64) -> Self {
        self.runtime.pending.turn_check = Some(PendingTurnCheck {
            previous_segment_id,
            activity_epoch: self.runtime.activity.segment_activity_epoch,
        });
        self
    }

    fn in_flight(self, request: AsrRequest) -> Self {
        self.runtime.requests.in_flight_request = Some(request);
        self
    }

    fn turn_revision(self, turn_id: u64, revision: u64) -> Self {
        self.runtime.turn_store.revisions.insert(turn_id, revision);
        self
    }

    fn last_recognition_route(self, route: RecognitionRoute) -> Self {
        self.runtime.turn_store.last_recognition_route = Some(route);
        self
    }

    fn next_runtime_tick(self, tick: u64) -> Self {
        self.runtime.counters.next_runtime_tick = tick;
        self
    }
}

struct RecognitionSessionTestBuilder {
    config: ParapperConfig,
    turn_session_id: u64,
    asr_runner: Box<dyn AsrRequestRunner>,
    turn_decision_runner: Box<dyn TurnDecisionRunner>,
    output_sink: Box<dyn TurnOutputSink>,
    language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
    language_id: Option<Box<dyn LanguageDetector>>,
}

impl RecognitionSessionTestBuilder {
    fn new() -> Self {
        Self {
            config: ParapperConfig::default(),
            turn_session_id: 1,
            asr_runner: Box::new(super::NoopAsrRequestRunner),
            turn_decision_runner: Box::new(super::NoopTurnDecisionRunner),
            output_sink: Box::new(super::NoopTurnOutputSink),
            language_id_runtime: None,
            language_id: None,
        }
    }

    // --- Main 4 flag axes (TurnDetector / interim / multilingual / rerec_full) ---

    fn turn_detector(mut self, td: TurnDetector) -> Self {
        self.config.turn.detector = td;
        self
    }

    fn interim_display(mut self, on: bool) -> Self {
        self.config.turn.interim_result_enabled = on;
        self
    }

    fn multilingual(mut self, on: bool) -> Self {
        self.config.asr.multilingual_enabled = on;
        self
    }

    fn rerecognize_full_on_complete(mut self, on: bool) -> Self {
        self.config.turn.rerecognize_full_on_complete = on;
        self
    }

    // --- Secondary config setters ---

    fn asr_model(mut self, model: AsrModel) -> Self {
        self.config.asr.model = model;
        self.config.asr.language = model.language();
        self
    }

    fn interim_asr_model(mut self, model: AsrModel) -> Self {
        self.config.asr.interim_model = Some(model);
        self
    }

    fn asr_language(mut self, lang: AsrLanguage) -> Self {
        self.config.asr.language = lang;
        self
    }

    fn enabled_asr_models(mut self, models: Vec<AsrModel>) -> Self {
        self.config.asr.enabled_models = models;
        self
    }

    fn vad_interval_ms(mut self, ms: u32) -> Self {
        self.config.segmentation.vad_interval_ms = ms;
        self
    }

    fn segment_start_speech_ms(mut self, ms: u32) -> Self {
        self.config.segmentation.segment_start_speech_ms = ms;
        self
    }

    fn interim_result_silence_ms(mut self, ms: u32) -> Self {
        self.config.turn.interim_result_silence_ms = ms;
        self
    }

    fn turn_check_silence_ms(mut self, ms: u32) -> Self {
        self.config.turn.check_silence_ms = ms;
        self
    }

    fn namo_turn_confidence_threshold(mut self, t: f32) -> Self {
        self.config.turn.namo_confidence_threshold = t;
        self
    }

    fn turn_session_id(mut self, id: u64) -> Self {
        self.turn_session_id = id;
        self
    }

    fn config_mut(&mut self) -> &mut ParapperConfig {
        &mut self.config
    }

    // --- IO injection (handle-returning setters use &mut self) ---

    fn asr_runner(mut self, runner: Box<dyn AsrRequestRunner>) -> Self {
        self.asr_runner = runner;
        self
    }

    fn scripted_asr_texts(mut self, texts: Vec<&str>) -> Self {
        self.asr_runner = Box::new(ScriptedAsrRunner::from_texts(texts));
        self
    }

    fn scripted_asr_transcripts(mut self, transcripts: Vec<AsrTranscript>) -> Self {
        self.asr_runner = Box::new(ScriptedAsrRunner::from_transcripts(transcripts));
        self
    }

    fn use_manual_asr(&mut self) -> ManualAsrHandle {
        let (runner, handle) = ManualAsrRunner::new();
        self.asr_runner = Box::new(runner);
        handle
    }

    fn use_scripted_decisions(&mut self, decisions: Vec<TurnDecision>) -> Arc<Mutex<Vec<String>>> {
        let texts = Arc::new(Mutex::new(Vec::new()));
        self.turn_decision_runner =
            Box::new(ScriptedTurnDecisionRunner::new(decisions, texts.clone()));
        texts
    }

    fn turn_decision_runner(mut self, runner: Box<dyn TurnDecisionRunner>) -> Self {
        self.turn_decision_runner = runner;
        self
    }

    fn output_sink(mut self, sink: Box<dyn TurnOutputSink>) -> Self {
        self.output_sink = sink;
        self
    }

    fn use_recording_sink(&mut self) -> Arc<Mutex<Vec<OutputSnapshot>>> {
        let outputs = Arc::new(Mutex::new(Vec::new()));
        self.output_sink = Box::new(RecordingOutputSink {
            outputs: outputs.clone(),
        });
        outputs
    }

    fn use_recording_phrase_sink(&mut self) -> Arc<Mutex<Vec<PhraseOutputSnapshot>>> {
        let outputs = Arc::new(Mutex::new(Vec::new()));
        self.output_sink = Box::new(RecordingPhraseOutputSink {
            outputs: outputs.clone(),
        });
        outputs
    }

    fn language_id_runtime(mut self) -> Self {
        self.language_id_runtime = Some(Box::new(TestLanguageIdRuntime));
        self
    }

    fn use_scripted_language_detector(&mut self, languages: Vec<&str>) -> Arc<Mutex<Vec<usize>>> {
        if self.language_id_runtime.is_none() {
            self.language_id_runtime = Some(Box::new(TestLanguageIdRuntime));
        }
        let call_audio_lens = Arc::new(Mutex::new(Vec::new()));
        self.language_id = Some(Box::new(ScriptedLanguageDetector {
            detected_languages: languages.into_iter().map(String::from).collect(),
            call_audio_lens: call_audio_lens.clone(),
        }));
        call_audio_lens
    }

    fn build(self) -> (RecognitionDriver, ParapperConfig) {
        let runtime = RecognitionSession::new_for_test_with_all_io(
            &self.config,
            self.turn_session_id,
            self.asr_runner,
            self.turn_decision_runner,
            self.output_sink,
            self.language_id_runtime,
            self.language_id,
        );
        (RecognitionDriver::new(runtime, &self.config), self.config)
    }
}

fn japanese_punctuation_transcript() -> AsrTranscript {
    AsrTranscript::from_parts(
        "はい。次です".to_string(),
        vec![
            "は".to_string(),
            "い".to_string(),
            "。".to_string(),
            "次".to_string(),
            "で".to_string(),
            "す".to_string(),
        ],
        Some(&[
            0.0,
            1.0 / 16_000.0,
            2.0 / 16_000.0,
            3.0 / 16_000.0,
            4.0 / 16_000.0,
            5.0 / 16_000.0,
        ]),
        Some(&[
            1.0 / 16_000.0,
            1.0 / 16_000.0,
            1.0 / 16_000.0,
            1.0 / 16_000.0,
            1.0 / 16_000.0,
            1.0 / 16_000.0,
        ]),
    )
}

fn pending_segment(
    segment_id: u64,
    previous_segment_id: Option<u64>,
    reason: SegmentCloseReason,
    range: std::ops::Range<u64>,
) -> PendingAsrSegment {
    let audio_len = usize::try_from(range.end.saturating_sub(range.start))
        .expect("test segment range should fit usize");
    let sample = f32::from(u16::try_from(segment_id).expect("test segment id should fit u16"));
    let audio = vec![sample; audio_len];
    let vad_results = vec![vad(true)];
    PendingAsrSegment {
        segment_id,
        previous_segment_id,
        source_audio: audio.clone(),
        source_vad_results: vad_results.clone(),
        audio,
        vad_results,
        reason,
        range: AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
        created_at_frame: VadFrameIndex(segment_id),
    }
}

fn recognized_turn_with_audio(turn_id: u64, text: &str, audio: &[f32]) -> Turn {
    let mut turn = Turn::new(format!("turn-1-{turn_id}-0"), 0);
    let vad_results = vec![vad(true); audio.len().max(1)];
    turn.draft_mut().append_recognized_segment(
        turn_id,
        None,
        audio,
        &vad_results,
        RecognitionRoute::from_language(AsrLanguage::Japanese),
        text.to_string(),
        0,
    );
    turn
}

fn recognized_turn_with_vad(
    turn_id: u64,
    text: &str,
    audio: &[f32],
    vad_results: &[VadResult],
) -> Turn {
    let mut turn = Turn::new(format!("turn-1-{turn_id}-0"), 0);
    turn.draft_mut().append_recognized_segment(
        turn_id,
        None,
        audio,
        vad_results,
        RecognitionRoute::from_language(AsrLanguage::Japanese),
        text.to_string(),
        0,
    );
    turn
}

fn recognized_turn_with_boundary_candidates(
    turn_id: u64,
    text: &str,
    audio: &[f32],
    vad_results: &[VadResult],
    boundary_candidates: Vec<TurnBoundaryCandidate>,
) -> Turn {
    let mut turn = recognized_turn_with_vad(turn_id, text, audio, vad_results);
    turn.draft_mut().boundary_candidates = boundary_candidates;
    turn
}

fn boundary_candidate(
    char_end_text: &str,
    sample_end: usize,
    prefix_audio_end: usize,
    suffix_audio_start: usize,
    class: GrammarBoundaryClass,
) -> TurnBoundaryCandidate {
    TurnBoundaryCandidate {
        char_end: char_end_text.chars().count(),
        sample_end,
        prefix_audio_end,
        suffix_audio_start,
        class,
    }
}

fn interim_request_for_turn(request_id: u64, turn_id: u64) -> AsrRequest {
    AsrRequest {
        request_id: AsrRequestId(request_id),
        kind: AsrTaskKind::InterimDisplay,
        target: AsrTarget::new(
            TurnId(turn_id),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(1)),
            Some(SegmentId(turn_id)),
            Some(SegmentId(turn_id)),
        ),
        route: RecognitionRoute::from_model(ParapperConfig::default().asr.model),
        detected_language: None,
        audio: vec![1.0],
        vad_results: vec![vad(true)],
        source_audio: vec![1.0],
        source_vad_results: vec![vad(true)],
        close_reason: Some(SegmentCloseReason::InterimResultSilenceReached),
        created_at_frame: VadFrameIndex(1),
    }
}

pub(crate) fn tauri_test_handle() -> tauri::AppHandle {
    let builder = tauri::Builder::default();
    #[cfg(any(windows, target_os = "linux"))]
    let builder = builder.any_thread();
    let app = builder
        .build(tauri::test::mock_context(tauri::test::noop_assets()))
        .expect("test app should build");
    app.handle().clone()
}

fn test_env_path(key: &str) -> PathBuf {
    std::env::var_os(key).map_or_else(
        || {
            control_test_dotenv().get(key).map_or_else(
                || {
                    panic!(
                        "{key} must be set in the process environment or a local .env file for this diagnostic test"
                    )
                },
                PathBuf::from,
            )
        },
        PathBuf::from,
    )
}

fn diagnostic_models_root() -> PathBuf {
    std::env::var_os("PARAPPER_MODELS_ROOT")
        .map_or_else(|| test_app_data_dir().join("models"), PathBuf::from)
}

fn test_app_data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .map(|path| path.join("com.parakeet-inc.parapper"))
            .expect("APPDATA must be set or PARAPPER_MODELS_ROOT must be provided")
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|path| {
                path.join("Library")
                    .join("Application Support")
                    .join("com.parakeet-inc.parapper")
            })
            .expect("HOME must be set or PARAPPER_MODELS_ROOT must be provided")
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        if let Some(path) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from) {
            return path.join("com.parakeet-inc.parapper");
        }
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|path| {
                path.join(".local")
                    .join("share")
                    .join("com.parakeet-inc.parapper")
            })
            .expect("XDG_DATA_HOME, HOME, or PARAPPER_MODELS_ROOT must be provided")
    }
}

fn control_test_dotenv() -> &'static HashMap<String, String> {
    static ENV: OnceLock<HashMap<String, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        control_test_dotenv_paths()
            .into_iter()
            .find_map(|path| path.is_file().then(|| parse_dotenv_file(&path)))
            .unwrap_or_default()
    })
}

fn control_test_dotenv_paths() -> [PathBuf; 2] {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        manifest_dir
            .parent()
            .expect("src-tauri should have a workspace parent")
            .join(".env"),
        manifest_dir.join(".env"),
    ]
}

fn parse_dotenv_file(path: &Path) -> HashMap<String, String> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    parse_dotenv_contents(&contents)
}

fn parse_dotenv_contents(contents: &str) -> HashMap<String, String> {
    contents
        .lines()
        .filter_map(parse_dotenv_line)
        .collect::<HashMap<_, _>>()
}

fn parse_dotenv_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line);
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), unquote_dotenv_value(value.trim())))
}

fn unquote_dotenv_value(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(value)
        .to_string()
}

#[test]
fn control_test_dotenv_parser_keeps_fleurs_windows_path() {
    let env = parse_dotenv_contents(
        r#"
        # local datasets
        FLEURS_R_ROOT=.\datasets\fleurs-r
        QUOTED=".\datasets with spaces\fleurs-r"
        export JVS_ROOT=.\datasets\jvs\jvs_ver1
        "#,
    );

    assert_eq!(
        env.get("FLEURS_R_ROOT").map(String::as_str),
        Some(r".\datasets\fleurs-r")
    );
    assert_eq!(
        env.get("QUOTED").map(String::as_str),
        Some(r".\datasets with spaces\fleurs-r")
    );
    assert_eq!(
        env.get("JVS_ROOT").map(String::as_str),
        Some(r".\datasets\jvs\jvs_ver1")
    );
}

struct JvsPart {
    id: String,
    text: String,
    samples: Vec<f32>,
    sample_rate: u32,
}

struct FleursPart {
    locale: String,
    wav_path: PathBuf,
    samples: Vec<f32>,
    sample_rate: u32,
}

fn read_jvs_nonparallel_part(id: &str) -> JvsPart {
    let jvs_root = test_env_path("JVS_ROOT");
    let nonpara = jvs_root.join("jvs001").join("nonpara30");
    assert!(
        nonpara.is_dir(),
        "JVS nonparallel directory does not exist: {}",
        nonpara.display()
    );
    let text = read_jvs_transcript(&nonpara.join("transcripts_utf8.txt"), id);
    let wav = read_pcm16_wav_mono_f32(&nonpara.join("wav24kHz16bit").join(format!("{id}.wav")));
    JvsPart {
        id: id.to_string(),
        text,
        samples: wav.0,
        sample_rate: wav.1,
    }
}

fn read_jvs_transcript(path: &Path, id: &str) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(line_id, text)| (line_id == id).then(|| text.to_string()))
        .unwrap_or_else(|| panic!("JVS transcript id {id} was not found in {}", path.display()))
}

fn read_short_fleurs_dev_parts(locale: &str, count: usize) -> Vec<FleursPart> {
    let fleurs_root = test_env_path("FLEURS_R_ROOT");
    let split_dir = fleurs_root.join(locale).join("dev").join("dev");
    assert!(
        split_dir.is_dir(),
        "FLEURS-R dev wav directory does not exist: {}",
        split_dir.display()
    );
    let mut wav_paths = fs::read_dir(&split_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", split_dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        })
        .collect::<Vec<_>>();
    wav_paths.sort_by_key(|path| {
        fs::metadata(path)
            .unwrap_or_else(|err| panic!("failed to stat {}: {err}", path.display()))
            .len()
    });

    wav_paths
        .into_iter()
        .take(count)
        .map(|wav_path| read_fleurs_part(locale, wav_path))
        .collect()
}

fn read_fleurs_part(locale: &str, wav_path: PathBuf) -> FleursPart {
    let (samples, sample_rate) = read_pcm16_wav_mono_f32(&wav_path);
    let samples = resample_linear_for_test(&samples, sample_rate, crate::audio::ASR_SAMPLE_RATE);
    FleursPart {
        locale: locale.to_string(),
        wav_path,
        samples,
        sample_rate: crate::audio::ASR_SAMPLE_RATE,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test WAV resampling converts bounded sample positions between integer indices and fractional interpolation weights"
)]
fn resample_linear_for_test(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let target_len = (samples.len() as u128 * u128::from(target_rate))
        .div_ceil(u128::from(source_rate)) as usize;
    (0..target_len)
        .map(|index| {
            let position = index as f64 * f64::from(source_rate) / f64::from(target_rate);
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len().saturating_sub(1));
            let fraction = (position - left as f64) as f32;
            samples.get(left).copied().unwrap_or(0.0) * (1.0 - fraction)
                + samples.get(right).copied().unwrap_or(0.0) * fraction
        })
        .collect()
}

fn read_pcm16_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let bytes =
        fs::read(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert!(bytes.len() >= 12, "wav is too short: {}", path.display());
    assert_eq!(
        &bytes[0..4],
        b"RIFF",
        "wav must be RIFF: {}",
        path.display()
    );
    assert_eq!(
        &bytes[8..12],
        b"WAVE",
        "wav must be WAVE: {}",
        path.display()
    );

    let mut cursor = 12;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = usize::try_from(u32::from_le_bytes(
            bytes[cursor + 4..cursor + 8]
                .try_into()
                .expect("chunk size should have 4 bytes"),
        ))
        .expect("wav chunk size should fit usize");
        cursor += 8;
        let chunk_end = cursor.saturating_add(chunk_size).min(bytes.len());
        match chunk_id {
            b"fmt " => {
                assert!(
                    chunk_size >= 16,
                    "fmt chunk is too small in {}",
                    path.display()
                );
                let audio_format = u16::from_le_bytes(
                    bytes[cursor..cursor + 2]
                        .try_into()
                        .expect("audio format should have 2 bytes"),
                );
                assert_eq!(audio_format, 1, "wav must be PCM: {}", path.display());
                channels = Some(u16::from_le_bytes(
                    bytes[cursor + 2..cursor + 4]
                        .try_into()
                        .expect("channels should have 2 bytes"),
                ));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[cursor + 4..cursor + 8]
                        .try_into()
                        .expect("sample rate should have 4 bytes"),
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    bytes[cursor + 14..cursor + 16]
                        .try_into()
                        .expect("bits per sample should have 2 bytes"),
                ));
            }
            b"data" => {
                data = Some(cursor..chunk_end);
            }
            _ => {}
        }
        cursor = chunk_end + (chunk_size % 2);
    }

    let channels = channels.expect("wav fmt chunk should define channels");
    let channel_count = usize::from(channels);
    let sample_rate = sample_rate.expect("wav fmt chunk should define sample rate");
    assert_eq!(
        bits_per_sample,
        Some(16),
        "wav must be 16-bit PCM: {}",
        path.display()
    );
    let data = data.unwrap_or_else(|| panic!("wav data chunk not found: {}", path.display()));
    let frame_bytes = channel_count * 2;
    let mut samples = Vec::with_capacity((data.end - data.start) / frame_bytes);
    for frame in bytes[data].chunks_exact(frame_bytes) {
        let mut sum = 0.0_f32;
        for channel in 0..channel_count {
            let offset = channel * 2;
            let sample = i16::from_le_bytes(
                frame[offset..offset + 2]
                    .try_into()
                    .expect("PCM16 sample should have 2 bytes"),
            );
            sum += f32::from(sample) / 32768.0;
        }
        samples.push(sum / f32::from(channels));
    }
    (samples, sample_rate)
}

fn push_jvs_speech_chunks(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    samples: &[f32],
    sample_rate: u32,
) {
    let chunk_len = frames_for_millis(sample_rate, config.segmentation.vad_interval_ms);
    for chunk in samples.chunks(chunk_len) {
        runtime.push_vad_frame(chunk, vad(true));
        runtime.step();
    }
}

fn push_fleurs_speech_chunks(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    part: &FleursPart,
) {
    push_jvs_speech_chunks(runtime, config, &part.samples, part.sample_rate);
}

fn push_silence_chunks(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    sample_rate: u32,
    chunks: usize,
) {
    let chunk_len = frames_for_millis(sample_rate, config.segmentation.vad_interval_ms);
    let silence = vec![0.0; chunk_len];
    for _ in 0..chunks {
        runtime.push_vad_frame(&silence, vad(false));
        runtime.step();
    }
}

fn frames_for_millis(sample_rate: u32, millis: u32) -> usize {
    usize::try_from((u64::from(sample_rate) * u64::from(millis)).div_ceil(1000))
        .expect("test sample count should fit usize")
}

fn assert_output_phrase_contains_jvs_parts(output: &PhraseOutputSnapshot, parts: &[JvsPart]) {
    let mut search_from = 0;
    for part in parts {
        assert!(
            output.text.contains(part.text.trim_end_matches('。'))
                || output.text.contains(&part.text),
            "UI text for turn {} segment {} should keep JVS part {} visible\ntext: {}",
            output.turn_id,
            output.segment_id,
            part.id,
            output.text
        );
        let fingerprint = jvs_audio_fingerprint(part);
        let position = find_subsequence_approx(&output.phrase, &fingerprint, search_from)
                .unwrap_or_else(|| {
                    panic!(
                        "UI phrase audio for turn {} segment {} is missing JVS part {} ({} samples); output phrase has {} samples",
                        output.turn_id,
                        output.segment_id,
                        part.id,
                        part.samples.len(),
                        output.phrase.len()
                    )
                });
        search_from = position + fingerprint.len();
    }
}

fn assert_output_phrase_contains_fleurs_parts(output: &PhraseOutputSnapshot, parts: &[FleursPart]) {
    let mut search_from = 0;
    for part in parts {
        for fingerprint in fleurs_audio_fingerprints(part) {
            let position = find_subsequence_approx(&output.phrase, &fingerprint, search_from)
                .unwrap_or_else(|| {
                    panic!(
                        "UI phrase audio for turn {} segment {} is missing FLEURS part {} ({}, {} samples); output phrase has {} samples",
                        output.turn_id,
                        output.segment_id,
                        part.locale,
                        part.wav_path.display(),
                        part.samples.len(),
                        output.phrase.len()
                    )
                });
            search_from = position + fingerprint.len();
        }
    }
}

fn jvs_audio_fingerprint(part: &JvsPart) -> Vec<f32> {
    let len = part.samples.len().min(2048);
    assert!(len > 0, "JVS part {} should not be empty", part.id);
    let start = part.samples.len() / 2 - len / 2;
    part.samples[start..start + len].to_vec()
}

fn fleurs_audio_fingerprints(part: &FleursPart) -> [Vec<f32>; 2] {
    [
        fleurs_audio_fingerprint_at(part, 1, 4),
        fleurs_audio_fingerprint_at(part, 1, 2),
    ]
}

fn fleurs_audio_fingerprint_at(
    part: &FleursPart,
    numerator: usize,
    denominator: usize,
) -> Vec<f32> {
    let len = part.samples.len().min(2048);
    assert!(
        len > 0,
        "FLEURS part {} should not be empty",
        part.wav_path.display()
    );
    let center = part.samples.len().saturating_mul(numerator) / denominator.max(1);
    let start = center
        .saturating_sub(len / 2)
        .min(part.samples.len().saturating_sub(len));
    part.samples[start..start + len].to_vec()
}

fn find_subsequence_approx(haystack: &[f32], needle: &[f32], start: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() || start > haystack.len() - needle.len() {
        return None;
    }
    (start..=haystack.len() - needle.len()).find(|&index| {
        haystack[index..index + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-6)
    })
}

#[test]
fn replay_vad_frames_for_runtime_preserves_fifo_order_for_test_harness() {
    let mut runtime = RecordingRecognitionSession::default();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &ParapperConfig::default(),
        vec![(vec![1.0], vad(true)), (vec![2.0], vad(false))],
    );

    assert_eq!(
        runtime.calls,
        vec![
            RuntimeCall::UpdateConfig,
            RuntimeCall::VadFrame {
                sample: 1.0,
                is_speech: true,
            },
            RuntimeCall::Step,
            RuntimeCall::VadFrame {
                sample: 2.0,
                is_speech: false,
            },
            RuntimeCall::Step,
        ]
    );
}
