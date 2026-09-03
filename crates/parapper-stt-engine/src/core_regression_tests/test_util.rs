use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};

use parapper_models::{
    asr::{AsrLanguage, AsrModel, AsrTranscript},
    td::TurnBoundaryCandidate,
};

use crate::{
    RecognitionDriver, RecognitionSession, RecognizedTurn as RecognizedTextOutput,
    SegmentCloseReason, SegmentId, SourceIdentitySnapshot, SttEngineConfig, TurnId, VadResult,
    ports::{
        AsrRequestRunner, LanguageDetectionWarningSink, LanguageDetector,
        RecognitionOutputSink as TurnOutputSink, TranscriptBoundaryDetector, TurnDecisionRunner,
    },
    transcription::{
        planner::PendingAsrSegment,
        preprocess::MIN_LANGUAGE_ID_SAMPLES,
        reducer::AsrResultRerecognitionPurpose as RerecognitionPurpose,
        route::RecognitionRoute,
        task::{
            AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AsrTarget, AsrTaskKind,
            AudioRange, GlobalSampleIndex, TurnRevision, VadFrameIndex,
        },
    },
    runtime::PendingTurnCheck,
    turn::{GrammarBoundaryClass, Turn, TurnDecision, TurnDetector},
};

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
    source_session: crate::SourceSessionKey,
    source_identity: crate::SourceIdentitySnapshot,
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
            source_session: output.meta.source().source_session_key(),
            source_identity: output.meta.source().identity.clone(),
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

fn vad(is_speech: bool) -> VadResult {
    VadResult {
        probability: if is_speech { 0.9 } else { 0.1 },
        is_speech,
    }
}

fn fixed_vad_frame(sample: f32, len: usize, is_speech: bool) -> (Vec<f32>, VadResult) {
    (vec![sample; len], vad(is_speech))
}

struct NoopAsrRequestRunner;

impl AsrRequestRunner for NoopAsrRequestRunner {
    fn submit(&mut self, _request: AsrRequest) -> bool {
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        None
    }
}

struct NoopTurnDecisionRunner;

impl TurnDecisionRunner for NoopTurnDecisionRunner {
    fn decide(
        &mut self,
        _route: RecognitionRoute,
        _text: &str,
        _max_context_tokens: u32,
    ) -> Result<TurnDecision> {
        Ok(TurnDecision {
            is_end_of_turn: false,
            confidence: 0.0,
        })
    }
}

struct NoopTurnOutputSink;

impl TurnOutputSink for NoopTurnOutputSink {
    fn emit(&mut self, _output: RecognizedTextOutput) {}
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
    config: SttEngineConfig,
    turn_session_id: u64,
    source_identity: Option<SourceIdentitySnapshot>,
    asr_runner: Box<dyn AsrRequestRunner>,
    turn_decision_runner: Box<dyn TurnDecisionRunner>,
    output_sink: Box<dyn TurnOutputSink>,
    language_id_runtime: Option<Box<dyn LanguageDetectionWarningSink>>,
    language_id: Option<Box<dyn LanguageDetector>>,
}

impl RecognitionSessionTestBuilder {
    fn new() -> Self {
        Self {
            config: SttEngineConfig::default(),
            turn_session_id: 1,
            source_identity: None,
            asr_runner: Box::new(NoopAsrRequestRunner),
            turn_decision_runner: Box::new(NoopTurnDecisionRunner),
            output_sink: Box::new(NoopTurnOutputSink),
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

    fn source_identity(mut self, identity: SourceIdentitySnapshot) -> Self {
        self.source_identity = Some(identity);
        self
    }

    fn config_mut(&mut self) -> &mut SttEngineConfig {
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

    fn build(self) -> (RecognitionDriver, SttEngineConfig) {
        let runtime = if let Some(source_identity) = self.source_identity {
            RecognitionSession::with_ports_and_source_identity(
                self.config.clone(),
                self.turn_session_id,
                source_identity,
                self.asr_runner,
                self.turn_decision_runner,
                self.output_sink,
                self.language_id_runtime,
                self.language_id,
                Some(Box::new(TestTranscriptBoundaryDetector)),
            )
        } else {
            RecognitionSession::with_ports(
                self.config.clone(),
                self.turn_session_id,
                self.asr_runner,
                self.turn_decision_runner,
                self.output_sink,
                self.language_id_runtime,
                self.language_id,
                Some(Box::new(TestTranscriptBoundaryDetector)),
            )
        };
        (RecognitionDriver::new(runtime, &self.config), self.config)
    }
}

struct TestTranscriptBoundaryDetector;

impl TranscriptBoundaryDetector for TestTranscriptBoundaryDetector {
    fn candidates_for_transcript(
        &self,
        language: AsrLanguage,
        transcript: &AsrTranscript,
        audio: &[f32],
        vad_results: &[VadResult],
    ) -> Vec<TurnBoundaryCandidate> {
        parapper_models::td::candidates_for_transcript_without_morph(
            language,
            transcript,
            audio,
            vad_results,
        )
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
        route: RecognitionRoute::from_model(SttEngineConfig::default().asr.model),
        detected_language: None,
        audio: vec![1.0],
        vad_results: vec![vad(true)],
        source_audio: vec![1.0],
        source_vad_results: vec![vad(true)],
        close_reason: Some(SegmentCloseReason::InterimResultSilenceReached),
        created_at_frame: VadFrameIndex(1),
    }
}

fn push_silence_chunks(
    runtime: &mut RecognitionDriver,
    config: &SttEngineConfig,
    sample_rate: u32,
    chunks: usize,
) {
    let chunk_len = usize::try_from(sample_rate).expect("sample rate should fit usize")
        * usize::try_from(config.segmentation.vad_interval_ms)
            .expect("VAD interval should fit usize")
        / 1_000;
    for _ in 0..chunks {
        runtime.push_vad_frame(&vec![0.0; chunk_len], vad(false));
        runtime.step();
    }
}
