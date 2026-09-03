use std::{
    collections::VecDeque,
    ops::Range,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use parapper_models::asr::{
    AsrEngine, AsrLanguage, AsrModel, AsrSpeechRangeSamples, AsrStreamConfig, AsrTranscript,
    StreamingSessionId,
};

use super::{AsrExecutionRuntime, AsrModelRegistry};
use crate::{
    SegmentCloseReason, SegmentId, SourceId, SourceSessionKey, SttAsrConfig, TurnId, VadResult,
    transcription::{
        route::RecognitionRoute,
        task::{
            AsrRequest, AsrRequestId, AsrTarget, AsrTaskKind, AudioRange, GlobalSampleIndex,
            TurnRevision, VadFrameIndex,
        },
    },
};

const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
const NEMOTRON_MODEL: AsrModel = AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8;

#[derive(Default)]
struct Calls {
    starts: Vec<StreamingSessionId>,
    stream_configs: Vec<AsrStreamConfig>,
    streaming_audio: Vec<Vec<f32>>,
    offline_audio: Vec<Vec<f32>>,
    finishes: Vec<StreamingSessionId>,
    cancelled: Vec<StreamingSessionId>,
    streaming_failures: VecDeque<String>,
}

struct RecordingEngine {
    calls: Arc<Mutex<Calls>>,
    transcript: AsrTranscript,
}

impl AsrEngine for RecordingEngine {
    fn recognize(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        self.calls
            .lock()
            .unwrap()
            .offline_audio
            .push(samples.to_vec());
        Ok(self.transcript.clone())
    }

    fn start_stream(&mut self, session: StreamingSessionId, config: AsrStreamConfig) -> Result<()> {
        let mut calls = self.calls.lock().unwrap();
        calls.starts.push(session);
        calls.stream_configs.push(config);
        Ok(())
    }

    fn push_stream(
        &mut self,
        _session: StreamingSessionId,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        let mut calls = self.calls.lock().unwrap();
        calls.streaming_audio.push(samples.to_vec());
        if let Some(reason) = calls.streaming_failures.pop_front() {
            return Err(anyhow!(reason));
        }
        Ok(self.transcript.clone())
    }

    fn finish_stream(&mut self, session: StreamingSessionId) -> Result<AsrTranscript> {
        self.calls.lock().unwrap().finishes.push(session);
        Ok(self.transcript.clone())
    }

    fn cancel_stream(&mut self, session: StreamingSessionId) {
        self.calls.lock().unwrap().cancelled.push(session);
    }
}

#[test]
fn streaming_requests_start_once_and_feed_only_raw_source_pcm_deltas() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let config = asr_config(NEMOTRON_MODEL, false);
    let first = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let second = nemotron_interim_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64, 2.0);

    runtime.execute(&config, &first).unwrap();
    runtime.execute(&config, &second).unwrap();

    let calls = calls.lock().unwrap();
    assert_eq!(calls.starts, vec![StreamingSessionId::new(1, Some(1))]);
    assert_eq!(
        calls.stream_configs,
        vec![AsrStreamConfig {
            speech_range_samples: Some(AsrSpeechRangeSamples {
                start: 0,
                end: NEMOTRON_CHUNK_SAMPLES,
            }),
            language_hint: None,
        }]
    );
    assert_eq!(calls.streaming_audio.len(), 2);
    assert_eq!(calls.streaming_audio[0], vec![1.0; NEMOTRON_CHUNK_SAMPLES]);
    assert_eq!(calls.streaming_audio[1], vec![2.0; NEMOTRON_CHUNK_SAMPLES]);
    assert!(calls.offline_audio.is_empty());
}

#[test]
fn multilingual_nemotron_receives_japanese_hint_for_japanese_completion_without_language_id() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let config = asr_config(AsrModel::ReazonSpeechK2V2, false);
    let request = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);

    runtime.execute(&config, &request).unwrap();

    let actual = calls.lock().unwrap().stream_configs.clone();
    drop(runtime);
    assert_eq!(
        actual,
        vec![AsrStreamConfig {
            speech_range_samples: Some(AsrSpeechRangeSamples {
                start: 0,
                end: NEMOTRON_CHUNK_SAMPLES,
            }),
            language_hint: Some(AsrLanguage::Japanese),
        }]
    );
}

#[test]
fn multilingual_nemotron_keeps_auto_prompt_when_language_id_is_enabled() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let mut config = asr_config(AsrModel::ReazonSpeechK2V2, false);
    config.multilingual_enabled = true;
    let request = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);

    runtime.execute(&config, &request).unwrap();

    let actual = calls.lock().unwrap().stream_configs.clone();
    drop(runtime);
    assert_eq!(
        actual,
        vec![AsrStreamConfig {
            speech_range_samples: Some(AsrSpeechRangeSamples {
                start: 0,
                end: NEMOTRON_CHUNK_SAMPLES,
            }),
            language_hint: None,
        }]
    );
}

#[test]
fn failed_streaming_delta_cancels_the_model_session_without_replaying_source_audio() {
    let (mut runtime, calls) =
        streaming_runtime(VecDeque::from(["streaming result unavailable".to_string()]));
    let config = asr_config(NEMOTRON_MODEL, false);
    let first = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let second = nemotron_interim_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64, 2.0);

    assert!(runtime.execute(&config, &first).is_err());
    runtime.execute(&config, &second).unwrap();

    let calls = calls.lock().unwrap();
    assert_eq!(calls.cancelled, vec![StreamingSessionId::new(1, Some(1))]);
    assert_eq!(
        calls.starts,
        vec![
            StreamingSessionId::new(1, Some(1)),
            StreamingSessionId::new(1, Some(1))
        ]
    );
    assert_eq!(calls.streaming_audio[1], vec![2.0; NEMOTRON_CHUNK_SAMPLES]);
}

#[test]
fn nemotron_completion_request_cancels_interim_but_is_not_promoted_to_offline_recognition() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let config = asr_config(NEMOTRON_MODEL, false);
    let interim = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let completion = nemotron_completion_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64);

    runtime.execute(&config, &interim).unwrap();
    let error = runtime.execute(&config, &completion).unwrap_err();

    let calls = calls.lock().unwrap();
    assert_eq!(calls.cancelled, vec![StreamingSessionId::new(1, Some(1))]);
    assert!(calls.offline_audio.is_empty());
    assert!(error.to_string().contains("interim-only"));
}

#[test]
fn source_reset_cancels_only_that_source_and_keeps_other_stream_active() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let config = asr_config(NEMOTRON_MODEL, false);
    let source_a = SourceSessionKey::new(7, SourceId::from("source-a"));
    let source_b = SourceSessionKey::new(7, SourceId::from("source-b"));
    let mut first_a = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let mut first_b = first_a.clone();
    first_a.target.source_session = source_a.clone();
    first_b.target.source_session = source_b.clone();

    runtime.execute(&config, &first_a).unwrap();
    runtime.execute(&config, &first_b).unwrap();
    runtime.reset_streaming_sessions_for_source(&source_a);

    let cancelled_after_a_reset = calls.lock().unwrap().cancelled.clone();
    assert_eq!(cancelled_after_a_reset.len(), 1);
    let starts = calls.lock().unwrap().starts.clone();
    assert_ne!(starts[0], starts[1]);

    let mut second_b = nemotron_interim_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64, 2.0);
    second_b.target.source_session = source_b;
    runtime.execute(&config, &second_b).unwrap();
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.starts.len(),
        2,
        "source B stream must not be restarted by source A reset"
    );
    assert_eq!(calls.streaming_audio[2], vec![2.0; NEMOTRON_CHUNK_SAMPLES]);
}

#[test]
fn explicit_stream_lifecycle_reports_duplicate_start_push_before_start_and_push_after_finish() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let key =
        nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0).streaming_session_key();

    assert!(matches!(
        runtime.push_stream(&key, &[0.0]),
        Err(crate::AsrStreamingLifecycleError::NotStarted(_))
    ));
    runtime
        .start_stream(key.clone(), AsrStreamConfig::default())
        .unwrap();
    assert!(matches!(
        runtime.start_stream(key.clone(), AsrStreamConfig::default()),
        Err(crate::AsrStreamingLifecycleError::AlreadyStarted(_))
    ));
    runtime.push_stream(&key, &[0.0]).unwrap();
    runtime.finish_stream(&key).unwrap();
    assert!(matches!(
        runtime.push_stream(&key, &[0.0]),
        Err(crate::AsrStreamingLifecycleError::NotStarted(_))
    ));
    assert_eq!(
        calls.lock().unwrap().finishes,
        vec![StreamingSessionId::new(1, Some(1))]
    );
}

#[test]
fn cancel_stream_discards_tail_without_calling_backend_finish() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let key =
        nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0).streaming_session_key();

    runtime
        .start_stream(key.clone(), AsrStreamConfig::default())
        .unwrap();
    runtime.push_stream(&key, &[1.0]).unwrap();
    runtime.cancel_stream(&key).unwrap();

    let calls = calls.lock().unwrap();
    assert!(calls.finishes.is_empty());
    assert_eq!(calls.cancelled, vec![StreamingSessionId::new(1, Some(1))]);
}

#[test]
fn offline_request_shifts_model_timestamps_by_inserted_leading_padding() {
    let calls = Arc::new(Mutex::new(Calls::default()));
    let mut models = AsrModelRegistry::default();
    models
        .insert(
            AsrModel::ReazonSpeechK2V2,
            Box::new(RecordingEngine {
                calls: calls.clone(),
                transcript: AsrTranscript::from_parts(
                    "後半",
                    vec!["後半".to_string()],
                    Some(&[1.0]),
                    Some(&[1.0]),
                ),
            }),
        )
        .unwrap();
    let mut runtime = AsrExecutionRuntime::new(models);
    let request = padded_interim_request();

    let transcript = runtime
        .execute(&asr_config(AsrModel::ReazonSpeechK2V2, false), &request)
        .unwrap();

    assert_eq!(calls.lock().unwrap().offline_audio[0].len(), 42_240);
    assert_eq!(transcript.text, "後半");
    let shifted_start = transcript.tokens[0].start_sec.unwrap();
    assert!((shifted_start - 0.68).abs() < 0.001);
}

#[test]
fn model_registry_rejects_replacing_an_already_registered_live_model() {
    let calls = Arc::new(Mutex::new(Calls::default()));
    let mut models = AsrModelRegistry::default();
    let first = RecordingEngine {
        calls: calls.clone(),
        transcript: AsrTranscript::from_text("first"),
    };
    let second = RecordingEngine {
        calls,
        transcript: AsrTranscript::from_text("second"),
    };

    models.insert(NEMOTRON_MODEL, Box::new(first)).unwrap();
    let error = models
        .insert(NEMOTRON_MODEL, Box::new(second))
        .expect_err("a live model must not be silently replaced");

    assert!(error.to_string().contains("already registered"));
}

#[test]
fn dropping_the_execution_runtime_cancels_each_active_model_stream() {
    let (mut runtime, calls) = streaming_runtime(VecDeque::new());
    let request = nemotron_interim_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);

    runtime
        .execute(&asr_config(NEMOTRON_MODEL, false), &request)
        .unwrap();
    drop(runtime);

    assert_eq!(
        calls.lock().unwrap().cancelled,
        vec![StreamingSessionId::new(1, Some(1))]
    );
}

fn streaming_runtime(failures: VecDeque<String>) -> (AsrExecutionRuntime, Arc<Mutex<Calls>>) {
    let calls = Arc::new(Mutex::new(Calls {
        streaming_failures: failures,
        ..Calls::default()
    }));
    let mut models = AsrModelRegistry::default();
    models
        .insert(
            NEMOTRON_MODEL,
            Box::new(RecordingEngine {
                calls: calls.clone(),
                transcript: AsrTranscript::from_text("interim"),
            }),
        )
        .unwrap();
    (AsrExecutionRuntime::new(models), calls)
}

fn asr_config(model: AsrModel, normalize_input_audio: bool) -> SttAsrConfig {
    SttAsrConfig {
        language: model.language(),
        model,
        interim_model: None,
        normalize_input_audio,
        multilingual_enabled: false,
        enabled_models: vec![model],
    }
}

fn padded_interim_request() -> AsrRequest {
    AsrRequest {
        request_id: AsrRequestId(1),
        kind: AsrTaskKind::InterimDisplay,
        target: AsrTarget::new(
            TurnId(1),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(32_000)),
            Some(SegmentId(1)),
            Some(SegmentId(1)),
        ),
        route: RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        detected_language: None,
        audio: vec![0.0; 32_000],
        vad_results: vec![vad(true), vad(true), vad(false), vad(true)],
        source_audio: vec![0.0; 32_000],
        source_vad_results: vec![vad(true), vad(true), vad(false), vad(true)],
        close_reason: Some(SegmentCloseReason::InterimResultSilenceReached),
        created_at_frame: VadFrameIndex(1),
    }
}

fn nemotron_interim_request(request_id: u64, range: Range<u64>, sample: f32) -> AsrRequest {
    let len = usize::try_from(range.end - range.start).unwrap();
    let chunk_len = NEMOTRON_CHUNK_SAMPLES.min(len);
    AsrRequest {
        request_id: AsrRequestId(request_id),
        kind: AsrTaskKind::InterimDisplay,
        target: AsrTarget::new(
            TurnId(1),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
            Some(SegmentId(1)),
            Some(SegmentId(1)),
        ),
        route: RecognitionRoute::from_model(NEMOTRON_MODEL),
        detected_language: None,
        audio: vec![sample; chunk_len],
        vad_results: vec![vad(true)],
        source_audio: vec![sample; len],
        source_vad_results: vec![vad(true)],
        close_reason: Some(SegmentCloseReason::InterimChunkReached),
        created_at_frame: VadFrameIndex(request_id),
    }
}

fn nemotron_completion_request(request_id: u64, range: Range<u64>) -> AsrRequest {
    let len = usize::try_from(range.end - range.start).unwrap();
    AsrRequest {
        request_id: AsrRequestId(request_id),
        kind: AsrTaskKind::CompletionCheck,
        target: AsrTarget::new(
            TurnId(1),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
            Some(SegmentId(1)),
            Some(SegmentId(1)),
        ),
        route: RecognitionRoute::from_model(NEMOTRON_MODEL),
        detected_language: None,
        audio: vec![1.0; len],
        vad_results: vec![vad(true), vad(false)],
        source_audio: vec![1.0; len],
        source_vad_results: vec![vad(true), vad(false)],
        close_reason: Some(SegmentCloseReason::EndSilenceReached),
        created_at_frame: VadFrameIndex(request_id),
    }
}

fn vad(is_speech: bool) -> VadResult {
    VadResult {
        probability: if is_speech { 0.9 } else { 0.0 },
        is_speech,
    }
}
