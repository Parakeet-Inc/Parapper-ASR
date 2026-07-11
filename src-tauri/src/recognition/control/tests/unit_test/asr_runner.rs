use super::super::*;

#[cfg(not(target_os = "macos"))]
#[test]
fn run_engine_asr_request_shifts_token_timestamps_without_rerecognition_and_runtime_emits_interim()
{
    let config = parapper_config! {
        segment_start_speech_ms: 500,
        ..ParapperConfig::default()
    };
    let request = padded_interim_asr_request(&config);
    let call_audio_lens = Arc::new(Mutex::new(Vec::new()));
    let mut asr = AsrEngineCache::default();
    asr.insert_engine_for_test(
        config.asr.model,
        Box::new(ScriptedAsrEngine {
            transcripts: vec![AsrTranscript::from_parts(
                "後半",
                vec!["後半".to_string()],
                Some(&[1.0]),
                Some(&[1.0]),
            )]
            .into(),
            call_audio_lens: call_audio_lens.clone(),
        }),
    );
    let handle = tauri_test_handle();

    let result = run_engine_asr_request(&handle, &config, &mut asr, &request);

    assert_eq!(
        *call_audio_lens
            .lock()
            .expect("ASR call lengths should be readable"),
        vec![42_240],
        "timestamp と VAD から欠落音声を推定して追加 ASR してはいけない"
    );
    let AsrResultStatus::Ok(transcript) = &result.status else {
        panic!("scripted ASR result should succeed");
    };
    assert_eq!(transcript.text, "後半");
    assert_eq!(transcript.tokens.len(), 1);
    let shifted_start = transcript.tokens[0]
        .start_sec
        .expect("token timestamp should be present");
    assert!(
        (shifted_start - 0.68).abs() < 0.001,
        "leading ASR padding should be subtracted from token timestamps"
    );

    let mut builder = RecognitionSessionTestBuilder::new().segment_start_speech_ms(500);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).in_flight(request);
    asr_handle.push_completed_result(result);

    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "後半...".to_string(),
            is_final: false,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn run_engine_asr_request_feeds_nemotron_interim_as_stateful_deltas() {
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let config = parapper_config! {
        asr_model: AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        asr_normalize_input_audio: false,
        ..ParapperConfig::default()
    };
    let streaming_call_audio = Arc::new(Mutex::new(Vec::new()));
    let stateless_call_audio = Arc::new(Mutex::new(Vec::new()));
    let clear_count = Arc::new(Mutex::new(0usize));
    let clear_session_count = Arc::new(Mutex::new(0usize));
    let streaming_failures = Arc::new(Mutex::new(std::collections::VecDeque::new()));
    let mut asr = AsrEngineCache::default();
    asr.insert_engine_for_test(
        config.asr.model,
        Box::new(RecordingStreamingAsrEngine {
            streaming_call_audio: streaming_call_audio.clone(),
            stateless_call_audio: stateless_call_audio.clone(),
            clear_count: clear_count.clone(),
            clear_session_count: clear_session_count.clone(),
            streaming_failures: streaming_failures.clone(),
        }),
    );
    let handle = tauri_test_handle();
    let first = nemotron_interim_chunk_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let second = nemotron_interim_chunk_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64, 2.0);

    let first_result = run_engine_asr_request(&handle, &config, &mut asr, &first);
    let second_result = run_engine_asr_request(&handle, &config, &mut asr, &second);

    assert!(matches!(first_result.status, AsrResultStatus::Ok(_)));
    assert!(matches!(second_result.status, AsrResultStatus::Ok(_)));
    let streaming_call_audio = streaming_call_audio
        .lock()
        .expect("streaming ASR calls should be readable");
    assert_eq!(streaming_call_audio.len(), 2);
    assert_eq!(
        streaming_call_audio[0].len(),
        NEMOTRON_CHUNK_SAMPLES * 2,
        "first Nemotron streaming chunk should include only the leading fade/adjustment window and the first real delta"
    );
    assert!(
        streaming_call_audio[0]
            .last()
            .is_some_and(|sample| *sample > 0.9),
        "first Nemotron streaming chunk must not append a tail fade or synthetic trailing silence before the next delta"
    );
    assert_eq!(
        streaming_call_audio[1],
        vec![2.0; NEMOTRON_CHUNK_SAMPLES],
        "second Nemotron streaming update must feed only the next 160ms delta"
    );
    assert!(
        stateless_call_audio
            .lock()
            .expect("stateless ASR calls should be readable")
            .is_empty(),
        "Nemotron interim chunks must not recreate a stateless stream per request"
    );
    assert_eq!(
        *clear_count.lock().expect("clear count should be readable"),
        0,
        "active Nemotron streaming state should stay alive across interim chunks"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn run_engine_asr_request_rebuilds_nemotron_stream_from_source_audio_after_delta_failure() {
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let config = parapper_config! {
        asr_model: AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        asr_normalize_input_audio: false,
        ..ParapperConfig::default()
    };
    let streaming_call_audio = Arc::new(Mutex::new(Vec::new()));
    let stateless_call_audio = Arc::new(Mutex::new(Vec::new()));
    let clear_count = Arc::new(Mutex::new(0usize));
    let clear_session_count = Arc::new(Mutex::new(0usize));
    let streaming_failures = Arc::new(Mutex::new(std::collections::VecDeque::from([
        "streaming result unavailable".to_string(),
    ])));
    let mut asr = AsrEngineCache::default();
    asr.insert_engine_for_test(
        config.asr.model,
        Box::new(RecordingStreamingAsrEngine {
            streaming_call_audio: streaming_call_audio.clone(),
            stateless_call_audio: stateless_call_audio.clone(),
            clear_count: clear_count.clone(),
            clear_session_count: clear_session_count.clone(),
            streaming_failures: streaming_failures.clone(),
        }),
    );
    let handle = tauri_test_handle();
    let first = nemotron_interim_chunk_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let second = nemotron_interim_chunk_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64, 2.0);

    let first_result = run_engine_asr_request(&handle, &config, &mut asr, &first);
    let second_result = run_engine_asr_request(&handle, &config, &mut asr, &second);

    assert!(matches!(first_result.status, AsrResultStatus::Failed(_)));
    assert!(matches!(second_result.status, AsrResultStatus::Ok(_)));
    assert_eq!(
        *clear_session_count
            .lock()
            .expect("clear-session count should be readable"),
        1,
        "a failed streaming delta must clear the partially-created inner stream session"
    );
    let streaming_call_audio = streaming_call_audio
        .lock()
        .expect("streaming ASR calls should be readable");
    assert_eq!(streaming_call_audio.len(), 2);
    assert_eq!(
        streaming_call_audio[1].len(),
        NEMOTRON_CHUNK_SAMPLES * 3,
        "after a streaming failure, the next request must rebuild the stream from cumulative source_audio instead of feeding only the next delta"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn run_engine_asr_request_clears_nemotron_stream_before_non_streaming_asr() {
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let config = parapper_config! {
        asr_model: AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        asr_normalize_input_audio: false,
        ..ParapperConfig::default()
    };
    let streaming_call_audio = Arc::new(Mutex::new(Vec::new()));
    let stateless_call_audio = Arc::new(Mutex::new(Vec::new()));
    let clear_count = Arc::new(Mutex::new(0usize));
    let clear_session_count = Arc::new(Mutex::new(0usize));
    let streaming_failures = Arc::new(Mutex::new(std::collections::VecDeque::new()));
    let mut asr = AsrEngineCache::default();
    asr.insert_engine_for_test(
        config.asr.model,
        Box::new(RecordingStreamingAsrEngine {
            streaming_call_audio: streaming_call_audio.clone(),
            stateless_call_audio: stateless_call_audio.clone(),
            clear_count: clear_count.clone(),
            clear_session_count: clear_session_count.clone(),
            streaming_failures: streaming_failures.clone(),
        }),
    );
    let handle = tauri_test_handle();
    let interim = nemotron_interim_chunk_request(1, 0..NEMOTRON_CHUNK_SAMPLES as u64, 1.0);
    let completion = nemotron_completion_request(2, 0..(NEMOTRON_CHUNK_SAMPLES * 2) as u64);

    let _ = run_engine_asr_request(&handle, &config, &mut asr, &interim);
    let completion_result = run_engine_asr_request(&handle, &config, &mut asr, &completion);

    assert!(matches!(completion_result.status, AsrResultStatus::Ok(_)));
    assert_eq!(
        *clear_count.lock().expect("clear count should be readable"),
        1,
        "completion ASR should reset the active Nemotron interim stream"
    );
    assert_eq!(
        stateless_call_audio
            .lock()
            .expect("stateless ASR calls should be readable")
            .len(),
        1,
        "completion ASR should still run through the regular full-request path"
    );
}

#[cfg(not(target_os = "macos"))]
fn padded_interim_asr_request(config: &ParapperConfig) -> AsrRequest {
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
        route: RecognitionRoute::from_model(config.asr.model),
        detected_language: None,
        audio: vec![0.0; 32_000],
        vad_results: vec![vad(true), vad(true), vad(false), vad(true)],
        source_audio: vec![0.0; 32_000],
        source_vad_results: vec![vad(true), vad(true), vad(false), vad(true)],
        close_reason: Some(SegmentCloseReason::InterimResultSilenceReached),
        created_at_frame: VadFrameIndex(1),
    }
}

#[cfg(not(target_os = "macos"))]
fn nemotron_interim_chunk_request(
    request_id: u64,
    range: std::ops::Range<u64>,
    sample: f32,
) -> AsrRequest {
    let len = (range.end - range.start) as usize;
    let chunk_len = 2_560.min(len);
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
        route: RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
        detected_language: None,
        audio: vec![sample; chunk_len],
        vad_results: vec![vad(true)],
        source_audio: vec![sample; len],
        source_vad_results: vec![vad(true)],
        close_reason: Some(SegmentCloseReason::InterimChunkReached),
        created_at_frame: VadFrameIndex(request_id),
    }
}

#[cfg(not(target_os = "macos"))]
fn nemotron_completion_request(request_id: u64, range: std::ops::Range<u64>) -> AsrRequest {
    let len = (range.end - range.start) as usize;
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
        route: RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
        detected_language: None,
        audio: vec![1.0; len],
        vad_results: vec![vad(true), vad(false)],
        source_audio: vec![1.0; len],
        source_vad_results: vec![vad(true), vad(false)],
        close_reason: Some(SegmentCloseReason::EndSilenceReached),
        created_at_frame: VadFrameIndex(request_id),
    }
}

#[cfg(not(target_os = "macos"))]
struct RecordingStreamingAsrEngine {
    streaming_call_audio: Arc<Mutex<Vec<Vec<f32>>>>,
    stateless_call_audio: Arc<Mutex<Vec<Vec<f32>>>>,
    clear_count: Arc<Mutex<usize>>,
    clear_session_count: Arc<Mutex<usize>>,
    streaming_failures: Arc<Mutex<std::collections::VecDeque<String>>>,
}

#[cfg(not(target_os = "macos"))]
impl AsrEngine for RecordingStreamingAsrEngine {
    fn transcribe(&mut self, samples: &[f32]) -> Result<AsrTranscript> {
        self.stateless_call_audio
            .lock()
            .expect("stateless ASR calls should be writable")
            .push(samples.to_vec());
        Ok(AsrTranscript::from_text("final"))
    }

    fn transcribe_streaming_delta(
        &mut self,
        _session: AsrStreamingSessionKey,
        samples: &[f32],
    ) -> Result<AsrTranscript> {
        self.streaming_call_audio
            .lock()
            .expect("streaming ASR calls should be writable")
            .push(samples.to_vec());
        if let Some(reason) = self
            .streaming_failures
            .lock()
            .expect("streaming failures should be writable")
            .pop_front()
        {
            return Err(anyhow::anyhow!(reason));
        }
        Ok(AsrTranscript::from_text("interim"))
    }

    fn clear_streaming_session(&mut self, _session: AsrStreamingSessionKey) {
        *self
            .clear_session_count
            .lock()
            .expect("clear-session count should be writable") += 1;
    }

    fn clear_streaming_sessions(&mut self) {
        *self
            .clear_count
            .lock()
            .expect("clear count should be writable") += 1;
    }
}
