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
