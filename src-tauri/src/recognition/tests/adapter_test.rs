// Desktop port wiring only. ASR execution policy is tested by parapper-stt-engine.
use super::*;

#[test]
fn completed_interim_result_from_asr_port_is_forwarded_to_runtime_output() {
    let request = AsrRequest {
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
        vad_results: vec![vad(true)],
        source_audio: vec![0.0; 32_000],
        source_vad_results: vec![vad(true)],
        close_reason: Some(SegmentCloseReason::InterimResultSilenceReached),
        created_at_frame: VadFrameIndex(1),
    };
    let result = AsrResult {
        request_id: request.request_id,
        kind: request.kind,
        target: request.target.clone(),
        route: request.route,
        status: AsrResultStatus::Ok(AsrTranscript::from_text("後半")),
        completed_at_frame: request.created_at_frame,
        elapsed_millis: 1,
    };
    let mut builder = RecognitionSessionTestBuilder::new().segment_start_speech_ms(500);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime.requests.in_flight_request = Some(request);
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
