use std::collections::VecDeque;

use parapper_models::asr::{AsrModel, AsrTranscript};
use parapper_stt_engine::{
    RecognitionConfig, RecognitionSegmentEngine, RecognizedTurn, SegmentBuilderEvent,
    SegmentCloseReason, VadResult,
    ports::{AsrRequestRunner, RecognitionOutputSink},
    transcription::{
        planner::{PendingAsrSegment, PlannerConfig, take_next_request_segment_plan},
        reducer::{
            AsrRequestStaleInput, AsrResultAction, AsrResultCompletionFailureAction,
            AsrResultReductionInput, reduce_asr_result,
        },
        route::{RecognitionRoute, RecognitionRouteSelection},
        task::{
            AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AudioRange, GlobalSampleIndex,
            VadFrameIndex,
        },
    },
    turn::TurnDraft,
};

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the headless contract is clearer as one end-to-end scenario"
)]
fn headless_host_drives_segment_asr_turn_and_structured_output_without_tauri() {
    let recognition_config = RecognitionConfig {
        segment_start_speech_ms: 1,
        turn_check_silence_ms: 32,
        ..RecognitionConfig::default()
    };
    let mut engine = RecognitionSegmentEngine::new(&recognition_config);
    let speech = vad(true);
    let silence = vad(false);

    let started = engine.push_vad_frame(&[0.5], speech);
    assert!(matches!(
        started.events.as_slice(),
        [SegmentBuilderEvent::SegmentStarted { segment_id: 1, .. }]
    ));
    let closed = engine.push_vad_frame(&[0.0], silence);
    let SegmentBuilderEvent::SegmentClosed {
        segment_id,
        previous_segment_id,
        full_audio,
        vad_results,
        source_audio,
        source_vad_results,
        reason,
    } = closed
        .events
        .into_iter()
        .find(|event| matches!(event, SegmentBuilderEvent::SegmentClosed { .. }))
        .expect("the real Segment state machine should close after turn-check silence")
    else {
        unreachable!()
    };
    assert_eq!(reason, SegmentCloseReason::EndSilenceReached);

    let audio_len = u64::try_from(full_audio.len()).unwrap();
    let mut pending = VecDeque::from([PendingAsrSegment {
        segment_id,
        previous_segment_id,
        audio: full_audio,
        vad_results,
        source_audio,
        source_vad_results,
        reason,
        range: AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(audio_len)),
        created_at_frame: VadFrameIndex(1),
    }]);
    let planner_config = PlannerConfig {
        can_connect_interim_after_completion: false,
        vad_interval_ms: recognition_config.vad_interval_ms,
    };
    let plan = take_next_request_segment_plan(&planner_config, &mut pending).unwrap();
    let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
    let request = plan.into_request(
        &planner_config,
        AsrRequestId(1),
        1,
        0,
        RecognitionRouteSelection {
            route,
            detected_language: Some("ja".to_owned()),
        },
    );

    let mut asr = ScriptedAsr::default();
    assert!(asr.submit(request.clone()));
    let result = asr.try_recv_result().unwrap();
    let action = reduce_asr_result(
        &result,
        &request,
        AsrResultReductionInput {
            stale_input: AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            },
            completion_has_non_empty_draft: false,
            completion_failure_action: AsrResultCompletionFailureAction::KeepOpen,
            completion_rerecognition_purpose: None,
            pending_rerecognition_purpose: None,
        },
    );
    let AsrResultAction::ApplyCompletionTranscript { transcript, .. } = action else {
        panic!("scripted completion should become a completion transcript action")
    };

    let mut draft = TurnDraft::new("turn-1-1-0".to_owned(), 0);
    draft.set_detected_language(request.detected_language.clone());
    draft.append_recognized_segment(
        segment_id,
        previous_segment_id,
        &request.source_audio,
        &request.source_vad_results,
        route,
        transcript.text,
        result.elapsed_millis,
    );
    let output = draft.confirm(1, 1, 1, route).unwrap().into_output();
    let mut sink = RecordingSink::default();
    sink.emit(output);

    assert_eq!(sink.outputs.len(), 1);
    assert_eq!(sink.outputs[0].text, "ヘッドレス認識。");
    assert_eq!(sink.outputs[0].meta.source.turn_id, 1);
    assert_eq!(sink.outputs[0].meta.source.segment_id, 1);
    assert!(sink.outputs[0].meta.is_final);
}

#[derive(Default)]
struct ScriptedAsr {
    completed: Option<AsrResult>,
}

impl AsrRequestRunner for ScriptedAsr {
    fn submit(&mut self, request: AsrRequest) -> bool {
        self.completed = Some(AsrResult {
            request_id: request.request_id,
            kind: request.kind,
            target: request.target,
            route: request.route,
            status: AsrResultStatus::Ok(AsrTranscript {
                text: "ヘッドレス認識".to_owned(),
                tokens: Vec::new(),
            }),
            completed_at_frame: request.created_at_frame,
            elapsed_millis: 3,
        });
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.completed.take()
    }
}

#[derive(Default)]
struct RecordingSink {
    outputs: Vec<RecognizedTurn>,
}

impl RecognitionOutputSink for RecordingSink {
    fn emit(&mut self, output: RecognizedTurn) {
        self.outputs.push(output);
    }
}

fn vad(is_speech: bool) -> VadResult {
    VadResult {
        probability: if is_speech { 0.9 } else { 0.0 },
        is_speech,
    }
}
