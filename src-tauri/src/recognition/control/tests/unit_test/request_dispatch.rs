use super::super::*;

struct RejectingAsrRunner;

impl AsrRequestRunner for RejectingAsrRunner {
    fn submit(&mut self, _request: AsrRequest) -> bool {
        false
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        None
    }
}

#[test]
fn turn_runtime_asr_submit_failure_does_not_occupy_in_flight_slot() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .asr_runner(Box::new(RejectingAsrRunner))
        .build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..10,
    );

    runtime.step();

    assert!(
        runtime.requests.in_flight_request.is_none(),
        "failed ASR submit must not leave the runtime waiting forever for a result"
    );
    assert!(
        runtime.requests.last_dispatched.is_none(),
        "failed ASR submit must not be recorded as a dispatched request"
    );
}

#[test]
fn turn_runtime_dispatches_completion_instead_of_covered_stale_interim() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..200,
        )
        .pending_segment(
            2,
            Some(1),
            SegmentCloseReason::InterimResultSilenceReached,
            200..250,
        )
        .pending_segment(3, Some(1), SegmentCloseReason::EndSilenceReached, 0..300);

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("covered interim requests should be dropped in favor of completion");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(300));
    assert!(
        runtime.pending.asr_segments.is_empty(),
        "covered interim segments must not remain queued to overwrite the completion later"
    );
}

#[test]
fn turn_runtime_builds_one_completion_request_with_following_interim_after_max_chunks_when_td_allows_it()
 {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(1, None, SegmentCloseReason::SegmentMaxChunksReached, 0..100)
        .pending_segment(
            2,
            Some(1),
            SegmentCloseReason::InterimResultSilenceReached,
            100..200,
        );

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("Namo should connect completion and following interim into one ASR request");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(200));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(2)));
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("connected request should be in flight");
    assert_eq!(
        request.close_reason,
        Some(SegmentCloseReason::SegmentMaxChunksReached)
    );
    assert_eq!(request.audio.len(), 200);
    assert!(runtime.pending.asr_segments.is_empty());
}

#[test]
fn turn_runtime_does_not_merge_completion_with_root_interim_without_segment_chain() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(1, None, SegmentCloseReason::EndSilenceReached, 0..100)
        .pending_segment(
            2,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..200,
        );

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("completion should dispatch without merging an unrelated root interim");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(100));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
    assert_eq!(
        runtime.pending.asr_segments.len(),
        1,
        "a root interim after completion must wait for TD/grammar, not request-level merge"
    );
}

#[test]
fn turn_runtime_keeps_completion_and_following_interim_separate_when_td_disallows_it() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(1, None, SegmentCloseReason::EndSilenceReached, 0..100)
        .pending_segment(
            2,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..200,
        );

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("Simple should dispatch completion first");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(100));
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("completion request should be in flight");
    assert_eq!(request.audio.len(), 100);
    assert_eq!(runtime.pending.asr_segments.len(), 1);
    assert_eq!(
        runtime
            .pending
            .asr_segments
            .front()
            .expect("interim should remain pending")
            .segment_id,
        2
    );
}

#[test]
fn turn_runtime_builds_one_interim_request_from_multiple_pending_interim_segments() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new().build();
    runtime_state(&mut runtime)
        .pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            0..100,
        )
        .pending_segment(
            2,
            Some(1),
            SegmentCloseReason::InterimResultSilenceReached,
            100..200,
        );

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("pending interim segments should be combined into one ASR request");
    assert_eq!(dispatched.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(200));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(2)));
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("combined interim request should be in flight");
    assert_eq!(
        request.source_audio,
        [vec![1.0; 100], vec![2.0; 100]].concat(),
        "interim ASR request padding must not change the source audio that will be persisted"
    );
    assert_eq!(request.target.first_segment_id, Some(SegmentId(1)));
    assert!(runtime.pending.asr_segments.is_empty());
}

#[test]
fn turn_runtime_does_not_merge_adjacent_root_interim_segments_without_segment_chain() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .interim_display(true)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            0..100,
        )
        .pending_segment(
            2,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..180,
        );

    runtime.step();

    let dispatched = runtime
        .take_last_dispatched()
        .expect("first root interim should dispatch alone");
    assert_eq!(dispatched.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("first root interim request should be in flight");
    assert_eq!(
        request.source_audio.len(),
        100,
        "adjacent root segments must not be silently merged into one ASR request"
    );
    assert_eq!(
        runtime.pending.asr_segments.len(),
        1,
        "the second root segment should remain pending for the next turn"
    );
}

#[test]
fn turn_runtime_batched_turn_check_promotes_pending_interim_to_completion_before_interim_asr() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .turn_check_silence_ms(64);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();

    runtime.push_vad_frame(&vec![1.0; 16_000], vad(true));
    runtime.push_vad_frame(&vec![0.0; 512], vad(false));
    runtime.push_vad_frame(&vec![0.0; 512], vad(false));

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(
        submitted[0].kind,
        AsrTaskKind::CompletionCheck,
        "when turn-check silence is already reached, the queued interim segment must become the completion ASR instead of dispatching interim first"
    );
    assert_eq!(
        submitted[0].close_reason,
        Some(SegmentCloseReason::EndSilenceReached)
    );
    assert!(
        runtime.pending.turn_check.is_none(),
        "the turn-check event must be consumed after promoting the queued interim segment"
    );
}

#[test]
fn turn_runtime_turn_check_promotes_real_interim_when_nemotron_streaming_chunk_is_queued_first() {
    let (mut runtime, _config) = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .interim_display(true)
        .build();
    runtime_state(&mut runtime)
        .pending_segment(1, None, SegmentCloseReason::InterimChunkReached, 0..160)
        .pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            0..320,
        )
        .pending_turn_check(1);

    runtime.step();

    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("turn-check should promote the real interim segment to completion");
    assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        request.close_reason,
        Some(SegmentCloseReason::EndSilenceReached)
    );
    assert_eq!(
        request.target.range,
        AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(320))
    );
    assert!(
        runtime.pending.asr_segments.is_empty(),
        "covered streaming chunks must be dropped before the promoted completion dispatches"
    );
}

#[test]
fn turn_runtime_new_root_interim_after_open_simple_turn_is_emitted_as_next_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();
    let mut previous_turn = Turn::new("turn-1-1-0".to_string(), 0);
    {
        let draft = previous_turn.draft_mut();
        draft.append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            RecognitionRoute::from_model(config.asr.model),
            "前の途中".to_string(),
            0,
        );
    }
    runtime_state(&mut runtime)
        .turn(1, previous_turn)
        .open_turn(1)
        .pending_segment(
            2,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..180,
        );

    runtime.step();
    let dispatched = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("new root interim should dispatch an ASR request");
    assert_eq!(
        dispatched.target.turn_id,
        TurnId(2),
        "a new root segment in Simple mode must not be attached to the previous open turn"
    );
    asr_handle.complete_request_with_text(&dispatched, "次の途中");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("次の途中...", false, 2, 2)]
    );
    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(2),
        "the following interim should become the currently displayed open turn"
    );
}

#[test]
fn turn_runtime_rerecognition_uses_global_audio_range_from_turn_sources() {
    let mut builder = RecognitionSessionTestBuilder::new().turn_detector(TurnDetector::Namo);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        7,
        None,
        SegmentCloseReason::EndSilenceReached,
        400..520,
    );
    runtime.step();
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(completion.target.range.start_sample, GlobalSampleIndex(400));
    assert_eq!(completion.target.range.end_sample, GlobalSampleIndex(520));
    asr_handle.complete_request_with_text(&completion, "範囲確認");

    runtime.step();

    let rerecognition = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("Namo completion should dispatch rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        rerecognition.target.range.start_sample,
        GlobalSampleIndex(400)
    );
    assert_eq!(
        rerecognition.target.range.end_sample,
        GlobalSampleIndex(520)
    );
}
