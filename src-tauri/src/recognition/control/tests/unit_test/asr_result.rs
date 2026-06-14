use super::super::*;

#[test]
fn turn_runtime_following_interim_keeps_previous_audio_visible_in_replaced_output() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .interim_display(true)
        .scripted_asr_texts(vec!["前半", "後半"]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();

    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        0..100,
    );
    runtime.step();
    runtime.step();

    runtime_state(&mut runtime).pending_segment(
        2,
        Some(1),
        SegmentCloseReason::InterimResultSilenceReached,
        100..180,
    );
    runtime.step();
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.segment_id
            ))
            .collect::<Vec<_>>(),
        vec![("前半...", false, 1, 1), ("前半後半...", false, 1, 2)]
    );
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.phrase.len())
            .collect::<Vec<_>>(),
        vec![100, 180],
        "a replaced interim output must still carry all previous turn audio"
    );
}

#[test]
fn turn_runtime_applies_asr_result_to_request_target_not_current_open_turn() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let request = interim_request_for_turn(1, 2);
    runtime_state(&mut runtime)
        .open_turn(1)
        .in_flight(request.clone());
    asr_handle.complete_request_with_text(&request, "target turn");

    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "target turn...".to_string(),
            is_final: false,
            turn_id: 2,
            segment_id: 2,
        }]
    );
    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(2),
        "open turn must follow the ASR request target after applying the result"
    );
}

#[test]
fn turn_runtime_completed_asr_result_without_in_flight_request_is_consumed_without_dispatching() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, _config) = builder.build();
    let request = interim_request_for_turn(1, 1);
    asr_handle.complete_request_with_text(&request, "late result");

    runtime.step();

    assert!(runtime.requests.in_flight_request.is_none());
    assert!(
        runtime.requests.last_dispatched.is_none(),
        "a late ASR result without an in-flight request must not synthesize a new dispatch"
    );
}

#[test]
fn turn_runtime_interim_output_uses_asr_result_elapsed_millis() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let request = interim_request_for_turn(1, 1);
    runtime_state(&mut runtime).in_flight(request.clone());
    asr_handle.complete_request_with_text_elapsed(&request, "処理時間あり", 37);

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "処理時間あり...");
    assert!(!outputs[0].is_final);
    assert_eq!(outputs[0].turn_id, 1);
    assert_eq!(outputs[0].segment_id, 1);
    assert_eq!(outputs[0].elapsed_millis, 37);
}

#[test]
fn turn_runtime_stale_asr_result_with_revision_mismatch_does_not_recreate_turn() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let request = interim_request_for_turn(1, 1);
    runtime_state(&mut runtime)
        .turn_revision(1, 1)
        .in_flight(request.clone());
    asr_handle.complete_request_with_text(&request, "古い途中表示");

    runtime.step();

    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "stale ASR result from a finalized turn must not overwrite the final output"
    );
    assert!(
        !runtime.turn_store.turns.contains_key(&1),
        "stale ASR result must not recreate a finalized turn draft"
    );
}

#[test]
fn turn_runtime_mismatched_asr_result_keeps_in_flight_request_for_later_match() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let request = interim_request_for_turn(1, 1);
    runtime_state(&mut runtime).in_flight(request.clone());
    asr_handle.push_completed_result(AsrResult {
        request_id: AsrRequestId(999),
        kind: request.kind,
        target: request.target.clone(),
        route: request.route,
        status: AsrResultStatus::Ok(AsrTranscript::from_text("古い結果")),
        completed_at_frame: VadFrameIndex(0),
        elapsed_millis: 0,
    });

    runtime.step();

    assert_eq!(
        runtime
            .requests
            .in_flight_request
            .as_ref()
            .map(|request| request.request_id),
        Some(request.request_id),
        "a mismatched result must not clear the current in-flight request"
    );
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "a mismatched result must not emit output"
    );

    asr_handle.complete_request_with_text(&request, "正しい結果");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("正しい結果...", false, 1, 1)]
    );
    assert!(runtime.requests.in_flight_request.is_none());
}

#[test]
fn turn_runtime_route_changed_before_result_marks_request_stale() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .asr_model(AsrModel::ReazonSpeechK2V2);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..100,
    );
    runtime.step();
    let old_route_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(
        old_route_request.route,
        RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2)
    );
    runtime.update_config(&parapper_config! {
        asr_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
        ..config
    });
    asr_handle.complete_request_with_text(&old_route_request, "古い経路");

    runtime.step();

    assert!(
        runtime.turn_store.turns.is_empty(),
        "an ASR result from the old route must not create or update a turn after route changes"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new()
    );
}

#[test]
fn turn_runtime_failed_completion_check_falls_back_to_existing_draft_final() {
    let mut builder = RecognitionSessionTestBuilder::new().turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    {
        let draft = turn.draft_mut();
        draft.append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            RecognitionRoute::from_language(crate::config::AsrLanguage::Japanese),
            "途中表示".to_string(),
            0,
        );
    }
    runtime_state(&mut runtime).turn(1, turn);
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..1,
    );
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
    asr_handle.fail_request(&request);

    runtime.step();

    assert!(runtime.requests.in_flight_request.is_none());
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("途中表示。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_failed_namo_completion_without_existing_draft_does_not_open_ghost_turn() {
    let mut builder = RecognitionSessionTestBuilder::new().turn_detector(TurnDetector::Namo);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..100,
    );
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
    asr_handle.fail_request(&request);

    runtime.step();

    assert!(runtime.requests.in_flight_request.is_none());
    assert!(
        runtime.turn_store.open_turn_id.is_none(),
        "a failed first completion with no draft text must not create a ghost open turn"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new()
    );
}

#[test]
fn turn_runtime_failed_interim_display_does_not_block_later_completion_final() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        0..100,
    );
    runtime.step();
    let interim_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("interim request should be in flight");
    assert_eq!(interim_request.kind, AsrTaskKind::InterimDisplay);
    asr_handle.fail_request(&interim_request);
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..100,
    );

    runtime.step();
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "failed interim must not emit a placeholder or create a broken open turn"
    );
    runtime.step();
    let completion_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion should still dispatch after interim failure");
    assert_eq!(completion_request.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text(&completion_request, "確定表示");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("確定表示。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_empty_interim_transcript_clears_in_flight_without_opening_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        0..100,
    );
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("interim ASR should be in flight");
    asr_handle.push_completed_result(AsrResult {
        request_id: request.request_id,
        kind: request.kind,
        target: request.target,
        route: request.route,
        status: AsrResultStatus::Ok(AsrTranscript::from_text("   ")),
        completed_at_frame: VadFrameIndex(0),
        elapsed_millis: 0,
    });

    runtime.step();

    assert!(
        runtime.requests.in_flight_request.is_none(),
        "empty ASR transcript must clear the in-flight request"
    );
    assert!(
        runtime.turn_store.open_turn_id.is_none(),
        "empty interim text must not create a ghost open turn"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new()
    );
}

#[test]
fn turn_runtime_completion_after_end_silence_emits_final_output() {
    let mut builder = RecognitionSessionTestBuilder::new().turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..100,
    );
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    asr_handle.complete_request_with_text(&request, "確定");

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "確定。");
}

#[test]
fn turn_runtime_failed_timeout_rerecognition_clears_purpose_and_finalizes_existing_draft() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_vad(1, "未確定", &[1.0, 2.0], &[vad(true), vad(false)]);
    let timeout_ticks = runtime.timeout_ticks();
    runtime_state(&mut runtime)
        .turn(1, turn)
        .open_turn_since(1, 0)
        .next_runtime_tick(timeout_ticks);

    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("timeout rerecognition request should be in flight");
    assert_eq!(request.kind, AsrTaskKind::Rerecognition);
    asr_handle.fail_request(&request);

    runtime.step();

    assert!(runtime.requests.in_flight_request.is_none());
    assert!(
        runtime.requests.pending_rerecognition_purpose.is_none(),
        "failed rerecognition must not leave a purpose for a later unrelated result"
    );
    assert_eq!(
        asr_handle.submitted_requests().len(),
        1,
        "a failed ASR result must not trigger another timeout rerecognition in the same step"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("未確定。", true, 1, 1)],
        "timeout rerecognition failure should fall back to the existing draft instead of hanging"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_failed_simple_turn_check_rerecognition_finalizes_existing_draft() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    {
        let draft = turn.draft_mut();
        draft.append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            RecognitionRoute::from_language(crate::config::AsrLanguage::Japanese),
            "簡易確定".to_string(),
            0,
        );
    }
    runtime_state(&mut runtime)
        .turn(1, turn)
        .open_turn(1)
        .pending_turn_check(1);
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("simple turn-check rerecognition should be in flight");
    asr_handle.fail_request(&request);

    runtime.step();

    assert!(runtime.requests.pending_rerecognition_purpose.is_none());
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("簡易確定。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_failed_grammar_rerecognition_uses_turn_decision_on_existing_draft() {
    let mut builder = RecognitionSessionTestBuilder::new().turn_detector(TurnDetector::Namo);
    let asr_handle = builder.use_manual_asr();
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..1,
    );
    runtime.step();
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text(&completion, "文法判定");
    runtime.step();
    let request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("grammar rerecognition should be in flight");
    asr_handle.fail_request(&request);

    runtime.step();

    assert!(runtime.requests.pending_rerecognition_purpose.is_none());
    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["文法判定".to_string()]
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("文法判定。", true, 1, 1)]
    );
}
