use super::super::*;

#[test]
fn turn_runtime_idle_steps_without_vad_frames_do_not_advance_open_turn_timeout() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    {
        let draft = turn.draft_mut();
        draft.append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            RecognitionRoute::from_model(config.asr.model),
            "まだ続く".to_string(),
            0,
        );
    }
    let next_runtime_tick = runtime.counters.next_runtime_tick;
    runtime_state(&mut runtime)
        .turn(1, turn)
        .open_turn_since(1, next_runtime_tick);

    for _ in 0..runtime.timeout_ticks().saturating_mul(2) {
        runtime.step();
    }

    assert!(
        asr_handle.submitted_requests().is_empty(),
        "idle loop steps without VAD frames must not advance open-turn timeout"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty()
    );
}

#[test]
fn turn_runtime_output_identity_includes_turn_session_id() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .turn_session_id(77);
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
    asr_handle.complete_request_with_text(&request, "セッション付き");

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].id, "turn-77-1-0");
    assert_eq!(outputs[0].turn_session_id, 77);
    assert_eq!(outputs[0].turn_id, 1);
    assert_eq!(outputs[0].output_sequence, 1);
}

#[test]
fn finalize_timeout_turn_after_rerecognition_emits_final_output() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).turn(
        1,
        recognized_turn_with_audio(1, "時間切れ", &[1.0, 2.0, 3.0]),
    );

    runtime.finalize_timeout_turn_after_rerecognition(1);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "時間切れ。");
}

#[test]
fn emit_stale_turn_finals_cleans_empty_text_turn_state_without_emitting_output() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime)
        .turn(1, Turn::new("turn-1-1-0".to_string(), 0))
        .turn_audio_range(1, 0..10);

    runtime.emit_stale_turn_finals(2);

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "empty stale turns must not emit a final text output"
    );
    assert!(
        !runtime.turn_store.turns.contains_key(&1),
        "empty stale turns must be removed instead of being retained forever"
    );
    assert!(
        !runtime.turn_store.audio_ranges.contains_key(&1),
        "empty stale turn audio ranges must be removed with the turn state"
    );
    assert_eq!(
        runtime.turn_store.confirmed_until_sample,
        GlobalSampleIndex(10),
        "stale audio must still advance confirmed_until_sample even when no text output is emitted"
    );
}

#[test]
fn turn_runtime_timeout_waits_for_in_flight_same_turn_asr_before_final_output() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .turn_check_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    let route = RecognitionRoute::from_model(config.asr.model);
    let request = interim_request_for_turn(1, 1);
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    turn.draft_mut().append_recognized_segment(
        1,
        None,
        &[1.0],
        &[vad(true)],
        route,
        "前半".to_string(),
        0,
    );
    let timeout_ticks = runtime.timeout_ticks();
    runtime_state(&mut runtime)
        .turn(1, turn)
        .turn_audio_range(1, 0..1)
        .open_turn_since(1, 0)
        .in_flight(request.clone())
        .next_runtime_tick(timeout_ticks);

    runtime.step();

    assert!(
        outputs
            .lock()
            .expect("phrase outputs should be readable")
            .is_empty(),
        "timeout final must wait for the in-flight ASR that still owns the same open turn"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));

    asr_handle.complete_request_with_text(&request, "後半");
    runtime.step();
    runtime.step();

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    let final_outputs = outputs
        .iter()
        .filter(|output| output.is_final)
        .collect::<Vec<_>>();
    assert_eq!(final_outputs.len(), 1);
    assert_eq!(final_outputs[0].text, "前半後半。");
    assert_eq!(
        final_outputs[0].phrase,
        vec![1.0, 1.0],
        "final phrase audio must include the late in-flight ASR source audio"
    );
}

#[test]
fn turn_runtime_simple_turn_check_waits_for_busy_asr_before_finalizing_next_turn_audio() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    let route = RecognitionRoute::from_model(config.asr.model);
    let mut first_turn = Turn::new("turn-1-1-0".to_string(), 0);
    first_turn.draft_mut().append_recognized_segment(
        1,
        None,
        &[1.0],
        &[vad(true)],
        route,
        "前半".to_string(),
        0,
    );
    let mut second_turn = Turn::new("turn-1-2-0".to_string(), 0);
    second_turn.draft_mut().append_recognized_segment(
        3,
        None,
        &[4.0],
        &[vad(true)],
        route,
        "後半".to_string(),
        0,
    );
    runtime_state(&mut runtime)
        .turn(1, first_turn)
        .turn_audio_range(1, 0..1)
        .open_turn(1)
        .pending_segment(
            2,
            Some(1),
            SegmentCloseReason::InterimResultSilenceReached,
            1..3,
        );
    runtime.step();
    let older_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("older interim request should be produced by runtime dispatch");
    assert_eq!(older_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(older_request.target.turn_id, TurnId(1));
    assert_eq!(older_request.source_audio, vec![2.0, 2.0]);

    runtime_state(&mut runtime)
        .turn(2, second_turn)
        .turn_audio_range(2, 3..4)
        .open_turn(2)
        .pending_turn_check(3);

    runtime.step();

    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "newer turn-check final must wait instead of finalizing an older turn with missing in-flight audio"
    );
    assert!(
        runtime.pending.turn_check.is_some(),
        "the turn-check must stay pending while ASR is busy"
    );

    asr_handle.complete_request_with_text(&older_request, "中間");
    runtime.step();
    runtime.step();
    let second_turn_rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("second turn check should dispatch after older ASR finishes");
    assert_eq!(second_turn_rerecognition.target.turn_id, TurnId(2));
    asr_handle.complete_request_with_text(&second_turn_rerecognition, "後半");
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    let final_outputs = outputs
        .iter()
        .filter(|output| output.is_final)
        .collect::<Vec<_>>();
    assert_eq!(
        final_outputs
            .iter()
            .map(|output| (output.turn_id, output.text.as_str()))
            .collect::<Vec<_>>(),
        vec![(1, "前半中間。"), (2, "後半。")],
        "the older turn must finalize before the pending newer turn-check final"
    );
    let final_first = final_outputs[0];
    assert_eq!(
        final_first.phrase,
        vec![1.0, 2.0, 2.0],
        "the final debug audio for the older turn must include the in-flight middle segment"
    );
}

#[test]
fn turn_runtime_stale_final_waits_for_in_flight_asr_for_older_turn() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();
    let route = RecognitionRoute::from_model(config.asr.model);
    let request = interim_request_for_turn(1, 1);
    runtime_state(&mut runtime).in_flight(request.clone());
    for (turn_id, text, range) in [(1, "低精度の古いターン", 0..1), (2, "新しいターン", 1..2)]
    {
        let mut turn = Turn::new(format!("turn-1-{turn_id}-0"), 0);
        turn.draft_mut().append_recognized_segment(
            turn_id,
            None,
            &[f32::from(
                u16::try_from(turn_id).expect("test turn id should fit u16"),
            )],
            &[vad(true)],
            route,
            text.to_string(),
            0,
        );
        runtime_state(&mut runtime)
            .turn(turn_id, turn)
            .turn_audio_range(turn_id, range);
    }

    runtime.complete_turn_without_grammar(2);
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "newer finalization must wait while an older in-flight ASR can still extend the stale turn"
    );

    asr_handle.complete_request_with_text(&request, "遅れてきた高精度");
    runtime.step();
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    let final_outputs = outputs
        .iter()
        .filter(|output| output.is_final)
        .collect::<Vec<_>>();
    assert_eq!(
        final_outputs
            .iter()
            .map(|output| (output.turn_id, output.text.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (1, "低精度の古いターン遅れてきた高精度。"),
            (2, "新しいターン。")
        ],
        "older in-flight ASR must update the stale turn before finalization advances"
    );
    assert!(!runtime.turn_store.turns.contains_key(&1));
    assert!(runtime.requests.in_flight_request.is_none());
}

#[test]
fn turn_runtime_pending_finalization_uses_latest_blocked_turn_to_avoid_orphaning_newer_turn() {
    let mut builder = RecognitionSessionTestBuilder::new();
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();
    let route = RecognitionRoute::from_model(config.asr.model);
    let older_request = interim_request_for_turn(1, 1);
    runtime_state(&mut runtime)
        .in_flight(older_request.clone())
        .open_turn(3);
    for (turn_id, text, range) in [
        (1, "古いターン", 0..1),
        (2, "中間ターン", 1..2),
        (3, "新しいターン", 2..3),
    ] {
        let mut turn = Turn::new(format!("turn-1-{turn_id}-0"), 0);
        turn.draft_mut().append_recognized_segment(
            turn_id,
            None,
            &[f32::from(
                u16::try_from(turn_id).expect("test turn id should fit u16"),
            )],
            &[vad(true)],
            route,
            text.to_string(),
            0,
        );
        runtime_state(&mut runtime)
            .turn(turn_id, turn)
            .turn_audio_range(turn_id, range);
    }

    runtime.complete_turn_without_grammar(2);
    runtime.complete_turn_without_grammar(3);
    asr_handle.complete_request_with_text(&older_request, "更新");
    runtime.step();
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    let final_outputs = outputs
        .iter()
        .filter(|output| output.is_final)
        .collect::<Vec<_>>();
    assert_eq!(
        final_outputs
            .iter()
            .map(|output| (output.turn_id, output.text.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (1, "古いターン更新。"),
            (2, "中間ターン。"),
            (3, "新しいターン。")
        ],
        "when multiple blocked finalizations are queued, the latest turn must finalize so stale turns are swept together"
    );
    assert!(runtime.turn_store.turns.is_empty());
    assert!(runtime.turn_store.open_turn_id.is_none());
}
