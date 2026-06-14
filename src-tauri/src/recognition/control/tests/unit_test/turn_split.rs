use super::super::*;

#[test]
fn turn_runtime_internal_grammar_boundary_keeps_turn_open_without_rerecognition() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "一。二。三",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[
            vad(true),
            vad(true),
            vad(false),
            vad(true),
            vad(true),
            vad(false),
            vad(true),
            vad(true),
            vad(true),
        ],
        vec![
            boundary_candidate("一。", 3, 3, 3, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("一。二。", 6, 6, 6, GrammarBoundaryClass::PredicateEnd),
        ],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("一。二。三...", false, 1, 1)],
        "internal grammar boundaries must not split before the completion ASR text end"
    );
    assert!(
        asr_handle.submitted_requests().is_empty(),
        "grammar boundary evaluation must not dispatch another ASR request"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
    assert_eq!(
        runtime
            .turn_store
            .turns
            .get(&1)
            .expect("turn should remain open")
            .draft()
            .combined_text,
        "一。二。三"
    );
}

#[test]
fn turn_runtime_internal_grammar_boundary_after_interim_reemits_same_open_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "一。二。三",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[
            vad(true),
            vad(true),
            vad(false),
            vad(true),
            vad(true),
            vad(false),
            vad(true),
            vad(true),
            vad(true),
        ],
        vec![
            boundary_candidate("一。", 3, 3, 3, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("一。二。", 6, 6, 6, GrammarBoundaryClass::PredicateEnd),
        ],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.emit_turn_output(1, false);
    runtime.process_grammar_boundaries_after_rerecognition(1);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.id.as_str(),
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.output_sequence,
            ))
            .collect::<Vec<_>>(),
        vec![
            ("turn-1-1-0", "一。二。三...", false, 1, 1),
            ("turn-1-1-0", "一。二。三...", false, 1, 2),
        ],
        "internal grammar boundary evaluation should keep replacing the same open turn row"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_timeout_after_internal_grammar_boundary_finalizes_same_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .scripted_asr_texts(vec!["明日は晴れですと思い"]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "明日は晴れですと思い",
        &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        &[
            vad(true),
            vad(true),
            vad(true),
            vad(false),
            vad(false),
            vad(false),
            vad(true),
            vad(true),
            vad(true),
        ],
        vec![boundary_candidate(
            "明日は晴れです",
            6,
            6,
            6,
            GrammarBoundaryClass::StrongEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);
    let timeout_chunks =
        usize::try_from(runtime.timeout_ticks()).expect("timeout ticks should fit usize");
    push_silence_chunks(&mut runtime, &config, 16_000, timeout_chunks);
    let timeout_rerecognition = runtime
        .take_last_dispatched()
        .expect("open turn timeout should dispatch rerecognition before final");
    assert_eq!(timeout_rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        timeout_rerecognition.target.turn_id,
        TurnId(1),
        "timeout ASR should still target the same open turn"
    );

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.id.as_str(),
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.output_sequence,
            ))
            .collect::<Vec<_>>(),
        vec![
            ("turn-1-1-0", "明日は晴れですと思い...", false, 1, 1),
            ("turn-1-1-0", "明日は晴れですと思い。", true, 1, 2),
        ],
        "timeout final should finalize the unsplit open turn"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_stale_finalization_after_internal_grammar_boundary_finalizes_unsplit_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = recognized_turn_with_boundary_candidates(
        1,
        "前文。後文",
        &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
        &[
            vad(true),
            vad(true),
            vad(false),
            vad(false),
            vad(true),
            vad(true),
        ],
        vec![boundary_candidate(
            "前文。",
            4,
            4,
            4,
            GrammarBoundaryClass::StrongEnd,
        )],
    );
    turn.draft_mut().processing_millis = 42;
    runtime_state(&mut runtime)
        .turn(1, turn)
        .turn_audio_range(1, 0..6);

    runtime.process_grammar_boundaries_after_rerecognition(1);
    runtime_state(&mut runtime)
        .turn(3, recognized_turn_with_audio(3, "次文", &[5.0, 6.0]))
        .turn_audio_range(3, 6..8);

    runtime.emit_stale_turn_finals(3);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.id.as_str(),
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.output_sequence,
                output.elapsed_millis,
                output.phrase.clone(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "turn-1-1-0",
                "前文。後文...",
                false,
                1,
                1,
                42,
                vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
            ),
            (
                "turn-1-1-0",
                "前文。後文。",
                true,
                1,
                2,
                42,
                vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
            ),
        ],
        "stale finalization should finalize the unsplit turn after an internal grammar boundary"
    );
    assert!(!runtime.turn_store.turns.contains_key(&1));
}

#[test]
fn turn_runtime_terminal_grammar_boundary_finalizes_whole_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = recognized_turn_with_boundary_candidates(
        1,
        "一。二。三。",
        &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0],
        &[
            vad(true),
            vad(true),
            vad(false),
            vad(false),
            vad(true),
            vad(true),
            vad(false),
            vad(false),
            vad(true),
            vad(true),
        ],
        vec![
            boundary_candidate("一。", 4, 4, 4, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("一。二。", 8, 8, 8, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("一。二。三。", 10, 10, 10, GrammarBoundaryClass::StrongEnd),
        ],
    );
    turn.draft_mut().processing_millis = 10;
    runtime_state(&mut runtime)
        .turn(1, turn)
        .turn_audio_range(1, 0..10);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.id.as_str(),
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.output_sequence,
                output.elapsed_millis,
                output.phrase.clone(),
            ))
            .collect::<Vec<_>>(),
        vec![(
            "turn-1-1-0",
            "一。二。三。",
            true,
            1,
            1,
            10,
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0]
        )],
        "terminal grammar boundary should finalize the whole completion ASR text"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_internal_grammar_boundary_does_not_drop_suffix_text() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let text = "はいはいはいはいはいはい";
    let turn = recognized_turn_with_boundary_candidates(
        1,
        text,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0],
        &[vad(true), vad(true), vad(true), vad(false)],
        vec![
            boundary_candidate("はい", 2, 12, 9, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("はいはい", 4, 12, 9, GrammarBoundaryClass::StrongEnd),
            boundary_candidate("はいはいはい", 6, 12, 9, GrammarBoundaryClass::StrongEnd),
            boundary_candidate(
                "はいはいはいはい",
                8,
                12,
                9,
                GrammarBoundaryClass::StrongEnd,
            ),
            boundary_candidate(
                "はいはいはいはいはい",
                10,
                12,
                9,
                GrammarBoundaryClass::StrongEnd,
            ),
            boundary_candidate(text, 12, 12, 9, GrammarBoundaryClass::StrongEnd),
        ],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.emit_turn_output(1, false);
    runtime.process_grammar_boundaries_after_rerecognition(1);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.id.as_str(),
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.output_sequence,
            ))
            .collect::<Vec<_>>(),
        vec![
            ("turn-1-1-0", "はいはいはいはいはいはい...", false, 1, 1),
            ("turn-1-1-0", "はいはいはいはいはいはい。", true, 1, 2),
        ],
        "an internal grammar candidate must not finalize only the prefix while suffix text remains"
    );
}
