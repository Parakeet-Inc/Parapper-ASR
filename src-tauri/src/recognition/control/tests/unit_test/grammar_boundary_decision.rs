use super::super::*;

#[test]
fn turn_runtime_grammar_boundary_reject_does_not_hide_terminal_strong_end() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "弱い確定",
        &[1.0, 2.0, 3.0, 4.0],
        &[vad(true), vad(true), vad(true), vad(false)],
        vec![
            boundary_candidate("弱い", 2, 2, 2, GrammarBoundaryClass::Reject),
            boundary_candidate("弱い確定", 4, 4, 4, GrammarBoundaryClass::StrongEnd),
        ],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("弱い確定。", true, 1, 1)],
        "a leading Reject candidate must not prevent a terminal StrongEnd from finalizing the turn"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_grammar_boundary_internal_predicate_before_connection_keeps_turn_open() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "しようとしたら",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[
            vad(true),
            vad(true),
            vad(true),
            vad(true),
            vad(true),
            vad(true),
        ],
        vec![boundary_candidate(
            "しよう",
            3,
            3,
            3,
            GrammarBoundaryClass::PredicateEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("しようとしたら...", false, 1, 1)],
        "an internal predicate end followed by a connective suffix must not split the turn"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_grammar_boundary_internal_normal_before_desu_kedo_keeps_turn_open_without_namo() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 1.0,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "〇〇なんですけど",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[
            vad(true),
            vad(true),
            vad(true),
            vad(true),
            vad(true),
            vad(true),
        ],
        vec![boundary_candidate(
            "〇〇",
            2,
            2,
            2,
            GrammarBoundaryClass::NormalEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        Vec::<String>::new(),
        "Namo must not be asked to complete an internal noun boundary before a connective suffix"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("〇〇なんですけど...", false, 1, 1)],
        "an internal noun boundary followed by なんですけど must not split the turn"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_grammar_boundary_terminal_predicate_end_finalizes_without_namo_decision() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "述語",
        &[1.0, 2.0, 3.0],
        &[vad(true), vad(true), vad(false)],
        vec![boundary_candidate(
            "述語",
            3,
            3,
            3,
            GrammarBoundaryClass::PredicateEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![PhraseOutputSnapshot {
            id: "turn-1-1-0".to_string(),
            text: "述語。".to_string(),
            is_final: true,
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            detected_language: None,
            turn_session_id: 1,
            turn_id: 1,
            segment_id: 1,
            output_sequence: 1,
            phrase: vec![1.0, 2.0, 3.0].into(),
            elapsed_millis: 0,
        },],
        "terminal predicate boundary should finalize the whole turn"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_grammar_boundary_clause_weak_keeps_turn_open() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "弱い節",
        &[1.0, 2.0, 3.0],
        &[vad(true), vad(false), vad(true)],
        vec![boundary_candidate(
            "弱い",
            2,
            2,
            2,
            GrammarBoundaryClass::ClauseWeak,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "弱い節...");
    assert!(!outputs[0].is_final);
    assert_eq!(outputs[0].turn_session_id, 1);
    assert_eq!(outputs[0].turn_id, 1);
    assert_eq!(outputs[0].segment_id, 1);
    assert_eq!(outputs[0].output_sequence, 1);
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_grammar_boundary_normal_end_low_confidence_keeps_turn_open() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .interim_display(true)
        .namo_turn_confidence_threshold(0.8)
        .scripted_asr_texts(Vec::new());
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.79,
    }]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "東京駅",
        &[1.0, 2.0, 3.0],
        &[vad(true), vad(true), vad(true)],
        vec![boundary_candidate(
            "東京駅",
            3,
            3,
            3,
            GrammarBoundaryClass::NormalEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["東京駅".to_string()]
    );
    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "東京駅...");
    assert!(!outputs[0].is_final);
    assert_eq!(outputs[0].turn_session_id, 1);
    assert_eq!(outputs[0].turn_id, 1);
    assert_eq!(outputs[0].segment_id, 1);
    assert_eq!(outputs[0].output_sequence, 1);
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_morph_terminal_normal_end_finalizes_without_namo_decision() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Morph)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 1.0,
    }]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "東京駅",
        &[1.0, 2.0, 3.0],
        &[vad(true), vad(true), vad(true)],
        vec![boundary_candidate(
            "東京駅",
            3,
            3,
            3,
            GrammarBoundaryClass::NormalEnd,
        )],
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        Vec::<String>::new(),
        "Morph must not ask Namo to confirm terminal NormalEnd"
    );
    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "東京駅。");
    assert!(outputs[0].is_final);
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_morph_no_grammar_candidate_keeps_turn_open_without_namo() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Morph)
        .interim_display(true)
        .scripted_asr_texts(Vec::new());
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 1.0,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn = recognized_turn_with_boundary_candidates(
        1,
        "境界なし",
        &[1.0, 2.0, 3.0],
        &[vad(true), vad(true), vad(true)],
        Vec::new(),
    );
    runtime_state(&mut runtime).turn(1, turn);

    runtime.process_grammar_boundaries_after_rerecognition(1);

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        Vec::<String>::new(),
        "Morph must not fall back to Namo when grammar has no candidate"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("境界なし...", false, 1, 1)]
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}
