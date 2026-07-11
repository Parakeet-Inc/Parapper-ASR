use super::super::*;

#[test]
fn route_selection_for_namo_root_segment_connected_to_open_turn_uses_open_turn_route() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let open_route = RecognitionRoute::from_language(crate::config::AsrLanguage::English);
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    {
        let draft = turn.draft_mut();
        draft.append_recognized_segment(
            1,
            None,
            &[1.0],
            &[vad(true)],
            open_route,
            "open turn".to_string(),
            0,
        );
    }
    runtime_state(&mut runtime)
        .turn(1, turn)
        .open_turn(1)
        .pending_segment(
            2,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            100..180,
        );

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].target.turn_id, TurnId(1));
    assert_eq!(
        submitted[0].route, open_route,
        "a root segment connected to an open Namo turn must use the open turn route"
    );
}

#[test]
fn namo_streaming_interim_open_turn_does_not_capture_next_root_before_td_continue() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B560MsInt8)
        .turn_detector(TurnDetector::Namo)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();

    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::InterimChunkReached,
        0..8_960,
    );
    runtime.step();
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("streaming interim should dispatch before the turn is checked");
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(interim.target.turn_id, TurnId(1));
    asr_handle.complete_request_with_text(&interim, "first draft");
    runtime.step();

    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(1),
        "interim display may keep the draft visible as an open turn"
    );

    runtime_state(&mut runtime).pending_segment(
        2,
        None,
        SegmentCloseReason::EndSilenceReached,
        10_000..12_000,
    );
    runtime.step();

    let completion = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("new root segment completion should dispatch");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        completion.target.turn_id,
        TurnId(2),
        "a mere interim display must not make the next root segment attach to the previous turn before TD Continue"
    );
}

#[test]
fn split_asr_interim_display_uses_interim_model() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8)
        .turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::InterimResultSilenceReached,
        0..16_000,
    );

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8),
        "interim display ASR must use the interim-only ASR model instead of the primary model"
    );
}

#[test]
fn split_asr_completion_uses_primary_model_even_after_interim_draft_route() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8)
        .turn_detector(TurnDetector::Namo);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    turn.draft_mut().append_recognized_segment(
        1,
        None,
        &[1.0],
        &[vad(true)],
        RecognitionRoute::from_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8),
        "interim".to_string(),
        0,
    );
    runtime_state(&mut runtime)
        .turn(1, turn)
        .open_turn(1)
        .pending_segment(2, Some(1), SegmentCloseReason::EndSilenceReached, 10..20);

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(submitted[0].target.turn_id, TurnId(1));
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        "completion ASR must switch from the interim draft route to the primary ASR model"
    );
}

#[test]
fn split_asr_rerecognition_uses_primary_model_even_after_interim_draft_route() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8)
        .turn_detector(TurnDetector::Namo);
    let _asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let mut turn = Turn::new("turn-1-1-0".to_string(), 0);
    turn.draft_mut().append_recognized_segment(
        1,
        None,
        &[1.0; 16],
        &[vad(true)],
        RecognitionRoute::from_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8),
        "interim".to_string(),
        0,
    );
    runtime_state(&mut runtime).turn(1, turn).open_turn(1);

    assert!(
        runtime
            .dispatch_rerecognition_for_turn_if_idle(1, RerecognitionPurpose::SimpleTurnCheckFinal)
    );

    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("rerecognition request should be in flight");
    assert_eq!(request.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        request.route,
        RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        "final rerecognition must prefer the primary ASR model over the interim draft route"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_completion_uses_sli_route_for_first_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..16_000,
    );

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![16_000],
        "completion ASR should run SLI on the same full-turn audio before selecting the ASR route"
    );
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        submitted[0].close_reason,
        Some(SegmentCloseReason::EndSilenceReached)
    );
    assert_eq!(submitted[0].detected_language.as_deref(), Some("en"));
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "multilingual ASR must dispatch the completion request to the route selected by SLI"
    );
}

#[test]
fn non_multilingual_japanese_parakeet_completion_keeps_selected_model_with_sli_runtime() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8)
        .language_id_runtime()
        .turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).pending_segment(
        1,
        None,
        SegmentCloseReason::EndSilenceReached,
        0..16_000,
    );

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8),
        "non-multilingual completion fallback must use the configured Japanese ASR model, not the Japanese default model"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_batched_turn_check_runs_sli_before_dispatching_completion() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .turn_check_silence_ms(64);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();

    runtime.push_vad_frame(&vec![1.0; 16_000], vad(true));
    runtime.push_vad_frame(&vec![0.0; 512], vad(false));
    runtime.push_vad_frame(&vec![0.0; 512], vad(false));

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![17_024],
        "SLI must run as soon as the completion silence is observed, before any interim ASR request is dispatched"
    );
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
    );
    assert_eq!(submitted[0].detected_language.as_deref(), Some("en"));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_interim_reuses_last_recognized_route_without_sli() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime)
        .last_recognition_route(RecognitionRoute::from_model(
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ))
        .pending_segment(
            1,
            None,
            SegmentCloseReason::InterimResultSilenceReached,
            20_000..20_512,
        );
    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        Vec::<usize>::new(),
        "interim ASR must not call SLI again"
    );
    assert_eq!(submitted.len(), 1);
    let interim = &submitted[0];
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(interim.detected_language, None);
    assert_eq!(
        interim.route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "a new interim segment should reuse the last recognized route"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_non_silence_completion_reuses_last_route_without_sli() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["ja"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime)
        .last_recognition_route(RecognitionRoute::from_model(
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ))
        .pending_segment(
            1,
            None,
            SegmentCloseReason::SegmentMaxChunksReached,
            0..MIN_LANGUAGE_ID_SAMPLES as u64,
        );

    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        Vec::<usize>::new(),
        "only end-silence completion checks should invoke SLI"
    );
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "non-silence completion should inherit the last recognized route without language detection"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn refresh_turn_route_with_sli_ignores_missing_turn_without_calling_language_id() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .scripted_asr_texts(Vec::new());
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();

    runtime.refresh_turn_route_with_sli(404);

    assert!(
        sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable")
            .is_empty(),
        "refreshing a missing turn must not call the language detector"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn refresh_turn_route_with_sli_ignores_empty_turn_audio_without_calling_language_id() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .scripted_asr_texts(Vec::new());
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime_state(&mut runtime).turn(1, Turn::new("turn-1-1-0".to_string(), 0));

    runtime.refresh_turn_route_with_sli(1);

    assert!(
        sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable")
            .is_empty(),
        "refreshing a turn without audio must not call the language detector"
    );
    assert!(
        runtime
            .turn_store
            .turns
            .get(&1)
            .expect("test turn should remain")
            .draft()
            .route
            .is_none(),
        "an empty-audio refresh must not invent a route on the draft"
    );
}

#[test]
fn refresh_turn_route_with_sli_ignores_runtime_without_language_id_runtime() {
    let sli_call_audio_lens = Arc::new(Mutex::new(Vec::new()));
    let mut builder = RecognitionSessionTestBuilder::new()
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .scripted_asr_texts(Vec::new());
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    runtime.io.language_id = Some(Box::new(ScriptedLanguageDetector {
        detected_languages: vec!["en".to_string()].into(),
        call_audio_lens: sli_call_audio_lens.clone(),
    }));
    let turn_audio = vec![1.0; MIN_LANGUAGE_ID_SAMPLES];
    runtime_state(&mut runtime).turn(
        1,
        recognized_turn_with_audio(1, "route refresh", &turn_audio),
    );

    runtime.refresh_turn_route_with_sli(1);

    assert!(
        sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable")
            .is_empty(),
        "a runtime without a SLI handle must not call a detector even if one is present"
    );
    assert_eq!(
        runtime
            .turn_store
            .turns
            .get(&1)
            .expect("test turn should remain")
            .draft()
            .route,
        Some(RecognitionRoute::from_language(AsrLanguage::Japanese))
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn refresh_turn_route_with_sli_updates_turn_route_from_full_audio() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .scripted_asr_texts(Vec::new());
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let turn_audio = vec![1.0; MIN_LANGUAGE_ID_SAMPLES];
    runtime_state(&mut runtime).turn(
        1,
        recognized_turn_with_audio(1, "route refresh", &turn_audio),
    );

    runtime.refresh_turn_route_with_sli(1);

    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![MIN_LANGUAGE_ID_SAMPLES],
        "route refresh must run SLI over the full accumulated turn audio"
    );
    let draft = runtime
        .turn_store
        .turns
        .get(&1)
        .expect("refreshed turn should remain")
        .draft();
    assert_eq!(
        draft.route,
        Some(RecognitionRoute::from_model(
            AsrModel::NemoParakeetTdt0_6BV2Int8
        ))
    );
    assert_eq!(draft.detected_language.as_deref(), Some("en"));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_timeout_refreshes_sli_route_before_rerecognition() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, _config) = builder.build();
    let timeout_ticks = runtime.timeout_ticks();
    let turn_audio = vec![1.0; MIN_LANGUAGE_ID_SAMPLES];
    runtime_state(&mut runtime)
        .turn(
            1,
            recognized_turn_with_audio(1, "timeout draft", &turn_audio),
        )
        .turn_audio_range(1, 0..MIN_LANGUAGE_ID_SAMPLES as u64)
        .open_turn_since(1, 0)
        .next_runtime_tick(timeout_ticks);

    assert!(runtime.handle_open_turn_timeout());

    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![MIN_LANGUAGE_ID_SAMPLES],
        "timeout finalization should classify the full open-turn audio before rerecognition"
    );
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "timeout rerecognition must use the SLI-selected route"
    );
    assert_eq!(submitted[0].detected_language.as_deref(), Some("en"));
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_update_config_disables_multilingual_asr_and_drops_language_id() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .scripted_asr_texts(Vec::new());
    let _ = builder.use_scripted_language_detector(Vec::new());
    let _outputs = builder.use_recording_sink();
    let (mut runtime, enabled_config) = builder.build();
    runtime_state(&mut runtime).last_recognition_route(RecognitionRoute::from_model(
        AsrModel::NemoParakeetTdt0_6BV2Int8,
    ));

    runtime.update_config(&parapper_config! {
        multilingual_asr_enabled: false,
        ..enabled_config
    });

    assert!(
        runtime.io.language_id.is_none(),
        "turning off multilingual ASR must discard the language detector"
    );
    assert!(
        runtime.turn_store.last_recognition_route.is_none(),
        "changing multilingual routing must invalidate the cached recognition route"
    );
}
