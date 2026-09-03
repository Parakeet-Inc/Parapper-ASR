use super::*;
use tauri::Manager as _;

#[test]
#[ignore = "diagnostic: loads production model resources and prints startup load timings"]
#[expect(
    clippy::too_many_lines,
    reason = "diagnostic test keeps measured startup steps in one printed sequence"
)]
fn measure_recognition_startup_load_times() {
    use std::{path::PathBuf, time::Instant};

    use anyhow::{Result, anyhow};
    use ort::session::Session;

    fn production_app_data_dir() -> PathBuf {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::generate_context!())
            .expect("Tauri production context should build");
        app.handle()
            .path()
            .app_data_dir()
            .expect("Tauri app data directory should resolve")
    }

    fn measure(label: &str, f: impl FnOnce() -> Result<()>) {
        let started_at = Instant::now();
        match f() {
            Ok(()) => println!(
                "{label}: {:.1} ms",
                started_at.elapsed().as_secs_f64() * 1000.0
            ),
            Err(err) => println!(
                "{label}: {:.1} ms ERROR: {err:#}",
                started_at.elapsed().as_secs_f64() * 1000.0
            ),
        }
    }

    let handle = tauri_test_handle();
    let app_data_dir = production_app_data_dir();
    let config_path = app_data_dir.join("config.json");
    let config = if config_path.is_file() {
        ParapperConfig::load(&config_path).expect("production config should load")
    } else {
        ParapperConfig::default()
    }
    .normalized();
    let models_root = app_data_dir.join("models");
    println!("config: {}", config_path.display());
    println!("models: {}", models_root.display());
    println!(
        "flags: turn_detector={:?} multilingual={} noise_cancellation={} asr_model={:?} enabled_asr_models={:?}",
        config.turn.detector,
        config.asr.multilingual_enabled,
        config.noise_cancellation.enabled,
        config.asr.model,
        config.asr.enabled_models
    );

    let mut no_noise_config = config.clone();
    no_noise_config.noise_cancellation.enabled = false;
    measure(
        "AudioInputProcessor::initialize without noise cancellation",
        || {
            let _processor = crate::audio::AudioInputProcessor::initialize(
                handle.clone(),
                &no_noise_config,
                48_000,
            )?;
            Ok(())
        },
    );

    measure("VAD OnnxRuntimeSileroVadEngine::new", || {
        let vad_path = models_root.join("silero_vad_v6").join("silero_vad.onnx");
        let _vad = parapper_models::vad::OnnxRuntimeSileroVadEngine::new(
            &vad_path,
            config.segmentation.vad_threshold,
        )?;
        Ok(())
    });

    measure("UL-UNAS noise cancellation ONNX session", || {
        let model_path = models_root
            .join("ul-unas")
            .join("ulunas_stream_simple.onnx");
        anyhow::ensure!(model_path.is_file(), "missing {}", model_path.display());
        let builder = Session::builder().map_err(|err| anyhow::anyhow!("{err}"))?;
        let builder = builder
            .with_intra_threads(1)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let builder = builder
            .with_inter_threads(1)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let builder = builder
            .with_parallel_execution(false)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let builder = builder
            .with_intra_op_spinning(false)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let mut builder = builder
            .with_inter_op_spinning(false)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let _session = builder
            .commit_from_file(&model_path)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        Ok(())
    });

    if config.turn_detector_model().is_some() {
        measure("NamoTurnDetectorEngine::new selected model", || {
            let model = crate::model::NamoTurnDetectorModel::for_asr_language(config.asr.language);
            let model_dir =
                crate::model::namo_turn_detector_model_dir_from_root(&models_root, model);
            let tokenizer_kind = match model {
                crate::model::NamoTurnDetectorModel::Japanese => {
                    parapper_models::td::namo::NamoTokenizerKind::Character
                }
                crate::model::NamoTurnDetectorModel::English
                | crate::model::NamoTurnDetectorModel::Multilingual => {
                    parapper_models::td::namo::NamoTokenizerKind::TokenizerJson
                }
            };
            let _engine =
                parapper_models::td::namo::NamoTurnDetectorEngine::new(&model_dir, tokenizer_kind)?;
            Ok(())
        });
    } else {
        println!("NamoTurnDetectorEngine::new selected model: skipped for Simple TD");
    }

    measure("SpokenLanguageIdentificationEngine::new", || {
        let model_dir = models_root.join("speechbrain-lang-id-voxlingua107-ecapa-onnx");
        let _engine = parapper_models::asr::SpokenLanguageIdentificationEngine::new(
            &model_dir,
            config.asr.num_threads.max(1),
        )?;
        Ok(())
    });

    measure("embedded JapaneseMorphAnalyzer initialization", || {
        crate::recognition::turn_adapter::load_japanese_morph_analyzer()
            .ok_or_else(|| anyhow!("embedded Japanese morph dictionary failed to load"))?;
        Ok(())
    });
}

#[test]
#[ignore = "requires local FLEURS-R wavs and installed production SLI model"]
#[expect(
    clippy::too_many_lines,
    reason = "diagnostic test keeps the alternating language scenario readable as one sequence"
)]
fn fleurs_r_alternating_languages_keep_detected_language_route_and_output_consistent() {
    use std::path::{Path, PathBuf};

    fn production_app_data_dir() -> PathBuf {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::generate_context!())
            .expect("Tauri production context should build");
        app.handle()
            .path()
            .app_data_dir()
            .expect("Tauri app data directory should resolve")
    }

    fn fleurs_root() -> PathBuf {
        test_env_path("FLEURS_R_ROOT")
    }

    fn first_fleurs_wav(root: &Path, locale: &str) -> FleursPart {
        let split_dir = root.join(locale).join("dev").join("dev");
        let wav_path = fs::read_dir(&split_dir)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", split_dir.display()))
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            })
            .min()
            .unwrap_or_else(|| panic!("no wav files found under {}", split_dir.display()));
        let (samples, sample_rate) = read_pcm16_wav_mono_f32(&wav_path);
        let samples =
            resample_linear_for_fleurs(&samples, sample_rate, crate::audio::ASR_SAMPLE_RATE);
        FleursPart {
            locale: locale.to_string(),
            wav_path,
            samples,
            sample_rate: crate::audio::ASR_SAMPLE_RATE,
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "diagnostic WAV resampling converts bounded sample positions between integer indices and fractional interpolation weights"
    )]
    fn resample_linear_for_fleurs(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
        if source_rate == target_rate {
            return samples.to_vec();
        }
        let target_len = (samples.len() as u128 * u128::from(target_rate))
            .div_ceil(u128::from(source_rate)) as usize;
        (0..target_len)
            .map(|index| {
                let position = index as f64 * f64::from(source_rate) / f64::from(target_rate);
                let left = position.floor() as usize;
                let right = (left + 1).min(samples.len().saturating_sub(1));
                let fraction = (position - left as f64) as f32;
                samples.get(left).copied().unwrap_or(0.0) * (1.0 - fraction)
                    + samples.get(right).copied().unwrap_or(0.0) * fraction
            })
            .collect()
    }

    fn expected_route_for(language: &str) -> RecognitionRoute {
        match language {
            "ja" => RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
            "en" => RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
            "fr" => RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV3Int8),
            _ => panic!("unexpected language in diagnostic: {language}"),
        }
    }

    let language_id_model_dir =
        diagnostic_models_root().join(crate::model::catalog::language_id_model_dir_name());
    let language_id =
        parapper_models::asr::SpokenLanguageIdentificationEngine::new(&language_id_model_dir, 1)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to load SLI model from {}: {err:#}",
                    language_id_model_dir.display()
                )
            });
    let root = fleurs_root();
    let sequence = [
        ("ja", first_fleurs_wav(&root, "ja_jp")),
        ("en", first_fleurs_wav(&root, "en_us")),
        ("fr", first_fleurs_wav(&root, "fr_fr")),
        ("ja", first_fleurs_wav(&root, "ja_jp")),
        ("en", first_fleurs_wav(&root, "en_us")),
        ("fr", first_fleurs_wav(&root, "fr_fr")),
    ];
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
            AsrModel::NemoParakeetTdt0_6BV3Int8,
        ])
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .segment_start_speech_ms(1)
        .turn_check_silence_ms(64)
        .interim_display(false)
        .language_id_runtime();
    builder.language_id = Some(Box::new(
        crate::recognition::language_adapter::AppLanguageDetector::new(language_id),
    ));
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    for (expected_language, part) in sequence {
        push_fleurs_speech_chunks(&mut runtime, &config, &part);
        push_silence_chunks(&mut runtime, &config, part.sample_rate, 3);
        let submitted = asr_handle.submitted_requests();
        let request = submitted
            .last()
            .unwrap_or_else(|| {
                panic!(
                    "ASR request was not submitted for {}",
                    part.wav_path.display()
                )
            })
            .clone();
        assert_eq!(request.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(
            request.detected_language.as_deref(),
            Some(expected_language)
        );
        assert_eq!(
            request.route,
            expected_route_for(expected_language),
            "SLI route mismatch for {} ({})",
            part.locale,
            part.wav_path.display()
        );
        println!(
            "{} {} -> detected={:?} route={:?}",
            part.locale,
            part.wav_path.display(),
            request.detected_language,
            request.route
        );

        let display_text = format!("{expected_language}-display");
        asr_handle.complete_next_with_text(&display_text);
        runtime.step();
        let outputs = outputs.lock().expect("phrase outputs should be readable");
        let latest = outputs.last().unwrap_or_else(|| {
            panic!(
                "final output was not emitted for {}",
                part.wav_path.display()
            )
        });
        assert!(latest.is_final);
        assert_eq!(latest.detected_language.as_deref(), Some(expected_language));
        assert_eq!(latest.source_asr_model, request.route.model);
        assert_eq!(latest.source_language, request.route.language);
        assert!(
            latest.text.starts_with(&display_text),
            "display text must come from the ASR request selected for {expected_language}, got {:?}",
            latest.text
        );
        drop(outputs);
    }
}

#[test]
#[ignore = "loads production model resources; use for local e2e smoke testing of production wiring"]
fn turn_runtime_production_e2e_smoke_accepts_vad_frames_and_shutdowns() {
    const SILERO_CHUNK_SAMPLES: usize = 512;
    let handle = tauri_test_handle();
    let config = parapper_config! {
        vad_interval_ms: 32,
        turn_check_silence_ms: 32,
        segment_start_speech_ms: 1,
        interim_result_enabled: false,
        ..ParapperConfig::default()
    };
    let mut runtime = RecognitionDriver::new_for_production(&handle, &config, None);
    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(0.1, SILERO_CHUNK_SAMPLES, true),
            fixed_vad_frame(0.0, SILERO_CHUNK_SAMPLES, false),
        ],
    );
    for _ in 0..4 {
        runtime.step();
    }
    let dispatched = runtime
        .requests
        .last_dispatched
        .as_ref()
        .expect("production runtime smoke should dispatch ASR work before shutdown");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    runtime.shutdown();
    assert!(
        runtime.pending.asr_segments.is_empty(),
        "closed segment should have been consumed into the dispatched ASR request"
    );
}

#[test]
fn turn_runtime_shutdown_flushes_active_segment_and_finalizes_tail_audio() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["最後まで保存"]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![2.0], vad(true)),
            (vec![3.0], vad(true)),
        ],
    );

    runtime.shutdown();

    assert_eq!(
        *outputs.lock().expect("phrase outputs should be readable"),
        vec![PhraseOutputSnapshot {
            id: "turn-1-1-0".to_string(),
            text: "最後まで保存。".to_string(),
            is_final: true,
            source_asr_model: config.asr.model,
            source_language: config.asr.language,
            detected_language: None,
            turn_session_id: 1,
            turn_id: 1,
            segment_id: 1,
            output_sequence: 1,
            phrase: vec![1.0, 2.0, 3.0],
            elapsed_millis: 0,
        }],
        "shutdown must flush an active segment so final text and saved phrase audio include the tail"
    );
}

#[test]
fn turn_runtime_shutdown_keeps_internal_grammar_boundary_in_same_turn_and_finalizes_tail_audio() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8)
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_transcripts(vec![
            AsrTranscript::from_text("前半後半"),
            japanese_timestamped_transcript("前半。後半"),
            AsrTranscript::from_text("追加"),
            AsrTranscript::from_text("前半。後半追加"),
        ]);
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![2.0], vad(true)),
            (vec![3.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![4.0], vad(true)),
            (vec![5.0], vad(true)),
            (vec![6.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );
    runtime.step();
    runtime.step();

    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(1),
        "internal grammar boundary should keep the original turn open before video-end shutdown"
    );
    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![7.0], vad(true)), (vec![8.0], vad(true))],
    );

    runtime.shutdown();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["前半。後半追加".to_string()],
        "shutdown should drive the flushed turn through Namo before final fallback"
    );
    let outputs = outputs.lock().expect("phrase outputs should be readable");
    assert_eq!(
        outputs
            .iter()
            .map(|output| (
                output.text.as_str(),
                output.is_final,
                output.turn_id,
                output.segment_id,
                output.phrase.clone(),
            ))
            .collect::<Vec<_>>(),
        vec![(
            "前半。後半追加。",
            true,
            1,
            2,
            vec![
                1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0, 8.0
            ]
        )],
        "Namo shutdown must finalize the same open turn and keep the active tail audio"
    );
}

fn japanese_timestamped_transcript(text: &str) -> AsrTranscript {
    let tokens = text.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
    let timestamps = (0..tokens.len())
        .map(|index| {
            f32::from(u16::try_from(index).expect("test transcript should have few tokens"))
                / 16_000.0
        })
        .collect::<Vec<_>>();
    let durations = vec![1.0 / 16_000.0; tokens.len()];
    AsrTranscript::from_parts(
        text.to_string(),
        tokens,
        Some(&timestamps),
        Some(&durations),
    )
}
