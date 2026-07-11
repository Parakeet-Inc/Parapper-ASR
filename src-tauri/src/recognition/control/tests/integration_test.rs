use super::*;
use tauri::Manager as _;

#[test]
#[ignore = "diagnostic: loads production model resources and prints startup load timings"]
#[expect(
    clippy::too_many_lines,
    reason = "diagnostic test keeps measured startup steps in one printed sequence"
)]
fn measure_recognition_startup_load_times() {
    use std::{
        fs,
        io::{Read as _, Write as _},
        path::{Path, PathBuf},
        time::Instant,
    };

    use anyhow::{Context as _, Result};
    use ort::session::Session;
    use vibrato_rkyv::{Dictionary, LoadMode, Tokenizer};

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

    fn measure_value<T>(label: &str, f: impl FnOnce() -> Result<T>) -> Option<T> {
        let started_at = Instant::now();
        match f() {
            Ok(value) => {
                println!(
                    "{label}: {:.1} ms",
                    started_at.elapsed().as_secs_f64() * 1000.0
                );
                Some(value)
            }
            Err(err) => {
                println!(
                    "{label}: {:.1} ms ERROR: {err:#}",
                    started_at.elapsed().as_secs_f64() * 1000.0
                );
                None
            }
        }
    }

    fn vibrato_rkyv_dictionary_compatible(path: &Path) -> Result<bool> {
        let mut file =
            fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut magic = vec![0; crate::model::catalog::VIBRATO_MODEL_MAGIC.len()];
        file.read_exact(&mut magic)
            .with_context(|| format!("failed to read magic from {}", path.display()))?;
        Ok(magic == crate::model::catalog::VIBRATO_MODEL_MAGIC)
    }

    fn transcode_legacy_zstd_dictionary_to_rkyv(
        compressed_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let input = fs::File::open(compressed_path)
            .with_context(|| format!("failed to open {}", compressed_path.display()))?;
        let decoder = zstd::Decoder::new(input)
            .with_context(|| format!("failed to decode {}", compressed_path.display()))?;
        let temporary_path = output_path.with_extension("dic.transcoding");
        let mut output = fs::File::create(&temporary_path)
            .with_context(|| format!("failed to create {}", temporary_path.display()))?;
        let dictionary = unsafe { Dictionary::from_legacy_reader(decoder) }.with_context(|| {
            format!(
                "failed to read legacy Vibrato dictionary from {}",
                compressed_path.display()
            )
        })?;
        dictionary
            .write(&mut output)
            .with_context(|| format!("failed to write {}", temporary_path.display()))?;
        output
            .flush()
            .with_context(|| format!("failed to flush {}", temporary_path.display()))?;
        drop(output);
        fs::rename(&temporary_path, output_path).with_context(|| {
            format!(
                "failed to move {} to {}",
                temporary_path.display(),
                output_path.display()
            )
        })?;
        Ok(())
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
        let _vad = crate::recognition::segmentation::vad::engine::OnnxRuntimeSileroVadEngine::new(
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
                    crate::recognition::turn::decision::engine::NamoTokenizerKind::Character
                }
                crate::model::NamoTurnDetectorModel::English
                | crate::model::NamoTurnDetectorModel::Multilingual => {
                    crate::recognition::turn::decision::engine::NamoTokenizerKind::TokenizerJson
                }
            };
            let _engine = crate::recognition::turn::decision::engine::NamoTurnDetectorEngine::new(
                &model_dir,
                tokenizer_kind,
            )?;
            Ok(())
        });
    } else {
        println!("NamoTurnDetectorEngine::new selected model: skipped for Simple TD");
    }

    measure("SpokenLanguageIdentificationEngine::new", || {
        let model_dir = models_root.join("speechbrain-lang-id-voxlingua107-ecapa-onnx");
        let _engine =
            crate::recognition::transcription::route::language_id::engine::SpokenLanguageIdentificationEngine::new(
                &model_dir,
                config.asr.num_threads.max(1),
            )?;
        Ok(())
    });

    let dictionary_path =
        crate::model::japanese_morph_dictionary_paths_from_root(&models_root)
            .into_iter()
            .find(|path| path.is_file())
            .or_else(|| {
                let compressed_path = models_root
                    .join("unidic-cwj-3_1_1")
                    .join("system.dic.zst");
                if !compressed_path.is_file() {
                    return None;
                }
                let diagnostic_path = std::env::current_dir()
                    .ok()?
                    .join("target")
                    .join("vibrato-rkyv-diagnostic")
                    .join("system.dic");
                println!(
                    "Japanese Vibrato rkyv dictionary path: not found; using diagnostic transcode from {}",
                    compressed_path.display()
                );
                let needs_transcode = !diagnostic_path.is_file()
                    || !vibrato_rkyv_dictionary_compatible(&diagnostic_path).unwrap_or(false);
                if needs_transcode {
                    measure("Vibrato-rkyv legacy zstd -> rkyv diagnostic transcode", || {
                        transcode_legacy_zstd_dictionary_to_rkyv(
                            &compressed_path,
                            &diagnostic_path,
                        )
                    });
                }
                diagnostic_path.is_file().then_some(diagnostic_path)
            });
    let Some(dictionary_path) = dictionary_path else {
        println!("Japanese Vibrato dictionary path: not found");
        return;
    };
    println!(
        "Japanese Vibrato dictionary path: {}",
        dictionary_path.display()
    );

    measure("JapaneseMorphAnalyzer::from_dictionary_path", || {
        let _analyzer =
            crate::recognition::turn::boundary::JapaneseMorphAnalyzer::from_dictionary_path(
                &dictionary_path,
            )?;
        Ok(())
    });

    let Some(_bytes) = measure_value("Vibrato dictionary fs::read", || {
        std::fs::read(&dictionary_path)
            .with_context(|| format!("failed to read {}", dictionary_path.display()))
    }) else {
        return;
    };
    let Some(dictionary) = measure_value("Vibrato-rkyv Dictionary::from_path", || {
        Dictionary::from_path(&dictionary_path, LoadMode::TrustCache)
            .context("failed to parse dictionary")
    }) else {
        return;
    };
    measure("Vibrato-rkyv Tokenizer::new", || {
        let _tokenizer = Tokenizer::new(dictionary);
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
        crate::recognition::transcription::route::language_id::engine::SpokenLanguageIdentificationEngine::new(
            &language_id_model_dir,
            1,
        )
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
    builder.language_id = Some(Box::new(language_id));
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
fn turn_runtime_completion_request_preserves_production_sized_vad_frame_range() {
    const SILERO_CHUNK_SAMPLES: usize = 512;
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, SILERO_CHUNK_SAMPLES, true),
            fixed_vad_frame(0.0, SILERO_CHUNK_SAMPLES, false),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("a production-sized speech/silence pair should dispatch completion ASR");
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(
        dispatched.target.range.end_sample,
        GlobalSampleIndex((SILERO_CHUNK_SAMPLES * 2) as u64)
    );
    let request = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("completion request should remain in flight");
    assert_eq!(request.audio.len(), SILERO_CHUNK_SAMPLES * 2);
    assert_eq!(request.vad_results.len(), 2);
}

#[test]
fn turn_runtime_enqueues_completion_asr_from_closed_segment() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("closed segment must enqueue and dispatch one ASR request");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    assert_eq!(dispatched.target.turn_revision, TurnRevision(0));
    assert_eq!(dispatched.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(dispatched.target.range.end_sample, GlobalSampleIndex(2));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
}

#[test]
fn turn_runtime_enqueues_interim_asr_without_finishing_turn() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("interim silence must enqueue and dispatch one interim ASR request");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(dispatched.target.turn_id, TurnId(1));
    assert_eq!(dispatched.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
}

#[test]
fn turn_runtime_nemotron_interim_dispatches_160ms_chunks_during_active_speech() {
    const FRAME_SAMPLES: usize = 256;
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..9).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );

    assert!(
        asr_handle.submitted_requests().is_empty(),
        "Nemotron interim must wait until the first full 160ms chunk is available"
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0; FRAME_SAMPLES], vad(true))],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    let request = &submitted[0];
    assert_eq!(request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        request.route,
        RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
    );
    assert_eq!(
        request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    assert_eq!(
        request.audio.len(),
        NEMOTRON_CHUNK_SAMPLES,
        "the runtime request should stay on the 160ms grid before Nemotron-specific worker padding"
    );
    assert_eq!(request.target.range.start_sample, GlobalSampleIndex(0));
    assert_eq!(
        request.target.range.end_sample,
        GlobalSampleIndex(NEMOTRON_CHUNK_SAMPLES as u64)
    );
}

#[test]
fn turn_runtime_streaming_interim_ignores_silence_snapshot_request() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 256], vad(true)),
            (vec![0.0; 256], vad(false)),
            (vec![2.0; 256], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert!(
        submitted.is_empty(),
        "streaming interim ASR must not also submit the non-streaming silence snapshot request"
    );
}

#[test]
fn turn_runtime_streaming_interim_silence_threshold_does_not_split_completion_request() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(64);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..10).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let streaming_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first streaming interim request should be in flight");
    assert_eq!(streaming_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        streaming_request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    asr_handle.complete_request_with_text(&streaming_request, "途中");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![0.0; FRAME_SAMPLES], vad(false)))
            .chain(std::iter::once((vec![2.0; FRAME_SAMPLES], vad(true))))
            .chain((0..4).map(|_| (vec![0.0; FRAME_SAMPLES], vad(false)))),
    );

    let completion = runtime
        .requests
        .in_flight_request
        .as_ref()
        .expect("turn-check silence should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        completion.target.first_segment_id,
        Some(SegmentId(1)),
        "streaming interim must not let interim_result_silence_ms split the logical completion segment"
    );
    assert_eq!(
        completion.target.last_segment_id,
        Some(SegmentId(1)),
        "completion should still target the original segment instead of a silence-threshold child segment"
    );
    assert_eq!(
        completion.source_audio.len(),
        FRAME_SAMPLES * 16,
        "completion source audio should cover the whole utterance regardless of interim_result_silence_ms"
    );
}

#[test]
fn turn_runtime_streaming_interim_continues_across_interim_silence_without_duplicate_prespeech() {
    const FRAME_SAMPLES: usize = 256;
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..10).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let first_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first 160ms streaming interim request should be in flight");
    asr_handle.complete_request_with_text(&first_request, "最初");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        std::iter::once((vec![0.0; FRAME_SAMPLES], vad(false)))
            .chain((0..9).map(|_| (vec![2.0; FRAME_SAMPLES], vad(true)))),
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 2);
    let second_request = &submitted[1];
    assert_eq!(second_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        second_request.close_reason,
        Some(SegmentCloseReason::InterimChunkReached)
    );
    assert_eq!(
        second_request.target.first_segment_id,
        Some(SegmentId(1)),
        "streaming interim should keep updating the logical utterance segment across interim-threshold silence"
    );
    assert_eq!(
        second_request.target.last_segment_id,
        Some(SegmentId(1)),
        "streaming interim output should replace the same draft segment instead of appending duplicate cumulative audio"
    );
    assert_eq!(second_request.audio.len(), NEMOTRON_CHUNK_SAMPLES);
    assert!(
        second_request.audio[..FRAME_SAMPLES]
            .iter()
            .all(|sample| *sample == 0.0),
        "the next streaming delta should include the real silence that occurred after the first chunk"
    );
    assert!(
        second_request.audio[FRAME_SAMPLES..]
            .iter()
            .all(|sample| *sample == 2.0),
        "interim-threshold silence must not create copied pre-speech audio in the streaming delta"
    );
    assert_eq!(
        second_request.source_audio.len(),
        NEMOTRON_CHUNK_SAMPLES * 2
    );
    assert!(
        second_request.source_audio[..NEMOTRON_CHUNK_SAMPLES]
            .iter()
            .all(|sample| *sample == 1.0)
    );
    assert!(
        second_request.source_audio[NEMOTRON_CHUNK_SAMPLES..NEMOTRON_CHUNK_SAMPLES + FRAME_SAMPLES]
            .iter()
            .all(|sample| *sample == 0.0)
    );
    assert!(
        second_request.source_audio[NEMOTRON_CHUNK_SAMPLES + FRAME_SAMPLES..]
            .iter()
            .all(|sample| *sample == 2.0)
    );
}

#[test]
fn turn_runtime_end_silence_discards_queued_nemotron_streaming_interim_chunks() {
    const FRAME_SAMPLES: usize = 256;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .turn_check_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..20).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );

    assert!(
        runtime.requests.in_flight_request.is_some(),
        "the first Nemotron interim chunk should already be in flight"
    );
    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .any(|segment| { segment.reason == SegmentCloseReason::InterimChunkReached }),
        "the second Nemotron interim chunk should be queued before the utterance closes"
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..2).map(|_| (vec![0.0; FRAME_SAMPLES], vad(false))),
    );

    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .all(|segment| { segment.reason != SegmentCloseReason::InterimChunkReached }),
        "queued Nemotron interim audio that was not submitted yet must be discarded when the utterance closes"
    );
    assert!(
        runtime
            .pending
            .asr_segments
            .iter()
            .any(|segment| { segment.reason == SegmentCloseReason::EndSilenceReached }),
        "the final completion candidate must remain queued after discarding interim-only audio"
    );
    assert_eq!(
        asr_handle.streaming_reset_count(),
        1,
        "closing the utterance must reset the Nemotron streaming cache before the next interim session"
    );
}

#[test]
fn turn_runtime_non_streaming_interim_keeps_silence_snapshot_request() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
        .vad_interval_ms(16)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(16)
        .turn_check_silence_ms(320);
    let asr_handle = builder.use_manual_asr();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 256], vad(true)),
            (vec![0.0; 256], vad(false)),
            (vec![2.0; 256], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        submitted[0].close_reason,
        Some(SegmentCloseReason::InterimResultSilenceReached)
    );
    assert_eq!(
        submitted[0].route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
    );
}

#[test]
fn turn_runtime_nemotron_interim_updates_same_segment_without_duplicating_turn_audio() {
    const FRAME_SAMPLES: usize = 256;
    const NEMOTRON_CHUNK_SAMPLES: usize = 2_560;
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .interim_asr_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8)
        .vad_interval_ms(16)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..10).map(|_| (vec![1.0; FRAME_SAMPLES], vad(true))),
    );
    let first_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("the first 160ms Nemotron interim request should be in flight");
    asr_handle.complete_request_with_text(&first_request, "あ");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        (0..10).map(|_| (vec![2.0; FRAME_SAMPLES], vad(true))),
    );
    runtime.step();
    let second_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("the second 160ms boundary should dispatch another interim request");
    assert_eq!(second_request.target.first_segment_id, Some(SegmentId(1)));
    assert_eq!(second_request.target.last_segment_id, Some(SegmentId(1)));
    assert_eq!(
        second_request.audio.len(),
        NEMOTRON_CHUNK_SAMPLES,
        "Nemotron streaming input must send only the next 160ms delta to the ASR worker"
    );
    assert_eq!(
        second_request.source_audio.len(),
        NEMOTRON_CHUNK_SAMPLES * 2,
        "Turn replacement must still keep the cumulative source audio for UI output"
    );
    asr_handle.complete_request_with_text(&second_request, "あいう");
    runtime.step();

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].text, "あ...");
    assert_eq!(outputs[0].phrase.len(), NEMOTRON_CHUNK_SAMPLES);
    assert_eq!(outputs[1].text, "あいう...");
    assert_eq!(
        outputs[1].phrase.len(),
        NEMOTRON_CHUNK_SAMPLES * 2,
        "same-segment interim updates must replace the previous source audio instead of appending duplicate audio"
    );
}

#[test]
fn turn_runtime_keeps_only_one_asr_request_in_flight() {
    let (mut runtime, config) = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
        ],
    );

    let dispatched = runtime
        .take_last_dispatched()
        .expect("first closed segment must dispatch");
    assert_eq!(dispatched.request_id, AsrRequestId(1));
    assert_eq!(dispatched.target.last_segment_id, Some(SegmentId(1)));
    runtime.step();
    assert!(
        runtime.take_last_dispatched().is_none(),
        "second closed segment must stay queued while the first ASR request is in flight"
    );
}

#[test]
fn turn_runtime_applies_interim_asr_result_to_output_sink() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .scripted_asr_texts(vec!["途中"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "途中...".to_string(),
            is_final: false,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn turn_runtime_interim_punctuation_does_not_run_grammar_finalization() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(320)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .scripted_asr_texts(vec!["はい。次です"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("はい。次です...", false, 1, 1)],
        "interim display must not run Mecab/grammar finalization even if punctuation is present"
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn turn_runtime_multilingual_turn_check_after_interim_uses_sli_route_for_rerecognition() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::ReazonSpeechK2V2)
        .multilingual(true)
        .enabled_asr_models(vec![
            AsrModel::ReazonSpeechK2V2,
            AsrModel::NemoParakeetTdt0_6BV2Int8,
        ])
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .segment_start_speech_ms(64)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .turn_check_silence_ms(64)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let sli_call_audio_lens = builder.use_scripted_language_detector(vec!["en"]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0; 16_000], vad(true)),
            (vec![1.0; 512], vad(true)),
            (vec![0.0; 512], vad(false)),
            (vec![1.0; 512], vad(true)),
        ],
    );
    let interim_request = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("interim ASR should dispatch before turn-check silence");
    assert_eq!(interim_request.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        interim_request.route,
        RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
        "interim display should keep the default route until turn-check SLI"
    );
    asr_handle.complete_request_with_text(&interim_request, "hello");
    runtime.step();

    replay_vad_frames_for_runtime(&mut runtime, &config, vec![(vec![0.0; 512], vad(false))]);

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        *sli_call_audio_lens
            .lock()
            .expect("SLI call lengths should be readable"),
        vec![17_024],
        "turn-check after interim must run SLI over the accumulated turn audio before rerecognition"
    );
    assert_eq!(
        rerecognition.route,
        RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8),
        "full-turn rerecognition must switch to the SLI-selected English route"
    );
}

#[test]
fn turn_runtime_namo_completion_and_rerecognition_elapsed_millis_are_accumulated() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8)
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅"]);
    builder.config_mut().asr.precision = AsrPrecision::Int8;
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion request should be in flight");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text_elapsed(&completion, "句読点つき。", 41);

    runtime.step();

    let rerecognition =
        runtime.requests.in_flight_request.clone().expect(
            "Namo completion must dispatch full-turn rerecognition even if punctuation exists",
        );
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        rerecognition.route.model,
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8
    );
    asr_handle.complete_request_with_text_elapsed(&rerecognition, "再認識後。", 59);

    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].text, "再認識後。");
    assert!(outputs[0].is_final);
    assert_eq!(
        outputs[0].elapsed_millis, 100,
        "final output should report completion ASR plus rerecognition ASR elapsed time"
    );
}

#[test]
fn turn_runtime_suppresses_late_interim_when_turn_check_already_reached() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty(),
        "silence that reaches turn-check must dispatch completion without first showing interim"
    );

    asr_handle.complete_next_with_text("hello");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "completion ASR must wait for full-turn rerecognition before final output"
    );
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);

    asr_handle.complete_next_with_text("hello");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("hello。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_interim_silence_does_not_emit_final_before_final_asr_result() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![2.0], vad(true)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)]
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)],
        "turn-check silence must not finalize text from the interim ASR result"
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です...", false, 1, 1)],
        "completion-check ASR must still wait for rerecognition before final output"
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![
            output_snapshot("五月五日はこどもの日です...", false, 1, 1),
            output_snapshot("五月五日はこどもの日です。", true, 1, 2),
        ]
    );
}

#[test]
fn turn_runtime_interim_display_asr_pads_edges_without_persisting_padding() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    const FADE_SAMPLES: usize = 160;
    let builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(2.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(
        interim.source_audio,
        [vec![1.0; CHUNK], vec![0.0; CHUNK]].concat(),
        "interim display must preserve only the observed source audio for the turn"
    );
    assert_eq!(
        interim.audio.len(),
        CHUNK * (EDGE_CHUNKS + 2 + EDGE_CHUNKS - 1),
        "interim ASR should add missing 320ms leading and trailing silence to the request audio"
    );
    assert!(
        interim.audio[..CHUNK * EDGE_CHUNKS]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "interim request should have synthetic leading silence"
    );
    assert!(
        interim.audio[CHUNK * (EDGE_CHUNKS + 2)..]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "interim request should have synthetic trailing silence after the natural silent chunk"
    );
    assert_sample_close(
        interim.audio[CHUNK * EDGE_CHUNKS],
        0.0,
        "speech edge should be faded in from the synthetic leading silence",
    );
    assert_sample_close(
        interim.audio[CHUNK * EDGE_CHUNKS + FADE_SAMPLES],
        1.0,
        "fade-in should not rewrite the steady speech body",
    );
    assert_eq!(
        interim.source_vad_results.len(),
        2,
        "synthetic ASR padding must not be persisted as turn-source VAD"
    );
    assert_eq!(
        interim.vad_results.len(),
        EDGE_CHUNKS + 2 + EDGE_CHUNKS - 1,
        "ASR VAD should cover the synthetic request padding"
    );
}

#[test]
fn turn_runtime_interim_display_rerecognition_uses_source_audio_before_padding() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(2.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");

    asr_handle.complete_request_with_text(&interim, "途中表示");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(0.0, CHUNK, false),
        ],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        completion.audio,
        [
            vec![0.0; CHUNK],
            vec![2.0; CHUNK],
            vec![0.0; CHUNK],
            vec![0.0; CHUNK],
        ]
        .concat(),
        "completion ASR may keep the copied trailing silence as local padding"
    );

    asr_handle.complete_request_with_text(&completion, "初回認識");
    runtime.step();

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion result should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        &rerecognition.audio[..CHUNK * 5],
        [
            vec![1.0; CHUNK],
            vec![0.0; CHUNK],
            vec![2.0; CHUNK],
            vec![0.0; CHUNK],
            vec![0.0; CHUNK],
        ]
        .concat()
        .as_slice(),
        "full-turn rerecognition must use the continuous turn audio, not stitched interim/completion ASR buffers"
    );
    assert_eq!(
        rerecognition.audio.len(),
        CHUNK * (5 + EDGE_CHUNKS - 2),
        "full-turn rerecognition should add only the missing trailing silence to the ASR request"
    );
    assert!(
        rerecognition.audio[CHUNK * 5..]
            .iter()
            .all(|sample| sample_is_zero(*sample)),
        "rerecognition padding should stay outside the persisted turn audio"
    );
}

fn sample_is_zero(sample: f32) -> bool {
    sample.abs() <= f32::EPSILON
}

fn assert_sample_close(actual: f32, expected: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= f32::EPSILON,
        "{context}: actual={actual}, expected={expected}"
    );
}

#[test]
fn turn_runtime_interim_open_turn_rerecognition_adds_missing_fixed_edge_silence() {
    const CHUNK: usize = 512;
    const EDGE_CHUNKS: usize = 10;
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(64)
        .interim_display(true)
        .interim_result_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let _outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(1.0, CHUNK, true),
            fixed_vad_frame(2.0, CHUNK, true),
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(3.0, CHUNK, true),
        ],
    );
    let interim = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("speech after interim silence should dispatch interim ASR");
    assert_eq!(interim.kind, AsrTaskKind::InterimDisplay);
    assert_eq!(interim.source_audio.len(), CHUNK * 3);

    asr_handle.complete_request_with_text(&interim, "interim");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            fixed_vad_frame(0.0, CHUNK, false),
            fixed_vad_frame(0.0, CHUNK, false),
        ],
    );
    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("turn-check silence after interim should dispatch full-turn rerecognition");

    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    assert_eq!(
        rerecognition.audio.len(),
        CHUNK * (3 + EDGE_CHUNKS - 1),
        "rerecognition should append the missing fixed 320ms edge silence instead of waiting for a completion segment"
    );
    assert_eq!(
        &rerecognition.audio[CHUNK * 3..],
        vec![0.0; CHUNK * (EDGE_CHUNKS - 1)].as_slice()
    );
    assert_eq!(
        rerecognition.target.range.end_sample,
        GlobalSampleIndex((CHUNK * 4) as u64),
        "synthetic ASR-only silence must not expand the source audio range"
    );
}

#[test]
fn turn_runtime_interim_disabled_waits_for_turn_check_before_completion_asr() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(64)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .interim_result_silence_ms(32);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );

    assert!(
        asr_handle.submitted_requests().is_empty(),
        "interim_result_enabled=false must not dispatch ASR at interim_result_silence_ms"
    );
    assert!(
        outputs
            .lock()
            .expect("outputs should be readable")
            .is_empty()
    );

    replay_vad_frames_for_runtime(&mut runtime, &config, vec![(vec![0.0], vad(false))]);

    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です。", true, 1, 1)]
    );
}

#[test]
fn turn_runtime_following_simple_interim_after_completed_turn_is_emitted_as_next_turn() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(96)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(32)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![1.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );

    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();
    asr_handle.complete_next_with_text("五月五日はこどもの日です");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![output_snapshot("五月五日はこどもの日です。", true, 1, 1)],
        "silence that reaches turn-check must finalize the first turn before the following root interim"
    );

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
            (vec![3.0], vad(true)),
        ],
    );
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
            (vec![0.0], vad(false)),
        ],
    );
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();
    asr_handle.complete_next_with_text("すごいね");
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![
            output_snapshot("五月五日はこどもの日です。", true, 1, 1),
            output_snapshot("すごいね...", false, 2, 2),
            output_snapshot("すごいね。", true, 2, 3),
        ]
    );
}

#[test]
#[ignore = "requires local JVS corpus; verifies UI output phrase audio coverage, not ASR accuracy"]
fn jvs_ui_phrase_audio_coverage_keeps_each_spoken_part_visible_across_interim_overwrites() {
    // This is a RecognitionSession/UI-output coverage test, not an ASR accuracy test.
    // ASR text is scripted so the assertion can focus on whether each JVS wav part
    // remains present in RecognizedTextOutput.phrase across interim replacement/finalization.
    let part_ids = ["BASIC5000_0408", "BASIC5000_1140"];
    let jvs_parts = part_ids
        .iter()
        .map(|id| read_jvs_nonparallel_part(id))
        .collect::<Vec<_>>();
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Simple)
        .vad_interval_ms(32)
        .turn_check_silence_ms(192)
        .segment_start_speech_ms(1)
        .interim_display(true)
        .interim_result_silence_ms(64)
        .rerecognize_full_on_complete(true);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();

    push_jvs_speech_chunks(
        &mut runtime,
        &config,
        &jvs_parts[0].samples,
        jvs_parts[0].sample_rate,
    );
    push_silence_chunks(&mut runtime, &config, jvs_parts[0].sample_rate, 2);
    let second_part_first_chunk_len = frames_for_millis(
        jvs_parts[1].sample_rate,
        config.segmentation.vad_interval_ms,
    )
    .min(jvs_parts[1].samples.len());
    runtime.push_vad_frame(
        &jvs_parts[1].samples[..second_part_first_chunk_len],
        vad(true),
    );
    runtime.step();
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::InterimDisplay);
    asr_handle.complete_next_with_text(&jvs_parts[0].text);
    runtime.step();

    {
        let current_outputs = outputs.lock().expect("outputs should be readable");
        let latest = current_outputs
            .last()
            .expect("interim output should be emitted");
        assert!(!latest.is_final);
        assert_output_phrase_contains_jvs_parts(latest, &jvs_parts[..1]);
    }

    if second_part_first_chunk_len < jvs_parts[1].samples.len() {
        push_jvs_speech_chunks(
            &mut runtime,
            &config,
            &jvs_parts[1].samples[second_part_first_chunk_len..],
            jvs_parts[1].sample_rate,
        );
    }

    push_silence_chunks(
        &mut runtime,
        &config,
        jvs_parts[0].sample_rate,
        chunks_for_ms(jvs_parts[0].sample_rate, config.turn.check_silence_ms),
    );
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_next_with_text(&jvs_parts[1].text);
    runtime.step();
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].kind, AsrTaskKind::Rerecognition);
    let final_text = jvs_parts
        .iter()
        .map(|part| part.text.as_str())
        .collect::<String>();
    asr_handle.complete_next_with_text(&final_text);
    runtime.step();

    let outputs = outputs.lock().expect("outputs should be readable");
    assert_eq!(outputs.len(), 2);
    assert!(!outputs[0].is_final);
    assert!(outputs[1].is_final);
    assert_output_phrase_contains_jvs_parts(&outputs[0], &jvs_parts[..1]);
    assert_output_phrase_contains_jvs_parts(&outputs[1], &jvs_parts[..2]);
    assert!(
        outputs[0].phrase.len() < outputs[1].phrase.len(),
        "UI phrase audio should grow across interim replacement and final: {:?}",
        outputs
            .iter()
            .map(|output| output.phrase.len())
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires local FLEURS-R corpus; verifies final saved phrase audio coverage across timing combinations, not ASR accuracy"]
fn fleurs_short_sentence_sequence_final_phrase_keeps_source_audio_in_order_for_timing_matrix() {
    let parts = read_short_fleurs_dev_parts("en_us", FLEURS_MATRIX_PART_COUNT);
    assert_eq!(parts.len(), FLEURS_MATRIX_PART_COUNT);

    for interim_silence_ms in [None, Some(64), Some(128)] {
        for turn_check_silence_ms in [128, 256, 512] {
            for gap_ms in [64, 128, 256] {
                run_fleurs_timing_matrix_case(
                    &parts,
                    FleursTimingCase {
                        interim_silence: interim_silence_ms,
                        turn_check_silence: turn_check_silence_ms,
                        gap: gap_ms,
                    },
                );
            }
        }
    }
}

const FLEURS_MATRIX_PART_COUNT: usize = 3;
const FLEURS_MATRIX_SEGMENT_START_MS: u32 = 128;

#[derive(Clone, Copy)]
struct FleursTimingCase {
    interim_silence: Option<u32>,
    turn_check_silence: u32,
    gap: u32,
}

fn run_fleurs_timing_matrix_case(parts: &[FleursPart], case: FleursTimingCase) {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(case.turn_check_silence)
        .segment_start_speech_ms(FLEURS_MATRIX_SEGMENT_START_MS)
        .interim_display(case.interim_silence.is_some())
        .rerecognize_full_on_complete(true);
    if let Some(interim_silence_ms) = case.interim_silence {
        builder = builder.interim_result_silence_ms(interim_silence_ms);
    }
    let internal_completion_count = usize::from(case.gap >= case.turn_check_silence) * 2;
    let mut decisions = vec![
        TurnDecision {
            is_end_of_turn: false,
            confidence: 0.99,
        };
        internal_completion_count
    ];
    decisions.push(TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    });
    let _ = builder.use_scripted_decisions(decisions);
    let asr_handle = builder.use_manual_asr();
    let outputs = builder.use_recording_phrase_sink();
    let (mut runtime, config) = builder.build();
    let chunk_len = frames_for_millis(parts[0].sample_rate, config.segmentation.vad_interval_ms);
    let expected_final_len = parts.iter().map(|part| part.samples.len()).sum::<usize>()
        + (chunks_for_ms(parts[0].sample_rate, case.gap) * 2
            + chunks_for_ms(parts[0].sample_rate, case.turn_check_silence))
            * chunk_len;

    push_fleurs_speech_chunks(&mut runtime, &config, &parts[0]);
    push_fleurs_gap_then_next_part(
        &mut runtime,
        &config,
        &asr_handle,
        &parts[1],
        case,
        "fleurs-0",
    );
    push_fleurs_gap_then_next_part(
        &mut runtime,
        &config,
        &asr_handle,
        &parts[2],
        case,
        "fleurs-1",
    );

    push_silence_chunks(
        &mut runtime,
        &config,
        parts[2].sample_rate,
        chunks_for_ms(parts[2].sample_rate, case.turn_check_silence),
    );
    complete_namo_turn_check_asr(
        &mut runtime,
        &asr_handle,
        case,
        "first pass",
        "fleurs final",
    );

    let outputs = outputs.lock().expect("phrase outputs should be readable");
    let final_output = outputs
        .last()
        .expect("final output should be emitted after rerecognition");
    assert!(final_output.is_final);
    assert_eq!(
        final_output.phrase.len(),
        expected_final_len,
        "{}: final phrase audio must contain observed FLEURS audio plus observed silence only",
        case.label()
    );
    assert_output_phrase_contains_fleurs_parts(final_output, parts);
    println!(
        "{} final phrase len={} parts={:?}",
        case.label(),
        final_output.phrase.len(),
        parts
            .iter()
            .map(|part| (part.wav_path.display().to_string(), part.samples.len()))
            .collect::<Vec<_>>()
    );
}

fn push_fleurs_gap_then_next_part(
    runtime: &mut RecognitionDriver,
    config: &ParapperConfig,
    asr_handle: &ManualAsrHandle,
    next_part: &FleursPart,
    case: FleursTimingCase,
    transcript: &str,
) {
    push_silence_chunks(
        runtime,
        config,
        next_part.sample_rate,
        chunks_for_ms(next_part.sample_rate, case.gap),
    );
    if case.gap >= case.turn_check_silence {
        complete_namo_turn_check_asr(runtime, asr_handle, case, transcript, transcript);
        push_fleurs_speech_chunks(runtime, config, next_part);
        return;
    }
    if case
        .interim_silence
        .is_some_and(|interim_silence_ms| case.gap >= interim_silence_ms)
    {
        push_first_fleurs_speech_chunk(runtime, config, next_part);
        assert_next_asr_kind(asr_handle, AsrTaskKind::InterimDisplay, &case);
        asr_handle.complete_next_with_text(transcript);
        runtime.step();
        push_remaining_fleurs_speech_chunks(runtime, config, next_part);
        return;
    }
    push_fleurs_speech_chunks(runtime, config, next_part);
}

fn complete_namo_turn_check_asr(
    runtime: &mut RecognitionDriver,
    asr_handle: &ManualAsrHandle,
    case: FleursTimingCase,
    completion_text: &str,
    rerecognition_text: &str,
) {
    assert_next_asr_kind(asr_handle, AsrTaskKind::CompletionCheck, &case);
    asr_handle.complete_next_with_text(completion_text);
    runtime.step();
    assert_next_asr_kind(asr_handle, AsrTaskKind::Rerecognition, &case);
    asr_handle.complete_next_with_text(rerecognition_text);
    runtime.step();
}

fn push_first_fleurs_speech_chunk(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    part: &FleursPart,
) {
    let chunk_len = frames_for_millis(part.sample_rate, config.segmentation.vad_interval_ms);
    let first_len = chunk_len.min(part.samples.len());
    runtime.push_vad_frame(&part.samples[..first_len], vad(true));
    runtime.step();
}

fn push_remaining_fleurs_speech_chunks(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    part: &FleursPart,
) {
    let chunk_len = frames_for_millis(part.sample_rate, config.segmentation.vad_interval_ms);
    if part.samples.len() <= chunk_len {
        return;
    }
    push_jvs_speech_chunks(
        runtime,
        config,
        &part.samples[chunk_len..],
        part.sample_rate,
    );
}

fn assert_next_asr_kind(
    asr_handle: &ManualAsrHandle,
    expected: AsrTaskKind,
    case: &FleursTimingCase,
) {
    let submitted = asr_handle.submitted_requests();
    assert_eq!(submitted.len(), 1, "{}", case.label());
    assert_eq!(submitted[0].kind, expected, "{}", case.label());
}

fn chunks_for_ms(sample_rate: u32, ms: u32) -> usize {
    let chunk_len = frames_for_millis(sample_rate, 32);
    frames_for_millis(sample_rate, ms).div_ceil(chunk_len)
}

impl FleursTimingCase {
    fn label(self) -> String {
        format!(
            "interim={:?}ms turn_check={}ms gap={}ms",
            self.interim_silence, self.turn_check_silence, self.gap
        )
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
            phrase: vec![1.0, 2.0, 3.0].into(),
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

#[test]
fn turn_runtime_applies_completion_asr_result_to_output_sink() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["完了"]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "完了。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn simple_completion_check_rerecognition_flag_controls_final_output_source() {
    for (case_name, rerecognize_full_on_complete) in [
        ("rerecognition disabled", false),
        ("rerecognition enabled", true),
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Simple)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false)
            .rerecognize_full_on_complete(rerecognize_full_on_complete);
        let asr_handle = builder.use_manual_asr();
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        let completion_request = runtime
            .requests
            .in_flight_request
            .clone()
            .unwrap_or_else(|| {
                panic!("{case_name}: closed segment should dispatch completion ASR")
            });
        assert_eq!(
            completion_request.kind,
            AsrTaskKind::CompletionCheck,
            "{case_name}"
        );

        asr_handle.complete_request_with_text(&completion_request, "初回結果");
        runtime.step();

        if rerecognize_full_on_complete {
            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                Vec::<OutputSnapshot>::new(),
                "{case_name}: completion output must wait for full-turn rerecognition"
            );
            let rerecognition_request =
                runtime
                    .requests
                    .in_flight_request
                    .clone()
                    .unwrap_or_else(|| {
                        panic!(
                            "{case_name}: completion result should dispatch full-turn rerecognition"
                        )
                    });
            assert_eq!(
                rerecognition_request.kind,
                AsrTaskKind::Rerecognition,
                "{case_name}"
            );
            assert_eq!(
                rerecognition_request.target.turn_id,
                TurnId(1),
                "{case_name}"
            );
            assert_eq!(
                &rerecognition_request.audio[..completion_request.source_audio.len()],
                completion_request.source_audio.as_slice(),
                "{case_name}: rerecognition should start from the continuous source audio"
            );
            assert!(
                rerecognition_request.audio[completion_request.source_audio.len()..]
                    .iter()
                    .all(|sample| *sample == 0.0),
                "{case_name}: missing ASR-only trailing silence should be synthesized"
            );

            asr_handle.complete_request_with_text(&rerecognition_request, "再認識結果");
            runtime.step();

            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                vec![output_snapshot("再認識結果。", true, 1, 1)],
                "{case_name}"
            );
        } else {
            assert_eq!(
                *outputs.lock().expect("outputs should be readable"),
                vec![output_snapshot("初回結果。", true, 1, 1)],
                "{case_name}"
            );
            assert!(
                runtime.requests.in_flight_request.is_none(),
                "{case_name}: disabled rerecognition must not dispatch a second ASR request"
            );
            assert_eq!(asr_handle.submitted_requests().len(), 1, "{case_name}");
        }
    }
}

#[test]
fn turn_runtime_parakeet_models_dispatch_rerecognition_after_namo_completion_check() {
    for model in [
        AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8,
        AsrModel::NemoParakeetTdt0_6BV2Int8,
        AsrModel::NemoParakeetTdt0_6BV3Int8,
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .asr_model(model)
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false);
        builder.config_mut().asr.precision = AsrPrecision::Int8;
        let asr_handle = builder.use_manual_asr();
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        let completion_request = runtime
            .requests
            .in_flight_request
            .clone()
            .expect("closed segment should dispatch completion ASR");
        assert_eq!(completion_request.kind, AsrTaskKind::CompletionCheck);
        assert_eq!(completion_request.route.model, model, "model={model:?}");

        asr_handle.complete_request_with_text(&completion_request, "first pass");
        runtime.step();

        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            Vec::<OutputSnapshot>::new(),
            "model={model:?} must wait for rerecognition before final output"
        );
        let rerecognition_request = runtime
            .requests
            .in_flight_request
            .clone()
            .expect("Namo completion result should dispatch full-turn rerecognition");
        assert_eq!(
            rerecognition_request.kind,
            AsrTaskKind::Rerecognition,
            "model={model:?}"
        );
        assert_eq!(rerecognition_request.route.model, model, "model={model:?}");
        assert_eq!(
            &rerecognition_request.audio[..completion_request.source_audio.len()],
            completion_request.source_audio.as_slice(),
            "single-segment model={model:?} should rerecognize the same full-turn source audio before ASR-only padding"
        );
        assert!(
            rerecognition_request.audio[completion_request.source_audio.len()..]
                .iter()
                .all(|sample| *sample == 0.0),
            "single-segment model={model:?} should synthesize only missing trailing silence"
        );
    }
}

#[test]
fn english_punctuation_after_rerecognition_finalizes_as_strong_end_without_namo() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .asr_model(AsrModel::NemoParakeetTdt0_6BV2Int8)
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(true);
    let asr_handle = builder.use_manual_asr();
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("closed English segment should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);

    asr_handle.complete_request_with_text(&completion, "we should keep going");
    runtime.step();

    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("Namo completion should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    asr_handle.push_completed_result(AsrResult {
        request_id: rerecognition.request_id,
        kind: rerecognition.kind,
        target: rerecognition.target.clone(),
        route: rerecognition.route,
        status: AsrResultStatus::Ok(english_sentence_end_transcript("We should keep going.")),
        completed_at_frame: VadFrameIndex(0),
        elapsed_millis: 0,
    });
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        Vec::<String>::new(),
        "English sentence punctuation should finalize as StrongEnd without asking Namo"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "We should keep going.".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }],
        "Namo must finalize English sentence punctuation as grammar StrongEnd"
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

fn english_sentence_end_transcript(text: &str) -> AsrTranscript {
    let tokens = text.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
    let timestamps = (0..tokens.len())
        .map(|index| {
            f32::from(u16::try_from(index).expect("test transcript should have few tokens")) / 100.0
        })
        .collect::<Vec<_>>();
    let durations = vec![0.01; tokens.len()];
    AsrTranscript::from_parts(
        text.to_string(),
        tokens,
        Some(&timestamps),
        Some(&durations),
    )
}

#[test]
fn turn_runtime_internal_strong_boundary_keeps_turn_open_until_terminal_candidate() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_transcripts(vec![
            AsrTranscript::from_text("はい次です"),
            japanese_punctuation_transcript(),
        ]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    let rerecognition = runtime
        .take_last_dispatched()
        .expect("completion result should dispatch full-turn rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);

    runtime.step();

    assert_eq!(*outputs.lock().expect("outputs should be readable"), vec![]);
    assert_eq!(
        runtime.turn_store.open_turn_id,
        Some(1),
        "internal grammar boundary must keep the full turn open"
    );
    assert!(
        runtime.turn_store.turns.contains_key(&1),
        "turn should remain open"
    );
}

#[test]
fn turn_runtime_namo_complete_without_boundary_emits_final() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅", "続き", "続き再認識", "さらに続き"]);
    let decision_texts = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: true,
        confidence: 0.99,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["東京駅".to_string()]
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "東京駅。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
}

#[test]
fn namo_decision_confidence_threshold_controls_finalization() {
    struct Case {
        name: &'static str,
        decision: TurnDecision,
        expected_output: Option<(&'static str, bool)>,
        expected_open_turn_id: Option<u64>,
    }

    let cases = vec![
        Case {
            name: "end decision at threshold finalizes",
            decision: TurnDecision {
                is_end_of_turn: true,
                confidence: 0.8,
            },
            expected_output: Some(("東京駅。", true)),
            expected_open_turn_id: None,
        },
        Case {
            name: "continue decision above threshold stays open",
            decision: TurnDecision {
                is_end_of_turn: false,
                confidence: 0.99,
            },
            expected_output: None,
            expected_open_turn_id: Some(1),
        },
        Case {
            name: "end decision below threshold stays open",
            decision: TurnDecision {
                is_end_of_turn: true,
                confidence: 0.79,
            },
            expected_output: None,
            expected_open_turn_id: Some(1),
        },
    ];

    for Case {
        name,
        decision,
        expected_output,
        expected_open_turn_id,
    } in cases
    {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(false)
            .namo_turn_confidence_threshold(0.8)
            .scripted_asr_texts(vec!["東京駅", "東京駅"]);
        let _ = builder.use_scripted_decisions(vec![decision]);
        let outputs = builder.use_recording_phrase_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        runtime.step();
        runtime.step();

        let outputs = outputs.lock().expect("outputs should be readable");
        match expected_output {
            Some((expected_text, expected_is_final)) => {
                assert_eq!(outputs.len(), 1, "{name}");
                assert_eq!(outputs[0].text, expected_text, "{name}");
                assert_eq!(outputs[0].is_final, expected_is_final, "{name}");
            }
            None => assert!(outputs.is_empty(), "{name}: got outputs {outputs:?}"),
        }
        assert_eq!(
            runtime.turn_store.open_turn_id, expected_open_turn_id,
            "{name}"
        );
    }
}

#[test]
fn namo_continue_interim_display_flag_controls_partial_output_while_turn_stays_open() {
    struct Case {
        name: &'static str,
        interim_display: bool,
        expected_output: Option<(&'static str, bool)>,
    }

    let cases = vec![
        Case {
            name: "interim display disabled",
            interim_display: false,
            expected_output: None,
        },
        Case {
            name: "interim display enabled",
            interim_display: true,
            expected_output: Some(("東京駅...", false)),
        },
    ];

    for Case {
        name,
        interim_display,
        expected_output,
    } in cases
    {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Namo)
            .vad_interval_ms(32)
            .turn_check_silence_ms(32)
            .segment_start_speech_ms(1)
            .interim_display(interim_display)
            .interim_result_silence_ms(32)
            .scripted_asr_texts(vec!["東京駅", "東京駅"]);
        let _ = builder.use_scripted_decisions(vec![TurnDecision {
            is_end_of_turn: false,
            confidence: 0.01,
        }]);
        let outputs = builder.use_recording_phrase_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
        );
        runtime.step();
        runtime.step();

        let outputs = outputs.lock().expect("outputs should be readable");
        match expected_output {
            Some((expected_text, expected_is_final)) => {
                assert_eq!(outputs.len(), 1, "{name}");
                assert_eq!(outputs[0].text, expected_text, "{name}");
                assert_eq!(outputs[0].is_final, expected_is_final, "{name}");
            }
            None => assert!(outputs.is_empty(), "{name}: got outputs {outputs:?}"),
        }
        assert_eq!(runtime.turn_store.open_turn_id, Some(1), "{name}");
    }
}

#[test]
fn namo_turn_decision_error_keeps_turn_open_without_final_output() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅"]);
    let decision_texts = builder.use_scripted_decisions(Vec::new());
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();

    assert_eq!(
        *decision_texts
            .lock()
            .expect("turn decision texts should be readable"),
        vec!["東京駅".to_string()],
        "the decision runner should receive the draft text before its error is handled"
    );
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new(),
        "a Namo decision error must continue the turn instead of finalizing with stale confidence"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
}

#[test]
fn turn_runtime_timeout_after_namo_continue_rerecognizes_then_finalizes() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false)
        .scripted_asr_texts(vec!["東京駅", "東京駅", "東京駅再認識"]);
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.01,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    runtime.step();
    runtime.step();
    let timeout_chunks =
        usize::try_from(runtime.timeout_ticks()).expect("timeout ticks should fit usize");
    push_silence_chunks(&mut runtime, &config, 16_000, timeout_chunks);
    let timeout_rerecognition = runtime
        .take_last_dispatched()
        .expect("timeout should dispatch rerecognition before final");
    assert_eq!(timeout_rerecognition.kind, AsrTaskKind::Rerecognition);
    runtime.step();

    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        vec![OutputSnapshot {
            text: "東京駅再認識。".to_string(),
            is_final: true,
            turn_id: 1,
            segment_id: 1,
        }]
    );
    assert!(runtime.turn_store.open_turn_id.is_none());
}

#[test]
fn turn_runtime_activity_after_namo_continue_delays_timeout() {
    let mut builder = RecognitionSessionTestBuilder::new()
        .turn_detector(TurnDetector::Namo)
        .vad_interval_ms(32)
        .turn_check_silence_ms(32)
        .segment_start_speech_ms(1)
        .interim_display(false);
    let asr_handle = builder.use_manual_asr();
    let _ = builder.use_scripted_decisions(vec![TurnDecision {
        is_end_of_turn: false,
        confidence: 0.01,
    }]);
    let outputs = builder.use_recording_sink();
    let (mut runtime, config) = builder.build();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![(vec![1.0], vad(true)), (vec![0.0], vad(false))],
    );
    let completion = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("first segment should dispatch completion ASR");
    assert_eq!(completion.kind, AsrTaskKind::CompletionCheck);
    asr_handle.complete_request_with_text(&completion, "東京駅");
    runtime.step();
    let rerecognition = runtime
        .requests
        .in_flight_request
        .clone()
        .expect("completion result should dispatch rerecognition");
    assert_eq!(rerecognition.kind, AsrTaskKind::Rerecognition);
    asr_handle.complete_request_with_text(&rerecognition, "東京駅");
    runtime.step();

    replay_vad_frames_for_runtime(
        &mut runtime,
        &config,
        vec![
            (vec![2.0], vad(true)),
            (vec![2.0], vad(true)),
            (vec![0.0], vad(false)),
        ],
    );

    let next_completion = runtime
        .take_last_dispatched()
        .expect("following active speech should close as the next segment before timeout");
    assert_eq!(next_completion.kind, AsrTaskKind::CompletionCheck);
    assert_eq!(
        next_completion.target.turn_id,
        TurnId(1),
        "the segment after Namo Continue must stay attached to the open turn"
    );
    assert_eq!(runtime.turn_store.open_turn_id, Some(1));
    assert_eq!(
        *outputs.lock().expect("outputs should be readable"),
        Vec::<OutputSnapshot>::new()
    );
}

#[test]
fn simple_turn_check_rerecognition_flag_controls_existing_interim_finalization() {
    for (case_name, rerecognize_full_on_complete) in [
        ("rerecognition disabled", false),
        ("rerecognition enabled", true),
    ] {
        let mut builder = RecognitionSessionTestBuilder::new()
            .turn_detector(TurnDetector::Simple)
            .vad_interval_ms(32)
            .turn_check_silence_ms(64)
            .segment_start_speech_ms(1)
            .interim_display(true)
            .interim_result_silence_ms(32)
            .rerecognize_full_on_complete(rerecognize_full_on_complete)
            .scripted_asr_texts(vec!["途中", "確定"]);
        let outputs = builder.use_recording_sink();
        let (mut runtime, config) = builder.build();

        replay_vad_frames_for_runtime(
            &mut runtime,
            &config,
            vec![
                (vec![1.0], vad(true)),
                (vec![0.0], vad(false)),
                (vec![0.0], vad(false)),
            ],
        );

        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            Vec::<OutputSnapshot>::new(),
            "{case_name}: turn check should not emit before the queued work is stepped"
        );

        runtime.step();

        let expected_after_first_step = if rerecognize_full_on_complete {
            Vec::<OutputSnapshot>::new()
        } else {
            vec![output_snapshot("途中。", true, 1, 1)]
        };
        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            expected_after_first_step,
            "{case_name}"
        );

        runtime.step();

        let expected_after_second_step = if rerecognize_full_on_complete {
            vec![output_snapshot("確定。", true, 1, 1)]
        } else {
            vec![output_snapshot("途中。", true, 1, 1)]
        };
        assert_eq!(
            *outputs.lock().expect("outputs should be readable"),
            expected_after_second_step,
            "{case_name}"
        );
    }
}
