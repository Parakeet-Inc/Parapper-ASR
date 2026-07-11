use std::{
    borrow::Cow,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        control::{
            engine_cache::AsrEngineCache, events::MissingModelKind,
            runtime_event::emit_missing_model_event,
        },
        segmentation::segment::builder::SegmentCloseReason,
        transcription::asr::{
            input::{
                PreparedAsrInput, emit_asr_warning,
                maybe_shift_transcript_timestamps_for_leading_padding, normalize_asr_input_audio,
                prepare_asr_input_audio, prepare_nemotron_input_audio,
                prepare_nemotron_streaming_bootstrap_audio,
            },
            task::{AsrRequest, AsrResult, AsrResultStatus, AsrTaskKind},
        },
    },
};

const ASR_WORKER_JOIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) trait AsrRequestRunner {
    fn update_config(&mut self, _config: &ParapperConfig) {}
    fn reset_streaming_sessions(&mut self) {}
    fn submit(&mut self, request: AsrRequest) -> bool;
    fn try_recv_result(&mut self) -> Option<AsrResult>;
    fn shutdown(&mut self) {}
}

pub(crate) type AsrWorkerStartupResult = Result<(), Vec<String>>;
pub(crate) type AsrWorkerStartupSender = Sender<AsrWorkerStartupResult>;

#[cfg(test)]
pub(crate) struct NoopAsrRequestRunner;

#[cfg(test)]
impl AsrRequestRunner for NoopAsrRequestRunner {
    fn submit(&mut self, _request: AsrRequest) -> bool {
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        None
    }
}

pub(crate) struct EngineAsrRequestRunner {
    request_sender: Option<Sender<AsrWorkerCommand>>,
    result_receiver: Receiver<AsrResult>,
    config: Arc<RwLock<ParapperConfig>>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

enum AsrWorkerCommand {
    Request(Box<AsrRequest>),
    ResetStreamingSessions,
}

impl EngineAsrRequestRunner {
    pub(crate) fn new(
        handle: AppHandle,
        config: &ParapperConfig,
        startup_sender: Option<AsrWorkerStartupSender>,
    ) -> Self {
        let (request_sender, request_receiver) = std::sync::mpsc::channel();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        let config = Arc::new(RwLock::new(config.clone()));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let startup_sender_for_spawn_error = startup_sender.clone();
        let join_handle = match thread::Builder::new()
            .name("parapper-next-asr-runner".to_string())
            .spawn(move || {
                run_engine_asr_request_worker(
                    &handle,
                    &worker_config,
                    &request_receiver,
                    &result_sender,
                    &worker_stop,
                    startup_sender,
                );
            }) {
            Ok(join_handle) => Some(join_handle),
            Err(err) => {
                let reason = format!("Failed to spawn ASR request worker: {err}");
                log::warn!("{reason}");
                if let Some(sender) = startup_sender_for_spawn_error {
                    let _ = sender.send(Err(vec![reason]));
                }
                None
            }
        };

        Self {
            request_sender: Some(request_sender),
            result_receiver,
            config,
            stop_requested,
            join_handle,
        }
    }
}

impl AsrRequestRunner for EngineAsrRequestRunner {
    fn update_config(&mut self, config: &ParapperConfig) {
        if let Ok(mut current) = self.config.write() {
            *current = config.clone();
        }
    }

    fn reset_streaming_sessions(&mut self) {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!(
                "Failed to reset ASR streaming sessions because the request sender is closed"
            );
            return;
        };
        if let Err(err) = sender.send(AsrWorkerCommand::ResetStreamingSessions) {
            log::warn!(
                "Failed to submit ASR streaming session reset to next runtime runner: {err}"
            );
        }
    }

    fn submit(&mut self, request: AsrRequest) -> bool {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!("Failed to submit ASR request because the request sender is closed");
            return false;
        };
        if let Err(err) = sender.send(AsrWorkerCommand::Request(Box::new(request))) {
            log::warn!("Failed to submit ASR request to next runtime runner: {err}");
            return false;
        }
        true
    }

    fn try_recv_result(&mut self) -> Option<AsrResult> {
        self.result_receiver.try_recv().ok()
    }

    fn shutdown(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        self.request_sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            let started_at = Instant::now();
            while !join_handle.is_finished() && started_at.elapsed() < ASR_WORKER_JOIN_TIMEOUT {
                thread::sleep(Duration::from_millis(1));
            }
            if join_handle.is_finished() {
                if let Err(err) = join_handle.join() {
                    log::warn!("RecognitionSession ASR runner thread panicked: {err:?}");
                }
            } else {
                log::warn!(
                    "Timed out waiting for recognition ASR runner shutdown; detaching worker"
                );
            }
        }
    }
}

impl Drop for EngineAsrRequestRunner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_engine_asr_request_worker(
    handle: &AppHandle,
    config: &Arc<RwLock<ParapperConfig>>,
    request_receiver: &Receiver<AsrWorkerCommand>,
    result_sender: &Sender<AsrResult>,
    stop_requested: &Arc<AtomicBool>,
    startup_sender: Option<AsrWorkerStartupSender>,
) {
    let startup_config = config
        .read()
        .map_or_else(|_| ParapperConfig::default(), |config| config.clone());
    let mut asr = AsrEngineCache::default();
    let startup_errors = asr.preload_required(handle, &startup_config);
    for reason in &startup_errors {
        log::warn!("{reason}");
        emit_missing_model_event(handle, MissingModelKind::Asr, reason.clone());
    }
    if let Some(sender) = startup_sender {
        let _ = sender.send(if startup_errors.is_empty() {
            Ok(())
        } else {
            Err(startup_errors)
        });
    }

    while !stop_requested.load(Ordering::Acquire) {
        let command = match request_receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(command) => command,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let AsrWorkerCommand::Request(request) = command else {
            asr.clear_streaming_sessions();
            continue;
        };
        let request = *request;
        let current_config = config
            .read()
            .map_or_else(|_| startup_config.clone(), |config| config.clone());
        for reason in asr.preload_required(handle, &current_config) {
            log::warn!("{reason}");
            emit_missing_model_event(handle, MissingModelKind::Asr, reason);
        }
        let result = run_engine_asr_request(handle, &current_config, &mut asr, &request);
        if result_sender.send(result).is_err() {
            break;
        }
    }
}

pub(crate) fn run_engine_asr_request(
    handle: &AppHandle,
    config: &ParapperConfig,
    asr: &mut AsrEngineCache,
    request: &AsrRequest,
) -> AsrResult {
    let request_id = request.request_id;
    let kind = request.kind;
    let target = request.target.clone();
    let route = request.route;
    let completed_at_frame = request.created_at_frame;
    let started_at = Instant::now();
    let status = if is_nemotron_streaming_interim_request(request) {
        let session = request.streaming_session_key();
        let existing_leading_padding = asr.streaming_leading_padding_samples(session);
        let prepared = if existing_leading_padding.is_some() {
            PreparedAsrInput {
                audio: Cow::Borrowed(request.audio.as_slice()),
                leading_padding_samples: 0,
            }
        } else {
            let bootstrap_audio = if request.source_audio.is_empty() {
                request.audio.as_slice()
            } else {
                request.source_audio.as_slice()
            };
            let bootstrap_vad_results = if request.source_vad_results.is_empty() {
                request.vad_results.as_slice()
            } else {
                request.source_vad_results.as_slice()
            };
            prepare_nemotron_streaming_bootstrap_audio(bootstrap_audio, bootstrap_vad_results)
        };
        let audio = normalize_asr_input_audio(config, prepared.audio.as_ref());
        match asr.transcribe_streaming_delta(
            route,
            session,
            audio.as_ref(),
            prepared.leading_padding_samples,
        ) {
            Ok((mut transcript, leading_padding_samples)) => {
                maybe_shift_transcript_timestamps_for_leading_padding(
                    &mut transcript,
                    leading_padding_samples,
                );
                AsrResultStatus::Ok(transcript)
            }
            Err(err) => {
                emit_asr_warning(handle, &err);
                AsrResultStatus::Failed(err.to_string())
            }
        }
    } else {
        asr.clear_streaming_sessions();
        let prepared = if route.model.is_nemotron() {
            prepare_nemotron_input_audio(&request.audio, &request.vad_results)
        } else {
            prepare_asr_input_audio(&request.audio, &request.vad_results)
        };
        let audio = normalize_asr_input_audio(config, prepared.audio.as_ref());
        match asr.transcribe(route, audio.as_ref()) {
            Ok(mut transcript) => {
                maybe_shift_transcript_timestamps_for_leading_padding(
                    &mut transcript,
                    prepared.leading_padding_samples,
                );
                AsrResultStatus::Ok(transcript)
            }
            Err(err) => {
                emit_asr_warning(handle, &err);
                AsrResultStatus::Failed(err.to_string())
            }
        }
    };

    AsrResult {
        request_id,
        kind,
        target,
        route,
        status,
        completed_at_frame,
        elapsed_millis: started_at.elapsed().as_millis(),
    }
}

fn is_nemotron_streaming_interim_request(request: &AsrRequest) -> bool {
    request.route.model.is_nemotron()
        && request.kind == AsrTaskKind::InterimDisplay
        && request.close_reason == Some(SegmentCloseReason::InterimChunkReached)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic,
        sync::mpsc,
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use crate::{
        config::AsrModel,
        recognition::{
            segmentation::{segment::builder::SegmentCloseReason, vad::engine::VadResult},
            transcription::{
                asr::task::{
                    AsrRequestId, AsrTarget, AsrTaskKind, AudioRange, GlobalSampleIndex, SegmentId,
                    TurnId, TurnRevision, VadFrameIndex,
                },
                route::RecognitionRoute,
            },
        },
    };

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_reports_initial_preload_failure_without_waiting_for_request() {
        let handle = tauri_test_handle();
        let config = Arc::new(RwLock::new(config_with_missing_model_dir(
            "worker-startup-signal",
        )));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (_request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let (startup_sender, startup_receiver) = mpsc::channel();
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                Some(startup_sender),
            );
        });

        let startup_result = startup_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker should report initial ASR preload before any request is submitted");

        stop_requested.store(true, Ordering::Release);
        worker.join().expect("worker should exit cleanly");
        assert!(
            matches!(startup_result, Err(ref errors) if errors.iter().any(|error| error.contains("Failed to preload"))),
            "missing ASR models must be reported through startup readiness, got {startup_result:?}"
        );
        assert!(
            result_receiver.try_recv().is_err(),
            "startup preload failure must not require or synthesize an ASR request result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_processes_request_and_returns_failed_result_when_engine_missing() {
        let handle = tauri_test_handle();
        let config = Arc::new(RwLock::new(config_with_missing_model_dir(
            "worker-processes-request",
        )));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                None,
            );
        });
        let request = test_asr_request(7);

        request_sender
            .send(AsrWorkerCommand::Request(Box::new(request.clone())))
            .expect("worker request channel should accept a request");
        let result = result_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker should send a result for the submitted request");
        stop_requested.store(true, Ordering::Release);
        drop(request_sender);
        worker.join().expect("worker should exit cleanly");

        assert_eq!(result.request_id, request.request_id);
        assert_eq!(result.kind, request.kind);
        assert_eq!(result.target, request.target);
        assert_eq!(result.route, request.route);
        assert!(
            matches!(result.status, AsrResultStatus::Failed(ref reason) if reason.contains("was not preloaded")),
            "a missing model must surface as a failed ASR result, got {:?}",
            result.status
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_exits_after_request_channel_disconnects() {
        let handle = tauri_test_handle();
        let config = Arc::new(RwLock::new(config_with_missing_model_dir(
            "worker-disconnect",
        )));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        drop(request_sender);

        run_engine_asr_request_worker(
            &handle,
            &config,
            &request_receiver,
            &result_sender,
            &stop_requested,
            None,
        );

        assert!(
            result_receiver.try_recv().is_err(),
            "disconnecting the request channel without a request must not produce a result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_worker_observes_stop_request_after_timeout_tick() {
        let handle = tauri_test_handle();
        let config = Arc::new(RwLock::new(config_with_missing_model_dir("worker-stop")));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let (_request_sender, request_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let worker_config = config.clone();
        let worker_stop = stop_requested.clone();
        let worker = thread::spawn(move || {
            run_engine_asr_request_worker(
                &handle,
                &worker_config,
                &request_receiver,
                &result_sender,
                &worker_stop,
                None,
            );
        });

        thread::sleep(Duration::from_millis(150));
        stop_requested.store(true, Ordering::Release);
        worker
            .join()
            .expect("worker should stop after a timeout tick");

        assert!(
            result_receiver.try_recv().is_err(),
            "a stop request without submitted ASR work must not produce a result"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn engine_asr_request_runner_shutdown_sets_stop_and_takes_worker_handles() {
        let handle = tauri_test_handle();
        let config = config_with_missing_model_dir("runner-shutdown");
        let mut runner = EngineAsrRequestRunner::new(handle, &config, None);

        runner.shutdown();
        runner.shutdown();

        assert!(runner.stop_requested.load(Ordering::Acquire));
        assert!(runner.request_sender.is_none());
        assert!(runner.join_handle.is_none());
    }

    #[test]
    fn engine_asr_request_runner_submit_sends_request_over_channel() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            config: Arc::new(RwLock::new(ParapperConfig::default())),
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
        };
        let request = test_asr_request(9);

        runner.submit(request.clone());

        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("submit should forward the ASR request to the worker channel");
        let AsrWorkerCommand::Request(submitted) = submitted else {
            panic!("submit must send an ASR request command");
        };
        assert_eq!(submitted.request_id, request.request_id);
        assert_eq!(submitted.target, request.target);
    }

    #[test]
    fn engine_asr_request_runner_reset_streaming_sessions_sends_reset_command() {
        let (request_sender, request_receiver) = mpsc::channel();
        let (_result_sender, result_receiver) = mpsc::channel();
        let mut runner = EngineAsrRequestRunner {
            request_sender: Some(request_sender),
            result_receiver,
            config: Arc::new(RwLock::new(ParapperConfig::default())),
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
        };

        runner.reset_streaming_sessions();

        let submitted = request_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("reset should forward the streaming reset command to the worker channel");
        assert!(matches!(
            submitted,
            AsrWorkerCommand::ResetStreamingSessions
        ));
    }

    #[test]
    fn engine_asr_request_runner_update_config_ignores_poisoned_config_lock() {
        let (_result_sender, result_receiver) = mpsc::channel();
        let config = Arc::new(RwLock::new(ParapperConfig::default()));
        let poison_target = config.clone();
        let previous_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let poison_result = thread::spawn(move || {
            let _guard = poison_target
                .write()
                .expect("test config lock should be writable before poisoning");
            panic!("poison config lock for update_config coverage");
        })
        .join();
        panic::set_hook(previous_hook);
        assert!(
            poison_result.is_err(),
            "test setup should poison the config lock"
        );
        let mut runner = EngineAsrRequestRunner {
            request_sender: None,
            result_receiver,
            config,
            stop_requested: Arc::new(AtomicBool::new(false)),
            join_handle: None,
        };
        let updated = parapper_config! {
            asr_model: AsrModel::NemoParakeetTdt0_6BV2Int8,
            ..ParapperConfig::default()
        };

        runner.update_config(&updated);

        let poisoned = runner
            .config
            .read()
            .expect_err("config lock should remain poisoned after update_config");
        assert_eq!(
            poisoned.get_ref().asr.model,
            ParapperConfig::default().asr.model
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "diagnostic: requires downloaded Nemotron 3.5 and Parakeet TDT CTC JA models"]
    fn measure_cpu4_rtf_current_worker_nemotron_streaming_delta_vs_parakeet_tdt_ctc_ja() {
        use std::time::{Duration, Instant};

        const NEMOTRON_CHUNK_SAMPLES: usize = crate::audio::ASR_SAMPLE_RATE as usize * 160 / 1_000;

        fn models_root_for_diagnostic() -> std::path::PathBuf {
            std::env::var_os("PARAPPER_MODELS_ROOT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(
                        std::env::var_os("APPDATA")
                            .expect("APPDATA or PARAPPER_MODELS_ROOT is required"),
                    )
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                })
        }

        fn diagnostic_config(model: AsrModel, models_root: &std::path::Path) -> ParapperConfig {
            let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
            parapper_config! {
                model_dir: Some(model_dir.to_string_lossy().into_owned()),
                asr_language: model.language(),
                asr_model: model,
                asr_num_threads: 4,
                asr_normalize_input_audio: false,
                ..ParapperConfig::default()
            }
            .normalized()
        }

        fn vad_results_for_sample_len(sample_len: usize) -> Vec<VadResult> {
            let frames = sample_len.div_ceil(NEMOTRON_CHUNK_SAMPLES).max(1);
            vec![
                VadResult {
                    probability: 0.9,
                    is_speech: true,
                };
                frames
            ]
        }

        fn nemotron_interim_request(
            request_id: u64,
            chunk: &[f32],
            source_audio: &[f32],
        ) -> AsrRequest {
            let end_sample = source_audio.len();
            AsrRequest {
                request_id: AsrRequestId(request_id),
                kind: AsrTaskKind::InterimDisplay,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(end_sample as u64)),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8),
                detected_language: None,
                audio: chunk.to_vec(),
                vad_results: vec![VadResult {
                    probability: 0.9,
                    is_speech: true,
                }],
                source_audio: source_audio.to_vec(),
                source_vad_results: vad_results_for_sample_len(end_sample),
                close_reason: Some(SegmentCloseReason::InterimChunkReached),
                created_at_frame: VadFrameIndex(request_id),
            }
        }

        fn parakeet_completion_request(samples: &[f32]) -> AsrRequest {
            AsrRequest {
                request_id: AsrRequestId(1),
                kind: AsrTaskKind::CompletionCheck,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(
                        GlobalSampleIndex(0),
                        GlobalSampleIndex(samples.len() as u64),
                    ),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8),
                detected_language: None,
                audio: samples.to_vec(),
                vad_results: vad_results_for_sample_len(samples.len()),
                source_audio: samples.to_vec(),
                source_vad_results: vad_results_for_sample_len(samples.len()),
                close_reason: Some(SegmentCloseReason::EndSilenceReached),
                created_at_frame: VadFrameIndex(1),
            }
        }

        fn measure_nemotron_worker_delta_batch(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            samples: &[f32],
            audio_sec: f64,
            label: &str,
            batch_samples: usize,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 0..=repeats {
                let mut asr = AsrEngineCache::default();
                let preload_errors = asr.preload_required(handle, config);
                assert!(
                    preload_errors.is_empty(),
                    "Nemotron preload should succeed: {preload_errors:?}"
                );
                let started_at = Instant::now();
                let mut request_id = 1_u64;
                let mut start = 0;
                while start < samples.len() {
                    let end = (start + batch_samples).min(samples.len());
                    let chunk = &samples[start..end];
                    let result = run_engine_asr_request(
                        handle,
                        config,
                        &mut asr,
                        &nemotron_interim_request(request_id, chunk, &samples[..end]),
                    );
                    let AsrResultStatus::Ok(transcript) = result.status else {
                        panic!("Nemotron delta request failed: {:?}", result.status);
                    };
                    last_text = transcript.text;
                    request_id += 1;
                    start = end;
                }
                let elapsed = started_at.elapsed();
                if iteration == 0 {
                    println!(
                        "nemotron_worker_delta_cpu4_{label} warmup requests={} text={last_text:?}",
                        request_id - 1
                    );
                    continue;
                }
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "nemotron_worker_delta_cpu4_{label} iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!("nemotron_worker_delta_cpu4_{label} avg_rtf={avg_rtf:.3} repeats={repeats}");
            (avg_rtf, last_text)
        }

        fn measure_parakeet_worker_full(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            samples: &[f32],
            audio_sec: f64,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut asr = AsrEngineCache::default();
            let preload_errors = asr.preload_required(handle, config);
            assert!(
                preload_errors.is_empty(),
                "Parakeet preload should succeed: {preload_errors:?}"
            );
            let warmup = run_engine_asr_request(
                handle,
                config,
                &mut asr,
                &parakeet_completion_request(samples),
            );
            let AsrResultStatus::Ok(transcript) = warmup.status else {
                panic!("Parakeet warmup failed: {:?}", warmup.status);
            };
            println!(
                "parakeet_tdt_ctc_ja_worker_full_cpu4 warmup text: {:?}",
                transcript.text
            );

            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 1..=repeats {
                let started_at = Instant::now();
                let result = run_engine_asr_request(
                    handle,
                    config,
                    &mut asr,
                    &parakeet_completion_request(samples),
                );
                let elapsed = started_at.elapsed();
                let AsrResultStatus::Ok(transcript) = result.status else {
                    panic!("Parakeet request failed: {:?}", result.status);
                };
                last_text = transcript.text;
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "parakeet_tdt_ctc_ja_worker_full_cpu4 iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!("parakeet_tdt_ctc_ja_worker_full_cpu4 avg_rtf={avg_rtf:.3} repeats={repeats}");
            (avg_rtf, last_text)
        }

        let handle = tauri_test_handle();
        let models_root = models_root_for_diagnostic();
        let nemotron_dir = models_root.join(crate::model::catalog::asr_model_dir_name(
            AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8,
        ));
        let wav_path = std::env::var_os("PARAPPER_ASR_RTF_WAV")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| nemotron_dir.join("test_wavs").join("ja.wav"));
        let wave = sherpa_onnx::Wave::read(&wav_path.display().to_string())
            .unwrap_or_else(|| panic!("failed to read {}", wav_path.display()));
        assert_eq!(
            wave.sample_rate(),
            i32::try_from(crate::audio::ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32")
        );
        let audio_sec = wave.samples().len() as f64 / f64::from(crate::audio::ASR_SAMPLE_RATE);
        println!(
            "current worker RTF input: {} samples={} audio_sec={audio_sec:.3}",
            wav_path.display(),
            wave.samples().len(),
        );

        let nemotron_config =
            diagnostic_config(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8, &models_root);
        let parakeet_config =
            diagnostic_config(AsrModel::NemoParakeetTdtCtc0_6BJa35000Int8, &models_root);
        let nemotron_measurements = [
            ("160ms", NEMOTRON_CHUNK_SAMPLES),
            ("320ms", NEMOTRON_CHUNK_SAMPLES * 2),
            ("640ms", NEMOTRON_CHUNK_SAMPLES * 4),
            ("1280ms", NEMOTRON_CHUNK_SAMPLES * 8),
            ("all", wave.samples().len()),
        ]
        .into_iter()
        .map(|(label, batch_samples)| {
            let (rtf, text) = measure_nemotron_worker_delta_batch(
                &handle,
                &nemotron_config,
                wave.samples(),
                audio_sec,
                label,
                batch_samples,
            );
            (label, rtf, text)
        })
        .collect::<Vec<_>>();
        let (parakeet_rtf, parakeet_text) =
            measure_parakeet_worker_full(&handle, &parakeet_config, wave.samples(), audio_sec);

        for (label, nemotron_rtf, _) in &nemotron_measurements {
            println!(
                "current worker RTF comparison cpu4: nemotron_delta_{label}={nemotron_rtf:.3} parakeet_tdt_ctc_ja={parakeet_rtf:.3} ratio={:.3}",
                nemotron_rtf / parakeet_rtf
            );
        }
        for (label, _, nemotron_text) in &nemotron_measurements {
            assert!(
                !nemotron_text.trim().is_empty(),
                "Nemotron worker delta {label} should produce non-empty text"
            );
        }
        assert!(
            !parakeet_text.trim().is_empty(),
            "Parakeet worker full should produce non-empty text"
        );
    }

    #[cfg(feature = "real-asr-tests")]
    #[test]
    #[ignore = "diagnostic: requires downloaded Nemotron English streaming model"]
    fn measure_cpu4_rtf_current_worker_nemotron_en_streaming_batches() {
        use std::time::{Duration, Instant};

        const NEMOTRON_CHUNK_SAMPLES: usize = crate::audio::ASR_SAMPLE_RATE as usize * 160 / 1_000;

        fn models_root_for_diagnostic() -> std::path::PathBuf {
            std::env::var_os("PARAPPER_MODELS_ROOT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(
                        std::env::var_os("APPDATA")
                            .expect("APPDATA or PARAPPER_MODELS_ROOT is required"),
                    )
                    .join("com.parakeet-inc.parapper")
                    .join("models")
                })
        }

        fn diagnostic_config(model: AsrModel, models_root: &std::path::Path) -> ParapperConfig {
            let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
            parapper_config! {
                model_dir: Some(model_dir.to_string_lossy().into_owned()),
                asr_language: model.language(),
                asr_model: model,
                asr_num_threads: 4,
                asr_normalize_input_audio: false,
                ..ParapperConfig::default()
            }
            .normalized()
        }

        fn vad_results_for_sample_len(sample_len: usize) -> Vec<VadResult> {
            let frames = sample_len.div_ceil(NEMOTRON_CHUNK_SAMPLES).max(1);
            vec![
                VadResult {
                    probability: 0.9,
                    is_speech: true,
                };
                frames
            ]
        }

        fn nemotron_interim_request(
            model: AsrModel,
            request_id: u64,
            chunk: &[f32],
            source_audio: &[f32],
        ) -> AsrRequest {
            let end_sample = source_audio.len();
            AsrRequest {
                request_id: AsrRequestId(request_id),
                kind: AsrTaskKind::InterimDisplay,
                target: AsrTarget::new(
                    TurnId(1),
                    TurnRevision(0),
                    AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(end_sample as u64)),
                    Some(SegmentId(1)),
                    Some(SegmentId(1)),
                ),
                route: RecognitionRoute::from_model(model),
                detected_language: None,
                audio: chunk.to_vec(),
                vad_results: vec![VadResult {
                    probability: 0.9,
                    is_speech: true,
                }],
                source_audio: source_audio.to_vec(),
                source_vad_results: vad_results_for_sample_len(end_sample),
                close_reason: Some(SegmentCloseReason::InterimChunkReached),
                created_at_frame: VadFrameIndex(request_id),
            }
        }

        fn measure_nemotron_worker_delta_batch(
            handle: &tauri::AppHandle,
            config: &ParapperConfig,
            model: AsrModel,
            samples: &[f32],
            audio_sec: f64,
            label: &str,
            batch_samples: usize,
        ) -> (f64, String) {
            let repeats = std::env::var("PARAPPER_ASR_RTF_REPEATS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3)
                .max(1);
            let mut total = Duration::ZERO;
            let mut last_text = String::new();
            for iteration in 0..=repeats {
                let mut asr = AsrEngineCache::default();
                let preload_errors = asr.preload_required(handle, config);
                assert!(
                    preload_errors.is_empty(),
                    "Nemotron preload should succeed: {preload_errors:?}"
                );
                let started_at = Instant::now();
                let mut request_id = 1_u64;
                let mut start = 0;
                while start < samples.len() {
                    let end = (start + batch_samples).min(samples.len());
                    let result = run_engine_asr_request(
                        handle,
                        config,
                        &mut asr,
                        &nemotron_interim_request(
                            model,
                            request_id,
                            &samples[start..end],
                            &samples[..end],
                        ),
                    );
                    let AsrResultStatus::Ok(transcript) = result.status else {
                        panic!("Nemotron {label} request failed: {:?}", result.status);
                    };
                    last_text = transcript.text;
                    request_id += 1;
                    start = end;
                }
                let elapsed = started_at.elapsed();
                if iteration == 0 {
                    println!(
                        "nemotron_en_worker_delta_cpu4_{label} warmup requests={} text={last_text:?}",
                        request_id - 1
                    );
                    continue;
                }
                let rtf = elapsed.as_secs_f64() / audio_sec;
                println!(
                    "nemotron_en_worker_delta_cpu4_{label} iter {iteration}: elapsed_ms={:.1} rtf={rtf:.3} text={last_text:?}",
                    elapsed.as_secs_f64() * 1000.0,
                );
                total += elapsed;
            }
            let avg_rtf = total.as_secs_f64() / repeats as f64 / audio_sec;
            println!(
                "nemotron_en_worker_delta_cpu4_{label} avg_rtf={avg_rtf:.3} repeats={repeats}"
            );
            (avg_rtf, last_text)
        }

        let model = AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8;
        let handle = tauri_test_handle();
        let models_root = models_root_for_diagnostic();
        let model_dir = models_root.join(crate::model::catalog::asr_model_dir_name(model));
        let wav_path = std::env::var_os("PARAPPER_NEMOTRON_EN_RTF_WAV")
            .or_else(|| std::env::var_os("PARAPPER_ASR_RTF_WAV"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| model_dir.join("test_wavs").join("0.wav"));
        let wave = sherpa_onnx::Wave::read(&wav_path.display().to_string())
            .unwrap_or_else(|| panic!("failed to read {}", wav_path.display()));
        assert_eq!(
            wave.sample_rate(),
            i32::try_from(crate::audio::ASR_SAMPLE_RATE).expect("ASR sample rate fits in i32")
        );
        let audio_sec = wave.samples().len() as f64 / f64::from(crate::audio::ASR_SAMPLE_RATE);
        println!(
            "current worker Nemotron EN RTF input: {} samples={} audio_sec={audio_sec:.3}",
            wav_path.display(),
            wave.samples().len(),
        );

        let config = diagnostic_config(model, &models_root);
        let measurements = [
            ("160ms", NEMOTRON_CHUNK_SAMPLES),
            ("320ms", NEMOTRON_CHUNK_SAMPLES * 2),
            ("640ms", NEMOTRON_CHUNK_SAMPLES * 4),
            ("1280ms", NEMOTRON_CHUNK_SAMPLES * 8),
            ("all", wave.samples().len()),
        ]
        .into_iter()
        .map(|(label, batch_samples)| {
            let (rtf, text) = measure_nemotron_worker_delta_batch(
                &handle,
                &config,
                model,
                wave.samples(),
                audio_sec,
                label,
                batch_samples,
            );
            (label, rtf, text)
        })
        .collect::<Vec<_>>();

        for (label, rtf, text) in &measurements {
            println!("current worker Nemotron EN RTF cpu4: delta_{label}={rtf:.3} text={text:?}");
            assert!(
                !text.trim().is_empty(),
                "Nemotron EN worker delta {label} should produce non-empty text"
            );
        }
    }

    fn test_asr_request(request_id: u64) -> AsrRequest {
        AsrRequest {
            request_id: AsrRequestId(request_id),
            kind: AsrTaskKind::CompletionCheck,
            target: AsrTarget::new(
                TurnId(1),
                TurnRevision(0),
                AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(4)),
                Some(SegmentId(1)),
                Some(SegmentId(1)),
            ),
            route: RecognitionRoute::from_model(ParapperConfig::default().asr.model),
            detected_language: None,
            audio: vec![0.0, 0.25, -0.25, 0.5],
            vad_results: vec![VadResult {
                probability: 0.9,
                is_speech: true,
            }],
            source_audio: vec![0.0, 0.25, -0.25, 0.5],
            source_vad_results: vec![VadResult {
                probability: 0.9,
                is_speech: true,
            }],
            close_reason: Some(SegmentCloseReason::EndSilenceReached),
            created_at_frame: VadFrameIndex(1),
        }
    }

    fn config_with_missing_model_dir(test_name: &str) -> ParapperConfig {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let missing_dir = std::env::temp_dir().join(format!(
            "parapper-missing-asr-model-{test_name}-{}-{unique}",
            std::process::id()
        ));
        parapper_config! {
            model_dir: Some(missing_dir.to_string_lossy().into_owned()),
            ..ParapperConfig::default()
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn tauri_test_handle() -> tauri::AppHandle {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("test app should build");
        app.handle().clone()
    }
}
