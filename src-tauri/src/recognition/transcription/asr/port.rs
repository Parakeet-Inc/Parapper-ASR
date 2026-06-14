use std::{
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
        transcription::asr::{
            input::{
                emit_asr_warning, maybe_shift_transcript_timestamps_for_leading_padding,
                normalize_asr_input_audio, prepare_asr_input_audio,
            },
            task::{AsrRequest, AsrResult, AsrResultStatus},
        },
    },
};

pub(crate) trait AsrRequestRunner {
    fn update_config(&mut self, _config: &ParapperConfig) {}
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
    request_sender: Option<Sender<AsrRequest>>,
    result_receiver: Receiver<AsrResult>,
    config: Arc<RwLock<ParapperConfig>>,
    stop_requested: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
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

    fn submit(&mut self, request: AsrRequest) -> bool {
        let Some(sender) = self.request_sender.as_ref() else {
            log::warn!("Failed to submit ASR request because the request sender is closed");
            return false;
        };
        if let Err(err) = sender.send(request) {
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
        if let Some(join_handle) = self.join_handle.take()
            && let Err(err) = join_handle.join()
        {
            log::warn!("RecognitionSession ASR runner thread panicked: {err:?}");
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
    request_receiver: &Receiver<AsrRequest>,
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
        let request = match request_receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(request) => request,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
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
    let prepared = prepare_asr_input_audio(&request.audio, &request.vad_results);
    let audio = normalize_asr_input_audio(config, prepared.audio.as_ref());
    let status = match asr.transcribe(route, audio.as_ref()) {
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
            .send(request.clone())
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
        assert_eq!(submitted.request_id, request.request_id);
        assert_eq!(submitted.target, request.target);
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
