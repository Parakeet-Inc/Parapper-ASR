use std::{
    ops::{Deref, DerefMut},
    thread,
    time::{Duration, Instant},
};

use parapper_models::vad::VadResult;
use tauri::AppHandle;

pub(crate) use parapper_stt_engine::RecognitionShutdownResult;

use super::{AsrRuntimePoolHandle, AsrWorkerStartupSender, RecognitionSession, TurnOutputSink};
use crate::{config::ParapperConfig, recognition::config::stt_engine_config};

pub(crate) trait RecognitionDriverHandle {
    fn update_runtime_parameters(&mut self, parameters: parapper_stt_engine::SttRuntimeParameters);
    fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult);
    fn step(&mut self);
    fn flush_input(&mut self);
    fn has_pending_work(&self) -> bool;
    fn finalize_open_turn_after_drain(&mut self);
    fn shutdown(&mut self) -> RecognitionShutdownResult;
    fn cancel(&mut self) {
        let _ = self.shutdown();
    }
}

pub(crate) struct RecognitionDriver {
    inner: parapper_stt_engine::RecognitionDriver,
}

const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

#[cfg(test)]
pub(crate) fn replay_vad_frames_for_runtime(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    frames: impl IntoIterator<Item = (Vec<f32>, VadResult)>,
) {
    runtime.update_runtime_parameters(crate::recognition::config::stt_runtime_parameters(config));
    for (samples, vad_result) in frames {
        runtime.push_vad_frame(&samples, vad_result);
        runtime.step();
    }
}

impl RecognitionDriver {
    #[cfg(test)]
    pub(crate) fn new_for_production(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
    ) -> Self {
        Self::new(
            RecognitionSession::new_for_production(handle, config, asr_startup_sender),
            config,
        )
    }

    pub(crate) fn new_for_production_with_output_sink(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        Self::new(
            RecognitionSession::new_for_production_with_output_sink(
                handle,
                config,
                asr_startup_sender,
                output_sink,
            ),
            config,
        )
    }

    pub(crate) fn new_for_production_with_output_sink_and_source_identity(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        Self::new(
            RecognitionSession::new_for_production_with_output_sink_and_source_identity(
                handle,
                config,
                asr_startup_sender,
                source_identity,
                output_sink,
            ),
            config,
        )
    }

    pub(crate) fn new_for_production_with_pool_and_output_sink_and_source_identity(
        handle: &AppHandle,
        config: &ParapperConfig,
        pool: &AsrRuntimePoolHandle,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        Self::new(
            RecognitionSession::new_for_production_with_pool_and_output_sink_and_source_identity(
                handle,
                config,
                pool,
                source_identity,
                output_sink,
            ),
            config,
        )
    }

    pub(in crate::recognition) fn new(
        runtime: RecognitionSession,
        config: &ParapperConfig,
    ) -> Self {
        Self {
            inner: parapper_stt_engine::RecognitionDriver::new(
                runtime.into_engine(),
                &stt_engine_config(config),
            ),
        }
    }

    #[cfg(test)]
    pub(in crate::recognition) fn take_last_dispatched(
        &mut self,
    ) -> Option<parapper_stt_engine::transcription::task::AsrInFlight> {
        self.inner.requests.last_dispatched.take()
    }

    fn shutdown_flush_and_drain(&mut self) -> RecognitionShutdownResult {
        self.inner.flush_input();
        let started_at = Instant::now();
        while self.inner.has_pending_work() {
            self.inner.step();
            if !self.inner.has_pending_work() {
                break;
            }
            if started_at.elapsed() >= SHUTDOWN_DRAIN_TIMEOUT {
                log::warn!("Timed out while draining recognition shutdown work");
                return RecognitionShutdownResult::TimedOut;
            }
            thread::sleep(SHUTDOWN_DRAIN_POLL_INTERVAL);
        }
        self.inner.finalize_open_turn_after_drain();
        RecognitionShutdownResult::Completed
    }
}

impl Deref for RecognitionDriver {
    type Target = parapper_stt_engine::RecognitionDriver;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for RecognitionDriver {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl RecognitionDriverHandle for RecognitionDriver {
    fn update_runtime_parameters(&mut self, parameters: parapper_stt_engine::SttRuntimeParameters) {
        self.inner.update_runtime_parameters(parameters);
    }

    fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult) {
        self.inner.push_vad_frame(samples, vad_result);
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn flush_input(&mut self) {
        self.inner.flush_input();
    }

    fn has_pending_work(&self) -> bool {
        self.inner.has_pending_work()
    }

    fn finalize_open_turn_after_drain(&mut self) {
        self.inner.finalize_open_turn_after_drain();
    }

    fn shutdown(&mut self) -> RecognitionShutdownResult {
        let result = self.shutdown_flush_and_drain();
        self.inner.shutdown_ports();
        result
    }

    fn cancel(&mut self) {
        self.inner.shutdown_ports();
    }
}
