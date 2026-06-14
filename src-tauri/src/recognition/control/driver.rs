use std::{
    ops::{Deref, DerefMut},
    thread,
    time::{Duration, Instant},
};

use tauri::AppHandle;

use super::{AsrWorkerStartupSender, RecognitionSession, pending::PendingTurnCheck};
use crate::{
    config::ParapperConfig,
    recognition::segmentation::{
        flow::{SegmentationFlow, SegmentationFrameEvents},
        segment::builder::SegmentBuilderEvent,
        vad::engine::VadResult,
    },
};

pub(crate) trait RecognitionDriverHandle {
    fn update_config(&mut self, config: &ParapperConfig);
    fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult);
    fn step(&mut self);
    fn shutdown(&mut self);
}

pub(crate) struct RecognitionDriver {
    runtime: RecognitionSession,
    segmentation_flow: SegmentationFlow,
}

const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

#[cfg(test)]
pub(crate) fn replay_vad_frames_for_runtime(
    runtime: &mut dyn RecognitionDriverHandle,
    config: &ParapperConfig,
    frames: impl IntoIterator<Item = (Vec<f32>, VadResult)>,
) {
    runtime.update_config(config);
    for (samples, vad_result) in frames {
        runtime.push_vad_frame(&samples, vad_result);
        runtime.step();
    }
}

impl RecognitionDriver {
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

    pub(in crate::recognition) fn new(
        runtime: RecognitionSession,
        config: &ParapperConfig,
    ) -> Self {
        Self {
            runtime,
            segmentation_flow: SegmentationFlow::new(config),
        }
    }

    fn shutdown_flush_and_drain(&mut self) {
        let frame_events = self.segmentation_flow.flush();
        self.runtime.push_segment_event_frame(frame_events);
        let started_at = Instant::now();
        // Shutdown is an input boundary, so drive queued ASR and pending finalization
        // far enough that an active tail segment is not dropped with the worker.
        while self.runtime.has_shutdown_drain_work() {
            self.step();
            if !self.runtime.has_shutdown_drain_work() {
                break;
            }
            if started_at.elapsed() >= SHUTDOWN_DRAIN_TIMEOUT {
                log::warn!("Timed out while draining recognition shutdown work");
                break;
            }
            thread::sleep(SHUTDOWN_DRAIN_POLL_INTERVAL);
        }
        if let Some(open_turn_id) = self.runtime.turn_store.open_turn_id
            && !self.runtime.has_shutdown_drain_work()
        {
            // No more audio can arrive after shutdown; an open Namo suffix must fall
            // back to final instead of waiting for runtime ticks that will never come.
            self.runtime
                .finalize_timeout_turn_after_rerecognition(open_turn_id);
        }
    }
}

impl Deref for RecognitionDriver {
    type Target = RecognitionSession;

    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

impl DerefMut for RecognitionDriver {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runtime
    }
}

impl RecognitionSession {
    fn update_config(&mut self, config: &ParapperConfig) {
        if self.config == *config {
            return;
        }
        let route_settings_changed = self.config.asr.language != config.asr.language
            || self.config.asr.model != config.asr.model
            || self.config.asr.multilingual_enabled != config.asr.multilingual_enabled
            || self.config.asr.enabled_models != config.asr.enabled_models;
        self.config = config.clone();
        if route_settings_changed {
            self.turn_store.last_recognition_route = None;
        }
        self.io.asr_runner.update_config(config);
        self.io.turn_decision_runner.update_config(config);
        self.io.output_sink.update_config(config);
        if !config.asr.multilingual_enabled {
            self.io.language_id = None;
        } else if self.io.language_id.is_none() {
            self.io.language_id = self
                .io
                .language_id_runtime
                .as_ref()
                .and_then(|runtime| runtime.build_language_id(config));
        }
    }

    fn advance_runtime_tick(&mut self) {
        self.counters.next_runtime_tick = self.counters.next_runtime_tick.saturating_add(1);
    }

    pub(in crate::recognition) fn push_segment_event_frame(
        &mut self,
        frame_events: SegmentationFrameEvents,
    ) {
        self.counters.global_sample_cursor = self
            .counters
            .global_sample_cursor
            .saturating_add(frame_events.samples_len as u64);
        self.counters.next_vad_frame_index = self.counters.next_vad_frame_index.saturating_add(1);

        for event in frame_events.events {
            match event {
                SegmentBuilderEvent::SegmentStarted { .. }
                | SegmentBuilderEvent::SegmentExtended { .. } => {
                    self.activity.segment_activity_epoch =
                        self.activity.segment_activity_epoch.saturating_add(1);
                }
                SegmentBuilderEvent::TurnCheckSilenceReached {
                    previous_segment_id,
                } => {
                    self.pending.turn_check = Some(PendingTurnCheck {
                        previous_segment_id,
                        activity_epoch: self.activity.segment_activity_epoch,
                    });
                }
                SegmentBuilderEvent::SegmentClosed {
                    segment_id,
                    previous_segment_id,
                    full_audio,
                    vad_results,
                    source_audio,
                    source_vad_results,
                    reason,
                } => {
                    self.record_segment_closed_asr_candidate(
                        segment_id,
                        previous_segment_id,
                        full_audio,
                        vad_results,
                        source_audio,
                        source_vad_results,
                        reason,
                    );
                }
            }
        }
    }
}

impl RecognitionDriverHandle for RecognitionDriver {
    fn update_config(&mut self, config: &ParapperConfig) {
        self.segmentation_flow.update_config(config);
        self.runtime.update_config(config);
    }

    fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult) {
        self.runtime.advance_runtime_tick();
        let frame_events = self.segmentation_flow.push_vad_frame(samples, vad_result);
        self.runtime.push_segment_event_frame(frame_events);
    }

    fn step(&mut self) {
        if self.runtime.apply_completed_asr_result_if_ready() {
            return;
        }

        if self.runtime.process_pending_finalization_if_ready() {
            return;
        }

        if let Some(turn_check) = self.runtime.pending.turn_check {
            if turn_check.activity_epoch != self.runtime.activity.segment_activity_epoch {
                self.runtime.pending.turn_check = None;
                return;
            }
            if self
                .runtime
                .handle_turn_check_silence_reached(turn_check.previous_segment_id)
            {
                self.runtime.pending.turn_check = None;
                return;
            }
            return;
        }

        if self.runtime.handle_open_turn_timeout() {
            return;
        }

        self.runtime.dispatch_next_asr_request_if_idle();
    }

    fn shutdown(&mut self) {
        self.shutdown_flush_and_drain();
        self.runtime.io.asr_runner.shutdown();
    }
}

impl RecognitionSession {
    fn has_shutdown_drain_work(&self) -> bool {
        self.requests.in_flight_request.is_some()
            || self.pending.turn_check.is_some()
            || self.pending.finalization.is_some()
            || !self.pending.asr_segments.is_empty()
    }
}
