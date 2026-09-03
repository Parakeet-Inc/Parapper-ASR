use std::ops::{Deref, DerefMut};

use crate::{
    RecognitionFrame, RecognitionSegmentEngine, RecognitionSession, SegmentBuilderEvent,
    SegmentCloseReason, SttEngineConfig, SttRuntimeParameters, VadResult,
    runtime::PendingTurnCheck,
};

/// Host-neutral STT coordinator.
///
/// Hosts own worker threads, clocks, sleeping, and shutdown deadlines. This type
/// only advances the deterministic recognition state machine.
pub struct RecognitionDriver {
    runtime: RecognitionSession,
    segmentation: RecognitionSegmentEngine,
}

impl RecognitionDriver {
    #[must_use]
    pub fn new(runtime: RecognitionSession, config: &SttEngineConfig) -> Self {
        Self {
            runtime,
            segmentation: RecognitionSegmentEngine::new(&config.recognition_config()),
        }
    }

    pub fn update_runtime_parameters(&mut self, parameters: SttRuntimeParameters) {
        if self.runtime.config.runtime_parameters() == parameters {
            return;
        }
        self.runtime.config.apply_runtime_parameters(parameters);
        self.segmentation
            .update_config(&self.runtime.config.recognition_config());
        self.runtime
            .io
            .asr_runner
            .set_normalize_input_audio(parameters.normalize_input_audio);
    }

    pub fn push_vad_frame(&mut self, samples: &[f32], vad_result: VadResult) {
        self.runtime.counters.next_runtime_tick =
            self.runtime.counters.next_runtime_tick.saturating_add(1);
        let frame = self.segmentation.push_vad_frame(samples, vad_result);
        self.push_segment_event_frame(frame);
    }

    pub fn step(&mut self) {
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
            }
            return;
        }
        if self.runtime.handle_open_turn_timeout() {
            return;
        }
        self.runtime.dispatch_next_asr_request_if_idle();
    }

    /// Closes the active segmentation tail without waiting for asynchronous ports.
    pub fn flush_input(&mut self) {
        let frame = self.segmentation.flush();
        self.push_segment_event_frame(frame);
    }

    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        self.runtime.requests.in_flight_request.is_some()
            || self.runtime.pending.turn_check.is_some()
            || self.runtime.pending.finalization.is_some()
            || !self.runtime.pending.asr_segments.is_empty()
    }

    /// Finalizes a Namo continuation once the host has drained every queued job.
    pub fn finalize_open_turn_after_drain(&mut self) {
        if self.has_pending_work() {
            return;
        }
        if let Some(turn_id) = self.runtime.turn_store.open_turn_id {
            self.runtime
                .finalize_timeout_turn_after_rerecognition(turn_id);
        }
    }

    pub fn shutdown_ports(&mut self) {
        self.runtime.io.asr_runner.shutdown();
    }

    #[cfg(test)]
    pub(crate) fn take_last_dispatched(
        &mut self,
    ) -> Option<crate::transcription::task::AsrInFlight> {
        self.runtime.requests.last_dispatched.take()
    }

    fn push_segment_event_frame(&mut self, frame: RecognitionFrame) {
        self.runtime.counters.global_sample_cursor = self
            .runtime
            .counters
            .global_sample_cursor
            .saturating_add(frame.samples_len as u64);
        self.runtime.counters.next_vad_frame_index =
            self.runtime.counters.next_vad_frame_index.saturating_add(1);

        for event in frame.events {
            match event {
                SegmentBuilderEvent::SegmentStarted {
                    segment_id,
                    previous_segment_id,
                    audio_so_far,
                    vad_results,
                } => {
                    self.runtime.activity.segment_activity_epoch = self
                        .runtime
                        .activity
                        .segment_activity_epoch
                        .saturating_add(1);
                    self.runtime.record_interim_segment_started(
                        segment_id,
                        previous_segment_id,
                        audio_so_far,
                        vad_results,
                    );
                }
                SegmentBuilderEvent::SegmentExtended {
                    segment_id,
                    previous_segment_id,
                    new_audio,
                    vad_result,
                } => {
                    self.runtime.activity.segment_activity_epoch = self
                        .runtime
                        .activity
                        .segment_activity_epoch
                        .saturating_add(1);
                    self.runtime.record_interim_segment_extended(
                        segment_id,
                        previous_segment_id,
                        new_audio,
                        vad_result,
                    );
                }
                SegmentBuilderEvent::TurnCheckSilenceReached {
                    previous_segment_id,
                } => {
                    self.runtime.pending.turn_check = Some(PendingTurnCheck {
                        previous_segment_id,
                        activity_epoch: self.runtime.activity.segment_activity_epoch,
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
                    if reason == SegmentCloseReason::EndSilenceReached {
                        self.runtime
                            .reset_interim_streaming_for_completion(segment_id);
                    }
                    self.runtime.record_segment_closed_asr_candidate(
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
