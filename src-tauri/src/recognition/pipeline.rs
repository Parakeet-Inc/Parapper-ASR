use std::sync::{Arc, RwLock};

use anyhow::Result;
use tauri::{AppHandle, Emitter};

use super::{
    events::{VadState, VadStateEvent},
    segment_builder::{SegmentBuilder, SegmentBuilderEvent},
    transcription::AsrWorker,
};
use crate::{
    config::ParapperConfig,
    model::vad_model_path,
    recognition::engines::{OnnxRuntimeSileroVadEngine, VadEngine},
};

trait SegmentBuilderEventSink {
    fn send_segment_activity(&self);
    fn send_segment_closed(
        &self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        reason: super::segment_builder::SegmentCloseReason,
    );
    fn send_turn_check_silence_reached(&self, previous_segment_id: u64);
}

impl SegmentBuilderEventSink for AsrWorker {
    fn send_segment_activity(&self) {
        AsrWorker::send_segment_activity(self);
    }

    fn send_segment_closed(
        &self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        full_audio: Vec<f32>,
        reason: super::segment_builder::SegmentCloseReason,
    ) {
        AsrWorker::send_segment_closed(self, segment_id, previous_segment_id, full_audio, reason);
    }

    fn send_turn_check_silence_reached(&self, previous_segment_id: u64) {
        AsrWorker::send_turn_check_silence_reached(self, previous_segment_id);
    }
}

pub struct RecognitionPipeline {
    handle: AppHandle,
    segment_builder: SegmentBuilder,
    vad: Box<dyn VadEngine>,
    asr_worker: AsrWorker,
}

impl RecognitionPipeline {
    pub fn new(
        handle: AppHandle,
        config: &ParapperConfig,
        runtime_config: &Arc<RwLock<ParapperConfig>>,
    ) -> Result<Self> {
        let vad_path = vad_model_path(&handle)?;
        let vad = OnnxRuntimeSileroVadEngine::new(&vad_path, config.vad_threshold)?;
        let asr_worker = AsrWorker::start(handle.clone(), config.clone(), runtime_config)?;

        Ok(Self {
            handle,
            segment_builder: SegmentBuilder::new(config),
            vad: Box::new(vad),
            asr_worker,
        })
    }

    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<()> {
        let vad_result = self.vad.process(samples)?;
        let state = if vad_result.is_speech {
            VadState::Speech
        } else {
            VadState::Silence
        };
        let _ = self.handle.emit(
            "parapper://vad-state",
            VadStateEvent {
                state,
                probability: vad_result.probability,
            },
        );

        let events = self.segment_builder.push(samples, vad_result);
        self.dispatch_segment_builder_events(events);

        Ok(())
    }

    pub fn tick(&mut self) {
        self.asr_worker.tick();
    }

    fn dispatch_segment_builder_events(&self, events: Vec<SegmentBuilderEvent>) {
        dispatch_segment_builder_events_to_sink(events, &self.asr_worker);
    }

    pub fn update_config(&mut self, config: &ParapperConfig) {
        self.segment_builder.update_config(config);
        self.vad.set_threshold(config.vad_threshold);
    }

    pub fn stop(mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        self.asr_worker.stop_inner();
    }
}

fn dispatch_segment_builder_events_to_sink(
    events: Vec<SegmentBuilderEvent>,
    sink: &impl SegmentBuilderEventSink,
) {
    for event in events {
        match event {
            SegmentBuilderEvent::SegmentStarted {
                segment_id: _,
                previous_segment_id: _,
                audio_so_far: _,
                vad_results: _,
            }
            | SegmentBuilderEvent::SegmentExtended {
                segment_id: _,
                previous_segment_id: _,
                new_audio: _,
                vad_result: _,
            } => {
                sink.send_segment_activity();
            }
            SegmentBuilderEvent::SegmentClosed {
                segment_id,
                previous_segment_id,
                full_audio,
                vad_results: _,
                reason,
            } => {
                sink.send_segment_closed(segment_id, previous_segment_id, full_audio, reason);
            }
            SegmentBuilderEvent::TurnCheckSilenceReached {
                previous_segment_id,
            } => {
                sink.send_turn_check_silence_reached(previous_segment_id);
            }
        }
    }
}

impl Drop for RecognitionPipeline {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{
        super::segment_builder::{SegmentBuilderEvent, SegmentCloseReason},
        SegmentBuilderEventSink, dispatch_segment_builder_events_to_sink,
    };
    use crate::recognition::engines::VadResult;

    #[derive(Debug, PartialEq)]
    enum DispatchedAsrInput {
        Activity,
        SegmentClosed {
            segment_id: u64,
            previous_segment_id: Option<u64>,
            full_audio: Vec<f32>,
            reason: SegmentCloseReason,
        },
        TurnCheckSilenceReached {
            previous_segment_id: u64,
        },
    }

    #[derive(Default)]
    struct RecordingSink {
        inputs: RefCell<Vec<DispatchedAsrInput>>,
    }

    impl SegmentBuilderEventSink for RecordingSink {
        fn send_segment_activity(&self) {
            self.inputs.borrow_mut().push(DispatchedAsrInput::Activity);
        }

        fn send_segment_closed(
            &self,
            segment_id: u64,
            previous_segment_id: Option<u64>,
            full_audio: Vec<f32>,
            reason: SegmentCloseReason,
        ) {
            self.inputs
                .borrow_mut()
                .push(DispatchedAsrInput::SegmentClosed {
                    segment_id,
                    previous_segment_id,
                    full_audio,
                    reason,
                });
        }

        fn send_turn_check_silence_reached(&self, previous_segment_id: u64) {
            self.inputs
                .borrow_mut()
                .push(DispatchedAsrInput::TurnCheckSilenceReached {
                    previous_segment_id,
                });
        }
    }

    fn vad(is_speech: bool) -> VadResult {
        VadResult {
            is_speech,
            probability: if is_speech { 0.9 } else { 0.1 },
        }
    }

    #[test]
    fn dispatcher_records_started_and_extended_as_activity_side_channel() {
        let sink = RecordingSink::default();

        dispatch_segment_builder_events_to_sink(
            vec![
                SegmentBuilderEvent::SegmentStarted {
                    segment_id: 1,
                    previous_segment_id: None,
                    audio_so_far: vec![1.0],
                    vad_results: vec![vad(true)],
                },
                SegmentBuilderEvent::SegmentExtended {
                    segment_id: 1,
                    previous_segment_id: None,
                    new_audio: vec![2.0],
                    vad_result: vad(true),
                },
                SegmentBuilderEvent::SegmentClosed {
                    segment_id: 1,
                    previous_segment_id: None,
                    full_audio: vec![1.0, 2.0, 0.0],
                    vad_results: vec![vad(true), vad(true), vad(false)],
                    reason: SegmentCloseReason::EndSilenceReached,
                },
                SegmentBuilderEvent::TurnCheckSilenceReached {
                    previous_segment_id: 1,
                },
            ],
            &sink,
        );

        assert_eq!(
            *sink.inputs.borrow(),
            vec![
                DispatchedAsrInput::Activity,
                DispatchedAsrInput::Activity,
                DispatchedAsrInput::SegmentClosed {
                    segment_id: 1,
                    previous_segment_id: None,
                    full_audio: vec![1.0, 2.0, 0.0],
                    reason: SegmentCloseReason::EndSilenceReached,
                },
                DispatchedAsrInput::TurnCheckSilenceReached {
                    previous_segment_id: 1,
                },
            ]
        );
    }
}
