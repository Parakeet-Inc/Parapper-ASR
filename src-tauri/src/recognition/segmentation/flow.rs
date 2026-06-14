use crate::config::ParapperConfig;

use super::{
    segment::builder::{SegmentBuilder, SegmentBuilderEvent},
    vad::engine::VadResult,
};

pub(in crate::recognition) struct SegmentationFlow {
    segment_builder: SegmentBuilder,
}

pub(in crate::recognition) struct SegmentationFrameEvents {
    pub(in crate::recognition) samples_len: usize,
    pub(in crate::recognition) events: Vec<SegmentBuilderEvent>,
}

impl SegmentationFlow {
    pub(in crate::recognition) fn new(config: &ParapperConfig) -> Self {
        Self {
            segment_builder: SegmentBuilder::new(config),
        }
    }

    pub(in crate::recognition) fn update_config(&mut self, config: &ParapperConfig) {
        self.segment_builder.update_config(config);
    }

    pub(in crate::recognition) fn push_vad_frame(
        &mut self,
        samples: &[f32],
        vad_result: VadResult,
    ) -> SegmentationFrameEvents {
        SegmentationFrameEvents {
            samples_len: samples.len(),
            events: self.segment_builder.push(samples, vad_result),
        }
    }

    pub(in crate::recognition) fn flush(&mut self) -> SegmentationFrameEvents {
        SegmentationFrameEvents {
            samples_len: 0,
            events: self.segment_builder.flush(),
        }
    }
}
