use std::collections::VecDeque;

use crate::recognition::segmentation::vad::engine::VadResult;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SegmentChunk {
    pub(crate) audio: Vec<f32>,
    pub(crate) vad: VadResult,
    // False only for copied ASR padding that must not be persisted into the continuous Turn audio.
    pub(crate) include_in_turn_audio: bool,
}

impl SegmentChunk {
    pub(super) fn new(audio: Vec<f32>, vad: VadResult) -> Self {
        Self {
            audio,
            vad,
            include_in_turn_audio: true,
        }
    }

    pub(super) fn into_asr_padding(mut self) -> Self {
        self.include_in_turn_audio = false;
        self
    }
}

#[derive(Debug, Clone)]
pub(super) struct PreSegmentBuffer {
    chunks: VecDeque<SegmentChunk>,
    max_chunks: usize,
}

impl PreSegmentBuffer {
    pub(super) fn new(max_chunks: usize) -> Self {
        Self {
            chunks: VecDeque::new(),
            max_chunks,
        }
    }

    pub(super) fn push(&mut self, samples: &[f32], vad: VadResult) {
        self.push_chunk(SegmentChunk::new(samples.to_vec(), vad));
    }

    pub(super) fn push_chunk(&mut self, chunk: SegmentChunk) {
        self.chunks.push_back(chunk);
        self.prune();
    }

    pub(super) fn drain_into(&mut self, audio: &mut Vec<f32>) -> Vec<SegmentChunk> {
        let mut chunks = Vec::with_capacity(self.chunks.len());
        for chunk in self.chunks.drain(..) {
            audio.extend_from_slice(&chunk.audio);
            chunks.push(chunk);
        }
        chunks
    }

    pub(super) fn update_max_chunks(&mut self, max_chunks: usize) {
        self.max_chunks = max_chunks;
        self.prune();
    }

    fn prune(&mut self) {
        while self.chunks.len() > self.max_chunks {
            self.chunks.pop_front();
        }
    }
}
