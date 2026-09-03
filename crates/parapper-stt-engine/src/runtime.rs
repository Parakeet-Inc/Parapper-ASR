use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    SegmentCloseReason, SegmentId, VadResult,
    transcription::{
        planner::PendingAsrSegment,
        route::RecognitionRoute,
        task::{AsrInFlight, AsrRequest, AudioRange, GlobalSampleIndex, VadFrameIndex},
    },
    turn::{RerecognitionPurpose, Turn},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingFinalization {
    turn_id: u64,
}

impl PendingFinalization {
    #[must_use]
    pub const fn new(turn_id: u64) -> Self {
        Self { turn_id }
    }

    #[must_use]
    pub const fn turn_id(self) -> u64 {
        self.turn_id
    }
}

#[derive(Clone, Copy)]
pub struct PendingTurnCheck {
    pub previous_segment_id: u64,
    pub activity_epoch: u64,
}

#[derive(Default)]
pub struct PendingRuntimeState {
    pub turn_check: Option<PendingTurnCheck>,
    pub finalization: Option<PendingFinalization>,
    pub asr_segments: VecDeque<PendingAsrSegment>,
    pub interim_asr: InterimAsrState,
}

#[derive(Default)]
pub struct InterimAsrState {
    streaming: StreamingInterimState,
}

#[derive(Default)]
struct StreamingInterimState {
    active: Option<StreamingInterimSegmentState>,
}

struct StreamingInterimSegmentState {
    display_segment_id: u64,
    current_segment_id: u64,
    chunks: Vec<StreamingInterimAudioChunk>,
    emitted_samples: usize,
    range_start: GlobalSampleIndex,
    created_at_frame: VadFrameIndex,
}

struct StreamingInterimAudioChunk {
    audio: Vec<f32>,
    vad: VadResult,
}

impl InterimAsrState {
    pub fn start_streaming_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        audio_so_far: Vec<f32>,
        vad_results: Vec<VadResult>,
        end_sample: GlobalSampleIndex,
        created_at_frame: VadFrameIndex,
    ) -> Vec<PendingAsrSegment> {
        if audio_so_far.is_empty() {
            return Vec::new();
        }
        if let Some(active) = self.streaming.active.as_mut()
            && active.can_continue_with(previous_segment_id)
        {
            let chunks = streaming_chunks_from_flattened_audio(audio_so_far.clone(), vad_results);
            let overlap_samples = active.suffix_prefix_overlap_samples(&audio_so_far);
            active.current_segment_id = segment_id;
            active.append_chunks(drop_prefix_from_chunks(chunks, overlap_samples));
            return self.take_pending_streaming_delta();
        }

        let audio_len = audio_so_far.len() as u64;
        let range_start = GlobalSampleIndex(end_sample.0.saturating_sub(audio_len));
        self.streaming.active = Some(StreamingInterimSegmentState {
            display_segment_id: segment_id,
            current_segment_id: segment_id,
            chunks: streaming_chunks_from_flattened_audio(audio_so_far, vad_results),
            emitted_samples: 0,
            range_start,
            created_at_frame,
        });
        self.take_pending_streaming_delta()
    }

    pub fn extend_streaming_segment(
        &mut self,
        segment_id: u64,
        previous_segment_id: Option<u64>,
        new_audio: Vec<f32>,
        vad_result: VadResult,
        end_sample: GlobalSampleIndex,
        created_at_frame: VadFrameIndex,
    ) -> Vec<PendingAsrSegment> {
        if new_audio.is_empty() {
            return Vec::new();
        }
        if let Some(active) = self.streaming.active.as_mut()
            && (active.current_segment_id == segment_id
                || active.can_continue_with(previous_segment_id))
        {
            active.current_segment_id = segment_id;
            active.chunks.push(StreamingInterimAudioChunk {
                audio: new_audio,
                vad: vad_result,
            });
        } else {
            let audio_len = new_audio.len() as u64;
            let range_start = GlobalSampleIndex(end_sample.0.saturating_sub(audio_len));
            self.streaming.active = Some(StreamingInterimSegmentState {
                display_segment_id: segment_id,
                current_segment_id: segment_id,
                chunks: vec![StreamingInterimAudioChunk {
                    audio: new_audio,
                    vad: vad_result,
                }],
                emitted_samples: 0,
                range_start,
                created_at_frame,
            });
        }
        self.take_pending_streaming_delta()
    }

    #[must_use]
    pub fn interim_request(
        &self,
        streaming_interim_enabled: bool,
        segment: PendingAsrSegment,
    ) -> Option<PendingAsrSegment> {
        debug_assert_eq!(
            segment.reason,
            SegmentCloseReason::InterimResultSilenceReached
        );
        (!streaming_interim_enabled).then_some(segment)
    }

    pub fn clear_streaming_if_segment(&mut self, segment_id: u64) -> Option<u64> {
        let active = self.streaming.active.as_ref()?;
        if active.current_segment_id != segment_id {
            return None;
        }
        let display_segment_id = active.display_segment_id;
        self.streaming.active = None;
        Some(display_segment_id)
    }

    pub fn clear_streaming(&mut self) {
        self.streaming.active = None;
    }

    /// Emits each newly received source PCM delta immediately. Native ASR
    /// windowing is backend-owned: this layer must not know the model chunk
    /// size or synthesize padding.
    fn take_pending_streaming_delta(&mut self) -> Vec<PendingAsrSegment> {
        let Some(active) = self.streaming.active.as_mut() else {
            return Vec::new();
        };
        let delta_start = active.emitted_samples;
        let end = active.audio_len();
        if delta_start == end {
            return Vec::new();
        }
        active.emitted_samples = end;
        let (source_audio, source_vad_results) = active.audio_and_vad_range(0, end);
        let (audio, vad_results) = active.audio_and_vad_range(delta_start, end);
        let range = AudioRange::new(
            active.range_start,
            GlobalSampleIndex(active.range_start.0 + end as u64),
        );
        vec![PendingAsrSegment {
            segment_id: active.display_segment_id,
            previous_segment_id: None,
            source_audio,
            source_vad_results,
            audio,
            vad_results,
            reason: SegmentCloseReason::InterimChunkReached,
            range,
            created_at_frame: active.created_at_frame,
        }]
    }
}

impl StreamingInterimSegmentState {
    fn can_continue_with(&self, previous_segment_id: Option<u64>) -> bool {
        previous_segment_id == Some(self.current_segment_id)
    }

    fn audio_len(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.audio.len()).sum()
    }

    fn append_chunks(&mut self, chunks: Vec<StreamingInterimAudioChunk>) {
        self.chunks.extend(chunks);
    }

    fn suffix_prefix_overlap_samples(&self, prefix_audio: &[f32]) -> usize {
        let max_overlap = self.audio_len().min(prefix_audio.len());
        if max_overlap == 0 {
            return 0;
        }
        let suffix_start = self.audio_len() - max_overlap;
        let (suffix_audio, _) = self.audio_and_vad_range(suffix_start, self.audio_len());
        (1..=max_overlap)
            .rev()
            .find(|overlap| {
                suffix_audio[max_overlap - overlap..]
                    .iter()
                    .zip(&prefix_audio[..*overlap])
                    .all(|(left, right)| left.to_bits() == right.to_bits())
            })
            .unwrap_or(0)
    }

    fn audio_and_vad_range(&self, start: usize, end: usize) -> (Vec<f32>, Vec<VadResult>) {
        let end = end.min(self.audio_len());
        if start >= end {
            return (Vec::new(), Vec::new());
        }
        let mut consumed = 0;
        let mut audio = Vec::with_capacity(end - start);
        let mut vad_results = Vec::new();
        for chunk in &self.chunks {
            let chunk_start = consumed;
            let chunk_end = consumed + chunk.audio.len();
            consumed = chunk_end;
            if chunk_end <= start {
                continue;
            }
            if chunk_start >= end {
                break;
            }
            let local_start = start.saturating_sub(chunk_start);
            let local_end = (end - chunk_start).min(chunk.audio.len());
            if local_start < local_end {
                audio.extend_from_slice(&chunk.audio[local_start..local_end]);
                vad_results.push(chunk.vad);
            }
        }
        (audio, vad_results)
    }
}

fn streaming_chunks_from_flattened_audio(
    audio: Vec<f32>,
    vad_results: Vec<VadResult>,
) -> Vec<StreamingInterimAudioChunk> {
    if audio.is_empty() {
        return Vec::new();
    }
    if vad_results.is_empty() {
        return vec![StreamingInterimAudioChunk {
            audio,
            vad: VadResult {
                probability: 1.0,
                is_speech: true,
            },
        }];
    }
    let Some(ranges) = even_chunk_ranges(audio.len(), vad_results.len()) else {
        return vec![StreamingInterimAudioChunk {
            audio,
            vad: vad_results
                .last()
                .copied()
                .expect("non-empty VAD results should have a last value"),
        }];
    };
    ranges
        .into_iter()
        .zip(vad_results)
        .filter_map(|(range, vad)| {
            (!range.is_empty()).then(|| StreamingInterimAudioChunk {
                audio: audio[range].to_vec(),
                vad,
            })
        })
        .collect()
}

fn drop_prefix_from_chunks(
    chunks: Vec<StreamingInterimAudioChunk>,
    mut samples_to_drop: usize,
) -> Vec<StreamingInterimAudioChunk> {
    chunks
        .into_iter()
        .filter_map(|chunk| {
            if samples_to_drop >= chunk.audio.len() {
                samples_to_drop -= chunk.audio.len();
                return None;
            }
            if samples_to_drop == 0 {
                return Some(chunk);
            }
            let audio = chunk.audio[samples_to_drop..].to_vec();
            samples_to_drop = 0;
            (!audio.is_empty()).then_some(StreamingInterimAudioChunk {
                audio,
                vad: chunk.vad,
            })
        })
        .collect()
}

fn even_chunk_ranges(audio_len: usize, chunk_count: usize) -> Option<Vec<std::ops::Range<usize>>> {
    if audio_len == 0 || chunk_count == 0 {
        return None;
    }
    let base = audio_len / chunk_count;
    if base == 0 {
        return None;
    }
    let remainder = audio_len % chunk_count;
    let mut start = 0;
    Some(
        (0..chunk_count)
            .map(|index| {
                let len = base + usize::from(index < remainder);
                let end = (start + len).min(audio_len);
                let range = start..end;
                start = end;
                range
            })
            .collect(),
    )
}

pub struct TurnStore {
    pub turns: HashMap<u64, Turn>,
    pub audio_ranges: HashMap<u64, AudioRange>,
    pub revisions: HashMap<u64, u64>,
    pub finalized_turns: HashSet<u64>,
    pub streaming_interim_ranges: HashMap<u64, AudioRange>,
    pub confirmed_until_sample: GlobalSampleIndex,
    pub last_recognition_route: Option<RecognitionRoute>,
    pub open_turn_id: Option<u64>,
    pub open_turn_accepts_root_segment: bool,
    /// Correlates requests sent before a New turn has an id with the turn id
    /// allocated by the first usable result. The key is source-scoped and
    /// stable across interim/completion requests for the same segment chain.
    pub pending_new_turns: HashMap<(crate::SourceSessionKey, Option<SegmentId>), u64>,
}

impl Default for TurnStore {
    fn default() -> Self {
        Self {
            turns: HashMap::new(),
            audio_ranges: HashMap::new(),
            revisions: HashMap::new(),
            finalized_turns: HashSet::new(),
            streaming_interim_ranges: HashMap::new(),
            confirmed_until_sample: GlobalSampleIndex(0),
            last_recognition_route: None,
            open_turn_id: None,
            open_turn_accepts_root_segment: false,
            pending_new_turns: HashMap::new(),
        }
    }
}

pub struct RuntimeCounters {
    pub turn_session_id: u64,
    pub next_turn_id: u64,
    pub next_output_sequence: u64,
    pub next_request_id: u64,
    pub next_vad_frame_index: u64,
    pub next_runtime_tick: u64,
    pub global_sample_cursor: u64,
}

impl RuntimeCounters {
    #[must_use]
    pub const fn new(turn_session_id: u64) -> Self {
        Self {
            turn_session_id,
            next_turn_id: 1,
            next_output_sequence: 1,
            next_request_id: 1,
            next_vad_frame_index: 0,
            next_runtime_tick: 0,
            global_sample_cursor: 0,
        }
    }
}

#[derive(Default)]
pub struct ActivityState {
    pub segment_activity_epoch: u64,
    pub open_turn_activity_epoch: u64,
    pub open_turn_since_tick: Option<u64>,
}

#[derive(Default)]
pub struct AsrRequestState {
    pub in_flight_request: Option<AsrRequest>,
    pub pending_rerecognition_purpose: Option<RerecognitionPurpose>,
    pub last_dispatched: Option<AsrInFlight>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_interim_forwards_each_source_delta_without_native_chunk_wait() {
        let mut state = InterimAsrState::default();
        let speech = VadResult {
            probability: 0.9,
            is_speech: true,
        };
        let first = state.start_streaming_segment(
            1,
            None,
            vec![1.0; 1],
            vec![speech],
            GlobalSampleIndex(1),
            VadFrameIndex(1),
        );
        let second = state.extend_streaming_segment(
            1,
            None,
            vec![2.0; 1],
            speech,
            GlobalSampleIndex(2),
            VadFrameIndex(2),
        );

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].audio, vec![1.0; 1]);
        assert_eq!(first[0].source_audio, vec![1.0; 1]);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].audio, vec![2.0; 1]);
        assert_eq!(second[0].source_audio, vec![1.0, 2.0]);
    }
}
