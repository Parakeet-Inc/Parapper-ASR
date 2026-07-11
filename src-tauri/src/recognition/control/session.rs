use std::collections::{HashMap, HashSet, VecDeque};

use super::pending::{PendingFinalization, PendingTurnCheck, RerecognitionPurpose};
use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    recognition::{
        segmentation::{segment::builder::SegmentCloseReason, vad::engine::VadResult},
        transcription::{
            asr::{
                input::NEMOTRON_CHUNK_MS,
                task::{AsrInFlight, AsrRequest, AudioRange, GlobalSampleIndex, VadFrameIndex},
            },
            planner::PendingAsrSegment,
            route::{RecognitionRoute, language_id::LanguageDetector},
        },
        turn::{Turn, boundary::JapaneseMorphAnalyzer},
    },
};

use super::{AsrRequestRunner, TurnDecisionRunner, TurnOutputSink};

pub(crate) struct RecognitionSession {
    pub(in crate::recognition) config: ParapperConfig,
    pub(in crate::recognition) pending: PendingRuntimeState,
    pub(in crate::recognition) io: RuntimeIo,
    pub(in crate::recognition) turn_store: TurnStore,
    pub(in crate::recognition) counters: RuntimeCounters,
    pub(in crate::recognition) activity: ActivityState,
    pub(in crate::recognition) requests: AsrRequestState,
}

pub(in crate::recognition) struct RuntimeIo {
    pub(in crate::recognition) asr_runner: Box<dyn AsrRequestRunner>,
    pub(in crate::recognition) turn_decision_runner: Box<dyn TurnDecisionRunner>,
    pub(in crate::recognition) output_sink: Box<dyn TurnOutputSink>,
    pub(in crate::recognition) language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
    pub(in crate::recognition) language_id: Option<Box<dyn LanguageDetector>>,
    pub(in crate::recognition) japanese_morph: Option<JapaneseMorphAnalyzer>,
}

#[derive(Default)]
pub(in crate::recognition) struct PendingRuntimeState {
    pub(in crate::recognition) turn_check: Option<PendingTurnCheck>,
    pub(in crate::recognition) finalization: Option<PendingFinalization>,
    pub(in crate::recognition) asr_segments: VecDeque<PendingAsrSegment>,
    pub(in crate::recognition) interim_asr: InterimAsrState,
}

#[derive(Default)]
pub(in crate::recognition) struct InterimAsrState {
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
    pub(in crate::recognition) fn start_streaming_segment(
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
            return self.take_ready_streaming_segments();
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
        self.take_ready_streaming_segments()
    }

    pub(in crate::recognition) fn extend_streaming_segment(
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
        self.take_ready_streaming_segments()
    }

    #[expect(
        clippy::unused_self,
        reason = "interim request policy stays behind the interim ASR state boundary even when the first branch only depends on the selected mode"
    )]
    pub(in crate::recognition) fn interim_request(
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

    pub(in crate::recognition) fn clear_streaming_if_segment(
        &mut self,
        segment_id: u64,
    ) -> Option<u64> {
        let active = self.streaming.active.as_ref()?;
        if active.current_segment_id != segment_id {
            return None;
        }
        let display_segment_id = active.display_segment_id;
        self.streaming.active = None;
        Some(display_segment_id)
    }

    pub(in crate::recognition) fn clear_streaming(&mut self) {
        self.streaming.active = None;
    }

    fn take_ready_streaming_segments(&mut self) -> Vec<PendingAsrSegment> {
        let Some(active) = self.streaming.active.as_mut() else {
            return Vec::new();
        };
        let mut segments = Vec::new();
        let chunk_samples = nemotron_interim_chunk_samples();
        while active.emitted_samples + chunk_samples <= active.audio_len() {
            let delta_start = active.emitted_samples;
            active.emitted_samples += chunk_samples;
            let (source_audio, source_vad_results) =
                active.audio_and_vad_range(0, active.emitted_samples);
            let (audio, vad_results) =
                active.audio_and_vad_range(delta_start, active.emitted_samples);
            let range = AudioRange::new(
                active.range_start,
                GlobalSampleIndex(active.range_start.0 + active.emitted_samples as u64),
            );
            segments.push(PendingAsrSegment {
                segment_id: active.display_segment_id,
                previous_segment_id: None,
                source_audio,
                source_vad_results,
                audio,
                vad_results,
                reason: SegmentCloseReason::InterimChunkReached,
                range,
                created_at_frame: active.created_at_frame,
            });
        }
        segments
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

fn nemotron_interim_chunk_samples() -> usize {
    ASR_SAMPLE_RATE as usize * NEMOTRON_CHUNK_MS / 1_000
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

pub(in crate::recognition) trait LanguageIdRuntime:
    crate::recognition::transcription::route::language_id::LanguageDetectionWarningSink
{
    fn build_language_id(&self, config: &ParapperConfig) -> Option<Box<dyn LanguageDetector>>;
}

pub(in crate::recognition) struct TurnStore {
    pub(in crate::recognition) turns: HashMap<u64, Turn>,
    pub(in crate::recognition) audio_ranges: HashMap<u64, AudioRange>,
    pub(in crate::recognition) revisions: HashMap<u64, u64>,
    pub(in crate::recognition) finalized_turns: HashSet<u64>,
    pub(in crate::recognition) streaming_interim_ranges: HashMap<u64, AudioRange>,
    pub(in crate::recognition) confirmed_until_sample: GlobalSampleIndex,
    pub(in crate::recognition) last_recognition_route: Option<RecognitionRoute>,
    pub(in crate::recognition) open_turn_id: Option<u64>,
    pub(in crate::recognition) open_turn_accepts_root_segment: bool,
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
        }
    }
}

pub(in crate::recognition) struct RuntimeCounters {
    pub(in crate::recognition) turn_session_id: u64,
    pub(in crate::recognition) next_turn_id: u64,
    pub(in crate::recognition) next_output_sequence: u64,
    pub(in crate::recognition) next_request_id: u64,
    pub(in crate::recognition) next_vad_frame_index: u64,
    pub(in crate::recognition) next_runtime_tick: u64,
    pub(in crate::recognition) global_sample_cursor: u64,
}

impl RuntimeCounters {
    pub(in crate::recognition) fn new(turn_session_id: u64) -> Self {
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
pub(in crate::recognition) struct ActivityState {
    pub(in crate::recognition) segment_activity_epoch: u64,
    pub(in crate::recognition) open_turn_activity_epoch: u64,
    pub(in crate::recognition) open_turn_since_tick: Option<u64>,
}

#[derive(Default)]
pub(in crate::recognition) struct AsrRequestState {
    pub(in crate::recognition) in_flight_request: Option<AsrRequest>,
    pub(in crate::recognition) pending_rerecognition_purpose: Option<RerecognitionPurpose>,
    pub(in crate::recognition) last_dispatched: Option<AsrInFlight>,
}
