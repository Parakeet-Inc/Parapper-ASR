use crate::recognition::{
    control::RecognitionSession,
    segmentation::segment::builder::SegmentCloseReason,
    segmentation::vad::engine::VadResult,
    transcription::asr::{
        engine::AsrTranscript,
        task::{AsrRequest, AsrTaskKind, AudioRange, GlobalSampleIndex},
    },
    turn::{Turn, boundary::candidates_for_transcript, turn_event_id},
};

impl RecognitionSession {
    pub(in crate::recognition) fn apply_segment_transcript(
        &mut self,
        request: &AsrRequest,
        transcript: AsrTranscript,
        elapsed_millis: u128,
    ) -> u64 {
        let turn_id = request.target.turn_id.0;
        self.counters.next_turn_id = self.counters.next_turn_id.max(turn_id.saturating_add(1));
        let completion_replaces_streaming_interim =
            self.completion_replaces_streaming_interim(turn_id, request);
        let streaming_interim_overlap_offset = (!completion_replaces_streaming_interim)
            .then(|| self.streaming_interim_completion_source_overlap_offset(turn_id, request))
            .flatten();
        let completion_is_duplicate_tail = streaming_interim_overlap_offset.is_some_and(|offset| {
            !vad_has_speech_after_sample(
                request.source_audio.len(),
                &request.source_vad_results,
                offset,
            )
        });
        self.merge_turn_audio_range(turn_id, request.target.range);
        if completion_is_duplicate_tail {
            return turn_id;
        }
        let revision = *self.turn_store.revisions.entry(turn_id).or_insert(0);
        let turn = self.turn_store.turns.entry(turn_id).or_insert_with(|| {
            Turn::new(
                turn_event_id(self.counters.turn_session_id, turn_id, revision),
                revision,
            )
        });
        let draft = turn.draft_mut();
        draft.set_detected_language(request.detected_language.clone());
        let segment_id = request
            .target
            .last_segment_id
            .map_or(request.target.turn_id.0, |segment_id| segment_id.0);
        let previous_segment_id = request.target.first_segment_id.and_then(|segment_id| {
            (Some(segment_id) != request.target.last_segment_id).then_some(segment_id.0)
        });
        let latest_segment_audio_is_prefix = || {
            let Some(latest_audio_len) = draft.segment_audio_lens.last().copied() else {
                return false;
            };
            if latest_audio_len > request.source_audio.len()
                || latest_audio_len > draft.full_audio.len()
            {
                return false;
            }
            let latest_start = draft.full_audio.len() - latest_audio_len;
            draft.full_audio[latest_start..]
                .iter()
                .zip(request.source_audio.iter())
                .take(latest_audio_len)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        };
        let replace_latest_segment = draft.latest_segment_id == Some(segment_id)
            && (request.close_reason == Some(SegmentCloseReason::InterimChunkReached)
                || (request.kind == AsrTaskKind::CompletionCheck
                    && latest_segment_audio_is_prefix()))
            || completion_replaces_streaming_interim;
        if replace_latest_segment {
            draft.replace_latest_recognized_segment(
                segment_id,
                previous_segment_id,
                &request.source_audio,
                &request.source_vad_results,
                request.route,
                transcript.text,
                elapsed_millis,
            );
        } else {
            let append_source_start = streaming_interim_overlap_offset.unwrap_or(0);
            let append_vad_results;
            let source_vad_results = if append_source_start == 0 {
                request.source_vad_results.as_slice()
            } else {
                append_vad_results = vad_suffix_after_sample(
                    request.source_audio.len(),
                    &request.source_vad_results,
                    append_source_start,
                );
                append_vad_results.as_slice()
            };
            draft.append_recognized_segment(
                segment_id,
                previous_segment_id,
                &request.source_audio[append_source_start..],
                source_vad_results,
                request.route,
                transcript.text,
                elapsed_millis,
            );
        }
        if request.close_reason == Some(SegmentCloseReason::InterimChunkReached) {
            self.turn_store
                .streaming_interim_ranges
                .entry(turn_id)
                .and_modify(|range| *range = range.merge(request.target.range))
                .or_insert(request.target.range);
        } else if completion_replaces_streaming_interim {
            self.turn_store.streaming_interim_ranges.remove(&turn_id);
        }
        self.turn_store.last_recognition_route = Some(request.route);
        turn_id
    }

    fn completion_replaces_streaming_interim(&self, turn_id: u64, request: &AsrRequest) -> bool {
        if request.kind != AsrTaskKind::CompletionCheck
            || request.close_reason != Some(SegmentCloseReason::EndSilenceReached)
        {
            return false;
        }
        let Some(streaming_range) = self
            .turn_store
            .streaming_interim_ranges
            .get(&turn_id)
            .copied()
        else {
            return false;
        };
        if request.target.range.start_sample >= streaming_range.end_sample {
            return false;
        }
        if request.target.range.start_sample > streaming_range.start_sample {
            return false;
        }
        let Some(draft) = self.turn_store.turns.get(&turn_id).map(Turn::draft) else {
            return false;
        };
        let Some(first_segment_id) = request
            .target
            .first_segment_id
            .map(|segment_id| segment_id.0)
        else {
            return false;
        };
        draft.latest_segment_id == Some(first_segment_id)
    }

    fn streaming_interim_completion_source_overlap_offset(
        &self,
        turn_id: u64,
        request: &AsrRequest,
    ) -> Option<usize> {
        if request.kind != AsrTaskKind::CompletionCheck
            || request.close_reason != Some(SegmentCloseReason::EndSilenceReached)
        {
            return None;
        }
        let Some(streaming_range) = self
            .turn_store
            .streaming_interim_ranges
            .get(&turn_id)
            .copied()
        else {
            return None;
        };
        if request.target.range.start_sample >= streaming_range.end_sample {
            return None;
        }
        let Some(draft) = self.turn_store.turns.get(&turn_id).map(Turn::draft) else {
            return None;
        };
        let Some(first_segment_id) = request
            .target
            .first_segment_id
            .map(|segment_id| segment_id.0)
        else {
            return None;
        };
        if request.target.last_segment_id == request.target.first_segment_id {
            return None;
        }
        if draft.latest_segment_id != Some(first_segment_id) {
            return None;
        }
        Some(
            samples_between(
                request.target.range.start_sample,
                streaming_range.end_sample,
            )
            .min(request.source_audio.len()),
        )
    }

    fn merge_turn_audio_range(&mut self, turn_id: u64, range: AudioRange) {
        self.turn_store
            .audio_ranges
            .entry(turn_id)
            .and_modify(|current| *current = current.merge(range))
            .or_insert(range);
    }

    pub(in crate::recognition) fn apply_rerecognition_transcript(
        &mut self,
        request: &AsrRequest,
        transcript: AsrTranscript,
        elapsed_millis: u128,
        refresh_boundary_candidates: bool,
    ) {
        let turn_id = request.target.turn_id.0;
        let candidates = refresh_boundary_candidates.then(|| {
            candidates_for_transcript(
                request.route.language,
                &transcript,
                &request.audio,
                &request.vad_results,
                self.io.japanese_morph.as_ref(),
            )
        });
        if let Some(turn) = self.turn_store.turns.get_mut(&turn_id) {
            let draft = turn.draft_mut();
            draft.set_detected_language(request.detected_language.clone());
            draft.replace_text_preserving_sources(request.route, transcript.text, elapsed_millis);
            if let Some(candidates) = candidates {
                draft.boundary_candidates = candidates;
            }
            self.turn_store.last_recognition_route = Some(request.route);
        }
    }
}

fn samples_between(start: GlobalSampleIndex, end: GlobalSampleIndex) -> usize {
    usize::try_from(end.0.saturating_sub(start.0)).unwrap_or(usize::MAX)
}

fn vad_has_speech_after_sample(
    audio_len: usize,
    vad_results: &[VadResult],
    start_sample: usize,
) -> bool {
    if start_sample >= audio_len {
        return false;
    }
    let Some(ranges) = even_chunk_ranges(audio_len, vad_results.len()) else {
        return true;
    };
    ranges
        .into_iter()
        .zip(vad_results)
        .any(|(range, vad)| vad.is_speech && range.end > start_sample)
}

fn vad_suffix_after_sample(
    audio_len: usize,
    vad_results: &[VadResult],
    start_sample: usize,
) -> Vec<VadResult> {
    if start_sample == 0 {
        return vad_results.to_vec();
    }
    if start_sample >= audio_len || vad_results.is_empty() {
        return Vec::new();
    }
    let Some(ranges) = even_chunk_ranges(audio_len, vad_results.len()) else {
        return vad_results.to_vec();
    };
    ranges
        .into_iter()
        .zip(vad_results)
        .filter_map(|(range, vad)| (range.end > start_sample).then_some(*vad))
        .collect()
}

fn even_chunk_ranges(audio_len: usize, chunk_count: usize) -> Option<Vec<std::ops::Range<usize>>> {
    if chunk_count == 0 || audio_len < chunk_count {
        return None;
    }
    let base = audio_len / chunk_count;
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
