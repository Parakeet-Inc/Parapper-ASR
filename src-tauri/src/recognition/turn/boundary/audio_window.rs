use std::ops::Range;

use crate::recognition::segmentation::vad::engine::VadResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BoundaryAudioWindow {
    pub(super) prefix_audio_end: usize,
    pub(super) suffix_audio_start: usize,
}

pub(super) fn audio_window_for_boundary(
    audio_len: usize,
    vad_results: &[VadResult],
    boundary_sample: usize,
) -> BoundaryAudioWindow {
    let Some(chunk_ranges) = chunk_ranges(audio_len, vad_results.len()) else {
        let boundary_sample = boundary_sample.min(audio_len);
        return BoundaryAudioWindow {
            prefix_audio_end: boundary_sample,
            suffix_audio_start: boundary_sample,
        };
    };
    let boundary_sample = boundary_sample.min(audio_len);
    let Some(start_index) = chunk_ranges
        .iter()
        .position(|range| range.end > boundary_sample)
        .or_else(|| chunk_ranges.len().checked_sub(1))
    else {
        return BoundaryAudioWindow {
            prefix_audio_end: boundary_sample,
            suffix_audio_start: boundary_sample,
        };
    };

    if !vad_results[start_index].is_speech {
        let mut silence_start = start_index;
        while silence_start > 0 && !vad_results[silence_start - 1].is_speech {
            silence_start -= 1;
        }
        let silence_end = silence_run_end(vad_results, start_index);
        return BoundaryAudioWindow {
            prefix_audio_end: chunk_ranges[silence_end].end,
            suffix_audio_start: chunk_ranges[silence_start].start,
        };
    }

    if let Some((index, _)) = vad_results
        .iter()
        .enumerate()
        .skip(start_index + 1)
        .find(|(_, vad)| !vad.is_speech)
    {
        let silence_end = silence_run_end(vad_results, index);
        return BoundaryAudioWindow {
            prefix_audio_end: chunk_ranges[silence_end].end,
            suffix_audio_start: chunk_ranges[index].start,
        };
    }

    BoundaryAudioWindow {
        prefix_audio_end: boundary_sample,
        suffix_audio_start: boundary_sample,
    }
}

fn silence_run_end(vad_results: &[VadResult], start_index: usize) -> usize {
    let mut silence_end = start_index;
    while silence_end + 1 < vad_results.len() && !vad_results[silence_end + 1].is_speech {
        silence_end += 1;
    }
    silence_end
}

fn chunk_ranges(audio_len: usize, chunk_count: usize) -> Option<Vec<Range<usize>>> {
    if audio_len == 0 || chunk_count == 0 {
        return None;
    }
    let base = audio_len / chunk_count;
    let remainder = audio_len % chunk_count;
    if base == 0 {
        return None;
    }
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
