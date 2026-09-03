// Fade ramps convert bounded audio-buffer indices to the engine's f32 sample type.
#![allow(clippy::cast_precision_loss)]

use crate::VadResult;

const ASR_EDGE_SILENCE_MS: usize = 320;
const ASR_EDGE_FADE_MS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsrRequestEdgePadding {
    Both,
    TrailingOnly,
}

pub fn ensure_asr_request_edge_silence(
    vad_interval_ms: u32,
    audio: &mut Vec<f32>,
    vad_results: &mut Vec<VadResult>,
    padding: AsrRequestEdgePadding,
) {
    if audio.is_empty() || vad_results.is_empty() {
        return;
    }
    let Some(chunk_samples) = estimated_vad_chunk_samples(audio.len(), vad_results.len()) else {
        return;
    };
    let required_silence = ASR_EDGE_SILENCE_MS
        .div_ceil(vad_interval_ms.max(1) as usize)
        .max(1)
        .saturating_mul(chunk_samples);
    let (leading_silence, trailing_silence) = vad_edge_silence_samples(audio.len(), vad_results);
    let missing_leading = match padding {
        AsrRequestEdgePadding::Both => required_silence.saturating_sub(leading_silence),
        AsrRequestEdgePadding::TrailingOnly => 0,
    };
    let missing_trailing = required_silence.saturating_sub(trailing_silence);
    if missing_leading == 0 && missing_trailing == 0 {
        return;
    }

    let fade_samples = chunk_samples
        .saturating_mul(ASR_EDGE_FADE_MS)
        .div_ceil(vad_interval_ms.max(1) as usize)
        .max(1)
        .min(audio.len());
    if missing_leading > 0 {
        apply_fade_in(audio, fade_samples);
    }
    if missing_trailing > 0 {
        apply_fade_out(audio, fade_samples);
    }
    if missing_leading > 0 {
        let mut padded = Vec::with_capacity(missing_leading + audio.len() + missing_trailing);
        padded.resize(missing_leading, 0.0);
        padded.extend_from_slice(audio);
        *audio = padded;
        prepend_silence_vad_frames(vad_results, missing_leading, chunk_samples);
    }
    if missing_trailing > 0 {
        audio.resize(audio.len() + missing_trailing, 0.0);
        append_silence_vad_frames(vad_results, missing_trailing, chunk_samples);
    }
}

fn estimated_vad_chunk_samples(audio_len: usize, vad_count: usize) -> Option<usize> {
    if audio_len == 0 || vad_count == 0 {
        return None;
    }
    Some(audio_len.div_ceil(vad_count).max(1))
}

fn prepend_silence_vad_frames(
    vad_results: &mut Vec<VadResult>,
    silence_samples: usize,
    chunk_samples: usize,
) {
    let added_vad_frames = silence_samples.div_ceil(chunk_samples).max(1);
    let mut padded = Vec::with_capacity(added_vad_frames + vad_results.len());
    padded.resize(
        added_vad_frames,
        VadResult {
            probability: 0.0,
            is_speech: false,
        },
    );
    padded.append(vad_results);
    *vad_results = padded;
}

fn append_silence_vad_frames(
    vad_results: &mut Vec<VadResult>,
    silence_samples: usize,
    chunk_samples: usize,
) {
    let added_vad_frames = silence_samples.div_ceil(chunk_samples).max(1);
    vad_results.resize(
        vad_results.len() + added_vad_frames,
        VadResult {
            probability: 0.0,
            is_speech: false,
        },
    );
}

fn vad_edge_silence_samples(audio_len: usize, vad_results: &[VadResult]) -> (usize, usize) {
    let Some(chunk_samples) = estimated_vad_chunk_samples(audio_len, vad_results.len()) else {
        return (0, 0);
    };
    let leading_frames = vad_results.iter().take_while(|vad| !vad.is_speech).count();
    let trailing_frames = vad_results
        .iter()
        .rev()
        .take_while(|vad| !vad.is_speech)
        .count();
    (
        leading_frames.saturating_mul(chunk_samples).min(audio_len),
        trailing_frames.saturating_mul(chunk_samples).min(audio_len),
    )
}

fn apply_fade_in(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    for (index, sample) in audio.iter_mut().take(fade_samples).enumerate() {
        *sample *= index as f32 / fade_samples.max(1) as f32;
    }
}

fn apply_fade_out(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    let audio_len = audio.len();
    for (index, sample) in audio.iter_mut().skip(audio_len - fade_samples).enumerate() {
        *sample *= (fade_samples - index) as f32 / fade_samples.max(1) as f32;
    }
}
