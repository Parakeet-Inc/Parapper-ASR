use std::borrow::Cow;

use tauri::AppHandle;

use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    error_event::{ErrorSeverity, ParapperErrorType, emit_parapper_error},
    recognition::{
        segmentation::vad::engine::VadResult, transcription::asr::engine::AsrTranscript,
    },
};

pub(crate) const MIN_LANGUAGE_ID_SAMPLES: usize = ASR_SAMPLE_RATE as usize;
const NORMALIZED_ASR_INPUT_PEAK: f32 = 0.95;
const ASR_EDGE_SILENCE_MS: usize = 320;
const ASR_EDGE_FADE_MS: usize = 10;

#[derive(Clone, Copy)]
pub(crate) enum AsrRequestEdgePadding {
    TrailingOnly,
    LeadingAndTrailing,
}

pub(crate) struct PreparedAsrInput<'a> {
    pub(crate) audio: Cow<'a, [f32]>,
    pub(crate) leading_padding_samples: usize,
}

pub(in crate::recognition) fn ensure_asr_request_edge_silence(
    config: &ParapperConfig,
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
    let required_silence = asr_request_edge_silence_chunks(config).saturating_mul(chunk_samples);
    let (leading_silence, trailing_silence) = vad_edge_silence_samples(audio.len(), vad_results);
    let missing_leading = match padding {
        AsrRequestEdgePadding::LeadingAndTrailing => {
            required_silence.saturating_sub(leading_silence)
        }
        AsrRequestEdgePadding::TrailingOnly => 0,
    };
    let missing_trailing = required_silence.saturating_sub(trailing_silence);
    if missing_leading == 0 && missing_trailing == 0 {
        return;
    }

    let fade_samples = fade_samples_for_vad_chunk(config, chunk_samples).min(audio.len());
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

pub(crate) fn prepare_asr_input_audio<'a>(
    audio: &'a [f32],
    vad_results: &[VadResult],
) -> PreparedAsrInput<'a> {
    if audio.is_empty() {
        return PreparedAsrInput {
            audio: Cow::Borrowed(audio),
            leading_padding_samples: 0,
        };
    }

    let required_silence = samples_for_millis(ASR_EDGE_SILENCE_MS);
    let fade_samples = samples_for_millis(ASR_EDGE_FADE_MS);
    let (leading_silence, trailing_silence) = vad_edge_silence_samples(audio.len(), vad_results);
    let missing_leading = required_silence.saturating_sub(leading_silence);
    let missing_trailing = required_silence.saturating_sub(trailing_silence);

    if missing_leading == 0 && missing_trailing == 0 {
        return PreparedAsrInput {
            audio: Cow::Borrowed(audio),
            leading_padding_samples: 0,
        };
    }

    let mut padded = Vec::with_capacity(missing_leading + audio.len() + missing_trailing);
    padded.resize(missing_leading, 0.0);
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + missing_trailing, 0.0);

    let original_start = missing_leading;
    let original_end = original_start + audio.len();
    apply_fade_in(&mut padded[original_start..original_end], fade_samples);
    apply_fade_out(&mut padded[original_start..original_end], fade_samples);

    PreparedAsrInput {
        audio: Cow::Owned(padded),
        leading_padding_samples: missing_leading,
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "ASR timestamps are seconds in f32, so inserted sample counts must be converted to f32 seconds."
)]
pub(crate) fn maybe_shift_transcript_timestamps_for_leading_padding(
    transcript: &mut AsrTranscript,
    leading_padding_samples: usize,
) {
    if leading_padding_samples == 0 {
        return;
    }
    let leading_padding_sec = leading_padding_samples as f32 / ASR_SAMPLE_RATE as f32;
    let Some(first_timestamp) = transcript.tokens.iter().find_map(|token| token.start_sec) else {
        return;
    };
    if first_timestamp < leading_padding_sec * 0.8 {
        return;
    }
    for token in &mut transcript.tokens {
        if let Some(start_sec) = &mut token.start_sec {
            *start_sec = (*start_sec - leading_padding_sec).max(0.0);
        }
    }
}

pub(crate) fn normalize_asr_input_audio<'a>(
    config: &ParapperConfig,
    audio: &'a [f32],
) -> Cow<'a, [f32]> {
    if !config.asr.normalize_input_audio {
        return Cow::Borrowed(audio);
    }

    let peak = audio
        .iter()
        .copied()
        .filter(|sample| sample.is_finite())
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if peak <= f32::EPSILON {
        return Cow::Borrowed(audio);
    }

    let gain = NORMALIZED_ASR_INPUT_PEAK / peak;
    if (gain - 1.0).abs() <= f32::EPSILON {
        return Cow::Borrowed(audio);
    }

    Cow::Owned(
        audio
            .iter()
            .copied()
            .map(|sample| {
                if sample.is_finite() {
                    sample * gain
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

pub(crate) fn emit_asr_warning(handle: &AppHandle, err: &anyhow::Error) {
    emit_parapper_error(
        handle,
        ParapperErrorType::Asr,
        ErrorSeverity::Warning,
        Some(err.to_string()),
    );
}

fn samples_for_millis(millis: usize) -> usize {
    ASR_SAMPLE_RATE as usize * millis / 1_000
}

fn asr_request_edge_silence_chunks(config: &ParapperConfig) -> usize {
    ASR_EDGE_SILENCE_MS
        .div_ceil(config.segmentation.vad_interval_ms.max(1) as usize)
        .max(1)
}

fn estimated_vad_chunk_samples(audio_len: usize, vad_count: usize) -> Option<usize> {
    if audio_len == 0 || vad_count == 0 {
        return None;
    }
    Some(audio_len.div_ceil(vad_count).max(1))
}

fn fade_samples_for_vad_chunk(config: &ParapperConfig, chunk_samples: usize) -> usize {
    let interval_ms = config.segmentation.vad_interval_ms.max(1) as usize;
    chunk_samples
        .saturating_mul(ASR_EDGE_FADE_MS)
        .div_ceil(interval_ms)
        .max(1)
}

fn prepend_silence_vad_frames(
    vad_results: &mut Vec<VadResult>,
    silence_samples: usize,
    chunk_samples: usize,
) {
    let added_vad_frames = silence_samples.div_ceil(chunk_samples).max(1);
    let mut padded = Vec::with_capacity(added_vad_frames + vad_results.len());
    padded.extend(std::iter::repeat_n(silence_vad_result(), added_vad_frames));
    padded.append(vad_results);
    *vad_results = padded;
}

fn append_silence_vad_frames(
    vad_results: &mut Vec<VadResult>,
    silence_samples: usize,
    chunk_samples: usize,
) {
    let added_vad_frames = silence_samples.div_ceil(chunk_samples).max(1);
    vad_results.extend(std::iter::repeat_n(silence_vad_result(), added_vad_frames));
}

fn silence_vad_result() -> VadResult {
    VadResult {
        probability: 0.0,
        is_speech: false,
    }
}

fn vad_edge_silence_samples(audio_len: usize, vad_results: &[VadResult]) -> (usize, usize) {
    let Some(ranges) = chunk_ranges(audio_len, vad_results.len()) else {
        return (0, 0);
    };
    let leading = vad_results
        .iter()
        .zip(ranges.iter())
        .take_while(|(vad, _)| !vad.is_speech)
        .map(|(_, range)| range.len())
        .sum();
    let trailing = vad_results
        .iter()
        .rev()
        .zip(ranges.iter().rev())
        .take_while(|(vad, _)| !vad.is_speech)
        .map(|(_, range)| range.len())
        .sum();
    (leading, trailing)
}

fn chunk_ranges(audio_len: usize, chunk_count: usize) -> Option<Vec<std::ops::Range<usize>>> {
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

#[expect(
    clippy::cast_precision_loss,
    reason = "Fade gains are intentionally computed as f32 ratios for audio samples."
)]
fn apply_fade_in(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    if fade_samples == 0 {
        return;
    }
    for (index, sample) in audio.iter_mut().take(fade_samples).enumerate() {
        let gain = index as f32 / fade_samples as f32;
        *sample *= gain;
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Fade gains are intentionally computed as f32 ratios for audio samples."
)]
fn apply_fade_out(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    if fade_samples == 0 {
        return;
    }
    let start = audio.len() - fade_samples;
    for (index, sample) in audio[start..].iter_mut().enumerate() {
        let gain = (fade_samples - index) as f32 / fade_samples as f32;
        *sample *= gain;
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{
        apply_fade_in, apply_fade_out, chunk_ranges,
        maybe_shift_transcript_timestamps_for_leading_padding, normalize_asr_input_audio,
        prepare_asr_input_audio,
    };
    use crate::{
        config::ParapperConfig,
        recognition::{
            segmentation::vad::engine::VadResult, transcription::asr::engine::AsrTranscript,
        },
    };

    #[test]
    fn normalize_asr_input_audio_scales_peak_to_target() {
        let config = parapper_config! {
            asr_normalize_input_audio: true,
            ..ParapperConfig::default()
        };
        let normalized = normalize_asr_input_audio(&config, &[0.0, 0.5, -0.25]);

        assert!(matches!(normalized, Cow::Owned(_)));
        assert!((normalized[1] - 0.95).abs() < 0.0001);
        assert!((normalized[2] + 0.475).abs() < 0.0001);
    }

    #[test]
    fn normalize_asr_input_audio_keeps_audio_when_disabled() {
        let config = parapper_config! {
            asr_normalize_input_audio: false,
            ..ParapperConfig::default()
        };
        let audio = [0.0, 0.5, -0.25];
        let normalized = normalize_asr_input_audio(&config, &audio);

        assert!(matches!(normalized, Cow::Borrowed(_)));
        assert_f32_slice_close(normalized.as_ref(), &audio, f32::EPSILON);
    }

    #[test]
    fn normalize_asr_input_audio_zeros_non_finite_samples_when_scaling_finite_peak() {
        let config = parapper_config! {
            asr_normalize_input_audio: true,
            ..ParapperConfig::default()
        };
        let normalized = normalize_asr_input_audio(&config, &[f32::NAN, 0.5, f32::INFINITY, -1.0]);

        assert!(matches!(normalized, Cow::Owned(_)));
        assert_f32_close(normalized[0], 0.0, f32::EPSILON);
        assert!((normalized[1] - 0.475).abs() < 0.0001);
        assert_f32_close(normalized[2], 0.0, f32::EPSILON);
        assert!((normalized[3] + 0.95).abs() < 0.0001);
    }

    #[test]
    fn normalize_asr_input_audio_keeps_silent_or_non_finite_only_audio_unscaled() {
        let config = parapper_config! {
            asr_normalize_input_audio: true,
            ..ParapperConfig::default()
        };
        for audio in [&[0.0, -0.0][..], &[f32::NAN, f32::INFINITY][..]] {
            let normalized = normalize_asr_input_audio(&config, audio);

            assert!(matches!(normalized, Cow::Borrowed(_)));
            assert_eq!(normalized.len(), audio.len());
            for (actual, expected) in normalized.iter().zip(audio.iter()) {
                if expected.is_finite() {
                    assert_f32_close(*actual, *expected, f32::EPSILON);
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn transcript_timestamp_shift_ignores_zero_padding_or_empty_tokens() {
        let mut zero_padding =
            AsrTranscript::from_parts("abc", vec!["abc".to_string()], Some(&[1.0]), Some(&[0.1]));
        maybe_shift_transcript_timestamps_for_leading_padding(&mut zero_padding, 0);
        assert_eq!(
            zero_padding.tokens[0].start_sec.map(f32::to_bits),
            Some(1.0_f32.to_bits())
        );

        let mut empty_tokens = AsrTranscript::from_text("abc");
        maybe_shift_transcript_timestamps_for_leading_padding(&mut empty_tokens, 16_000);
        assert!(empty_tokens.tokens.is_empty());
    }

    #[test]
    fn transcript_timestamp_shift_only_compensates_when_first_token_includes_padding() {
        let mut already_compensated =
            AsrTranscript::from_parts("abc", vec!["abc".to_string()], Some(&[0.79]), Some(&[0.1]));
        maybe_shift_transcript_timestamps_for_leading_padding(&mut already_compensated, 16_000);
        assert_f32_close_with_context(
            already_compensated.tokens[0]
                .start_sec
                .expect("timestamp should be preserved"),
            0.79,
            f32::EPSILON,
            "timestamps already near the original audio start must not be shifted",
        );

        let mut includes_padding =
            AsrTranscript::from_parts("abc", vec!["abc".to_string()], Some(&[0.8]), Some(&[0.1]));
        maybe_shift_transcript_timestamps_for_leading_padding(&mut includes_padding, 16_000);
        assert_f32_close_with_context(
            includes_padding.tokens[0]
                .start_sec
                .expect("timestamp should be shifted"),
            0.0,
            f32::EPSILON,
            "timestamps at the padding threshold should be shifted back to the original audio",
        );
    }

    #[test]
    fn prepare_asr_input_audio_adds_missing_edge_silence_and_fades_original_audio() {
        let audio = vec![1.0; 320];

        let prepared = prepare_asr_input_audio(&audio, &vads(&[true]));

        assert!(matches!(prepared.audio, Cow::Owned(_)));
        assert_eq!(prepared.leading_padding_samples, 5_120);
        assert_eq!(prepared.audio.len(), 10_560);
        assert_f32_close(prepared.audio[0], 0.0, f32::EPSILON);
        assert_f32_close(prepared.audio[5_120], 0.0, f32::EPSILON);
        assert!(
            prepared.audio[5_121] > 0.0 && prepared.audio[5_121] < 1.0,
            "the original audio should fade in after inserted silence"
        );
        assert_f32_close(prepared.audio[5_280], 1.0, f32::EPSILON);
        assert!(
            prepared.audio[5_439] > 0.0 && prepared.audio[5_439] < 1.0,
            "the original audio should fade out before appended silence"
        );
        assert_f32_close(prepared.audio[5_440], 0.0, f32::EPSILON);
    }

    #[test]
    fn prepare_asr_input_audio_uses_vad_silence_instead_of_sample_amplitude_for_edge_padding() {
        let mut audio = vec![1.0; 5_120];
        audio.extend(vec![1.0; 5_120]);
        audio.extend(vec![1.0; 5_120]);

        let prepared = prepare_asr_input_audio(&audio, &vads(&[false, true, false]));

        assert!(matches!(prepared.audio, Cow::Borrowed(_)));
        assert_eq!(prepared.leading_padding_samples, 0);
        assert_eq!(
            prepared
                .audio
                .iter()
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>(),
            audio
                .iter()
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn chunk_ranges_returns_none_when_vad_chunks_are_more_granular_than_audio_samples() {
        assert_eq!(chunk_ranges(2, 3), None);
    }

    #[test]
    fn fade_in_and_out_clamp_to_short_audio_and_allow_zero_duration() {
        let mut no_fade = vec![1.0, 1.0];
        apply_fade_in(&mut no_fade, 0);
        apply_fade_out(&mut no_fade, 0);
        assert_f32_slice_close(&no_fade, &[1.0, 1.0], f32::EPSILON);

        let mut faded = vec![1.0, 1.0, 1.0];
        apply_fade_in(&mut faded, 10);
        assert_f32_slice_close_with_context(
            &faded,
            &[0.0, 1.0 / 3.0, 2.0 / 3.0],
            f32::EPSILON,
            "fade-in should clamp the requested duration to the available samples",
        );
        apply_fade_out(&mut faded, 10);
        assert!((faded[0] - 0.0).abs() < f32::EPSILON);
        assert!((faded[1] - 2.0 / 9.0).abs() < 0.0001);
        assert!((faded[2] - 2.0 / 9.0).abs() < 0.0001);
    }

    fn vads(pattern: &[bool]) -> Vec<VadResult> {
        pattern
            .iter()
            .map(|is_speech| VadResult {
                is_speech: *is_speech,
                probability: if *is_speech { 0.9 } else { 0.0 },
            })
            .collect()
    }

    fn assert_f32_close(actual: f32, expected: f32, tolerance: f32) {
        assert_f32_close_with_context(actual, expected, tolerance, "");
    }

    fn assert_f32_close_with_context(actual: f32, expected: f32, tolerance: f32, context: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context} actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_f32_slice_close_with_context(actual, expected, tolerance, "");
    }

    fn assert_f32_slice_close_with_context(
        actual: &[f32],
        expected: &[f32],
        tolerance: f32,
        context: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "{context} index={index}, actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}
