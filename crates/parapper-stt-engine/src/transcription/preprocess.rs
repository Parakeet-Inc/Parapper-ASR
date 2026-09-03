use std::borrow::Cow;

use parapper_models::asr::{AsrSpeechRangeSamples, AsrStreamConfig};

use crate::{AsrTranscript, SAMPLE_RATE_HZ, VadResult};

const ASR_SAMPLE_RATE: u32 = SAMPLE_RATE_HZ;

pub const MIN_LANGUAGE_ID_SAMPLES: usize = ASR_SAMPLE_RATE as usize;
const NORMALIZED_ASR_INPUT_PEAK: f32 = 0.95;
const ASR_EDGE_SILENCE_MS: usize = 320;
const ASR_EDGE_FADE_MS: usize = 10;
pub const NEMOTRON_CHUNK_MS: usize = 160;
const NEMOTRON_EDGE_FADE_MS: usize = 80;

pub struct PreparedAsrInput<'a> {
    pub audio: Cow<'a, [f32]>,
    pub leading_padding_samples: usize,
}

#[must_use]
pub fn prepare_asr_input_audio<'a>(
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

#[must_use]
pub fn prepare_nemotron_input_audio<'a>(
    audio: &'a [f32],
    vad_results: &[VadResult],
) -> PreparedAsrInput<'a> {
    prepare_nemotron_input_audio_with_tail(audio, vad_results, NemotronTailPadding::AdjustmentOnly)
}

#[must_use]
pub fn prepare_nemotron_streaming_bootstrap_audio<'a>(
    audio: &'a [f32],
    vad_results: &[VadResult],
) -> PreparedAsrInput<'a> {
    prepare_nemotron_input_audio_with_tail(audio, vad_results, NemotronTailPadding::None)
}

/// Reports only the segmenter-derived boundary that a model backend needs to
/// reproduce its native stream bootstrap. VAD samples themselves never cross
/// the model boundary.
#[must_use]
pub fn asr_stream_config_for_source_audio(
    source_audio: &[f32],
    source_vad_results: &[VadResult],
) -> AsrStreamConfig {
    AsrStreamConfig {
        // The old preparation helper treated a missing speech classification
        // as the whole first delta. Preserve that observable behavior without
        // passing VAD frames across the backend boundary.
        speech_range_samples: (!source_audio.is_empty()).then(|| {
            let (start, end) = speech_sample_range(source_audio.len(), source_vad_results)
                .unwrap_or((0, source_audio.len()));
            AsrSpeechRangeSamples { start, end }
        }),
        language_hint: None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NemotronTailPadding {
    None,
    AdjustmentOnly,
}

fn prepare_nemotron_input_audio_with_tail<'a>(
    audio: &'a [f32],
    vad_results: &[VadResult],
    tail: NemotronTailPadding,
) -> PreparedAsrInput<'a> {
    if audio.is_empty() {
        return PreparedAsrInput {
            audio: Cow::Borrowed(audio),
            leading_padding_samples: 0,
        };
    }

    let chunk_samples = samples_for_millis(NEMOTRON_CHUNK_MS);
    let fade_samples = samples_for_millis(NEMOTRON_EDGE_FADE_MS).min(audio.len());
    let (speech_start, speech_end) =
        speech_sample_range(audio.len(), vad_results).unwrap_or((0, audio.len()));
    let speech_len = speech_end.saturating_sub(speech_start);
    let required_prefix =
        fade_samples + alignment_padding_samples(fade_samples + speech_len, chunk_samples);
    let leading_available = speech_start.min(audio.len());
    let (copy_start, leading_padding_samples) = if leading_available >= required_prefix {
        (speech_start - required_prefix, 0)
    } else {
        (0, required_prefix - leading_available)
    };

    let required_tail = match tail {
        NemotronTailPadding::None => 0,
        NemotronTailPadding::AdjustmentOnly => samples_for_millis(NEMOTRON_EDGE_FADE_MS),
    };
    let trailing_available = audio.len().saturating_sub(speech_end);
    let tail_padding_samples = required_tail.saturating_sub(trailing_available);

    let copied_len = audio.len().saturating_sub(copy_start);
    let mut output = Vec::with_capacity(
        leading_padding_samples + copied_len + tail_padding_samples + chunk_samples,
    );
    output.resize(leading_padding_samples, 0.0);
    let copied_start = output.len();
    output.extend_from_slice(&audio[copy_start..]);
    let copied_end = output.len();
    if copied_start < copied_end {
        apply_fade_in(&mut output[copied_start..copied_end], fade_samples);
        if tail != NemotronTailPadding::None && tail_padding_samples > 0 {
            apply_fade_out(&mut output[copied_start..copied_end], fade_samples);
        }
    }
    output.resize(output.len() + tail_padding_samples, 0.0);

    let end_alignment = alignment_padding_samples(output.len(), chunk_samples);
    output.resize(output.len() + end_alignment, 0.0);

    PreparedAsrInput {
        audio: Cow::Owned(output),
        leading_padding_samples,
    }
}

fn alignment_padding_samples(len: usize, chunk_samples: usize) -> usize {
    let remainder = len % chunk_samples;
    if remainder == 0 {
        0
    } else {
        chunk_samples - remainder
    }
}

fn speech_sample_range(audio_len: usize, vad_results: &[VadResult]) -> Option<(usize, usize)> {
    if audio_len == 0 || vad_results.is_empty() {
        return None;
    }
    let chunk_samples = estimated_vad_chunk_samples(audio_len, vad_results.len())?;
    let first_speech = vad_results.iter().position(|vad| vad.is_speech)?;
    let last_speech = vad_results.iter().rposition(|vad| vad.is_speech)?;
    let start = first_speech.saturating_mul(chunk_samples).min(audio_len);
    let end = (last_speech + 1)
        .saturating_mul(chunk_samples)
        .min(audio_len);
    Some((start, end.max(start)))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "ASR timestamps are seconds in f32, so inserted sample counts must be converted to f32 seconds."
)]
pub fn maybe_shift_transcript_timestamps_for_leading_padding(
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

pub fn normalize_asr_input_audio(enabled: bool, audio: &[f32]) -> Cow<'_, [f32]> {
    if !enabled {
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

fn samples_for_millis(millis: usize) -> usize {
    ASR_SAMPLE_RATE as usize * millis / 1_000
}

fn estimated_vad_chunk_samples(audio_len: usize, vad_count: usize) -> Option<usize> {
    if audio_len == 0 || vad_count == 0 {
        return None;
    }
    Some(audio_len.div_ceil(vad_count).max(1))
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
        NEMOTRON_CHUNK_MS, NEMOTRON_EDGE_FADE_MS, apply_fade_in, apply_fade_out, chunk_ranges,
        maybe_shift_transcript_timestamps_for_leading_padding, normalize_asr_input_audio,
        prepare_asr_input_audio, prepare_nemotron_input_audio,
        prepare_nemotron_streaming_bootstrap_audio, samples_for_millis,
    };
    use crate::{AsrTranscript, VadResult};

    #[test]
    fn normalize_asr_input_audio_scales_peak_to_target() {
        let normalized = normalize_asr_input_audio(true, &[0.0, 0.5, -0.25]);

        assert!(matches!(normalized, Cow::Owned(_)));
        assert!((normalized[1] - 0.95).abs() < 0.0001);
        assert!((normalized[2] + 0.475).abs() < 0.0001);
    }

    #[test]
    fn normalize_asr_input_audio_keeps_audio_when_disabled() {
        let audio = [0.0, 0.5, -0.25];
        let normalized = normalize_asr_input_audio(false, &audio);

        assert!(matches!(normalized, Cow::Borrowed(_)));
        assert_f32_slice_close(normalized.as_ref(), &audio, f32::EPSILON);
    }

    #[test]
    fn normalize_asr_input_audio_zeros_non_finite_samples_when_scaling_finite_peak() {
        let normalized = normalize_asr_input_audio(true, &[f32::NAN, 0.5, f32::INFINITY, -1.0]);

        assert!(matches!(normalized, Cow::Owned(_)));
        assert_f32_close(normalized[0], 0.0, f32::EPSILON);
        assert!((normalized[1] - 0.475).abs() < 0.0001);
        assert_f32_close(normalized[2], 0.0, f32::EPSILON);
        assert!((normalized[3] + 0.95).abs() < 0.0001);
    }

    #[test]
    fn normalize_asr_input_audio_keeps_silent_or_non_finite_only_audio_unscaled() {
        for audio in [&[0.0, -0.0][..], &[f32::NAN, f32::INFINITY][..]] {
            let normalized = normalize_asr_input_audio(true, audio);

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
    fn prepare_nemotron_input_audio_aligns_to_160ms_and_adds_tail_adjustment() {
        let audio = vec![1.0; 1_600];
        let prepared = prepare_nemotron_input_audio(&audio, &vads(&[true]));

        assert_eq!(
            prepared.audio.len() % samples_for_millis(NEMOTRON_CHUNK_MS),
            0,
            "Nemotron ASR input must be aligned to the 160ms chunk grid"
        );
        assert!(
            prepared.leading_padding_samples >= samples_for_millis(NEMOTRON_EDGE_FADE_MS),
            "speech without pre-roll should receive at least the 80ms leading fade/padding window"
        );
        assert!(
            prepared.audio.len()
                >= prepared.leading_padding_samples
                    + audio.len()
                    + samples_for_millis(NEMOTRON_EDGE_FADE_MS),
            "Nemotron input must include the 80ms tail adjustment window"
        );
    }

    #[test]
    fn prepare_nemotron_input_audio_reuses_pre_speech_audio_before_inserting_silence() {
        let audio = vec![1.0; 5_120];
        let prepared =
            prepare_nemotron_input_audio(&audio, &vads(&[false, false, true, true, true]));

        assert_eq!(
            prepared.leading_padding_samples, 0,
            "available non-speech pre-roll should be preferred over inserting synthetic silence"
        );
        assert_eq!(
            prepared.audio.len() % samples_for_millis(NEMOTRON_CHUNK_MS),
            0
        );
    }

    #[test]
    fn prepare_nemotron_streaming_bootstrap_audio_does_not_append_tail_adjustment() {
        let audio = vec![1.0; samples_for_millis(NEMOTRON_CHUNK_MS)];
        let prepared = prepare_nemotron_streaming_bootstrap_audio(&audio, &vads(&[true]));

        assert_eq!(
            prepared.audio.len(),
            samples_for_millis(NEMOTRON_CHUNK_MS) * 2,
            "streaming bootstrap should contain the leading fade/alignment window and the real audio, without a synthetic tail window"
        );
        assert_f32_close(
            *prepared
                .audio
                .last()
                .expect("streaming bootstrap audio should not be empty"),
            1.0,
            f32::EPSILON,
        );
    }

    #[test]
    fn prepare_nemotron_input_audio_aligns_start_padding_to_vad_speech_not_tail_silence() {
        let audio = vec![1.0; 5_120];
        let prepared =
            prepare_nemotron_input_audio(&audio, &vads(&[false, true, true, false, false]));
        let vad_chunk_samples = audio.len() / 5;
        let speech_start = vad_chunk_samples;
        let speech_len = vad_chunk_samples * 2;

        assert_eq!(
            (prepared.leading_padding_samples + speech_start + speech_len)
                % samples_for_millis(NEMOTRON_CHUNK_MS),
            0,
            "start-side fade and adjustment must align with the VAD speech interval, not trailing silence"
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
