// Streaming NeMo extraction intentionally performs bounded index and f32 DSP conversions.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use realfft::{RealFftPlanner, RealToComplex};

use crate::{AsrSpeechRangeSamples, AsrStreamConfig, SAMPLE_RATE_HZ};

const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 400;
const WINDOW_STRIDE: usize = 160;
const FIRST_FRAME_OFFSET: isize = -120;
const PREEMPHASIS: f32 = 0.97;
const MEL_BINS: usize = 128;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const SLANEY_LOG_STEP: f64 = 0.068_751_777_420_949_12;

/// Owns Nemotron-specific stream bootstrap and tail completion around the
/// generic PCM delta stream exposed to hosts.
pub struct NemoStreamingAdapter {
    frontend: NemoStreamingFrontend,
    chunk_samples: usize,
    fade_samples: usize,
    stream_config: AsrStreamConfig,
    leading_padding_samples: usize,
    received_pcm: bool,
}

impl NemoStreamingAdapter {
    #[must_use]
    pub fn new(
        window_frames: usize,
        shift_frames: usize,
        chunk_samples: usize,
        fade_samples: usize,
    ) -> Self {
        Self {
            frontend: NemoStreamingFrontend::new(window_frames, shift_frames),
            chunk_samples,
            fade_samples,
            stream_config: AsrStreamConfig::default(),
            leading_padding_samples: 0,
            received_pcm: false,
        }
    }

    pub fn start(&mut self, config: AsrStreamConfig) {
        self.stream_config = config;
    }

    /// Adds one unmodified source PCM delta to this adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when the frontend cannot extract a model window.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<NemoStreamingWindow>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        if self.received_pcm {
            return self.frontend.push(samples);
        }
        self.received_pcm = true;
        let Some(speech_range) = self.stream_config.speech_range_samples else {
            return self.frontend.push(samples);
        };
        let bootstrap = self.bootstrap_audio(samples, speech_range);
        self.frontend.push(&bootstrap)
    }

    /// Completes the model-native tail exactly once for an active PCM stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the frontend cannot extract the final window.
    pub fn finish(&mut self) -> Result<Vec<NemoStreamingWindow>> {
        self.frontend.finish(self.received_pcm)
    }

    #[must_use]
    pub const fn leading_padding_samples(&self) -> usize {
        self.leading_padding_samples
    }

    fn bootstrap_audio(
        &mut self,
        samples: &[f32],
        speech_range: AsrSpeechRangeSamples,
    ) -> Vec<f32> {
        let speech_start = speech_range.start.min(samples.len());
        let speech_end = speech_range.end.clamp(speech_start, samples.len());
        let speech_len = speech_end.saturating_sub(speech_start);
        let fade_samples = self.fade_samples.min(samples.len());
        let required_prefix = fade_samples
            + alignment_padding_samples(
                fade_samples.saturating_add(speech_len),
                self.chunk_samples,
            );
        let (copy_start, leading_padding_samples) = if speech_start >= required_prefix {
            (speech_start - required_prefix, 0)
        } else {
            (0, required_prefix - speech_start)
        };
        self.leading_padding_samples = leading_padding_samples;

        let mut output = Vec::with_capacity(
            leading_padding_samples + samples.len().saturating_sub(copy_start) + self.chunk_samples,
        );
        output.resize(leading_padding_samples, 0.0);
        let copied_start = output.len();
        output.extend_from_slice(&samples[copy_start..]);
        apply_fade_in(&mut output[copied_start..], fade_samples);
        let alignment = alignment_padding_samples(output.len(), self.chunk_samples);
        output.resize(output.len() + alignment, 0.0);
        output
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NemoStreamingWindow {
    /// Contiguous `[1, 128, window_frames]` values.
    pub values: Vec<f32>,
    pub frames: usize,
}

pub struct NemoStreamingFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    samples: Vec<f32>,
    features: Vec<f32>,
    feature_frames: usize,
    processed_frames: usize,
    window_frames: usize,
    shift_frames: usize,
}

impl std::fmt::Debug for NemoStreamingFrontend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NemoStreamingFrontend")
            .field("feature_frames", &self.feature_frames)
            .field("processed_frames", &self.processed_frames)
            .field("window_frames", &self.window_frames)
            .field("shift_frames", &self.shift_frames)
            .finish_non_exhaustive()
    }
}

impl NemoStreamingFrontend {
    #[must_use]
    /// Creates a streaming frontend with an explicit window and shift.
    ///
    /// # Panics
    ///
    /// Panics unless `window_frames > shift_frames > 0`.
    pub fn new(window_frames: usize, shift_frames: usize) -> Self {
        assert!(window_frames > shift_frames && shift_frames > 0);
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = (0..WINDOW_SIZE)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * index as f32 / (WINDOW_SIZE - 1) as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            fft,
            window,
            mel_filters: librosa_slaney_mel_filters(),
            samples: Vec::new(),
            features: Vec::new(),
            feature_frames: 0,
            processed_frames: 0,
            window_frames,
            shift_frames,
        }
    }

    /// Adds audio and returns every newly complete inference window.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite input or an FFT processing failure.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<NemoStreamingWindow>> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("Nemotron input contains a non-finite sample");
        }
        self.samples.extend_from_slice(samples);
        self.extend_complete_features()?;
        let mut windows = Vec::new();
        // This strict inequality matches the online feature readiness contract
        // used by the exported Nemotron models.
        while self.processed_frames + self.window_frames < self.feature_frames {
            let mut values = vec![0.0; MEL_BINS * self.window_frames];
            for feature in 0..MEL_BINS {
                for time in 0..self.window_frames {
                    values[feature * self.window_frames + time] =
                        self.features[(self.processed_frames + time) * MEL_BINS + feature];
                }
            }
            windows.push(NemoStreamingWindow {
                values,
                frames: self.window_frames,
            });
            self.processed_frames += self.shift_frames;
        }
        Ok(windows)
    }

    fn finish(&mut self, received_pcm: bool) -> Result<Vec<NemoStreamingWindow>> {
        if !received_pcm {
            return Ok(Vec::new());
        }
        let required_feature_frames = self
            .processed_frames
            .saturating_add(self.window_frames)
            .saturating_add(1);
        let required_samples = 280_usize.saturating_add(
            required_feature_frames
                .saturating_sub(1)
                .saturating_mul(WINDOW_STRIDE),
        );
        if required_samples <= self.samples.len() {
            return Ok(Vec::new());
        }
        self.push(&vec![0.0; required_samples - self.samples.len()])
    }

    fn extend_complete_features(&mut self) -> Result<()> {
        let complete_frames = if self.samples.len() < 280 {
            0
        } else {
            1 + (self.samples.len() - 280) / WINDOW_STRIDE
        };
        let mut fft_input = self.fft.make_input_vec();
        let mut fft_output = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        while self.feature_frames < complete_frames {
            let frame = self.feature_frames;
            let frame_start = FIRST_FRAME_OFFSET + (frame * WINDOW_STRIDE) as isize;
            let mut raw_window = [0.0_f32; WINDOW_SIZE];
            for (index, value) in raw_window.iter_mut().enumerate() {
                let source = reflect_index(frame_start + index as isize, self.samples.len());
                *value = self.samples[source];
            }
            // Kaldi/NeMo streaming feature extraction applies preemphasis to
            // the reflected frame, including x[0] *= 1 - coefficient.
            for index in (1..WINDOW_SIZE).rev() {
                raw_window[index] -= PREEMPHASIS * raw_window[index - 1];
            }
            raw_window[0] *= 1.0 - PREEMPHASIS;

            fft_input.fill(0.0);
            for index in 0..WINDOW_SIZE {
                fft_input[index] = raw_window[index] * self.window[index];
            }
            self.fft
                .process_with_scratch(&mut fft_input, &mut fft_output, &mut scratch)
                .map_err(|error| anyhow!("Nemotron FFT failed: {error}"))?;
            for mel in 0..MEL_BINS {
                let filter =
                    &self.mel_filters[mel * fft_output.len()..(mel + 1) * fft_output.len()];
                let energy = fft_output
                    .iter()
                    .zip(filter)
                    .map(|(value, &weight)| value.norm_sqr() * weight)
                    .sum::<f32>();
                self.features.push((energy + LOG_GUARD).ln());
            }
            self.feature_frames += 1;
        }
        Ok(())
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

fn apply_fade_in(audio: &mut [f32], fade_samples: usize) {
    let fade_samples = fade_samples.min(audio.len());
    if fade_samples == 0 {
        return;
    }
    for (index, sample) in audio.iter_mut().take(fade_samples).enumerate() {
        *sample *= index as f32 / fade_samples as f32;
    }
}

fn reflect_index(mut index: isize, sample_count: usize) -> usize {
    debug_assert!(sample_count > 0);
    let sample_count = sample_count as isize;
    while index < 0 || index >= sample_count {
        index = if index < 0 {
            -index - 1
        } else {
            2 * sample_count - 1 - index
        };
    }
    index as usize
}

fn librosa_slaney_mel_filters() -> Vec<f32> {
    let frequencies = (0..=FFT_SIZE / 2)
        .map(|index| index as f64 * f64::from(SAMPLE_RATE_HZ) / FFT_SIZE as f64)
        .collect::<Vec<_>>();
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(f64::from(SAMPLE_RATE_HZ) / 2.0);
    let mel_frequencies = (0..MEL_BINS + 2)
        .map(|index| {
            let ratio = index as f64 / (MEL_BINS + 1) as f64;
            mel_to_hz(min_mel + ratio * (max_mel - min_mel))
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0; MEL_BINS * frequencies.len()];
    for mel in 0..MEL_BINS {
        let lower = mel_frequencies[mel];
        let center = mel_frequencies[mel + 1];
        let upper = mel_frequencies[mel + 2];
        let norm = 2.0 / (upper - lower);
        for (fft, &frequency) in frequencies.iter().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            filters[mel * frequencies.len() + fft] =
                (lower_slope.min(upper_slope).max(0.0) * norm) as f32;
        }
    }
    filters
}

fn hz_to_mel(frequency: f64) -> f64 {
    const SPACING: f64 = 200.0 / 3.0;
    const LOG_HZ: f64 = 1_000.0;
    const LOG_MEL: f64 = LOG_HZ / SPACING;
    if frequency >= LOG_HZ {
        LOG_MEL + (frequency / LOG_HZ).ln() / SLANEY_LOG_STEP
    } else {
        frequency / SPACING
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const SPACING: f64 = 200.0 / 3.0;
    const LOG_HZ: f64 = 1_000.0;
    const LOG_MEL: f64 = LOG_HZ / SPACING;
    if mel >= LOG_MEL {
        LOG_HZ * (SLANEY_LOG_STEP * (mel - LOG_MEL)).exp()
    } else {
        SPACING * mel
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsrSpeechRangeSamples, AsrStreamConfig};

    use super::{NemoStreamingAdapter, NemoStreamingFrontend};

    #[test]
    fn streaming_frontend_emits_only_complete_strictly_ready_windows() {
        let mut frontend = NemoStreamingFrontend::new(25, 16);
        assert!(frontend.push(&vec![0.0; 4_000]).unwrap().is_empty());
        let windows = frontend.push(&vec![0.0; 1_120]).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].values.len(), 128 * 25);
        assert!(frontend.push(&[]).unwrap().is_empty());
    }

    #[test]
    fn streaming_adapter_bootstrap_matches_legacy_prepared_audio_including_trailing_silence() {
        let cases = [
            ("no_pre_roll", vec![1.0; 2_560], 0, 2_560),
            (
                "available_pre_roll",
                (0..5_120).map(|index| index as f32).collect(),
                2_560,
                5_120,
            ),
            (
                "trailing_silence_is_not_counted_as_speech",
                (0..5_120).map(|index| index as f32).collect(),
                1_024,
                3_072,
            ),
        ];

        for (name, audio, speech_start, speech_end) in cases {
            let mut adapter = NemoStreamingAdapter::new(25, 16, 2_560, 1_280);
            let range = AsrSpeechRangeSamples {
                start: speech_start,
                end: speech_end,
            };
            let actual = adapter.bootstrap_audio(&audio, range);
            let (expected, expected_leading_padding) =
                legacy_streaming_bootstrap_audio(&audio, speech_start, speech_end, 2_560, 1_280);

            assert_eq!(
                (actual, adapter.leading_padding_samples()),
                (expected, expected_leading_padding),
                "{name} must keep the pre-adapter host contract byte-for-byte"
            );
        }
    }

    #[test]
    fn streaming_adapter_owns_native_chunk_and_finish_tail_boundaries() {
        let chunk = 2_560;
        let cases = [
            ("zero", 0, 0, 0, 0),
            ("one_sample_short", chunk - 1, chunk - 1, 1, 1),
            ("exact_chunk", chunk, chunk, 1, 1),
            ("multiple_chunks", chunk * 2, chunk * 2, 2, 1),
            ("speech_with_tail", chunk + 333, chunk, 2, 1),
        ];

        for (name, len, speech_end, expected_push_windows, expected_finish_windows) in cases {
            let mut adapter = NemoStreamingAdapter::new(25, 16, chunk, 1_280);
            adapter.start(AsrStreamConfig {
                speech_range_samples: (len > 0).then_some(AsrSpeechRangeSamples {
                    start: 0,
                    end: speech_end,
                }),
                language_hint: None,
            });
            let pushed = adapter.push(&vec![1.0; len]).unwrap();
            let finished = adapter.finish().unwrap();

            assert_eq!(pushed.len(), expected_push_windows, "{name}");
            assert_eq!(finished.len(), expected_finish_windows, "{name}");
        }
    }

    #[test]
    fn separate_stream_adapters_do_not_mix_source_tail_state() {
        let config = AsrStreamConfig {
            speech_range_samples: Some(AsrSpeechRangeSamples {
                start: 0,
                end: 2_560,
            }),
            language_hint: None,
        };
        let mut source_a = NemoStreamingAdapter::new(25, 16, 2_560, 1_280);
        let mut source_b = NemoStreamingAdapter::new(25, 16, 2_560, 1_280);
        let mut source_b_alone = NemoStreamingAdapter::new(25, 16, 2_560, 1_280);
        source_a.start(config);
        source_b.start(config);
        source_b_alone.start(config);

        source_a.push(&vec![1.0; 2_560]).unwrap();
        let source_b_push = source_b.push(&vec![2.0; 2_560]).unwrap();
        let source_b_finish = source_b.finish().unwrap();

        assert_eq!(
            (source_b_push, source_b_finish),
            (
                source_b_alone.push(&vec![2.0; 2_560]).unwrap(),
                source_b_alone.finish().unwrap(),
            ),
            "Source A's unflushed native tail must not affect Source B"
        );
    }

    fn legacy_streaming_bootstrap_audio(
        audio: &[f32],
        speech_start: usize,
        speech_end: usize,
        chunk_samples: usize,
        requested_fade_samples: usize,
    ) -> (Vec<f32>, usize) {
        let speech_start = speech_start.min(audio.len());
        let speech_end = speech_end.clamp(speech_start, audio.len());
        let fade_samples = requested_fade_samples.min(audio.len());
        let speech_len = speech_end - speech_start;
        let prefix_remainder = (fade_samples + speech_len) % chunk_samples;
        let required_prefix = fade_samples
            + if prefix_remainder == 0 {
                0
            } else {
                chunk_samples - prefix_remainder
            };
        let (copy_start, leading_padding) = if speech_start >= required_prefix {
            (speech_start - required_prefix, 0)
        } else {
            (0, required_prefix - speech_start)
        };
        let mut output = vec![0.0; leading_padding];
        let copied_start = output.len();
        output.extend_from_slice(&audio[copy_start..]);
        let actual_fade = fade_samples.min(output.len() - copied_start);
        for index in 0..actual_fade {
            output[copied_start + index] *= index as f32 / actual_fade as f32;
        }
        let end_remainder = output.len() % chunk_samples;
        if end_remainder != 0 {
            output.resize(output.len() + chunk_samples - end_remainder, 0.0);
        }
        (output, leading_padding)
    }
}
