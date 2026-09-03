// NeMo feature extraction intentionally performs bounded index and f32 DSP conversions.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use realfft::{RealFftPlanner, RealToComplex};

use crate::SAMPLE_RATE_HZ;

const FFT_SIZE: usize = 512;
const WINDOW_SIZE: usize = 400;
const WINDOW_STRIDE: usize = 160;
const WINDOW_OFFSET: usize = (FFT_SIZE - WINDOW_SIZE) / 2;
const CENTER_PADDING: isize = (FFT_SIZE / 2) as isize;
const PREEMPHASIS: f32 = 0.97;
const CTC_NUM_MEL_BINS: usize = 80;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const SLANEY_LOG_STEP: f64 = 0.068_751_777_420_949_12;
const NORMALIZE_EPSILON: f32 = 1.0e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct NemoFeatures {
    /// Contiguous `[1, mel_bins, frames]` data, with time as the innermost axis.
    pub values: Vec<f32>,
    pub mel_bins: usize,
    pub frames: usize,
    pub valid_frames: usize,
}

pub struct NemoMelFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    num_mel_bins: usize,
}

impl std::fmt::Debug for NemoMelFrontend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NemoMelFrontend")
            .field("fft_size", &FFT_SIZE)
            .field("window_size", &WINDOW_SIZE)
            .field("num_mel_bins", &self.num_mel_bins)
            .finish_non_exhaustive()
    }
}

impl Default for NemoMelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl NemoMelFrontend {
    #[must_use]
    pub fn new() -> Self {
        Self::with_mel_bins(CTC_NUM_MEL_BINS)
    }

    #[must_use]
    /// Creates a frontend with a custom positive mel-bin count.
    ///
    /// # Panics
    ///
    /// Panics when `num_mel_bins` is zero.
    pub fn with_mel_bins(num_mel_bins: usize) -> Self {
        assert!(num_mel_bins > 0, "NeMo mel bin count must be positive");
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
            mel_filters: librosa_slaney_mel_filters(num_mel_bins),
            num_mel_bins,
        }
    }

    /// Reproduces the pinned `NeMo` `AudioToMelSpectrogramPreprocessor` contract:
    /// 16 kHz, 25 ms Hann, 10 ms stride, FFT 512, 80 Slaney-normalized
    /// librosa mel bins, power 2, natural log, and per-feature sample stddev.
    ///
    /// # Errors
    ///
    /// Returns an error for insufficient or non-finite audio, an invalid feature shape, or an FFT
    /// processing failure.
    pub fn process(&self, samples: &[f32]) -> Result<NemoFeatures> {
        if samples.len() < WINDOW_STRIDE {
            bail!(
                "NeMo CTC requires at least {WINDOW_STRIDE} samples, received {}",
                samples.len()
            );
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("NeMo CTC input contains a non-finite sample");
        }

        let emphasized = preemphasize(samples);
        let valid_frames = samples.len() / WINDOW_STRIDE;
        // torch.stft(center=true) returns one extra frame. NeMo masks that
        // frame and passes `valid_frames` as the model length.
        let frames = valid_frames + 1;
        let mut time_major = self.log_mel(&emphasized, frames)?;
        normalize_like_pinned_nemo(&mut time_major, self.num_mel_bins, frames, valid_frames)?;
        let mut values = vec![0.0; self.num_mel_bins * frames];
        for frame in 0..frames {
            for feature in 0..self.num_mel_bins {
                values[feature * frames + frame] = time_major[frame * self.num_mel_bins + feature];
            }
        }
        Ok(NemoFeatures {
            values,
            mel_bins: self.num_mel_bins,
            frames,
            valid_frames,
        })
    }

    fn log_mel(&self, emphasized: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut time_major = vec![0.0; frames * self.num_mel_bins];
        let mut fft_input = self.fft.make_input_vec();
        let mut fft_output = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();

        for frame_index in 0..frames {
            fft_input.fill(0.0);
            let frame_start = (frame_index * WINDOW_STRIDE) as isize - CENTER_PADDING;
            for window_index in 0..WINDOW_SIZE {
                let fft_index = WINDOW_OFFSET + window_index;
                let source_index = frame_start + fft_index as isize;
                if let Ok(source_index) = usize::try_from(source_index)
                    && let Some(&sample) = emphasized.get(source_index)
                {
                    fft_input[fft_index] = sample * self.window[window_index];
                }
            }
            self.fft
                .process_with_scratch(&mut fft_input, &mut fft_output, &mut scratch)
                .map_err(|error| anyhow!("NeMo FFT failed: {error}"))?;

            for mel_index in 0..self.num_mel_bins {
                let filter = &self.mel_filters
                    [mel_index * fft_output.len()..(mel_index + 1) * fft_output.len()];
                let energy = fft_output
                    .iter()
                    .zip(filter)
                    .map(|(value, &weight)| value.norm_sqr() * weight)
                    .sum::<f32>();
                time_major[frame_index * self.num_mel_bins + mel_index] = (energy + LOG_GUARD).ln();
            }
        }
        Ok(time_major)
    }
}

fn preemphasize(samples: &[f32]) -> Vec<f32> {
    let mut emphasized = Vec::with_capacity(samples.len());
    emphasized.push(samples[0]);
    emphasized.extend(
        samples
            .windows(2)
            .map(|pair| pair[1] - PREEMPHASIS * pair[0]),
    );
    emphasized
}

fn normalize_like_pinned_nemo(
    features: &mut [f32],
    num_mel_bins: usize,
    frames: usize,
    valid_frames: usize,
) -> Result<()> {
    if frames < valid_frames || features.len() != frames * num_mel_bins {
        bail!("invalid NeMo feature shape");
    }
    if valid_frames <= 1 {
        bail!("NeMo per-feature normalization requires at least two valid frames");
    }
    for feature in 0..num_mel_bins {
        let mean = (0..valid_frames)
            .map(|frame| features[frame * num_mel_bins + feature])
            .sum::<f32>()
            / valid_frames as f32;
        // The pinned NVIDIA implementation divides by N-1. sherpa-onnx
        // v1.13.3 C++ divides by N, which is a known compatibility delta.
        let variance = (0..valid_frames)
            .map(|frame| {
                let centered = features[frame * num_mel_bins + feature] - mean;
                centered * centered
            })
            .sum::<f32>()
            / (valid_frames - 1) as f32;
        let standard_deviation = variance.sqrt() + NORMALIZE_EPSILON;
        for frame in 0..valid_frames {
            let index = frame * num_mel_bins + feature;
            features[index] = (features[index] - mean) / standard_deviation;
        }
        for frame in valid_frames..frames {
            features[frame * num_mel_bins + feature] = 0.0;
        }
    }
    Ok(())
}

fn librosa_slaney_mel_filters(num_mel_bins: usize) -> Vec<f32> {
    let fft_frequencies = (0..=FFT_SIZE / 2)
        .map(|index| index as f64 * f64::from(SAMPLE_RATE_HZ) / FFT_SIZE as f64)
        .collect::<Vec<_>>();
    let min_mel = hz_to_slaney_mel(0.0);
    let max_mel = hz_to_slaney_mel(f64::from(SAMPLE_RATE_HZ) / 2.0);
    let mel_frequencies = (0..num_mel_bins + 2)
        .map(|index| {
            let ratio = index as f64 / (num_mel_bins + 1) as f64;
            slaney_mel_to_hz(min_mel + ratio * (max_mel - min_mel))
        })
        .collect::<Vec<_>>();

    let mut filters = vec![0.0; num_mel_bins * fft_frequencies.len()];
    for mel_index in 0..num_mel_bins {
        let lower = mel_frequencies[mel_index];
        let center = mel_frequencies[mel_index + 1];
        let upper = mel_frequencies[mel_index + 2];
        let slaney_norm = 2.0 / (upper - lower);
        for (fft_index, &frequency) in fft_frequencies.iter().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            filters[mel_index * fft_frequencies.len() + fft_index] =
                (lower_slope.min(upper_slope).max(0.0) * slaney_norm) as f32;
        }
    }
    filters
}

fn hz_to_slaney_mel(frequency: f64) -> f64 {
    const FREQ_SPACING: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / FREQ_SPACING;
    if frequency >= MIN_LOG_HZ {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    } else {
        frequency / FREQ_SPACING
    }
}

fn slaney_mel_to_hz(mel: f64) -> f64 {
    const FREQ_SPACING: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / FREQ_SPACING;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (SLANEY_LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        FREQ_SPACING * mel
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use serde_json::Value;

    use super::{CTC_NUM_MEL_BINS, NemoMelFrontend, normalize_like_pinned_nemo, preemphasize};

    #[test]
    fn per_feature_normalization_uses_nvidia_sample_stddev_not_sherpa_population_stddev() {
        let mut features = vec![0.0; 3 * CTC_NUM_MEL_BINS];
        for feature in 0..CTC_NUM_MEL_BINS {
            features[feature] = 1.0;
            features[CTC_NUM_MEL_BINS + feature] = 3.0;
            features[2 * CTC_NUM_MEL_BINS + feature] = 100.0;
        }

        normalize_like_pinned_nemo(&mut features, CTC_NUM_MEL_BINS, 3, 2).unwrap();

        let expected = 1.0 / (2.0_f32.sqrt() + 1.0e-5);
        assert!((features[0] + expected).abs() < 1.0e-6);
        assert!((features[CTC_NUM_MEL_BINS] - expected).abs() < 1.0e-6);
        assert_eq!(
            features[2 * CTC_NUM_MEL_BINS].to_bits(),
            0.0_f32.to_bits(),
            "masked frames must be exact positive zero"
        );
        assert!(
            (features[0] + 1.0).abs() > 0.25,
            "population standard deviation would reproduce sherpa's incompatible value"
        );
    }

    #[test]
    fn frontend_matches_the_pinned_nvidia_torch_librosa_tensor() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../diagnostics/nemo-reference/fixtures/nemo-ctc-frontend.json");
        let fixture: Value =
            serde_json::from_str(&fs::read_to_string(fixture_path).unwrap()).unwrap();
        let samples = fixture["input"]["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let frontend = NemoMelFrontend::new();
        let actual = frontend.process(&samples).unwrap();
        let expected_shape = fixture["output"]["shape"].as_array().unwrap();
        let expected = fixture["output"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let tolerance = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;

        assert_eq!(expected_shape[0].as_u64(), Some(1));
        assert_eq!(expected_shape[1].as_u64(), Some(CTC_NUM_MEL_BINS as u64));
        assert_eq!(expected_shape[2].as_u64(), Some(actual.frames as u64));
        assert_eq!(
            fixture["output"]["valid_frames"].as_u64(),
            Some(actual.valid_frames as u64)
        );
        assert_eq!(actual.values.len(), expected.len());
        let expected_logs = fixture["output"]["log_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let actual_logs = frontend
            .log_mel(&preemphasize(&samples), actual.frames)
            .unwrap();
        for feature in 0..CTC_NUM_MEL_BINS {
            for frame in 0..actual.frames {
                let rust_index = frame * CTC_NUM_MEL_BINS + feature;
                let python_index = feature * actual.frames + frame;
                assert!(
                    (actual_logs[rust_index] - expected_logs[python_index]).abs() <= tolerance,
                    "log_feature[{feature}, {frame}] differs: Rust={}, NVIDIA={}, tolerance={tolerance}",
                    actual_logs[rust_index],
                    expected_logs[python_index]
                );
            }
        }
        for (index, (&actual, &expected)) in actual.values.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "feature[{index}] differs: Rust={actual}, NVIDIA={expected}, tolerance={tolerance}"
            );
        }
    }
}
