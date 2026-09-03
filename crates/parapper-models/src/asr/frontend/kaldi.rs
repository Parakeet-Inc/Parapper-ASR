// Kaldi feature extraction intentionally performs bounded index and f32 DSP conversions.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use realfft::{RealFftPlanner, RealToComplex};

use crate::SAMPLE_RATE_HZ;

const FFT_SIZE: usize = 512;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FIRST_FRAME_OFFSET: isize = -120;
const MEL_BINS: usize = 80;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 7_600.0;
const PREEMPHASIS: f32 = 0.97;

#[derive(Debug, Clone, PartialEq)]
pub struct KaldiFeatures {
    /// Contiguous `[1, frames, 80]` values.
    pub values: Vec<f32>,
    pub frames: usize,
}

pub struct KaldiFbankFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
}

impl std::fmt::Debug for KaldiFbankFrontend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KaldiFbankFrontend")
            .field("mel_bins", &MEL_BINS)
            .finish_non_exhaustive()
    }
}

impl Default for KaldiFbankFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl KaldiFbankFrontend {
    #[must_use]
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = (0..FRAME_LENGTH)
            .map(|index| {
                // kaldi-native-fbank computes the Povey window in double and
                // stores it as float. Keep that rounding boundary exact.
                let phase = 2.0 * std::f64::consts::PI * index as f64 / (FRAME_LENGTH - 1) as f64;
                (0.5 - 0.5 * phase.cos()).powf(0.85) as f32
            })
            .collect();
        Self {
            fft,
            window,
            mel_filters: kaldi_mel_filters(),
        }
    }

    /// Computes Kaldi-compatible filter-bank features.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite input or an FFT processing failure.
    pub fn process(&self, samples: &[f32]) -> Result<KaldiFeatures> {
        if samples.is_empty() {
            return Ok(KaldiFeatures {
                values: Vec::new(),
                frames: 0,
            });
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("Kaldi fbank input contains a non-finite sample");
        }
        let frames = (samples.len() + FRAME_SHIFT / 2) / FRAME_SHIFT;
        let mut values = Vec::with_capacity(frames * MEL_BINS);
        let mut fft_input = self.fft.make_input_vec();
        let mut fft_output = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        for frame in 0..frames {
            let start = FIRST_FRAME_OFFSET + (frame * FRAME_SHIFT) as isize;
            let mut windowed = [0.0_f32; FRAME_LENGTH];
            for (index, value) in windowed.iter_mut().enumerate() {
                *value = samples[reflect_index(start + index as isize, samples.len())];
            }
            let mean = windowed.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for value in &mut windowed {
                *value -= mean;
            }
            for index in (1..FRAME_LENGTH).rev() {
                windowed[index] -= PREEMPHASIS * windowed[index - 1];
            }
            windowed[0] *= 1.0 - PREEMPHASIS;
            fft_input.fill(0.0);
            for index in 0..FRAME_LENGTH {
                fft_input[index] = windowed[index] * self.window[index];
            }
            self.fft
                .process_with_scratch(&mut fft_input, &mut fft_output, &mut scratch)
                .map_err(|error| anyhow!("Kaldi fbank FFT failed: {error}"))?;
            for mel in 0..MEL_BINS {
                let filter =
                    &self.mel_filters[mel * fft_output.len()..(mel + 1) * fft_output.len()];
                let energy = fft_output
                    .iter()
                    .zip(filter)
                    .map(|(value, &weight)| value.norm_sqr() * weight)
                    .sum::<f32>()
                    .max(f32::EPSILON);
                values.push(energy.ln());
            }
        }
        Ok(KaldiFeatures { values, frames })
    }
}

fn reflect_index(mut index: isize, sample_count: usize) -> usize {
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

fn kaldi_mel_filters() -> Vec<f32> {
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(HIGH_FREQ);
    let delta = (mel_high - mel_low) / (MEL_BINS + 1) as f32;
    let fft_bins = FFT_SIZE / 2 + 1;
    let mut filters = vec![0.0; MEL_BINS * fft_bins];
    for mel in 0..MEL_BINS {
        let left = mel_low + mel as f32 * delta;
        let center = left + delta;
        let right = center + delta;
        for fft in 0..fft_bins {
            let frequency = fft as f32 * SAMPLE_RATE_HZ as f32 / FFT_SIZE as f32;
            let value = mel_scale(frequency);
            let weight = if value > left && value < right {
                if value <= center {
                    (value - left) / (center - left)
                } else {
                    (right - value) / (right - center)
                }
            } else {
                0.0
            };
            filters[mel * fft_bins + fft] = weight;
        }
    }
    filters
}

fn mel_scale(frequency: f32) -> f32 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

#[cfg(test)]
mod tests {
    use super::KaldiFbankFrontend;

    #[test]
    fn kaldi_fbank_uses_nonsnipping_reflected_frame_count() {
        let features = KaldiFbankFrontend::new()
            .process(&vec![0.0; 16_000])
            .unwrap();
        assert_eq!(features.frames, 100);
        assert_eq!(features.values.len(), 8_000);
        assert!(features.values.iter().all(|value| value.is_finite()));
    }
}
