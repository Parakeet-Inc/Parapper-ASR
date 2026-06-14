use std::collections::VecDeque;

use anyhow::{Context, Result};
use rubato::{
    Async, FixedAsync, PolynomialDegree, Resampler,
    audioadapter::{Adapter, AdapterMut},
};

const DEFAULT_VAD_INTERVAL_MS: u32 = 32;

pub(crate) struct MonoFastFixedInResampler {
    inner: MonoFastFixedInResamplerInner,
    pending: VecDeque<f32>,
    input_buffer: Vec<f32>,
    input_chunk_size: usize,
}

enum MonoFastFixedInResamplerInner {
    Identity,
    FastFixedIn {
        resampler: Async<f32>,
        output: Vec<f32>,
    },
}

impl MonoFastFixedInResampler {
    pub(crate) fn new(
        source_sample_rate: u32,
        target_sample_rate: u32,
        chunk_millis: u32,
    ) -> Result<Self> {
        let input_chunk_size = frames_for_millis(source_sample_rate, chunk_millis);
        if source_sample_rate == target_sample_rate {
            return Ok(Self {
                inner: MonoFastFixedInResamplerInner::Identity,
                pending: VecDeque::new(),
                input_buffer: Vec::with_capacity(input_chunk_size),
                input_chunk_size,
            });
        }

        let resampler = Async::<f32>::new_poly(
            f64::from(target_sample_rate) / f64::from(source_sample_rate),
            1.0,
            PolynomialDegree::Cubic,
            input_chunk_size,
            1,
            FixedAsync::Input,
        )
        .context("Failed to create FastFixedIn resampler")?;
        let output = vec![0.0; resampler.output_frames_max()];

        Ok(Self {
            inner: MonoFastFixedInResamplerInner::FastFixedIn { resampler, output },
            pending: VecDeque::new(),
            input_buffer: Vec::with_capacity(input_chunk_size),
            input_chunk_size,
        })
    }

    pub(crate) fn push_into(&mut self, samples: &[f32], chunks: &mut Vec<Vec<f32>>) -> Result<()> {
        self.pending.extend(samples.iter().copied());
        chunks.clear();

        while self.pending.len() >= self.input_chunk_size {
            self.input_buffer.clear();
            self.input_buffer
                .extend(self.pending.drain(..self.input_chunk_size));
            match &mut self.inner {
                MonoFastFixedInResamplerInner::Identity => chunks.push(self.input_buffer.clone()),
                MonoFastFixedInResamplerInner::FastFixedIn { resampler, output } => {
                    let input_adapter = SingleChannelInputAdapter::new(&self.input_buffer);
                    let mut output_adapter = SingleChannelOutputAdapter::new(output);
                    let (_, written) =
                        resampler.process_into_buffer(&input_adapter, &mut output_adapter, None)?;
                    chunks.push(output[..written].to_vec());
                }
            }
        }

        Ok(())
    }
}

pub(crate) fn validated_vad_interval_ms(value: u32) -> u32 {
    match value {
        32 => value,
        _ => DEFAULT_VAD_INTERVAL_MS,
    }
}

fn frames_for_millis(sample_rate: u32, millis: u32) -> usize {
    ((u64::from(sample_rate) * u64::from(millis)) / 1000)
        .try_into()
        .unwrap_or(1)
}

struct SingleChannelInputAdapter<'a> {
    data: &'a [f32],
}

impl<'a> SingleChannelInputAdapter<'a> {
    fn new(data: &'a [f32]) -> Self {
        Self { data }
    }
}

impl<'a> Adapter<'a, f32> for SingleChannelInputAdapter<'a> {
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> f32 {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one channel.
        unsafe { *self.data.get_unchecked(frame) }
    }

    fn channels(&self) -> usize {
        1
    }

    fn frames(&self) -> usize {
        self.data.len()
    }
}

struct SingleChannelOutputAdapter<'a> {
    data: &'a mut [f32],
}

impl<'a> SingleChannelOutputAdapter<'a> {
    fn new(data: &'a mut [f32]) -> Self {
        Self { data }
    }
}

impl<'a> Adapter<'a, f32> for SingleChannelOutputAdapter<'a> {
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> f32 {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one channel.
        unsafe { *self.data.get_unchecked(frame) }
    }

    fn channels(&self) -> usize {
        1
    }

    fn frames(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AdapterMut<'a, f32> for SingleChannelOutputAdapter<'a> {
    unsafe fn write_sample_unchecked(&mut self, channel: usize, frame: usize, value: &f32) -> bool {
        debug_assert_eq!(channel, 0);
        // SAFETY: rubato calls this adapter with frame < self.frames() and the
        // adapter exposes exactly one mutable output channel.
        unsafe {
            *self.data.get_unchecked_mut(frame) = *value;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{MonoFastFixedInResampler, frames_for_millis};
    use crate::audio::ASR_SAMPLE_RATE;

    #[test]
    fn frames_for_millis_returns_expected_count() {
        assert_eq!(frames_for_millis(48_000, 100), 4_800);
        assert_eq!(frames_for_millis(16_000, 100), 1_600);
    }

    #[test]
    fn identity_resampler_outputs_100ms_chunks() {
        let mut resampler =
            MonoFastFixedInResampler::new(ASR_SAMPLE_RATE, ASR_SAMPLE_RATE, 100).unwrap();
        let mut chunks = Vec::new();
        resampler.push_into(&vec![0.0; 1_600], &mut chunks).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1_600);
    }

    #[test]
    fn fast_fixed_in_resampler_converts_48khz_to_16khz() {
        let mut resampler = MonoFastFixedInResampler::new(48_000, ASR_SAMPLE_RATE, 100).unwrap();
        let mut chunks = Vec::new();
        resampler.push_into(&vec![0.0; 4_800], &mut chunks).unwrap();

        assert_eq!(chunks.len(), 1);
        assert!((1_590..=1_610).contains(&chunks[0].len()));
    }
}
