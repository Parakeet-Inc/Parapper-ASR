use std::sync::mpsc::SyncSender;

use anyhow::{Context, Result};
use cpal::{Device, Sample, SampleFormat, SizedSample, Stream, StreamConfig, traits::DeviceTrait};
use dasp_sample::ToSample;

#[derive(Debug)]
pub(crate) struct InputChunk {
    pub samples: Vec<f32>,
}

pub(crate) fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: SyncSender<InputChunk>,
) -> Result<Stream> {
    crate::dispatch_cpal_sample_format!(
        sample_format,
        build_input_stream_inner,
        device,
        config,
        sender;
        unsupported => anyhow::bail!("Unsupported input sample format: {sample_format:?}")
    )
}

fn build_input_stream_inner<T>(
    device: &Device,
    config: &StreamConfig,
    sender: SyncSender<InputChunk>,
) -> Result<Stream>
where
    T: Sample + SizedSample + ToSample<f32>,
{
    let channels = usize::from(config.channels);
    let err_fn = |err| log::warn!("Audio input stream error: {err}");
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if channels == 0 || data.is_empty() {
                    return;
                }

                let samples = interleaved_to_mono(data, channels);
                let chunk = InputChunk { samples };
                let _ = sender.try_send(chunk);
            },
            err_fn,
            None,
        )
        .context("Failed to build input stream")
}

#[expect(clippy::cast_precision_loss)]
fn interleaved_to_mono<T>(data: &[T], channels: usize) -> Vec<f32>
where
    T: Sample + ToSample<f32>,
{
    data.chunks(channels)
        .map(|frame| {
            let sum = frame
                .iter()
                .fold(0.0_f32, |acc, sample| acc + sample.to_sample());
            sum / frame.len() as f32
        })
        .collect()
}

pub(crate) fn peak_level(samples: &[f32]) -> f32 {
    samples
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()))
}
