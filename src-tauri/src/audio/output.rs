use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use anyhow::{Context, Result, bail};
use cpal::{
    FromSample, Sample, SizedSample,
    traits::{DeviceTrait, StreamTrait},
};

const MIN_TTS_GAIN: f32 = 0.1;
const MAX_TTS_GAIN: f32 = 10.0;

pub fn play_mono_samples(
    samples: &[f32],
    sample_rate: i32,
    volume: f32,
    output_device_host: Option<&str>,
    output_device_id: Option<&str>,
) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }
    let sample_rate = u32::try_from(sample_rate).context("TTS sample rate must be positive")?;
    let device =
        super::device::selected_output_device_by_id(output_device_host, output_device_id)?.device;
    let supported_config = device.default_output_config()?;
    let output_sample_rate = supported_config.sample_rate();
    let channels = usize::from(supported_config.channels()).max(1);
    let stream_config = supported_config.config();
    let prepared = Arc::new(scale_samples(
        &resample_linear(samples, sample_rate, output_sample_rate),
        volume,
    ));
    let cursor = Arc::new(AtomicUsize::new(0));
    let (done_sender, done_receiver) = mpsc::sync_channel::<()>(1);

    let stream = crate::dispatch_cpal_sample_format!(
        supported_config.sample_format(),
        build_output_stream,
        &device,
        &stream_config,
        prepared.clone(),
        cursor.clone(),
        channels,
        done_sender;
        unsupported => bail!("unsupported output sample format: {:?}", supported_config.sample_format()),
    )?;
    stream.play().context("Failed to start TTS output stream")?;

    let estimated_millis =
        (prepared.len() as u128 * 1_000).div_ceil(u128::from(output_sample_rate));
    let timeout_millis = estimated_millis
        .saturating_add(5_000)
        .min(u128::from(u64::MAX));
    let timeout = Duration::from_millis(u64::try_from(timeout_millis).unwrap_or(u64::MAX));
    done_receiver
        .recv_timeout(timeout)
        .context("Timed out while playing TTS audio")?;
    Ok(())
}

fn build_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
    channels: usize,
    done_sender: mpsc::SyncSender<()>,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    let mut sent_done = false;
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                for frame in data.chunks_mut(channels) {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let sample = samples.get(index).copied().unwrap_or_default();
                    for output in frame {
                        *output = T::from_sample(sample);
                    }
                }
                if !sent_done && cursor.load(Ordering::Relaxed) >= samples.len() {
                    sent_done = true;
                    let _ = done_sender.try_send(());
                }
            },
            |err| log::warn!("TTS output stream error: {err}"),
            None,
        )
        .context("Failed to build TTS output stream")
}

fn scale_samples(samples: &[f32], volume: f32) -> Vec<f32> {
    let volume = if volume.is_finite() {
        volume.clamp(MIN_TTS_GAIN, MAX_TTS_GAIN)
    } else {
        1.0
    };
    samples
        .iter()
        .map(|sample| (sample * volume).clamp(-1.0, 1.0))
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == output_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(input_rate) / f64::from(output_rate);
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let source_pos = index as f64 * ratio;
        let left_index = source_pos.floor() as usize;
        let right_index = (left_index + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_index as f64) as f32;
        let left = samples[left_index];
        let right = samples[right_index];
        output.push(left + (right - left) * fraction);
    }
    output
}
