use std::sync::mpsc::Sender;

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
    sender: Sender<InputChunk>,
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
    sender: Sender<InputChunk>,
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
                enqueue_input_chunk(&sender, chunk);
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

fn enqueue_input_chunk(sender: &Sender<InputChunk>, chunk: InputChunk) {
    // Receiver drop is the shutdown signal; there is nothing useful for the realtime
    // callback to do once the recognition worker has gone away.
    let _ = sender.send(chunk);
}

pub(crate) fn peak_level(samples: &[f32]) -> f32 {
    samples
        .iter()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::{InputChunk, enqueue_input_chunk};

    #[test]
    fn input_queue_keeps_all_chunks_in_fifo_order_when_producer_gets_ahead() {
        let (sender, receiver) = mpsc::channel();

        for sample in 0_u16..32 {
            enqueue_input_chunk(
                &sender,
                InputChunk {
                    samples: vec![f32::from(sample)],
                },
            );
        }
        drop(sender);

        let captured_chunks = receiver
            .iter()
            .map(|chunk| chunk.samples[0].to_bits())
            .collect::<Vec<_>>();
        let expected = (0_u16..32)
            .map(|sample| f32::from(sample).to_bits())
            .collect::<Vec<_>>();

        assert_eq!(captured_chunks, expected);
    }
}
