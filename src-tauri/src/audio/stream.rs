use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc::Sender,
};
use std::{error::Error, fmt};

use anyhow::{Context, Result};
use cpal::{Device, Sample, SampleFormat, SizedSample, Stream, StreamConfig, traits::DeviceTrait};
use dasp_sample::ToSample;

/// Monotonic sequence assigned by a capture endpoint to one callback payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CaptureSequence(pub u64);

/// PCM frames from one capture callback before any channel selection or mixing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InterleavedPcmChunk {
    pub(crate) capture_sequence: CaptureSequence,
    pub(crate) channel_count: u16,
    pub(crate) samples: Vec<f32>,
}

impl InterleavedPcmChunk {
    #[must_use]
    pub(crate) fn new(
        capture_sequence: CaptureSequence,
        channel_count: u16,
        samples: Vec<f32>,
    ) -> Self {
        Self {
            capture_sequence,
            channel_count,
            samples,
        }
    }
}

/// Mono PCM for one explicitly selected physical channel from a capture chunk.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChannelMonoChunk {
    pub(crate) capture_sequence: CaptureSequence,
    pub(crate) channel_index: u16,
    pub(crate) samples: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelDemuxError {
    ZeroChannelCount,
    ChannelIndexOutOfRange {
        channel_index: u16,
        channel_count: u16,
    },
    IncompleteInterleavedFrame {
        samples_len: usize,
        channel_count: u16,
    },
}

impl fmt::Display for ChannelDemuxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroChannelCount => formatter.write_str("interleaved capture has zero channels"),
            Self::ChannelIndexOutOfRange {
                channel_index,
                channel_count,
            } => write!(
                formatter,
                "selected channel {channel_index} is outside capture layout with {channel_count} channels"
            ),
            Self::IncompleteInterleavedFrame {
                samples_len,
                channel_count,
            } => write!(
                formatter,
                "interleaved capture has {samples_len} samples, not a whole number of {channel_count}-channel frames"
            ),
        }
    }
}

impl Error for ChannelDemuxError {}

/// Separates only the requested channels without changing their frame order.
///
/// The legacy all-channel average-mono path deliberately does not use this function yet.
pub(crate) fn demux_selected_channels(
    chunk: &InterleavedPcmChunk,
    selected_channels: &[u16],
) -> std::result::Result<Vec<ChannelMonoChunk>, ChannelDemuxError> {
    if chunk.channel_count == 0 {
        return Err(ChannelDemuxError::ZeroChannelCount);
    }
    if !chunk
        .samples
        .len()
        .is_multiple_of(usize::from(chunk.channel_count))
    {
        return Err(ChannelDemuxError::IncompleteInterleavedFrame {
            samples_len: chunk.samples.len(),
            channel_count: chunk.channel_count,
        });
    }
    if let Some(&channel_index) = selected_channels
        .iter()
        .find(|&&channel_index| channel_index >= chunk.channel_count)
    {
        return Err(ChannelDemuxError::ChannelIndexOutOfRange {
            channel_index,
            channel_count: chunk.channel_count,
        });
    }

    let frame_count = chunk.samples.len() / usize::from(chunk.channel_count);
    let mut outputs = selected_channels
        .iter()
        .map(|&channel_index| ChannelMonoChunk {
            capture_sequence: chunk.capture_sequence,
            channel_index,
            samples: Vec::with_capacity(frame_count),
        })
        .collect::<Vec<_>>();
    for frame in chunk.samples.chunks_exact(usize::from(chunk.channel_count)) {
        for output in &mut outputs {
            output
                .samples
                .push(frame[usize::from(output.channel_index)]);
        }
    }
    Ok(outputs)
}

#[derive(Debug)]
pub(crate) struct InputChunk {
    pub samples: Vec<f32>,
    #[allow(
        dead_code,
        reason = "preserved at the demux/lane boundary for source metrics and discontinuity detection"
    )]
    pub(crate) capture_sequence: Option<CaptureSequence>,
    _queue_permit: Option<InputQueuePermit>,
}

#[derive(Debug)]
struct InputQueuePermit {
    queued_samples: Arc<AtomicUsize>,
    samples: usize,
}

impl Drop for InputQueuePermit {
    fn drop(&mut self) {
        self.queued_samples
            .fetch_sub(self.samples, Ordering::AcqRel);
    }
}

impl InputChunk {
    pub(crate) fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            capture_sequence: None,
            _queue_permit: None,
        }
    }

    pub(crate) fn with_queue_permit(samples: Vec<f32>, queued_samples: Arc<AtomicUsize>) -> Self {
        let sample_count = samples.len();
        Self {
            samples,
            capture_sequence: None,
            _queue_permit: Some(InputQueuePermit {
                queued_samples,
                samples: sample_count,
            }),
        }
    }

    pub(crate) fn with_capture_sequence_and_queue_permit(
        samples: Vec<f32>,
        capture_sequence: CaptureSequence,
        queued_samples: Arc<AtomicUsize>,
    ) -> Self {
        let sample_count = samples.len();
        Self {
            samples,
            capture_sequence: Some(capture_sequence),
            _queue_permit: Some(InputQueuePermit {
                queued_samples,
                samples: sample_count,
            }),
        }
    }
}

pub(crate) fn build_interleaved_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: Sender<InterleavedPcmChunk>,
) -> Result<Stream> {
    crate::dispatch_cpal_sample_format!(
        sample_format,
        build_interleaved_input_stream_inner,
        device,
        config,
        sender;
        unsupported => anyhow::bail!("Unsupported input sample format: {sample_format:?}")
    )
}

fn build_interleaved_input_stream_inner<T>(
    device: &Device,
    config: &StreamConfig,
    sender: Sender<InterleavedPcmChunk>,
) -> Result<Stream>
where
    T: Sample + SizedSample + ToSample<f32>,
{
    let channel_count = config.channels;
    let next_sequence = AtomicU64::new(1);
    let err_fn = |err| log::warn!("Explicit audio input stream error: {err}");
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                enqueue_interleaved_capture(data, channel_count, &next_sequence, &sender);
            },
            err_fn,
            None,
        )
        .context("Failed to build explicit input stream")
}

fn enqueue_interleaved_capture<T>(
    data: &[T],
    channel_count: u16,
    next_sequence: &AtomicU64,
    sender: &Sender<InterleavedPcmChunk>,
) where
    T: Sample + ToSample<f32>,
{
    if data.is_empty() {
        return;
    }

    let capture_sequence = CaptureSequence(next_sequence.fetch_add(1, Ordering::Relaxed));
    let samples = data.iter().map(|sample| sample.to_sample()).collect();
    let _ = sender.send(InterleavedPcmChunk::new(
        capture_sequence,
        channel_count,
        samples,
    ));
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
                let chunk = InputChunk::new(samples);
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
    use std::sync::{atomic::AtomicU64, mpsc};

    use super::{
        CaptureSequence, ChannelDemuxError, ChannelMonoChunk, InputChunk, InterleavedPcmChunk,
        demux_selected_channels, enqueue_input_chunk, enqueue_interleaved_capture,
    };
    use cpal::Sample;

    #[test]
    fn callback_adapter_enqueues_one_interleaved_chunk_per_callback_with_monotonic_sequence() {
        let (sender, receiver) = mpsc::channel();
        let next_sequence = AtomicU64::new(1);

        enqueue_interleaved_capture(&[1_i16, 10, 2, 20], 2, &next_sequence, &sender);
        enqueue_interleaved_capture(&[3_i16, 30], 2, &next_sequence, &sender);

        assert_eq!(
            receiver.try_iter().collect::<Vec<_>>(),
            vec![
                InterleavedPcmChunk::new(
                    CaptureSequence(1),
                    2,
                    vec![
                        1_i16.to_sample(),
                        10_i16.to_sample(),
                        2_i16.to_sample(),
                        20_i16.to_sample(),
                    ],
                ),
                InterleavedPcmChunk::new(
                    CaptureSequence(2),
                    2,
                    vec![3_i16.to_sample(), 30_i16.to_sample(),],
                ),
            ],
            "a callback must enqueue one central interleaved chunk without channel work"
        );
    }

    #[test]
    fn stereo_demux_keeps_each_channel_sample_order_and_capture_sequence() {
        let chunk = InterleavedPcmChunk::new(
            CaptureSequence(41),
            2,
            vec![0.1, 10.1, 0.2, 10.2, 0.3, 10.3],
        );

        let output = demux_selected_channels(&chunk, &[0, 1]).expect("valid stereo PCM");

        assert_eq!(
            output,
            vec![
                super::ChannelMonoChunk {
                    capture_sequence: CaptureSequence(41),
                    channel_index: 0,
                    samples: vec![0.1, 0.2, 0.3],
                },
                super::ChannelMonoChunk {
                    capture_sequence: CaptureSequence(41),
                    channel_index: 1,
                    samples: vec![10.1, 10.2, 10.3],
                },
            ]
        );
    }

    #[test]
    fn demux_outputs_only_the_explicitly_selected_channels() {
        let chunk = InterleavedPcmChunk::new(CaptureSequence(9), 2, vec![0.1, 10.1, 0.2, 10.2]);

        let output = demux_selected_channels(&chunk, &[1]).expect("channel one is valid");

        assert_eq!(
            output,
            vec![ChannelMonoChunk {
                capture_sequence: CaptureSequence(9),
                channel_index: 1,
                samples: vec![10.1, 10.2],
            }]
        );
    }

    #[test]
    fn demux_rejects_zero_channel_capture_layout() {
        let chunk = InterleavedPcmChunk::new(CaptureSequence(1), 0, vec![0.1]);

        assert_eq!(
            demux_selected_channels(&chunk, &[0]),
            Err(ChannelDemuxError::ZeroChannelCount)
        );
    }

    #[test]
    fn demux_rejects_selected_channel_outside_capture_layout() {
        let chunk = InterleavedPcmChunk::new(CaptureSequence(2), 2, vec![0.1, 10.1]);

        assert_eq!(
            demux_selected_channels(&chunk, &[2]),
            Err(ChannelDemuxError::ChannelIndexOutOfRange {
                channel_index: 2,
                channel_count: 2,
            })
        );
    }

    #[test]
    fn demux_rejects_incomplete_interleaved_frame_without_emitting_audio() {
        let chunk = InterleavedPcmChunk::new(CaptureSequence(3), 2, vec![0.1, 10.1, 0.2]);

        assert_eq!(
            demux_selected_channels(&chunk, &[0]),
            Err(ChannelDemuxError::IncompleteInterleavedFrame {
                samples_len: 3,
                channel_count: 2,
            })
        );
    }

    #[test]
    fn input_queue_keeps_all_chunks_in_fifo_order_when_producer_gets_ahead() {
        let (sender, receiver) = mpsc::channel();

        for sample in 0_u16..32 {
            enqueue_input_chunk(&sender, InputChunk::new(vec![f32::from(sample)]));
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

    #[test]
    fn legacy_and_websocket_input_chunk_constructors_have_no_capture_sequence() {
        let queued_samples = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1));

        assert_eq!(InputChunk::new(vec![1.0]).capture_sequence, None);
        assert_eq!(
            InputChunk::with_queue_permit(vec![1.0], queued_samples).capture_sequence,
            None
        );
    }
}
