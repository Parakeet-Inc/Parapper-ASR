use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc::{Receiver, Sender, channel},
};

use anyhow::Result;

use crate::{
    audio::{InputChunk, RunningAudioInput},
    config::ParapperConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputSourceConfig {
    Physical,
    Loopback,
}

impl InputSourceConfig {
    pub(crate) fn from_config(config: &ParapperConfig) -> Self {
        if config
            .input
            .device_host
            .as_deref()
            .is_some_and(|host| host.ends_with(" (Loopback)"))
        {
            Self::Loopback
        } else {
            Self::Physical
        }
    }

    pub(crate) fn start(self, config: &ParapperConfig) -> Result<RunningInputSource> {
        match self {
            Self::Physical | Self::Loopback => RunningInputSource::desktop(config),
        }
    }
}

pub(crate) struct RunningInputSource {
    parts: RunningInputSourceParts,
}

pub(crate) struct RunningInputSourceParts {
    pub(crate) lifetime: InputSourceLifetime,
    pub(crate) receiver: Receiver<InputChunk>,
    pub(crate) source_sample_rate: u32,
    pub(crate) disconnect_policy: InputDisconnectPolicy,
}

pub(crate) struct InputSourceLifetime {
    _audio_input: Option<RunningAudioInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputDisconnectPolicy {
    Graceful,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedInputSendError {
    Overrun,
    Disconnected,
}

#[derive(Clone)]
pub(crate) struct BoundedInputSender {
    sender: Sender<InputChunk>,
    queued_samples: Arc<AtomicUsize>,
    max_queued_samples: usize,
}

impl BoundedInputSender {
    pub(crate) fn try_send(&self, samples: Vec<f32>) -> Result<(), BoundedInputSendError> {
        let sample_count = samples.len();
        let reserved =
            self.queued_samples
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                    queued
                        .checked_add(sample_count)
                        .filter(|next| *next <= self.max_queued_samples)
                });
        if reserved.is_err() {
            return Err(BoundedInputSendError::Overrun);
        }

        let chunk = InputChunk::with_queue_permit(samples, self.queued_samples.clone());
        match self.sender.send(chunk) {
            Ok(()) => Ok(()),
            Err(_error) => Err(BoundedInputSendError::Disconnected),
        }
    }

    #[cfg(test)]
    fn queued_samples(&self) -> usize {
        self.queued_samples.load(Ordering::Acquire)
    }
}

impl RunningInputSource {
    fn desktop(config: &ParapperConfig) -> Result<Self> {
        let startup = RunningAudioInput::start(config)?;
        Ok(Self {
            parts: RunningInputSourceParts {
                lifetime: InputSourceLifetime {
                    _audio_input: Some(startup.input),
                },
                receiver: startup.receiver,
                source_sample_rate: startup.source_sample_rate,
                disconnect_policy: InputDisconnectPolicy::Graceful,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn channel(source_sample_rate: u32) -> (Sender<InputChunk>, Self) {
        let (sender, receiver) = channel();
        (
            sender,
            Self::from_receiver(
                receiver,
                source_sample_rate,
                InputDisconnectPolicy::Graceful,
            ),
        )
    }

    pub(crate) fn from_receiver(
        receiver: Receiver<InputChunk>,
        source_sample_rate: u32,
        disconnect_policy: InputDisconnectPolicy,
    ) -> Self {
        Self {
            parts: RunningInputSourceParts {
                lifetime: InputSourceLifetime { _audio_input: None },
                receiver,
                source_sample_rate,
                disconnect_policy,
            },
        }
    }

    pub(crate) fn bounded_channel(
        source_sample_rate: u32,
        max_queued_samples: usize,
    ) -> (BoundedInputSender, Self) {
        // The atomic sample reservation is the queue bound. A second message-count
        // bound would reject valid small frames before the documented audio budget.
        let (sender, receiver) = channel();
        let queued_samples = Arc::new(AtomicUsize::new(0));
        (
            BoundedInputSender {
                sender,
                queued_samples: queued_samples.clone(),
                max_queued_samples,
            },
            Self::from_receiver(receiver, source_sample_rate, InputDisconnectPolicy::Cancel),
        )
    }

    pub(crate) fn into_parts(self) -> RunningInputSourceParts {
        self.parts
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{BoundedInputSendError, InputSourceConfig, RunningInputSource};
    use crate::{
        audio::{ASR_SAMPLE_RATE, AudioInputProcessor, InputChunk},
        config::ParapperConfig,
    };

    #[test]
    fn legacy_device_fields_select_physical_or_loopback_without_a_new_required_field() {
        let cases = [
            (None, InputSourceConfig::Physical),
            (Some("WASAPI"), InputSourceConfig::Physical),
            (Some("WASAPI (Loopback)"), InputSourceConfig::Loopback),
        ];

        for (host, expected) in cases {
            let mut config = ParapperConfig::default();
            config.input.device_host = host.map(str::to_string);

            assert_eq!(InputSourceConfig::from_config(&config), expected);
        }
    }

    #[test]
    fn channel_source_preserves_chunk_order_sample_rate_and_disconnect() {
        let (sender, source) = RunningInputSource::channel(8_000);
        sender.send(InputChunk::new(vec![1.0])).unwrap();
        sender.send(InputChunk::new(vec![2.0])).unwrap();
        drop(sender);
        let parts = source.into_parts();

        assert_eq!(parts.source_sample_rate, 8_000);
        assert_eq!(
            parts
                .receiver
                .iter()
                .map(|chunk| chunk.samples[0].to_bits())
                .collect::<Vec<_>>(),
            vec![1.0_f32.to_bits(), 2.0_f32.to_bits()]
        );
        assert!(
            parts
                .receiver
                .recv_timeout(Duration::from_millis(1))
                .is_err()
        );
    }

    #[test]
    fn channel_source_uses_the_same_gain_and_resampling_processor_as_desktop_audio() {
        let handle = tauri_test_handle();
        let mut config = ParapperConfig::default();
        config.input.volume_db = 6.020_6;
        let (sender, source) = RunningInputSource::channel(ASR_SAMPLE_RATE);
        sender.send(InputChunk::new(vec![0.25; 512])).unwrap();
        drop(sender);
        let parts = source.into_parts();
        let mut processor =
            AudioInputProcessor::initialize(handle, &config, parts.source_sample_rate).unwrap();
        let mut processed = Vec::new();

        for chunk in parts.receiver {
            processor.process(&chunk, &config, |samples| processed.push(samples));
        }

        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0].len(), 512);
        for sample in &processed[0] {
            assert!((*sample - 0.5).abs() < 0.001, "sample={sample}");
        }
    }

    #[test]
    fn bounded_source_counts_pending_samples_until_each_chunk_is_processed() {
        let (sender, source) = RunningInputSource::bounded_channel(16_000, 32_000);
        sender.try_send(vec![0.0; 16_000]).unwrap();
        sender.try_send(vec![0.0; 16_000]).unwrap();
        assert_eq!(sender.queued_samples(), 32_000);
        assert_eq!(
            sender.try_send(vec![0.0; 2]),
            Err(BoundedInputSendError::Overrun)
        );
        let parts = source.into_parts();

        let first = parts.receiver.recv().unwrap();
        assert_eq!(sender.queued_samples(), 32_000);
        drop(first);
        assert_eq!(sender.queued_samples(), 16_000);
        sender.try_send(vec![0.0; 16_000]).unwrap();
        assert_eq!(sender.queued_samples(), 32_000);
    }

    #[test]
    fn bounded_source_accepts_more_than_64_small_frames_while_sample_budget_remains() {
        let (sender, _source) = RunningInputSource::bounded_channel(16_000, 32_000);

        for frame_index in 0..100 {
            sender
                .try_send(vec![0.0; 160])
                .unwrap_or_else(|error| panic!("frame {frame_index} failed early: {error:?}"));
        }

        assert_eq!(sender.queued_samples(), 16_000);
    }

    fn tauri_test_handle() -> tauri::AppHandle {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("test app should build");
        app.handle().clone()
    }
}
