mod device;
mod dispatch;
mod input;
mod loopback_permission;
mod output;
mod resampler;
mod stream;

pub use device::{DeviceInfo, collect_input_devices, collect_output_devices};
pub use input::{ASR_SAMPLE_RATE, RunningAudioInput};
#[allow(
    unused_imports,
    reason = "the explicit capture API is shared with recognition runtime integration"
)]
pub(crate) use input::{
    AudioInputProcessor, ExplicitAudioLaneStartup, PreparedExplicitAudioInput, ProcessedAudioChunk,
    SourceQueueOverrun,
};
pub(crate) use loopback_permission::ensure_system_audio_permission;
pub use loopback_permission::{open_system_audio_settings, request_system_audio_permission};
pub(crate) use output::play_mono_samples;
#[allow(
    unused_imports,
    reason = "capture sequence metadata is shared with recognition runtime integration"
)]
pub(crate) use stream::{CaptureSequence, InputChunk};
