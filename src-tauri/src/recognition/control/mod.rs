mod construction;
mod driver;
pub(crate) mod engine_cache;
pub(crate) mod events;
pub(crate) mod input;
mod input_source;
mod pending;
pub(crate) mod runtime_event;
mod session;

#[cfg(test)]
pub(crate) use crate::recognition::segmentation::segment::builder::SegmentCloseReason;
pub(crate) use crate::recognition::transcription::asr::port::{
    AsrRequestRunner, AsrWorkerStartupResult, AsrWorkerStartupSender, EngineAsrRequestRunner,
};
#[cfg(test)]
pub(crate) use crate::recognition::transcription::asr::port::{
    NoopAsrRequestRunner, run_engine_asr_request,
};
#[cfg(test)]
use crate::recognition::transcription::planner::PendingAsrSegment;
#[cfg(test)]
pub(crate) use crate::recognition::turn::decision::port::NoopTurnDecisionRunner;
pub(crate) use crate::recognition::turn::decision::port::{
    EngineTurnDecisionRunner, TurnDecisionRunner,
};
#[cfg(test)]
pub(crate) use crate::recognition::turn::port::output_sink::NoopTurnOutputSink;
pub(crate) use crate::recognition::turn::port::output_sink::{
    DeliveryTurnOutputSink, TurnOutputSink,
};
#[cfg(test)]
pub(crate) use driver::replay_vad_frames_for_runtime;
pub(crate) use driver::{RecognitionDriver, RecognitionDriverHandle, RecognitionShutdownResult};
pub(crate) use input_source::{BoundedInputSendError, BoundedInputSender, RunningInputSource};
pub(in crate::recognition) use pending::PendingFinalization;
#[cfg(test)]
use pending::PendingTurnCheck;
pub(in crate::recognition) use pending::RerecognitionPurpose;
#[cfg(test)]
pub(in crate::recognition) use session::LanguageIdRuntime;
pub(crate) use session::RecognitionSession;

#[cfg(test)]
pub(crate) mod tests;
