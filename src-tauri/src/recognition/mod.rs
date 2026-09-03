mod asr_worker;
mod config;
mod driver;
pub(crate) mod events;
mod input;
mod input_source;
mod language_adapter;
mod model_factory;
mod output_sink;
mod session;
mod streaming;
mod turn_adapter;

#[cfg(test)]
pub(crate) use asr_worker::{AsrRequestRunner, NoopAsrRequestRunner, run_engine_asr_request};
pub(crate) use asr_worker::{
    AsrRuntimePool, AsrRuntimePoolHandle, AsrWorkerStartupReport, AsrWorkerStartupSender,
};
#[cfg(test)]
pub(crate) use driver::replay_vad_frames_for_runtime;
pub(crate) use driver::{RecognitionDriver, RecognitionDriverHandle, RecognitionShutdownResult};
pub use events::RecognitionStatus;
pub use input::{RecognitionStartError, RunningRecognitionInput};
pub(crate) use input::{RecognitionStreamEvent, RuntimeConfigState};
pub(crate) use input_source::{BoundedInputSendError, BoundedInputSender, RunningInputSource};
#[cfg(test)]
pub(in crate::recognition) use language_adapter::LanguageIdRuntime;
#[cfg(test)]
pub(crate) use output_sink::NoopTurnOutputSink;
pub(crate) use output_sink::{
    CompositeTurnOutputSink, DeliveryTurnOutputSink, TurnOutputSink, WebSocketTurnOutputSink,
};
pub(crate) use session::RecognitionSession;
pub(crate) use streaming::{
    NetworkOutputMode, StreamingRecognitionServer, StreamingRecognitionServerConfig,
};
#[cfg(test)]
pub(crate) use turn_adapter::{
    EngineTurnDecisionRunner, NoopTurnDecisionRunner, TurnDecisionRunner,
};

#[cfg(test)]
pub(in crate::recognition) use parapper_stt_engine::{
    SegmentCloseReason, runtime::PendingTurnCheck, transcription::planner::PendingAsrSegment,
    turn::RerecognitionPurpose,
};

#[cfg(test)]
mod tests;
