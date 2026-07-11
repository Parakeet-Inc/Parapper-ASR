use std::sync::atomic::{AtomicU64, Ordering};

use tauri::AppHandle;

#[cfg(test)]
use super::DeliveryTurnOutputSink;
use super::{
    AsrRequestRunner, AsrWorkerStartupSender, EngineAsrRequestRunner, EngineTurnDecisionRunner,
    RecognitionSession, TurnDecisionRunner, TurnOutputSink,
    session::{
        ActivityState, AsrRequestState, LanguageIdRuntime, PendingRuntimeState, RuntimeCounters,
        RuntimeIo, TurnStore,
    },
};
#[cfg(test)]
use super::{NoopAsrRequestRunner, NoopTurnDecisionRunner, NoopTurnOutputSink};
#[cfg(test)]
use crate::recognition::transcription::asr::task::AsrInFlight;
use crate::{
    config::ParapperConfig,
    recognition::transcription::asr::input::emit_asr_warning,
    recognition::{
        transcription::route::language_id::{LanguageDetectionWarningSink, LanguageDetector},
        transcription::route::selection::build_id_detector,
        turn::boundary::load_japanese_morph_analyzer,
    },
};

static NEXT_TURN_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn take_next_turn_session_id() -> u64 {
    NEXT_TURN_SESSION_ID.fetch_add(1, Ordering::Relaxed)
}

impl RecognitionSession {
    #[cfg(test)]
    pub(crate) fn new(config: &ParapperConfig) -> Self {
        Self::with_io_and_session_id(
            config,
            1,
            Box::new(NoopAsrRequestRunner),
            Box::new(NoopTurnDecisionRunner),
            Box::new(NoopTurnOutputSink),
            None,
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_for_production(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
    ) -> Self {
        Self::new_for_production_with_output_sink(
            handle,
            config,
            asr_startup_sender,
            Box::new(DeliveryTurnOutputSink::new(handle.clone(), config)),
        )
    }

    /// Builds a production session whose recognized-text output goes to the
    /// provided sink instead of the global delivery fan-out. Used by the
    /// external STT server so its results never reach translation, synthesis,
    /// the desktop UI, or YNC.
    pub(crate) fn new_for_production_with_output_sink(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        let mut runtime = Self::with_io_and_session_id(
            config,
            take_next_turn_session_id(),
            Box::new(EngineAsrRequestRunner::new(
                handle.clone(),
                config,
                asr_startup_sender,
            )),
            Box::new(EngineTurnDecisionRunner::new(handle, config)),
            output_sink,
            Some(Box::new(TauriLanguageIdRuntime {
                handle: handle.clone(),
            })),
            build_id_detector(handle, config),
        );
        if config.requires_japanese_morph_analyzer() {
            runtime.io.japanese_morph = load_japanese_morph_analyzer(handle);
        }
        runtime
    }

    #[cfg(test)]
    pub(in crate::recognition) fn new_for_test_with_all_io(
        config: &ParapperConfig,
        turn_session_id: u64,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
        language_id: Option<Box<dyn LanguageDetector>>,
    ) -> Self {
        Self::with_io_and_session_id(
            config,
            turn_session_id,
            asr_runner,
            turn_decision_runner,
            output_sink,
            language_id_runtime,
            language_id,
        )
    }

    fn with_io_and_session_id(
        config: &ParapperConfig,
        turn_session_id: u64,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
        language_id: Option<Box<dyn LanguageDetector>>,
    ) -> Self {
        Self {
            config: config.clone(),
            pending: PendingRuntimeState::default(),
            io: RuntimeIo {
                asr_runner,
                turn_decision_runner,
                output_sink,
                language_id_runtime,
                language_id,
                japanese_morph: None,
            },
            turn_store: TurnStore::default(),
            counters: RuntimeCounters::new(turn_session_id),
            activity: ActivityState::default(),
            requests: AsrRequestState::default(),
        }
    }

    #[cfg(test)]
    pub(in crate::recognition) fn take_last_dispatched(&mut self) -> Option<AsrInFlight> {
        self.requests.last_dispatched.take()
    }
}

struct TauriLanguageIdRuntime {
    handle: AppHandle,
}

impl LanguageDetectionWarningSink for TauriLanguageIdRuntime {
    fn emit_language_detection_warning(&self, err: &anyhow::Error) {
        emit_asr_warning(&self.handle, err);
    }
}

impl LanguageIdRuntime for TauriLanguageIdRuntime {
    fn build_language_id(&self, config: &ParapperConfig) -> Option<Box<dyn LanguageDetector>> {
        build_id_detector(&self.handle, config)
    }
}

#[cfg(test)]
mod tests {
    use super::take_next_turn_session_id;

    #[test]
    fn production_turn_session_ids_are_monotonic() {
        let first = take_next_turn_session_id();
        let second = take_next_turn_session_id();

        assert!(
            second > first,
            "new production runtimes must use increasing turn session ids"
        );
    }
}
