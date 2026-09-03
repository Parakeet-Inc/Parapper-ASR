use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

use tauri::AppHandle;

use crate::{
    config::ParapperConfig,
    recognition::{
        asr_worker::{AsrRuntimePoolHandle, AsrWorkerStartupSender, EngineAsrRequestRunner},
        config::stt_engine_config,
        language_adapter::{
            LanguageWarningAdapter, build_id_detector, tauri_language_warning_runtime,
        },
        output_sink::TurnOutputSink,
        turn_adapter::{AppTranscriptBoundaryDetector, EngineTurnDecisionRunner},
    },
};

#[cfg(test)]
use crate::recognition::{
    asr_worker::{AsrRequestRunner, NoopAsrRequestRunner},
    language_adapter::LanguageIdRuntime,
    output_sink::{DeliveryTurnOutputSink, NoopTurnOutputSink},
    turn_adapter::{NoopTurnDecisionRunner, TurnDecisionRunner},
};

static NEXT_TURN_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn take_next_turn_session_id() -> u64 {
    NEXT_TURN_SESSION_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) struct RecognitionSession {
    inner: parapper_stt_engine::RecognitionSession,
}

impl RecognitionSession {
    fn from_engine(inner: parapper_stt_engine::RecognitionSession) -> Self {
        Self { inner }
    }

    pub(super) fn into_engine(self) -> parapper_stt_engine::RecognitionSession {
        self.inner
    }

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

    /// Resolves Tauri-owned paths and constructs native model adapters before
    /// handing a Tauri-free set of ports to `parapper-stt-engine`.
    pub(crate) fn new_for_production_with_output_sink(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        Self::new_for_production_with_output_sink_and_source_identity(
            handle,
            config,
            asr_startup_sender,
            parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
            output_sink,
        )
    }

    /// Starts one source runtime with an immutable identity snapshot supplied by
    /// the application layer. A process-wide session id is allocated exactly
    /// once for this runtime.
    pub(crate) fn new_for_production_with_output_sink_and_source_identity(
        handle: &AppHandle,
        config: &ParapperConfig,
        asr_startup_sender: Option<AsrWorkerStartupSender>,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        let language_runtime = tauri_language_warning_runtime(handle);
        let language_id = build_id_detector(handle, config);
        Self::with_ports_and_new_session_id(
            stt_engine_config(config),
            source_identity,
            Box::new(EngineAsrRequestRunner::new(
                handle.clone(),
                config,
                asr_startup_sender,
            )),
            Box::new(EngineTurnDecisionRunner::new(handle, config)),
            output_sink,
            Some(Box::new(LanguageWarningAdapter(language_runtime))),
            language_id,
            Some(Box::new(AppTranscriptBoundaryDetector::new(handle, config))),
        )
    }

    pub(crate) fn new_for_production_with_pool_and_output_sink_and_source_identity(
        handle: &AppHandle,
        config: &ParapperConfig,
        pool: &AsrRuntimePoolHandle,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        output_sink: Box<dyn TurnOutputSink>,
    ) -> Self {
        let language_runtime = tauri_language_warning_runtime(handle);
        let language_id = build_id_detector(handle, config);
        Self::with_ports_and_new_session_id(
            stt_engine_config(config),
            source_identity,
            Box::new(EngineAsrRequestRunner::from_pool(pool)),
            Box::new(EngineTurnDecisionRunner::new(handle, config)),
            output_sink,
            Some(Box::new(LanguageWarningAdapter(language_runtime))),
            language_id,
            Some(Box::new(AppTranscriptBoundaryDetector::new(handle, config))),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor makes every engine port explicit at the application boundary"
    )]
    fn with_ports_and_new_session_id(
        config: parapper_stt_engine::SttEngineConfig,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        asr_runner: Box<dyn parapper_stt_engine::ports::AsrRequestRunner>,
        turn_decision_runner: Box<dyn parapper_stt_engine::ports::TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        warning_sink: Option<Box<dyn parapper_stt_engine::ports::LanguageDetectionWarningSink>>,
        language_id: Option<Box<dyn parapper_stt_engine::ports::LanguageDetector>>,
        boundary_detector: Option<Box<dyn parapper_stt_engine::ports::TranscriptBoundaryDetector>>,
    ) -> Self {
        Self::from_engine(
            parapper_stt_engine::RecognitionSession::with_ports_and_source_identity(
                config,
                take_next_turn_session_id(),
                source_identity,
                asr_runner,
                turn_decision_runner,
                output_sink,
                warning_sink,
                language_id,
                boundary_detector,
            ),
        )
    }

    #[cfg(test)]
    pub(in crate::recognition) fn new_for_test_with_all_io(
        config: &ParapperConfig,
        turn_session_id: u64,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
        language_id: Option<Box<dyn parapper_stt_engine::ports::LanguageDetector>>,
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

    #[cfg(test)]
    fn with_io_and_session_id(
        config: &ParapperConfig,
        turn_session_id: u64,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
        language_id: Option<Box<dyn parapper_stt_engine::ports::LanguageDetector>>,
    ) -> Self {
        let warning_sink = language_id_runtime.map(|runtime| {
            Box::new(LanguageWarningAdapter(runtime))
                as Box<dyn parapper_stt_engine::ports::LanguageDetectionWarningSink>
        });
        Self::from_engine(parapper_stt_engine::RecognitionSession::with_ports(
            stt_engine_config(config),
            turn_session_id,
            asr_runner,
            turn_decision_runner,
            output_sink,
            warning_sink,
            language_id,
            Some(Box::new(AppTranscriptBoundaryDetector::without_morph())),
        ))
    }

    #[cfg(test)]
    fn with_io_and_new_session_id_and_source_identity(
        config: &ParapperConfig,
        source_identity: parapper_stt_engine::SourceIdentitySnapshot,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn TurnOutputSink>,
        language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
        language_id: Option<Box<dyn parapper_stt_engine::ports::LanguageDetector>>,
    ) -> Self {
        let warning_sink = language_id_runtime.map(|runtime| {
            Box::new(LanguageWarningAdapter(runtime))
                as Box<dyn parapper_stt_engine::ports::LanguageDetectionWarningSink>
        });
        Self::with_ports_and_new_session_id(
            stt_engine_config(config),
            source_identity,
            asr_runner,
            turn_decision_runner,
            output_sink,
            warning_sink,
            language_id,
            Some(Box::new(AppTranscriptBoundaryDetector::without_morph())),
        )
    }

    #[cfg(test)]
    pub(in crate::recognition) fn take_last_dispatched(
        &mut self,
    ) -> Option<parapper_stt_engine::transcription::task::AsrInFlight> {
        self.requests.last_dispatched.take()
    }
}

impl Deref for RecognitionSession {
    type Target = parapper_stt_engine::RecognitionSession;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for RecognitionSession {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NoopAsrRequestRunner, NoopTurnDecisionRunner, NoopTurnOutputSink, RecognitionSession,
        take_next_turn_session_id,
    };
    use crate::config::ParapperConfig;
    use parapper_stt_engine::{SourceId, SourceIdentitySnapshot};

    #[test]
    fn production_turn_session_ids_are_monotonic() {
        let first = take_next_turn_session_id();
        let second = take_next_turn_session_id();

        assert!(
            second > first,
            "new production runtimes must use increasing turn session ids"
        );
    }

    #[test]
    fn source_identity_constructor_preserves_each_source_and_allocates_distinct_sessions() {
        let first_identity = SourceIdentitySnapshot::new(
            SourceId::from("microphone-left"),
            "Speaker left".to_owned(),
            "interface-main".to_owned(),
            Some(0),
        );
        let second_identity = SourceIdentitySnapshot::new(
            SourceId::from("microphone-right"),
            "Speaker right".to_owned(),
            "interface-main".to_owned(),
            Some(1),
        );

        let first = RecognitionSession::with_io_and_new_session_id_and_source_identity(
            &ParapperConfig::default(),
            first_identity.clone(),
            Box::new(NoopAsrRequestRunner),
            Box::new(NoopTurnDecisionRunner),
            Box::new(NoopTurnOutputSink),
            None,
            None,
        );
        let second = RecognitionSession::with_io_and_new_session_id_and_source_identity(
            &ParapperConfig::default(),
            second_identity.clone(),
            Box::new(NoopAsrRequestRunner),
            Box::new(NoopTurnDecisionRunner),
            Box::new(NoopTurnOutputSink),
            None,
            None,
        );

        assert_eq!(first.source_identity(), &first_identity);
        assert_eq!(second.source_identity(), &second_identity);
        assert_eq!(first.source_session().source_id, first_identity.source_id);
        assert_eq!(second.source_session().source_id, second_identity.source_id);
        assert_ne!(
            first.source_session().turn_session_id,
            second.source_session().turn_session_id,
            "every source runtime must have a process-unique session id"
        );
    }
}
