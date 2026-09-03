use crate::{
    SourceIdentitySnapshot, SourceSessionKey, SttEngineConfig,
    ports::{
        AsrRequestRunner, LanguageDetectionWarningSink, LanguageDetector, RecognitionOutputSink,
        TranscriptBoundaryDetector, TurnDecisionRunner,
    },
    runtime::{ActivityState, AsrRequestState, PendingRuntimeState, RuntimeCounters, TurnStore},
};

pub struct RecognitionSession {
    pub config: SttEngineConfig,
    source_session: SourceSessionKey,
    source_identity: SourceIdentitySnapshot,
    pub pending: PendingRuntimeState,
    pub io: RuntimeIo,
    pub turn_store: TurnStore,
    pub counters: RuntimeCounters,
    pub activity: ActivityState,
    pub requests: AsrRequestState,
}

pub struct RuntimeIo {
    pub asr_runner: Box<dyn AsrRequestRunner>,
    pub turn_decision_runner: Box<dyn TurnDecisionRunner>,
    pub output_sink: Box<dyn RecognitionOutputSink>,
    pub language_warning_sink: Option<Box<dyn LanguageDetectionWarningSink>>,
    pub language_id: Option<Box<dyn LanguageDetector>>,
    pub boundary_detector: Option<Box<dyn TranscriptBoundaryDetector>>,
}

impl RecognitionSession {
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor makes every host capability explicit at the engine boundary"
    )]
    pub fn with_ports(
        config: SttEngineConfig,
        turn_session_id: u64,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn RecognitionOutputSink>,
        language_warning_sink: Option<Box<dyn LanguageDetectionWarningSink>>,
        language_id: Option<Box<dyn LanguageDetector>>,
        boundary_detector: Option<Box<dyn TranscriptBoundaryDetector>>,
    ) -> Self {
        Self::with_ports_and_source_identity(
            config,
            turn_session_id,
            SourceIdentitySnapshot::legacy_single_source(),
            asr_runner,
            turn_decision_runner,
            output_sink,
            language_warning_sink,
            language_id,
            boundary_detector,
        )
    }

    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor makes every host capability explicit at the engine boundary"
    )]
    pub fn with_ports_and_source_identity(
        config: SttEngineConfig,
        turn_session_id: u64,
        source_identity: SourceIdentitySnapshot,
        asr_runner: Box<dyn AsrRequestRunner>,
        turn_decision_runner: Box<dyn TurnDecisionRunner>,
        output_sink: Box<dyn RecognitionOutputSink>,
        language_warning_sink: Option<Box<dyn LanguageDetectionWarningSink>>,
        language_id: Option<Box<dyn LanguageDetector>>,
        boundary_detector: Option<Box<dyn TranscriptBoundaryDetector>>,
    ) -> Self {
        let source_session =
            SourceSessionKey::new(turn_session_id, source_identity.source_id.clone());
        Self {
            config,
            source_session,
            source_identity,
            pending: PendingRuntimeState::default(),
            io: RuntimeIo {
                asr_runner,
                turn_decision_runner,
                output_sink,
                language_warning_sink,
                language_id,
                boundary_detector,
            },
            turn_store: TurnStore::default(),
            counters: RuntimeCounters::new(turn_session_id),
            activity: ActivityState::default(),
            requests: AsrRequestState::default(),
        }
    }

    #[must_use]
    pub const fn source_session(&self) -> &SourceSessionKey {
        &self.source_session
    }

    #[must_use]
    pub const fn source_identity(&self) -> &SourceIdentitySnapshot {
        &self.source_identity
    }

    #[cfg(test)]
    pub(crate) fn new(config: &SttEngineConfig) -> Self {
        Self::with_ports(
            config.clone(),
            1,
            Box::new(NoopAsrRunner),
            Box::new(NoopTurnDecisionRunner),
            Box::new(NoopOutputSink),
            None,
            None,
            None,
        )
    }
}

#[cfg(test)]
struct NoopAsrRunner;

#[cfg(test)]
impl AsrRequestRunner for NoopAsrRunner {
    fn submit(&mut self, _request: crate::transcription::task::AsrRequest) -> bool {
        true
    }

    fn try_recv_result(&mut self) -> Option<crate::transcription::task::AsrResult> {
        None
    }
}

#[cfg(test)]
struct NoopTurnDecisionRunner;

#[cfg(test)]
impl TurnDecisionRunner for NoopTurnDecisionRunner {
    fn decide(
        &mut self,
        _route: crate::transcription::route::RecognitionRoute,
        _text: &str,
        _max_context_tokens: u32,
    ) -> anyhow::Result<crate::turn::TurnDecision> {
        Ok(crate::turn::TurnDecision {
            is_end_of_turn: false,
            confidence: 0.0,
        })
    }
}

#[cfg(test)]
struct NoopOutputSink;

#[cfg(test)]
impl RecognitionOutputSink for NoopOutputSink {
    fn emit(&mut self, _output: crate::RecognizedTurn) {}
}
