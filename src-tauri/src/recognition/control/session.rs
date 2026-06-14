use std::collections::{HashMap, VecDeque};

use super::pending::{PendingFinalization, PendingTurnCheck, RerecognitionPurpose};
use crate::{
    config::ParapperConfig,
    recognition::{
        transcription::{
            asr::task::{AsrInFlight, AsrRequest, AudioRange, GlobalSampleIndex},
            planner::PendingAsrSegment,
            route::{RecognitionRoute, language_id::LanguageDetector},
        },
        turn::{Turn, boundary::JapaneseMorphAnalyzer},
    },
};

use super::{AsrRequestRunner, TurnDecisionRunner, TurnOutputSink};

pub(crate) struct RecognitionSession {
    pub(in crate::recognition) config: ParapperConfig,
    pub(in crate::recognition) pending: PendingRuntimeState,
    pub(in crate::recognition) io: RuntimeIo,
    pub(in crate::recognition) turn_store: TurnStore,
    pub(in crate::recognition) counters: RuntimeCounters,
    pub(in crate::recognition) activity: ActivityState,
    pub(in crate::recognition) requests: AsrRequestState,
}

pub(in crate::recognition) struct RuntimeIo {
    pub(in crate::recognition) asr_runner: Box<dyn AsrRequestRunner>,
    pub(in crate::recognition) turn_decision_runner: Box<dyn TurnDecisionRunner>,
    pub(in crate::recognition) output_sink: Box<dyn TurnOutputSink>,
    pub(in crate::recognition) language_id_runtime: Option<Box<dyn LanguageIdRuntime>>,
    pub(in crate::recognition) language_id: Option<Box<dyn LanguageDetector>>,
    pub(in crate::recognition) japanese_morph: Option<JapaneseMorphAnalyzer>,
}

#[derive(Default)]
pub(in crate::recognition) struct PendingRuntimeState {
    pub(in crate::recognition) turn_check: Option<PendingTurnCheck>,
    pub(in crate::recognition) finalization: Option<PendingFinalization>,
    pub(in crate::recognition) asr_segments: VecDeque<PendingAsrSegment>,
}

pub(in crate::recognition) trait LanguageIdRuntime:
    crate::recognition::transcription::route::language_id::LanguageDetectionWarningSink
{
    fn build_language_id(&self, config: &ParapperConfig) -> Option<Box<dyn LanguageDetector>>;
}

pub(in crate::recognition) struct TurnStore {
    pub(in crate::recognition) turns: HashMap<u64, Turn>,
    pub(in crate::recognition) audio_ranges: HashMap<u64, AudioRange>,
    pub(in crate::recognition) revisions: HashMap<u64, u64>,
    pub(in crate::recognition) confirmed_until_sample: GlobalSampleIndex,
    pub(in crate::recognition) last_recognition_route: Option<RecognitionRoute>,
    pub(in crate::recognition) open_turn_id: Option<u64>,
}

impl Default for TurnStore {
    fn default() -> Self {
        Self {
            turns: HashMap::new(),
            audio_ranges: HashMap::new(),
            revisions: HashMap::new(),
            confirmed_until_sample: GlobalSampleIndex(0),
            last_recognition_route: None,
            open_turn_id: None,
        }
    }
}

pub(in crate::recognition) struct RuntimeCounters {
    pub(in crate::recognition) turn_session_id: u64,
    pub(in crate::recognition) next_turn_id: u64,
    pub(in crate::recognition) next_output_sequence: u64,
    pub(in crate::recognition) next_request_id: u64,
    pub(in crate::recognition) next_vad_frame_index: u64,
    pub(in crate::recognition) next_runtime_tick: u64,
    pub(in crate::recognition) global_sample_cursor: u64,
}

impl RuntimeCounters {
    pub(in crate::recognition) fn new(turn_session_id: u64) -> Self {
        Self {
            turn_session_id,
            next_turn_id: 1,
            next_output_sequence: 1,
            next_request_id: 1,
            next_vad_frame_index: 0,
            next_runtime_tick: 0,
            global_sample_cursor: 0,
        }
    }
}

#[derive(Default)]
pub(in crate::recognition) struct ActivityState {
    pub(in crate::recognition) segment_activity_epoch: u64,
    pub(in crate::recognition) open_turn_activity_epoch: u64,
    pub(in crate::recognition) open_turn_since_tick: Option<u64>,
}

#[derive(Default)]
pub(in crate::recognition) struct AsrRequestState {
    pub(in crate::recognition) in_flight_request: Option<AsrRequest>,
    pub(in crate::recognition) pending_rerecognition_purpose: Option<RerecognitionPurpose>,
    pub(in crate::recognition) last_dispatched: Option<AsrInFlight>,
}
