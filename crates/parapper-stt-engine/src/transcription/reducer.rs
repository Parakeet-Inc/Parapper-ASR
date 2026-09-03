use parapper_models::asr::AsrTranscript;

use super::{
    route::RecognitionRoute,
    task::{AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AsrTaskKind, GlobalSampleIndex},
};

pub use crate::turn::RerecognitionPurpose as AsrResultRerecognitionPurpose;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsrResultCompletionAfterTranscript {
    RerecognizeIfIdle(AsrResultRerecognitionPurpose),
    CompleteWithoutGrammar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsrResultCompletionFailureAction {
    DecideWithNamo,
    CompleteWithoutGrammar,
    KeepOpen,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsrResultAction {
    KeepInFlightForMismatchedResult {
        result_request_id: AsrRequestId,
        in_flight_request_id: AsrRequestId,
    },
    DropStaleResult,
    DropUnusableInterim,
    DropUnusableCompletionWithoutDraft,
    FallbackCompletionWithNamo {
        turn_id: u64,
    },
    FallbackCompletionWithoutGrammar {
        turn_id: u64,
    },
    FallbackCompletionKeepOpen {
        turn_id: u64,
    },
    FallbackRerecognition {
        turn_id: u64,
        purpose: AsrResultRerecognitionPurpose,
    },
    ApplyInterimTranscript {
        transcript: AsrTranscript,
        elapsed_millis: u128,
    },
    ApplyCompletionTranscript {
        transcript: AsrTranscript,
        elapsed_millis: u128,
        after_transcript: AsrResultCompletionAfterTranscript,
    },
    ApplyRerecognitionTranscript {
        transcript: AsrTranscript,
        elapsed_millis: u128,
        purpose: AsrResultRerecognitionPurpose,
    },
}

#[derive(Clone, Copy)]
pub struct AsrRequestStaleInput {
    pub current_revision: u64,
    pub confirmed_until_sample: GlobalSampleIndex,
    pub target_turn_is_finalized: bool,
    pub turn_route: Option<RecognitionRoute>,
    pub last_recognition_route: Option<RecognitionRoute>,
    pub default_route: RecognitionRoute,
    pub split_route: Option<RecognitionRoute>,
}

#[derive(Clone, Copy)]
pub struct AsrResultReductionInput {
    pub stale_input: AsrRequestStaleInput,
    pub completion_has_non_empty_draft: bool,
    pub completion_failure_action: AsrResultCompletionFailureAction,
    pub completion_rerecognition_purpose: Option<AsrResultRerecognitionPurpose>,
    pub pending_rerecognition_purpose: Option<AsrResultRerecognitionPurpose>,
}

#[must_use]
pub fn result_matches_in_flight_request(result: &AsrResult, request: &AsrRequest) -> bool {
    result.request_id == request.request_id
        && result.kind == request.kind
        && result.target == request.target
        && result.route == request.route
}

#[must_use]
pub fn reduce_asr_result(
    result: &AsrResult,
    request: &AsrRequest,
    input: AsrResultReductionInput,
) -> AsrResultAction {
    if !result_matches_in_flight_request(result, request) {
        return AsrResultAction::KeepInFlightForMismatchedResult {
            result_request_id: result.request_id,
            in_flight_request_id: request.request_id,
        };
    }
    if is_stale_asr_request(request, input.stale_input) {
        return AsrResultAction::DropStaleResult;
    }

    let transcript = match &result.status {
        AsrResultStatus::Ok(transcript) if !transcript.text.trim().is_empty() => transcript.clone(),
        AsrResultStatus::Ok(_) | AsrResultStatus::Failed(_) => {
            return unusable_result_action(request, input);
        }
    };

    match request.kind {
        AsrTaskKind::InterimDisplay => AsrResultAction::ApplyInterimTranscript {
            transcript,
            elapsed_millis: result.elapsed_millis,
        },
        AsrTaskKind::CompletionCheck => AsrResultAction::ApplyCompletionTranscript {
            transcript,
            elapsed_millis: result.elapsed_millis,
            after_transcript: input.completion_rerecognition_purpose.map_or(
                AsrResultCompletionAfterTranscript::CompleteWithoutGrammar,
                AsrResultCompletionAfterTranscript::RerecognizeIfIdle,
            ),
        },
        AsrTaskKind::Rerecognition => AsrResultAction::ApplyRerecognitionTranscript {
            transcript,
            elapsed_millis: result.elapsed_millis,
            purpose: input
                .pending_rerecognition_purpose
                .unwrap_or(AsrResultRerecognitionPurpose::GrammarAfterCompletion),
        },
    }
}

/// Reduces a result whose request was sent as `New` after its correlation has
/// already been resolved by an earlier usable result. Both envelopes are
/// normalized to `Existing` before matching so failed/empty completion results
/// can still run the normal draft fallback path.
#[must_use]
pub fn reduce_asr_result_for_resolved_turn(
    result: &AsrResult,
    request: &AsrRequest,
    input: AsrResultReductionInput,
    resolved_turn_id: Option<u64>,
) -> AsrResultAction {
    let Some(turn_id) = resolved_turn_id else {
        return reduce_asr_result(result, request, input);
    };
    if !matches!(
        request.target.turn_target,
        crate::transcription::task::AsrTurnTarget::New
    ) {
        return reduce_asr_result(result, request, input);
    }
    let mut resolved_request = request.clone();
    resolved_request.target.turn_target =
        crate::transcription::task::AsrTurnTarget::Existing(crate::TurnId(turn_id));
    resolved_request.target.turn_id = crate::TurnId(turn_id);
    let mut resolved_result = result.clone();
    resolved_result.target = resolved_request.target.clone();
    reduce_asr_result(&resolved_result, &resolved_request, input)
}

fn unusable_result_action(request: &AsrRequest, input: AsrResultReductionInput) -> AsrResultAction {
    if matches!(
        request.target.turn_target,
        crate::transcription::task::AsrTurnTarget::New
    ) {
        return match request.kind {
            AsrTaskKind::InterimDisplay => AsrResultAction::DropUnusableInterim,
            AsrTaskKind::CompletionCheck | AsrTaskKind::Rerecognition => {
                AsrResultAction::DropUnusableCompletionWithoutDraft
            }
        };
    }
    let turn_id = request.target.turn_id.0;
    match request.kind {
        AsrTaskKind::InterimDisplay => AsrResultAction::DropUnusableInterim,
        AsrTaskKind::CompletionCheck => {
            if !input.completion_has_non_empty_draft {
                return AsrResultAction::DropUnusableCompletionWithoutDraft;
            }
            match input.completion_failure_action {
                AsrResultCompletionFailureAction::DecideWithNamo => {
                    AsrResultAction::FallbackCompletionWithNamo { turn_id }
                }
                AsrResultCompletionFailureAction::CompleteWithoutGrammar => {
                    AsrResultAction::FallbackCompletionWithoutGrammar { turn_id }
                }
                AsrResultCompletionFailureAction::KeepOpen => {
                    AsrResultAction::FallbackCompletionKeepOpen { turn_id }
                }
            }
        }
        AsrTaskKind::Rerecognition => AsrResultAction::FallbackRerecognition {
            turn_id,
            purpose: input
                .pending_rerecognition_purpose
                .unwrap_or(AsrResultRerecognitionPurpose::GrammarAfterCompletion),
        },
    }
}

#[must_use]
pub fn is_stale_asr_request(request: &AsrRequest, input: AsrRequestStaleInput) -> bool {
    if input.current_revision != request.target.turn_revision.0 {
        return true;
    }
    if input.target_turn_is_finalized {
        return true;
    }
    if request.target.range.end_sample <= input.confirmed_until_sample {
        return true;
    }
    if input.split_route == Some(request.route) {
        return false;
    }
    if let Some(route) = input.turn_route {
        return route != request.route;
    }
    if input.last_recognition_route == Some(request.route) || request.detected_language.is_some() {
        return false;
    }
    input.default_route != request.route
}

#[cfg(test)]
mod tests {
    use parapper_models::asr::AsrModel;

    use super::*;
    use crate::{SegmentId, TurnId, transcription::task::*};

    #[test]
    fn revision_range_and_route_mismatch_all_make_a_result_stale() {
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let request = request(route);
        let baseline = AsrRequestStaleInput {
            current_revision: 0,
            confirmed_until_sample: GlobalSampleIndex(0),
            target_turn_is_finalized: false,
            turn_route: None,
            last_recognition_route: Some(route),
            default_route: route,
            split_route: None,
        };

        for input in [
            AsrRequestStaleInput {
                current_revision: 1,
                ..baseline
            },
            AsrRequestStaleInput {
                confirmed_until_sample: GlobalSampleIndex(10),
                ..baseline
            },
            AsrRequestStaleInput {
                target_turn_is_finalized: true,
                ..baseline
            },
            AsrRequestStaleInput {
                turn_route: Some(RecognitionRoute::from_model(
                    AsrModel::NemoParakeetTdt0_6BV2Int8,
                )),
                ..baseline
            },
        ] {
            assert!(is_stale_asr_request(&request, input));
        }
    }

    #[test]
    fn matching_nonempty_result_is_reduced_to_the_task_specific_action() {
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let request = request(route);
        let result = AsrResult {
            request_id: request.request_id,
            kind: request.kind,
            target: request.target.clone(),
            route,
            status: AsrResultStatus::Ok(AsrTranscript {
                text: "認識結果".to_owned(),
                tokens: Vec::new(),
            }),
            completed_at_frame: VadFrameIndex(2),
            elapsed_millis: 12,
        };

        assert!(matches!(
            reduce_asr_result(
                &result,
                &request,
                AsrResultReductionInput {
                    stale_input: AsrRequestStaleInput {
                        current_revision: 0,
                        confirmed_until_sample: GlobalSampleIndex(0),
                        target_turn_is_finalized: false,
                        turn_route: None,
                        last_recognition_route: Some(route),
                        default_route: route,
                        split_route: None,
                    },
                    completion_has_non_empty_draft: false,
                    completion_failure_action: AsrResultCompletionFailureAction::KeepOpen,
                    completion_rerecognition_purpose: None,
                    pending_rerecognition_purpose: None,
                }
            ),
            AsrResultAction::ApplyInterimTranscript {
                transcript,
                elapsed_millis: 12,
            } if transcript.text == "認識結果"
        ));
    }

    fn request(route: RecognitionRoute) -> AsrRequest {
        AsrRequest {
            request_id: AsrRequestId(1),
            kind: AsrTaskKind::InterimDisplay,
            target: AsrTarget::new(
                TurnId(1),
                TurnRevision(0),
                AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(10)),
                Some(SegmentId(1)),
                Some(SegmentId(1)),
            ),
            route,
            detected_language: None,
            audio: vec![0.0; 10],
            vad_results: Vec::new(),
            source_audio: vec![0.0; 10],
            source_vad_results: Vec::new(),
            close_reason: None,
            created_at_frame: VadFrameIndex(1),
        }
    }
}

#[cfg(test)]
#[path = "reducer_regression_tests.rs"]
mod regression_tests;
