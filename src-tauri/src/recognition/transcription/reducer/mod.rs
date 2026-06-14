use crate::recognition::transcription::{
    asr::{
        engine::AsrTranscript,
        task::{
            AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AsrTaskKind, GlobalSampleIndex,
        },
    },
    route::RecognitionRoute,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) enum AsrResultRerecognitionPurpose {
    GrammarAfterCompletion,
    SimpleTurnCheckFinal,
    TimeoutFinal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) enum AsrResultCompletionAfterTranscript {
    RerecognizeIfIdle(AsrResultRerecognitionPurpose),
    CompleteWithoutGrammar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::recognition) enum AsrResultCompletionFailureAction {
    DecideWithNamo,
    CompleteWithoutGrammar,
    KeepOpen,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::recognition) enum AsrResultAction {
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
        emit_interim: bool,
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
pub(in crate::recognition) struct AsrRequestStaleInput {
    pub(in crate::recognition) current_revision: u64,
    pub(in crate::recognition) confirmed_until_sample: GlobalSampleIndex,
    pub(in crate::recognition) turn_route: Option<RecognitionRoute>,
    pub(in crate::recognition) last_recognition_route: Option<RecognitionRoute>,
    pub(in crate::recognition) default_route: RecognitionRoute,
}

#[derive(Clone, Copy)]
pub(in crate::recognition) struct AsrResultReductionInput {
    pub(in crate::recognition) stale_input: AsrRequestStaleInput,
    pub(in crate::recognition) interim_is_covered_by_completion: bool,
    pub(in crate::recognition) completion_has_non_empty_draft: bool,
    pub(in crate::recognition) completion_failure_action: AsrResultCompletionFailureAction,
    pub(in crate::recognition) completion_rerecognition_purpose:
        Option<AsrResultRerecognitionPurpose>,
    pub(in crate::recognition) pending_rerecognition_purpose: Option<AsrResultRerecognitionPurpose>,
}

pub(in crate::recognition) fn result_matches_in_flight_request(
    result: &AsrResult,
    request: &AsrRequest,
) -> bool {
    result.request_id == request.request_id
        && result.kind == request.kind
        && result.target == request.target
        && result.route == request.route
}

pub(in crate::recognition) fn reduce_asr_result(
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
            emit_interim: !input.interim_is_covered_by_completion,
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

fn unusable_result_action(request: &AsrRequest, input: AsrResultReductionInput) -> AsrResultAction {
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

pub(in crate::recognition) fn is_stale_asr_request(
    request: &AsrRequest,
    input: AsrRequestStaleInput,
) -> bool {
    if input.current_revision != request.target.turn_revision.0 {
        return true;
    }
    if request.target.range.end_sample <= input.confirmed_until_sample {
        return true;
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
    use super::*;
    use crate::{
        config::{AsrModel, ParapperConfig},
        recognition::{
            segmentation::{segment::builder::SegmentCloseReason, vad::engine::VadResult},
            transcription::{
                asr::{
                    engine::AsrTranscript,
                    task::{
                        AsrRequestId, AsrResultStatus, AsrTarget, AsrTaskKind, AudioRange,
                        SegmentId, TurnId, TurnRevision, VadFrameIndex,
                    },
                },
                route::RecognitionRoute,
            },
        },
    };

    #[test]
    fn result_matching_requires_request_id_kind_target_and_route() {
        let request = asr_request(
            AsrTaskKind::CompletionCheck,
            RecognitionRoute::from_model(ParapperConfig::default().asr.model),
            None,
            0..10,
        );
        assert!(result_matches_in_flight_request(
            &asr_result_from_request(&request),
            &request
        ));

        let mut wrong_kind = asr_result_from_request(&request);
        wrong_kind.kind = AsrTaskKind::InterimDisplay;
        let mut wrong_target = asr_result_from_request(&request);
        wrong_target.target = AsrTarget::new(
            TurnId(2),
            TurnRevision(0),
            AudioRange::new(GlobalSampleIndex(0), GlobalSampleIndex(10)),
            Some(SegmentId(1)),
            Some(SegmentId(1)),
        );
        let mut wrong_route = asr_result_from_request(&request);
        wrong_route.route = RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8);

        for result in [wrong_kind, wrong_target, wrong_route] {
            assert!(
                !result_matches_in_flight_request(&result, &request),
                "matching request_id alone must not apply a result when kind, target, or route differs"
            );
        }
    }

    #[test]
    fn stale_request_rejects_revision_range_and_route_mismatch() {
        let route = RecognitionRoute::from_model(ParapperConfig::default().asr.model);
        let request = asr_request(AsrTaskKind::InterimDisplay, route, None, 0..10);

        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 1,
                confirmed_until_sample: GlobalSampleIndex(0),
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
            }
        ));
        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(10),
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
            }
        ));
        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                turn_route: Some(RecognitionRoute::from_model(
                    AsrModel::NemoParakeetTdt0_6BV2Int8
                )),
                last_recognition_route: Some(route),
                default_route: route,
            }
        ));
    }

    #[test]
    fn stale_request_accepts_sli_selected_or_cached_non_default_route() {
        let route = RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8);
        let default_route = RecognitionRoute::from_model(ParapperConfig::default().asr.model);

        assert!(!is_stale_asr_request(
            &asr_request(
                AsrTaskKind::CompletionCheck,
                route,
                Some("en".to_string()),
                0..10
            ),
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                turn_route: None,
                last_recognition_route: None,
                default_route,
            }
        ));
        assert!(!is_stale_asr_request(
            &asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                turn_route: None,
                last_recognition_route: Some(route),
                default_route,
            }
        ));
    }

    #[test]
    fn reduce_success_result_selects_action_from_request_kind() {
        let route = RecognitionRoute::from_model(ParapperConfig::default().asr.model);
        let cases = [
            (
                asr_request(AsrTaskKind::InterimDisplay, route, None, 0..10),
                reduction_input_for(route),
                AsrResultAction::ApplyInterimTranscript {
                    transcript: AsrTranscript::from_text("hello"),
                    elapsed_millis: 1,
                    emit_interim: true,
                },
            ),
            (
                asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
                AsrResultReductionInput {
                    completion_rerecognition_purpose: Some(
                        AsrResultRerecognitionPurpose::SimpleTurnCheckFinal,
                    ),
                    ..reduction_input_for(route)
                },
                AsrResultAction::ApplyCompletionTranscript {
                    transcript: AsrTranscript::from_text("hello"),
                    elapsed_millis: 1,
                    after_transcript: AsrResultCompletionAfterTranscript::RerecognizeIfIdle(
                        AsrResultRerecognitionPurpose::SimpleTurnCheckFinal,
                    ),
                },
            ),
            (
                asr_request(AsrTaskKind::Rerecognition, route, None, 0..10),
                AsrResultReductionInput {
                    pending_rerecognition_purpose: Some(
                        AsrResultRerecognitionPurpose::TimeoutFinal,
                    ),
                    ..reduction_input_for(route)
                },
                AsrResultAction::ApplyRerecognitionTranscript {
                    transcript: AsrTranscript::from_text("hello"),
                    elapsed_millis: 1,
                    purpose: AsrResultRerecognitionPurpose::TimeoutFinal,
                },
            ),
        ];

        for (request, input, expected) in cases {
            assert_eq!(
                reduce_asr_result(&asr_result_from_request(&request), &request, input),
                expected
            );
        }
    }

    #[test]
    fn reduce_unusable_result_selects_fallback_action_from_request_kind_and_runtime_state() {
        let route = RecognitionRoute::from_model(ParapperConfig::default().asr.model);
        let cases = [
            (
                asr_request(AsrTaskKind::InterimDisplay, route, None, 0..10),
                AsrResultStatus::Failed("failed".to_string()),
                reduction_input_for(route),
                AsrResultAction::DropUnusableInterim,
            ),
            (
                asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
                AsrResultStatus::Failed("failed".to_string()),
                reduction_input_for(route),
                AsrResultAction::DropUnusableCompletionWithoutDraft,
            ),
            (
                asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
                AsrResultStatus::Ok(AsrTranscript::from_text("   ")),
                AsrResultReductionInput {
                    completion_has_non_empty_draft: true,
                    ..reduction_input_for(route)
                },
                AsrResultAction::FallbackCompletionWithoutGrammar { turn_id: 1 },
            ),
            (
                asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
                AsrResultStatus::Failed("namo fallback".to_string()),
                AsrResultReductionInput {
                    completion_has_non_empty_draft: true,
                    completion_failure_action: AsrResultCompletionFailureAction::DecideWithNamo,
                    ..reduction_input_for(route)
                },
                AsrResultAction::FallbackCompletionWithNamo { turn_id: 1 },
            ),
            (
                asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
                AsrResultStatus::Failed("morph fallback".to_string()),
                AsrResultReductionInput {
                    completion_has_non_empty_draft: true,
                    completion_failure_action: AsrResultCompletionFailureAction::KeepOpen,
                    ..reduction_input_for(route)
                },
                AsrResultAction::FallbackCompletionKeepOpen { turn_id: 1 },
            ),
            (
                asr_request(AsrTaskKind::Rerecognition, route, None, 0..10),
                AsrResultStatus::Failed("failed".to_string()),
                AsrResultReductionInput {
                    pending_rerecognition_purpose: Some(
                        AsrResultRerecognitionPurpose::TimeoutFinal,
                    ),
                    ..reduction_input_for(route)
                },
                AsrResultAction::FallbackRerecognition {
                    turn_id: 1,
                    purpose: AsrResultRerecognitionPurpose::TimeoutFinal,
                },
            ),
        ];

        for (request, status, input, expected) in cases {
            let mut result = asr_result_from_request(&request);
            result.status = status;

            assert_eq!(reduce_asr_result(&result, &request, input), expected);
        }
    }

    fn asr_result_from_request(request: &AsrRequest) -> AsrResult {
        AsrResult {
            request_id: request.request_id,
            kind: request.kind,
            target: request.target.clone(),
            route: request.route,
            status: AsrResultStatus::Ok(AsrTranscript {
                text: "hello".to_string(),
                tokens: Vec::new(),
            }),
            completed_at_frame: VadFrameIndex(2),
            elapsed_millis: 1,
        }
    }

    fn reduction_input_for(route: RecognitionRoute) -> AsrResultReductionInput {
        AsrResultReductionInput {
            stale_input: AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
            },
            interim_is_covered_by_completion: false,
            completion_has_non_empty_draft: false,
            completion_failure_action: AsrResultCompletionFailureAction::CompleteWithoutGrammar,
            completion_rerecognition_purpose: None,
            pending_rerecognition_purpose: None,
        }
    }

    fn asr_request(
        kind: AsrTaskKind,
        route: RecognitionRoute,
        detected_language: Option<String>,
        range: std::ops::Range<u64>,
    ) -> AsrRequest {
        let audio = vec![1.0; usize::try_from(range.end - range.start).unwrap()];
        let vad_results = vec![VadResult {
            probability: 0.9,
            is_speech: true,
        }];
        AsrRequest {
            request_id: AsrRequestId(1),
            kind,
            target: AsrTarget::new(
                TurnId(1),
                TurnRevision(0),
                AudioRange::new(GlobalSampleIndex(range.start), GlobalSampleIndex(range.end)),
                Some(SegmentId(1)),
                Some(SegmentId(1)),
            ),
            route,
            detected_language,
            source_audio: audio.clone(),
            source_vad_results: vad_results.clone(),
            audio,
            vad_results,
            close_reason: Some(SegmentCloseReason::EndSilenceReached),
            created_at_frame: VadFrameIndex(1),
        }
    }
}
