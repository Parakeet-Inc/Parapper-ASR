#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{
        SegmentCloseReason, SegmentId, TurnId,
        transcription::{
            route::RecognitionRoute,
            task::{
                AsrRequest, AsrRequestId, AsrResult, AsrResultStatus, AsrTarget, AsrTaskKind,
                AudioRange, GlobalSampleIndex, TurnRevision, VadFrameIndex,
            },
        },
    };
    use parapper_models::{
        asr::{AsrModel, AsrTranscript},
        vad::VadResult,
    };

    #[test]
    fn result_matching_requires_request_id_kind_target_and_route() {
        let request = asr_request(
            AsrTaskKind::CompletionCheck,
            RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
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
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let request = asr_request(AsrTaskKind::InterimDisplay, route, None, 0..10);

        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 1,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            }
        ));
        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(10),
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            }
        ));
        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: true,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            }
        ));
        assert!(is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: Some(RecognitionRoute::from_model(
                    AsrModel::NemoParakeetTdt0_6BV2Int8
                )),
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            }
        ));
    }

    #[test]
    fn stale_request_accepts_configured_split_route_over_existing_turn_route() {
        let interim_route = RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8);
        let final_route =
            RecognitionRoute::from_model(AsrModel::NemotronSpeechStreamingEn0_6B160MsInt8);
        let request = asr_request(AsrTaskKind::CompletionCheck, final_route, None, 0..10);

        assert!(!is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: Some(interim_route),
                last_recognition_route: Some(interim_route),
                default_route: RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
                split_route: Some(final_route),
            }
        ));
    }

    #[test]
    fn stale_request_accepts_configured_split_route_before_turn_route_exists() {
        let split_route =
            RecognitionRoute::from_model(AsrModel::Nemotron3_5AsrStreaming0_6B160MsInt8);
        let request = asr_request(AsrTaskKind::InterimDisplay, split_route, None, 0..10);

        assert!(!is_stale_asr_request(
            &request,
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: None,
                default_route: RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2),
                split_route: Some(split_route),
            }
        ));
    }

    #[test]
    fn stale_request_accepts_sli_selected_or_cached_non_default_route() {
        let route = RecognitionRoute::from_model(AsrModel::NemoParakeetTdt0_6BV2Int8);
        let default_route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);

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
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: None,
                default_route,
                split_route: None,
            }
        ));
        assert!(!is_stale_asr_request(
            &asr_request(AsrTaskKind::CompletionCheck, route, None, 0..10),
            AsrRequestStaleInput {
                current_revision: 0,
                confirmed_until_sample: GlobalSampleIndex(0),
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route,
                split_route: None,
            }
        ));
    }

    #[test]
    fn reduce_success_result_selects_action_from_request_kind() {
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
        let cases = [
            (
                asr_request(AsrTaskKind::InterimDisplay, route, None, 0..10),
                reduction_input_for(route),
                AsrResultAction::ApplyInterimTranscript {
                    transcript: AsrTranscript::from_text("hello"),
                    elapsed_millis: 1,
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
        let route = RecognitionRoute::from_model(AsrModel::ReazonSpeechK2V2);
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
                target_turn_is_finalized: false,
                turn_route: None,
                last_recognition_route: Some(route),
                default_route: route,
                split_route: None,
            },
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
