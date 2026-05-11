use std::collections::VecDeque;

use tauri::AppHandle;

use super::request::TranslationRequest;

pub(super) struct QueuedTranslationRequest {
    pub(super) handle: AppHandle,
    pub(super) request: TranslationRequest,
}

pub(super) struct TranslationQueueState {
    pub(super) queue: VecDeque<QueuedTranslationRequest>,
    pub(super) worker_started: bool,
}

impl TranslationQueueState {
    pub(super) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            worker_started: false,
        }
    }
}

pub(super) fn push_translation_request(
    state: &mut TranslationQueueState,
    handle: AppHandle,
    request: TranslationRequest,
) {
    remove_stale_translation_jobs(&mut state.queue, &request);
    state
        .queue
        .push_back(QueuedTranslationRequest { handle, request });
}

fn remove_stale_translation_jobs(
    queue: &mut VecDeque<QueuedTranslationRequest>,
    request: &TranslationRequest,
) {
    queue.retain(|queued| !translation_job_is_stale(&queued.request, request));
}

fn translation_job_is_stale(queued: &TranslationRequest, next: &TranslationRequest) -> bool {
    same_translation_source(queued, next) && (next.is_final || !queued.is_final)
}

fn same_translation_source(left: &TranslationRequest, right: &TranslationRequest) -> bool {
    left.source_meta.turn_session_id == right.source_meta.turn_session_id
        && left.source_meta.turn_id == right.source_meta.turn_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{AsrModel, ParapperConfig},
        delivery::RecognitionSourceMeta,
        recognition::events::RecognizedTextUpdateMode,
    };

    fn source_meta(
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            turn_session_id,
            turn_id,
            turn_revision: 0,
            output_sequence,
            segment_id: output_sequence,
            previous_segment_id: output_sequence.checked_sub(1),
        }
    }

    fn translation_request(id: &str, turn_id: u64, is_final: bool) -> TranslationRequest {
        translation_request_with_source(id, source_meta(1, turn_id, 1), is_final)
    }

    fn translation_request_with_source(
        id: &str,
        source_meta: RecognitionSourceMeta,
        is_final: bool,
    ) -> TranslationRequest {
        TranslationRequest {
            config: ParapperConfig::default(),
            source_recognition_id: id.to_string(),
            source_meta,
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_text: id.to_string(),
            source_detected_language: None,
            targets: vec!["en_US".to_string()],
            is_final,
            update_mode: RecognizedTextUpdateMode::Replace,
        }
    }

    #[test]
    fn translation_stale_decision_table() {
        let cases = [
            (
                "interim replaces same-turn interim",
                1,
                false,
                1,
                false,
                true,
            ),
            ("final replaces same-turn interim", 1, false, 1, true, true),
            (
                "interim does not replace same-turn final",
                1,
                true,
                1,
                false,
                false,
            ),
            ("final replaces same-turn final", 1, true, 1, true, true),
            (
                "final does not replace another turn",
                1,
                false,
                2,
                true,
                false,
            ),
        ];

        for (name, queued_turn, queued_final, next_turn, next_final, expected) in cases {
            let queued = translation_request("queued", queued_turn, queued_final);
            let next = translation_request("next", next_turn, next_final);

            assert_eq!(
                translation_job_is_stale(&queued, &next),
                expected,
                "case={name}"
            );
        }
    }

    #[test]
    fn translation_stale_decision_uses_structured_turn_identity_not_event_id_revision() {
        let queued = translation_request_with_source("turn-1-1-0", source_meta(7, 1, 1), false);
        let next = translation_request_with_source("turn-1-1-1", source_meta(7, 1, 2), true);
        let different_session =
            translation_request_with_source("turn-8-1-0", source_meta(8, 1, 2), true);

        assert!(translation_job_is_stale(&queued, &next));
        assert!(!translation_job_is_stale(&queued, &different_session));
    }
}
