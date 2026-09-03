use std::collections::{HashMap, HashSet, VecDeque};

use tauri::AppHandle;

use super::request::TranslationRequest;

type SourceSessionKey = parapper_stt_engine::SourceSessionKey;

pub(super) struct QueuedTranslationRequest {
    pub(super) handle: AppHandle,
    pub(super) request: TranslationRequest,
}

pub(super) struct TranslationQueueState {
    by_source: HashMap<SourceSessionKey, VecDeque<QueuedTranslationRequest>>,
    ready_sources: VecDeque<SourceSessionKey>,
    ready_set: HashSet<SourceSessionKey>,
    pub(super) worker_started: bool,
}

impl TranslationQueueState {
    pub(super) fn new() -> Self {
        Self {
            by_source: HashMap::new(),
            ready_sources: VecDeque::new(),
            ready_set: HashSet::new(),
            worker_started: false,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_source.is_empty()
    }

    pub(super) fn pop_next(&mut self) -> Option<QueuedTranslationRequest> {
        while let Some(source) = self.ready_sources.pop_front() {
            self.ready_set.remove(&source);
            let Some(queue) = self.by_source.get_mut(&source) else {
                continue;
            };
            let Some(item) = queue.pop_front() else {
                self.by_source.remove(&source);
                continue;
            };
            if queue.is_empty() {
                self.by_source.remove(&source);
            } else {
                self.ready_set.insert(source.clone());
                self.ready_sources.push_back(source);
            }
            return Some(item);
        }
        None
    }
}

pub(super) fn push_translation_request(
    state: &mut TranslationQueueState,
    handle: AppHandle,
    request: TranslationRequest,
) {
    let source = request.source_meta.source_session_key();
    let queue = state.by_source.entry(source.clone()).or_default();
    remove_stale_translation_jobs(queue, &request);
    queue.push_back(QueuedTranslationRequest { handle, request });
    if state.ready_set.insert(source.clone()) {
        state.ready_sources.push_back(source);
    }
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
    left.source_meta.source_session_key() == right.source_meta.source_session_key()
        && left.source_meta.turn_id == right.source_meta.turn_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{AsrLanguage, AsrModel, ParapperConfig, TranslationLanguage},
        delivery::{
            RecognitionSourceMeta,
            common::{TranslationProviderId, TranslationTarget},
        },
        recognition::events::RecognizedTextUpdateMode,
    };
    use tauri::AppHandle;

    fn source_meta(
        turn_session_id: u64,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
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
            delivery_route: ParapperConfig::default().legacy_delivery_route(),
            source_recognition_id: id.to_string(),
            source_meta,
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            source_text: id.to_string(),
            source_detected_language: None,
            targets: vec![TranslationTarget {
                provider_id: TranslationProviderId::Ync,
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            is_final,
            update_mode: RecognizedTextUpdateMode::Replace,
        }
    }

    fn test_handle() -> AppHandle {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("test app should build")
            .handle()
            .clone()
    }

    fn source_meta_for(
        source_id: &str,
        turn_id: u64,
        output_sequence: u64,
    ) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                source_id.into(),
                format!("Speaker {source_id}"),
                "capture-1".to_owned(),
                Some(0),
            ),
            turn_session_id: 1,
            turn_id,
            turn_revision: 0,
            output_sequence,
            segment_id: output_sequence,
            previous_segment_id: output_sequence.checked_sub(1),
        }
    }

    #[test]
    fn translation_queue_pop_next_preserves_source_fifo_and_ready_source_round_robin() {
        let handle = test_handle();
        let mut state = TranslationQueueState::new();
        for (source, id, turn_id) in [
            ("source-a", "A1", 1),
            ("source-a", "A2", 2),
            ("source-b", "B1", 1),
            ("source-b", "B2", 2),
        ] {
            let mut request = translation_request(id, turn_id, true);
            request.source_meta = source_meta_for(source, turn_id, turn_id);
            push_translation_request(&mut state, handle.clone(), request);
        }

        let ids = (0..4)
            .map(|_| {
                state
                    .pop_next()
                    .expect("queued translation request")
                    .request
                    .source_recognition_id
            })
            .collect::<Vec<_>>();

        assert_eq!(ids, ["A1", "B1", "A2", "B2"]);
        assert!(state.is_empty());
        assert!(state.pop_next().is_none());
    }

    #[test]
    fn translation_queue_stale_replacement_keeps_ready_bookkeeping_without_duplicate_source_turns()
    {
        let handle = test_handle();
        let mut state = TranslationQueueState::new();
        let mut interim = translation_request("interim", 1, false);
        interim.source_meta = source_meta_for("source-a", 1, 1);
        push_translation_request(&mut state, handle.clone(), interim);
        let mut other = translation_request("other", 1, true);
        other.source_meta = source_meta_for("source-b", 1, 1);
        push_translation_request(&mut state, handle.clone(), other);
        let mut final_request = translation_request("final", 1, true);
        final_request.source_meta = source_meta_for("source-a", 1, 2);
        push_translation_request(&mut state, handle, final_request);

        assert_eq!(
            state
                .pop_next()
                .expect("source A final should replace interim")
                .request
                .source_recognition_id,
            "final"
        );
        assert_eq!(
            state
                .pop_next()
                .expect("source B should remain ready")
                .request
                .source_recognition_id,
            "other"
        );
        assert!(state.is_empty());
        assert!(state.ready_sources.is_empty());
        assert!(state.ready_set.is_empty());
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

    #[test]
    fn translation_stale_decision_keeps_same_turn_from_different_source() {
        let queued = translation_request_with_source("source-a", source_meta(7, 1, 1), false);
        let next = translation_request_with_source(
            "source-b",
            RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                    "channel-2".into(),
                    "Speaker 2".to_string(),
                    "interface-1".to_string(),
                    Some(1),
                ),
                ..source_meta(7, 1, 2)
            },
            true,
        );

        assert!(!translation_job_is_stale(&queued, &next));
    }
}
