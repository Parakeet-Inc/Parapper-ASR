use std::collections::{HashMap, VecDeque};

use parapper_stt_engine::SourceSessionKey;
use tauri::AppHandle;

use super::request::QueuedSpeechRequest;

pub(super) struct QueuedTtsRequest {
    pub(super) handle: Option<AppHandle>,
    pub(super) request: QueuedSpeechRequest,
}

pub(super) struct TtsQueueState {
    pub(super) queue: SourceRoundRobinQueue<QueuedTtsRequest>,
    pub(super) worker_started: bool,
}

impl TtsQueueState {
    pub(super) fn new() -> Self {
        Self {
            queue: SourceRoundRobinQueue::new(),
            worker_started: false,
        }
    }
}

pub(super) fn push_tts_requests(
    state: &mut TtsQueueState,
    handle: Option<&AppHandle>,
    requests: Vec<QueuedSpeechRequest>,
) {
    for request in requests {
        remove_stale_tts_jobs(&mut state.queue, &request);
        let source = request.source_meta.source_session_key();
        state.queue.push(
            source,
            QueuedTtsRequest {
                handle: handle.cloned(),
                request,
            },
        );
    }
}

fn remove_stale_tts_jobs(
    queue: &mut SourceRoundRobinQueue<QueuedTtsRequest>,
    request: &QueuedSpeechRequest,
) {
    queue.retain(|queued| !tts_job_is_stale(&queued.request, request));
}

pub(super) fn tts_job_is_stale(queued: &QueuedSpeechRequest, next: &QueuedSpeechRequest) -> bool {
    same_tts_source(queued, next)
        && queued.source_kind == next.source_kind
        && queued.target_lang == next.target_lang
        && queued.id != next.id
}

fn same_tts_source(left: &QueuedSpeechRequest, right: &QueuedSpeechRequest) -> bool {
    left.source_meta.source_session_key() == right.source_meta.source_session_key()
        && left.source_meta.turn_id == right.source_meta.turn_id
}

/// A fair, source-scoped scheduler shared by external and local TTS queues.
/// Each source session keeps FIFO ordering; ready sessions take one item per
/// round so a long source backlog cannot monopolize a provider queue.
pub(super) struct SourceRoundRobinQueue<T> {
    by_source: HashMap<SourceSessionKey, VecDeque<T>>,
    ready_sources: VecDeque<SourceSessionKey>,
}

impl<T> SourceRoundRobinQueue<T> {
    pub(super) fn new() -> Self {
        Self {
            by_source: HashMap::new(),
            ready_sources: VecDeque::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ready_sources.is_empty()
    }

    pub(super) fn push(&mut self, source: SourceSessionKey, item: T) {
        let queue = self.by_source.entry(source.clone()).or_default();
        if queue.is_empty() {
            self.ready_sources.push_back(source);
        }
        queue.push_back(item);
    }

    pub(super) fn pop(&mut self) -> Option<T> {
        let source = self.next_ready_source()?;
        let (item, empty) = {
            let queue = self.by_source.get_mut(&source)?;
            let item = queue.pop_front()?;
            (item, queue.is_empty())
        };
        if empty {
            self.by_source.remove(&source);
        } else {
            self.ready_sources.push_back(source);
        }
        Some(item)
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.by_source.retain(|_, queue| {
            queue.retain(|item| keep(item));
            !queue.is_empty()
        });
        self.ready_sources
            .retain(|source| self.by_source.contains_key(source));
    }

    fn next_ready_source(&mut self) -> Option<SourceSessionKey> {
        let selected = self.ready_sources.pop_front()?;
        let older_session = self
            .ready_sources
            .iter()
            .position(|candidate| {
                candidate.source_id == selected.source_id
                    && candidate.turn_session_id < selected.turn_session_id
            })
            .and_then(|index| self.ready_sources.remove(index));
        if let Some(older_session) = older_session {
            // A source restart must not overtake its still-pending prior
            // session, while unrelated sources remain independently fair.
            self.ready_sources.push_front(selected);
            Some(older_session)
        } else {
            Some(selected)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{LocalTtsVoice, SpeechBackend, SpeechSourceKind},
        delivery::RecognitionSourceMeta,
    };

    fn source_meta(turn_id: u64, output_sequence: u64) -> RecognitionSourceMeta {
        RecognitionSourceMeta {
            identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
            turn_session_id: 1,
            turn_id,
            turn_revision: 0,
            output_sequence,
            segment_id: output_sequence,
            previous_segment_id: output_sequence.checked_sub(1),
        }
    }

    fn local_tts_request_with_voice(voice: LocalTtsVoice) -> QueuedSpeechRequest {
        QueuedSpeechRequest {
            port: 0,
            id: "speech-test".to_string(),
            source_event_id: "turn-1-1-0".to_string(),
            source_meta: source_meta(1, 1),
            source_kind: SpeechSourceKind::Recognition,
            target_lang: None,
            text: "test".to_string(),
            backend: SpeechBackend::LocalTts,
            talker: String::new(),
            local_tts_voice: Some(voice),
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }
    }

    fn request_for_source(
        id: &str,
        source_id: &str,
        turn_session_id: u64,
        output_sequence: u64,
    ) -> QueuedSpeechRequest {
        let mut request = local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx);
        request.id = id.to_owned();
        request.source_event_id = id.to_owned();
        request.source_meta = RecognitionSourceMeta {
            identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                source_id.into(),
                source_id.to_owned(),
                "interface-1".to_owned(),
                None,
            ),
            turn_session_id,
            turn_id: output_sequence,
            turn_revision: 0,
            output_sequence,
            segment_id: output_sequence,
            previous_segment_id: output_sequence.checked_sub(1),
        };
        request
    }

    #[test]
    fn external_tts_queue_round_robins_ready_sources_without_reordering_each_source_fifo() {
        let mut state = TtsQueueState::new();
        push_tts_requests(
            &mut state,
            None,
            vec![
                request_for_source("A1", "source-a", 10, 1),
                request_for_source("A2", "source-a", 10, 2),
                request_for_source("B1", "source-b", 20, 1),
                request_for_source("B2", "source-b", 20, 2),
            ],
        );

        let scheduled = (0..4)
            .map(|_| state.queue.pop().expect("queued TTS request").request.id)
            .collect::<Vec<_>>();

        assert_eq!(scheduled, vec!["A1", "B1", "A2", "B2"]);
    }

    #[test]
    fn tts_stale_decision_table() {
        struct TtsStaleCase {
            name: &'static str,
            queued: QueuedSpeechRequest,
            next: QueuedSpeechRequest,
            expected: bool,
        }

        let base = local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx);
        let cases = [
            TtsStaleCase {
                name: "same turn kind and target replaces older request",
                queued: QueuedSpeechRequest {
                    id: "speech-old".to_string(),
                    ..base.clone()
                },
                next: QueuedSpeechRequest {
                    id: "speech-new".to_string(),
                    ..base.clone()
                },
                expected: true,
            },
            TtsStaleCase {
                name: "same id is not stale because it is the same request",
                queued: base.clone(),
                next: base.clone(),
                expected: false,
            },
            TtsStaleCase {
                name: "different turn is not stale",
                queued: QueuedSpeechRequest {
                    id: "speech-old".to_string(),
                    ..base.clone()
                },
                next: QueuedSpeechRequest {
                    id: "speech-new".to_string(),
                    source_meta: source_meta(2, 1),
                    ..base.clone()
                },
                expected: false,
            },
            TtsStaleCase {
                name: "different source kind is not stale",
                queued: QueuedSpeechRequest {
                    id: "speech-old".to_string(),
                    ..base.clone()
                },
                next: QueuedSpeechRequest {
                    id: "speech-new".to_string(),
                    source_kind: SpeechSourceKind::Translation,
                    target_lang: Some("en_US".to_string()),
                    ..base.clone()
                },
                expected: false,
            },
            TtsStaleCase {
                name: "different translation target is not stale",
                queued: QueuedSpeechRequest {
                    id: "speech-old".to_string(),
                    source_kind: SpeechSourceKind::Translation,
                    target_lang: Some("en_US".to_string()),
                    ..base.clone()
                },
                next: QueuedSpeechRequest {
                    id: "speech-new".to_string(),
                    source_kind: SpeechSourceKind::Translation,
                    target_lang: Some("fr_FR".to_string()),
                    ..base.clone()
                },
                expected: false,
            },
        ];

        for case in cases {
            assert_eq!(
                tts_job_is_stale(&case.queued, &case.next),
                case.expected,
                "case={}",
                case.name
            );
        }
    }

    #[test]
    fn tts_queue_replaces_pending_request_for_same_structured_source() {
        let mut queue = SourceRoundRobinQueue::new();
        let old_request = QueuedSpeechRequest {
            id: "speech-turn-1-old".to_string(),
            source_event_id: "turn-1".to_string(),
            ..local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx)
        };
        queue.push(
            old_request.source_meta.source_session_key(),
            QueuedTtsRequest {
                handle: None,
                request: old_request,
            },
        );

        remove_stale_tts_jobs(
            &mut queue,
            &QueuedSpeechRequest {
                id: "speech-turn-1-new".to_string(),
                source_event_id: "turn-1".to_string(),
                ..local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx)
            },
        );

        assert!(queue.is_empty());
    }

    #[test]
    fn tts_stale_decision_uses_structured_turn_identity_not_event_id_revision() {
        let mut queue = SourceRoundRobinQueue::new();
        let old_request = QueuedSpeechRequest {
            id: "speech-turn-old".to_string(),
            source_event_id: "turn-1-1-0".to_string(),
            source_meta: source_meta(10, 1),
            ..local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx)
        };
        queue.push(
            old_request.source_meta.source_session_key(),
            QueuedTtsRequest {
                handle: None,
                request: old_request,
            },
        );

        remove_stale_tts_jobs(
            &mut queue,
            &QueuedSpeechRequest {
                id: "speech-turn-new".to_string(),
                source_event_id: "turn-1-1-1".to_string(),
                source_meta: source_meta(10, 2),
                ..local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx)
            },
        );

        assert!(queue.is_empty());
    }

    #[test]
    fn tts_stale_decision_keeps_same_turn_from_different_source() {
        let queued = local_tts_request_with_voice(LocalTtsVoice::Supertonic2Onnx);
        let next = QueuedSpeechRequest {
            id: "speech-source-b".to_string(),
            source_meta: RecognitionSourceMeta {
                identity: parapper_stt_engine::SourceIdentitySnapshot::new(
                    "channel-2".into(),
                    "Speaker 2".to_string(),
                    "interface-1".to_string(),
                    Some(1),
                ),
                ..source_meta(1, 2)
            },
            ..queued.clone()
        };

        assert!(!tts_job_is_stale(&queued, &next));
    }
}
