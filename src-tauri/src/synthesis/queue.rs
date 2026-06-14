use std::collections::VecDeque;

use tauri::AppHandle;

use super::request::QueuedSpeechRequest;

pub(super) struct QueuedTtsRequest {
    pub(super) handle: Option<AppHandle>,
    pub(super) request: QueuedSpeechRequest,
}

pub(super) struct TtsQueueState {
    pub(super) queue: VecDeque<QueuedTtsRequest>,
    pub(super) worker_started: bool,
}

impl TtsQueueState {
    pub(super) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
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
        state.queue.push_back(QueuedTtsRequest {
            handle: handle.cloned(),
            request,
        });
    }
}

fn remove_stale_tts_jobs(queue: &mut VecDeque<QueuedTtsRequest>, request: &QueuedSpeechRequest) {
    queue.retain(|queued| !tts_job_is_stale(&queued.request, request));
}

fn tts_job_is_stale(queued: &QueuedSpeechRequest, next: &QueuedSpeechRequest) -> bool {
    same_tts_source(queued, next)
        && queued.source_kind == next.source_kind
        && queued.target_lang == next.target_lang
        && queued.id != next.id
}

fn same_tts_source(left: &QueuedSpeechRequest, right: &QueuedSpeechRequest) -> bool {
    left.source_meta.turn_session_id == right.source_meta.turn_session_id
        && left.source_meta.turn_id == right.source_meta.turn_id
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SpeechOrderKey {
    turn_session_id: u64,
    output_sequence: u64,
}

pub(super) fn speech_order_key_for_request(request: &QueuedSpeechRequest) -> SpeechOrderKey {
    SpeechOrderKey {
        turn_session_id: request.source_meta.turn_session_id,
        output_sequence: request.source_meta.output_sequence,
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

    #[test]
    fn tts_stale_decision_table() {
        struct TtsStaleCase {
            name: &'static str,
            queued: QueuedSpeechRequest,
            next: QueuedSpeechRequest,
            expected: bool,
        }

        let base = local_tts_request_with_voice(LocalTtsVoice::Kristin);
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
        let mut queue = VecDeque::new();
        queue.push_back(QueuedTtsRequest {
            handle: None,
            request: QueuedSpeechRequest {
                id: "speech-turn-1-old".to_string(),
                source_event_id: "turn-1".to_string(),
                ..local_tts_request_with_voice(LocalTtsVoice::Kristin)
            },
        });

        remove_stale_tts_jobs(
            &mut queue,
            &QueuedSpeechRequest {
                id: "speech-turn-1-new".to_string(),
                source_event_id: "turn-1".to_string(),
                ..local_tts_request_with_voice(LocalTtsVoice::Kristin)
            },
        );

        assert!(queue.is_empty());
    }

    #[test]
    fn tts_stale_decision_uses_structured_turn_identity_not_event_id_revision() {
        let mut queue = VecDeque::new();
        queue.push_back(QueuedTtsRequest {
            handle: None,
            request: QueuedSpeechRequest {
                id: "speech-turn-old".to_string(),
                source_event_id: "turn-1-1-0".to_string(),
                source_meta: source_meta(10, 1),
                ..local_tts_request_with_voice(LocalTtsVoice::Kristin)
            },
        });

        remove_stale_tts_jobs(
            &mut queue,
            &QueuedSpeechRequest {
                id: "speech-turn-new".to_string(),
                source_event_id: "turn-1-1-1".to_string(),
                source_meta: source_meta(10, 2),
                ..local_tts_request_with_voice(LocalTtsVoice::Kristin)
            },
        );

        assert!(queue.is_empty());
    }
}
