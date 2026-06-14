use super::{GrammarBoundaryClass, TurnBoundaryCandidate, TurnDraft};
use crate::{config::AsrLanguage, recognition::transcription::route::RecognitionRoute};

#[test]
fn new_turn_draft_starts_empty() {
    let draft = TurnDraft::new("turn-1".to_string(), 0);

    assert_eq!(draft.event_id, "turn-1");
    assert!(draft.combined_text.is_empty());
    assert!(draft.full_audio.is_empty());
}

#[test]
fn replace_with_full_turn_transcription_keeps_full_audio_and_replaces_text() {
    let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
    let mut draft = TurnDraft::new("turn-1".to_string(), 0);
    draft.append_recognized_segment(1, None, &[1.0, 2.0], &[], route, "今日は".to_string(), 10);
    draft.append_recognized_segment(2, Some(1), &[3.0], &[], route, "晴れです".to_string(), 5);

    draft.replace_with_full_turn_transcription(route, "今日は晴れです".to_string(), 20);

    assert_eq!(draft.full_audio, vec![1.0, 2.0, 3.0]);
    assert_eq!(draft.segment_texts, vec!["今日は晴れです"]);
    assert_eq!(draft.combined_text, "今日は晴れです");
    assert_eq!(draft.processing_millis, 35);
}

#[test]
fn spans_multiple_source_segments_only_changes_when_segment_ids_differ() {
    let route = RecognitionRoute::from_language(AsrLanguage::Japanese);
    let mut draft = TurnDraft::new("turn-1".to_string(), 0);
    assert!(!draft.spans_multiple_source_segments());

    draft.append_recognized_segment(1, None, &[1.0], &[], route, "同じ".to_string(), 0);
    draft.append_recognized_segment(1, Some(1), &[2.0], &[], route, "ターン".to_string(), 0);
    assert!(!draft.spans_multiple_source_segments());

    draft.append_recognized_segment(2, Some(1), &[3.0], &[], route, "続き".to_string(), 0);
    assert!(draft.spans_multiple_source_segments());
}

#[test]
fn boundary_candidate_offset_moves_text_and_audio_coordinates_together() {
    let candidate = TurnBoundaryCandidate {
        char_end: 4,
        sample_end: 160,
        prefix_audio_end: 120,
        suffix_audio_start: 180,
        class: GrammarBoundaryClass::NormalEnd,
    };

    assert_eq!(
        candidate.offset_by(10, 1_000),
        TurnBoundaryCandidate {
            char_end: 14,
            sample_end: 1_160,
            prefix_audio_end: 1_120,
            suffix_audio_start: 1_180,
            class: GrammarBoundaryClass::NormalEnd,
        }
    );
}
