use super::TurnDraft;

#[test]
fn new_turn_draft_starts_empty() {
    let draft = TurnDraft::new("turn-1".to_string(), 0);

    assert_eq!(draft.event_id, "turn-1");
    assert!(draft.combined_text.is_empty());
    assert!(draft.full_audio.is_empty());
}

#[test]
fn replace_with_full_turn_transcription_keeps_full_audio_and_replaces_text() {
    let route = crate::recognition::route::RecognitionRoute::from_language(
        crate::config::AsrLanguage::Japanese,
    );
    let mut draft = TurnDraft::new("turn-1".to_string(), 0);
    draft.append_recognized_segment(1, None, &[1.0, 2.0], route, "今日は".to_string(), 10);
    draft.append_recognized_segment(2, Some(1), &[3.0], route, "晴れです".to_string(), 5);

    draft.replace_with_full_turn_transcription(route, "今日は晴れです".to_string(), 20);

    assert_eq!(draft.full_audio, vec![1.0, 2.0, 3.0]);
    assert_eq!(draft.segment_texts, vec!["今日は晴れです"]);
    assert_eq!(draft.combined_text, "今日は晴れです");
    assert_eq!(draft.processing_millis, 35);
}
