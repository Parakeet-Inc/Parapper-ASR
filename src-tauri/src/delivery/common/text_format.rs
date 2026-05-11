use crate::config::AsrLanguage;

pub(crate) fn join_turn_segments(segments: &[String], language: AsrLanguage) -> String {
    let mut text = String::new();
    for segment in segments
        .iter()
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
    {
        if text.is_empty() {
            text.push_str(segment);
            continue;
        }
        if language == AsrLanguage::Japanese {
            text = trim_japanese_sentence_end(&text).to_string();
            text.push_str(segment);
        } else {
            text.push(' ');
            text.push_str(segment);
        }
    }
    text
}

pub(crate) fn continuing_turn_text(text: &str) -> String {
    let text = trim_continuation_marker(text.trim());
    if text.is_empty() {
        return String::new();
    }
    format!("{text}...")
}

pub(crate) fn finalize_turn_text(text: &str, language: AsrLanguage) -> String {
    let text = trim_continuation_marker(text.trim());
    if language != AsrLanguage::Japanese || text.is_empty() || has_japanese_sentence_end(text) {
        return text.to_string();
    }
    format!("{text}。")
}

fn trim_japanese_sentence_end(text: &str) -> &str {
    text.trim_end_matches(['。', '、', '！', '？'])
}

pub(crate) fn trim_continuation_marker(text: &str) -> &str {
    text.trim_end_matches("...")
}

fn has_japanese_sentence_end(text: &str) -> bool {
    text.chars()
        .last()
        .is_some_and(|character| matches!(character, '。' | '！' | '？'))
}
