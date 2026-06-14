use super::{
    audio_window::{BoundaryAudioWindow, audio_window_for_boundary},
    candidates_for_transcript,
    japanese::{JapaneseMorphToken, has_pos1, is_nominal_suffix, japanese_morph_candidates},
    sample_end_for_char_end, seconds_to_sample, slice_chars,
};
use crate::{
    config::AsrLanguage,
    recognition::{
        segmentation::vad::engine::VadResult,
        transcription::asr::engine::{AsrToken, AsrTranscript},
        turn::GrammarBoundaryClass,
    },
};

#[test]
fn english_period_question_and_exclamation_create_strong_end_candidates_but_comma_does_not() {
    let transcript = AsrTranscript::from_parts(
        "Hello, wait. Really?".to_string(),
        vec![
            " Hello".to_string(),
            ",".to_string(),
            " wait".to_string(),
            ".".to_string(),
            " Really".to_string(),
            "?".to_string(),
        ],
        Some(&[0.0, 0.2, 0.4, 0.6, 1.0, 1.4]),
        None,
    );

    let candidates = candidates_for_transcript(
        AsrLanguage::English,
        &transcript,
        &vec![0.0; 32_000],
        &vads(&[true, true]),
        None,
    );

    assert_eq!(candidates.len(), 2);
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.class)
            .collect::<Vec<_>>(),
        vec![
            GrammarBoundaryClass::StrongEnd,
            GrammarBoundaryClass::StrongEnd,
        ]
    );
    assert_eq!(
        slice_chars(&transcript.text, 0..candidates[0].char_end),
        "Hello, wait."
    );
}

#[test]
fn english_punctuation_inside_word_tokens_still_creates_candidates() {
    let transcript = AsrTranscript::from_parts(
        "Hello. Really?".to_string(),
        vec![" Hello.".to_string(), " Really?".to_string()],
        Some(&[0.0, 1.0]),
        Some(&[0.5, 0.5]),
    );

    let candidates = candidates_for_transcript(
        AsrLanguage::English,
        &transcript,
        &vec![0.0; 32_000],
        &vads(&[true, true]),
        None,
    );

    assert_eq!(
        candidates
            .iter()
            .map(|candidate| slice_chars(&transcript.text, 0..candidate.char_end))
            .collect::<Vec<_>>(),
        vec!["Hello.".to_string(), "Hello. Really?".to_string()]
    );
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.class)
            .collect::<Vec<_>>(),
        vec![
            GrammarBoundaryClass::StrongEnd,
            GrammarBoundaryClass::StrongEnd,
        ]
    );
}

#[test]
fn boundary_candidates_require_every_visible_token_to_have_start_timestamp() {
    let transcript = AsrTranscript {
        text: "Hello.".to_string(),
        tokens: vec![
            AsrToken {
                text: "Hello".to_string(),
                start_sec: Some(0.0),
                duration_sec: Some(0.2),
                char_range: Some(0..5),
            },
            AsrToken {
                text: ".".to_string(),
                start_sec: None,
                duration_sec: Some(0.1),
                char_range: Some(5..6),
            },
        ],
    };

    let candidates = candidates_for_transcript(
        AsrLanguage::English,
        &transcript,
        &vec![0.0; 16_000],
        &vads(&[true]),
        None,
    );

    assert!(
        candidates.is_empty(),
        "punctuation boundaries must not be inferred when a visible token lacks alignment"
    );
}

#[test]
fn sample_end_uses_next_token_start_when_duration_is_missing_or_non_positive() {
    let missing_duration = AsrTranscript::from_parts(
        "Hello.",
        vec!["Hello".to_string(), ".".to_string()],
        Some(&[0.1, 0.4]),
        None,
    );
    assert_eq!(
        sample_end_for_char_end(&missing_duration, 5, 16_000),
        Some(6_400),
        "a missing duration should fall back to the next token start"
    );

    let zero_duration = AsrTranscript::from_parts(
        "Hello.",
        vec!["Hello".to_string(), ".".to_string()],
        Some(&[0.1, 0.4]),
        Some(&[0.0, 0.2]),
    );
    assert_eq!(
        sample_end_for_char_end(&zero_duration, 5, 16_000),
        Some(6_400),
        "a non-positive duration should use the same next-token-start fallback"
    );

    let final_token_without_duration = AsrTranscript::from_parts(
        "Hello.",
        vec!["Hello".to_string(), ".".to_string()],
        Some(&[0.1, 0.4]),
        Some(&[0.2, 0.0]),
    );
    assert_eq!(
        sample_end_for_char_end(&final_token_without_duration, 6, 16_000),
        Some(6_400),
        "the last token without duration should end at its own start timestamp"
    );
}

#[test]
fn seconds_to_sample_clamps_non_finite_negative_and_oversized_values() {
    assert_eq!(seconds_to_sample(f32::NAN, 16_000), 0);
    assert_eq!(seconds_to_sample(f32::INFINITY, 16_000), 0);
    assert_eq!(seconds_to_sample(-1.0, 16_000), 0);
    assert_eq!(seconds_to_sample(99.0, 16_000), 16_000);
}

#[test]
fn slice_chars_truncates_ranges_that_extend_past_text_end() {
    assert_eq!(slice_chars("あいう", 1..99), "いう");
    assert_eq!(slice_chars("あいう", 99..100), "");
}

#[test]
fn japanese_morph_final_particle_and_terminal_form_create_candidates() {
    let transcript = AsrTranscript::from_parts(
        "そうですね行きます".to_string(),
        vec![
            "そ".to_string(),
            "う".to_string(),
            "で".to_string(),
            "す".to_string(),
            "ね".to_string(),
            "行".to_string(),
            "き".to_string(),
            "ま".to_string(),
            "す".to_string(),
        ],
        Some(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0, 1.1]),
        None,
    );
    let morph_tokens = vec![
        JapaneseMorphToken {
            surface: "です".to_string(),
            char_range: 0..4,
            feature: "助動詞,*,*,*,助動詞-デス,終止形-一般".to_string(),
        },
        JapaneseMorphToken {
            surface: "ね".to_string(),
            char_range: 4..5,
            feature: "助詞,終助詞,*,*,*".to_string(),
        },
    ];

    let candidates =
        japanese_morph_candidates(&transcript, 32_000, &vads(&[true, false]), &morph_tokens);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].class, GrammarBoundaryClass::StrongEnd);
}

#[test]
fn japanese_morph_terminal_predicate_at_text_end_is_predicate_end() {
    let transcript = AsrTranscript::from_parts(
        "行きます".to_string(),
        vec![
            "行".to_string(),
            "き".to_string(),
            "ま".to_string(),
            "す".to_string(),
        ],
        Some(&[0.0, 0.1, 0.2, 0.3]),
        None,
    );
    let morph_tokens = vec![JapaneseMorphToken {
        surface: "行きます".to_string(),
        char_range: 0..4,
        feature: "動詞,一般,*,*,五段-カ行,終止形-一般".to_string(),
    }];

    let candidates = japanese_morph_candidates(&transcript, 32_000, &vads(&[true]), &morph_tokens);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].class, GrammarBoundaryClass::PredicateEnd);
}

#[test]
fn japanese_morph_terminal_nominal_particle_and_comma_classes_are_distinct() {
    let cases = [
        (
            "東京駅",
            "名詞,固有名詞,地名,*,*",
            GrammarBoundaryClass::NormalEnd,
        ),
        ("を", "助詞,格助詞,*,*,*", GrammarBoundaryClass::Reject),
        (
            "、",
            "補助記号,読点,*,*,*",
            GrammarBoundaryClass::ClauseWeak,
        ),
    ];

    for (surface, feature, class) in cases {
        let transcript = AsrTranscript::from_parts(
            surface.to_string(),
            vec![surface.to_string()],
            Some(&[0.0]),
            Some(&[0.1]),
        );
        let morph_tokens = vec![JapaneseMorphToken {
            surface: surface.to_string(),
            char_range: 0..surface.chars().count(),
            feature: feature.to_string(),
        }];

        let candidates =
            japanese_morph_candidates(&transcript, 32_000, &vads(&[true]), &morph_tokens);

        assert_eq!(
            candidates.len(),
            1,
            "{surface} should produce one candidate"
        );
        assert_eq!(candidates[0].class, class);
    }
}

#[test]
fn japanese_morph_boundary_classes_use_token_aligned_transcript_timestamps() {
    let cases = [
        ("ね", "助詞,終助詞,*,*,*", GrammarBoundaryClass::StrongEnd),
        (
            "行く",
            "動詞,一般,*,*,五段-カ行,終止形-一般",
            GrammarBoundaryClass::PredicateEnd,
        ),
        (
            "東京駅",
            "名詞,固有名詞,地名,*,*",
            GrammarBoundaryClass::NormalEnd,
        ),
        ("を", "助詞,格助詞,*,*,*", GrammarBoundaryClass::Reject),
        (
            "、",
            "補助記号,読点,*,*,*",
            GrammarBoundaryClass::ClauseWeak,
        ),
    ];

    for (surface, feature, expected_class) in cases {
        let transcript = token_aligned_transcript(surface);
        let morph_tokens = vec![JapaneseMorphToken {
            surface: surface.to_string(),
            char_range: 0..surface.chars().count(),
            feature: feature.to_string(),
        }];

        let candidates =
            japanese_morph_candidates(&transcript, 16_000, &vads(&[true]), &morph_tokens);

        assert_eq!(
            candidates.len(),
            1,
            "{surface} should produce exactly one boundary candidate"
        );
        let candidate = &candidates[0];
        assert_eq!(candidate.class, expected_class, "{surface}");
        assert_eq!(candidate.char_end, surface.chars().count(), "{surface}");
        assert_eq!(
            candidate.sample_end,
            surface.chars().count() * 1_600,
            "{surface} should derive sample_end from token timestamps"
        );
    }
}

#[test]
fn unidic_pos1_matching_does_not_use_substring_fallback() {
    assert!(has_pos1("動詞,一般,*,*,五段-カ行,終止形-一般", "動詞"));
    assert!(has_pos1("助動詞,*,*,*,助動詞-デス,終止形-一般", "助動詞"));
    assert!(!has_pos1("助動詞,*,*,*,助動詞-デス,終止形-一般", "動詞"));
    assert!(!has_pos1("感動詞,一般,*,*,*,*", "動詞"));
}

#[test]
fn unidic_suffix_normal_end_requires_nominal_pos2() {
    assert!(is_nominal_suffix("接尾辞,名詞的,一般,*,*,*"));
    assert!(is_nominal_suffix("接尾辞-名詞的,一般,*,*,*"));
    assert!(!is_nominal_suffix("接尾辞,形容詞的,*,*,*,*"));
}

#[test]
fn boundary_audio_window_duplicates_silence_after_boundary_into_both_turns() {
    let audio_len = 8;
    let vad_results = vads(&[true, true, false, false]);

    assert_eq!(
        audio_window_for_boundary(audio_len, &vad_results, 3),
        BoundaryAudioWindow {
            prefix_audio_end: 8,
            suffix_audio_start: 4,
        }
    );
}

#[test]
fn boundary_audio_window_duplicates_current_silence_run_when_boundary_is_in_silence() {
    let audio_len = 8;
    let vad_results = vads(&[true, false, false, true]);

    assert_eq!(
        audio_window_for_boundary(audio_len, &vad_results, 3),
        BoundaryAudioWindow {
            prefix_audio_end: 6,
            suffix_audio_start: 2,
        }
    );
}

fn vads(pattern: &[bool]) -> Vec<VadResult> {
    pattern
        .iter()
        .map(|is_speech| VadResult {
            is_speech: *is_speech,
            probability: if *is_speech { 0.9 } else { 0.0 },
        })
        .collect()
}

fn token_aligned_transcript(text: &str) -> AsrTranscript {
    let token_texts = text
        .chars()
        .map(|character| character.to_string())
        .collect::<Vec<_>>();
    let timestamps = (0..token_texts.len())
        .map(|index| {
            f32::from(u16::try_from(index).expect("test token index should fit u16")) * 0.1
        })
        .collect::<Vec<_>>();
    let durations = vec![0.1; token_texts.len()];
    AsrTranscript::from_parts(
        text.to_string(),
        token_texts,
        Some(&timestamps),
        Some(&durations),
    )
}
