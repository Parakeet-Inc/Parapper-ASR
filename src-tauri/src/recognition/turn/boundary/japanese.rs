use std::{ops::Range, path::Path};

use anyhow::{Context, Result};
use tauri::AppHandle;
use vibrato_rkyv::{Dictionary, LoadMode, Tokenizer};

use super::{audio_window::audio_window_for_boundary, sample_end_for_char_end};
use crate::{
    model::{japanese_morph_dictionary_paths_from_root, models_root},
    recognition::{
        segmentation::vad::engine::VadResult,
        transcription::asr::engine::AsrTranscript,
        turn::{GrammarBoundaryClass, TurnBoundaryCandidate},
    },
};

pub(crate) struct JapaneseMorphAnalyzer {
    tokenizer: Tokenizer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct JapaneseMorphToken {
    pub(super) surface: String,
    pub(super) char_range: Range<usize>,
    pub(super) feature: String,
}

impl JapaneseMorphAnalyzer {
    pub(crate) fn from_dictionary_path(path: &Path) -> Result<Self> {
        let dict = Dictionary::from_path(path, LoadMode::TrustCache)
            .with_context(|| format!("Failed to read Vibrato dictionary: {}", path.display()))?;
        Ok(Self {
            tokenizer: Tokenizer::new(dict),
        })
    }

    pub(super) fn analyze(&self, text: &str) -> Vec<JapaneseMorphToken> {
        let mut worker = self.tokenizer.new_worker();
        worker.reset_sentence(text);
        worker.tokenize();
        (0..worker.num_tokens())
            .map(|index| {
                let token = worker.token(index);
                JapaneseMorphToken {
                    surface: token.surface().to_string(),
                    char_range: token.range_char(),
                    feature: token.feature().to_string(),
                }
            })
            .collect()
    }
}

pub(crate) fn load_japanese_morph_analyzer(handle: &AppHandle) -> Option<JapaneseMorphAnalyzer> {
    let root = match models_root(handle) {
        Ok(root) => root,
        Err(err) => {
            log::warn!("Failed to resolve model root for Japanese boundary analyzer: {err}");
            return None;
        }
    };
    let mut last_error = None;
    for path in japanese_morph_dictionary_paths_from_root(&root)
        .into_iter()
        .filter(|path| path.is_file())
    {
        match JapaneseMorphAnalyzer::from_dictionary_path(&path) {
            Ok(analyzer) => return Some(analyzer),
            Err(err) => {
                last_error = Some((path, err));
            }
        }
    }
    if let Some((path, err)) = last_error {
        log::warn!(
            "Failed to initialize Japanese boundary analyzer from {}: {err}",
            path.display()
        );
    }
    None
}

pub(super) fn japanese_morph_candidates(
    transcript: &AsrTranscript,
    audio_len: usize,
    vad_results: &[VadResult],
    morph_tokens: &[JapaneseMorphToken],
) -> Vec<TurnBoundaryCandidate> {
    let text_len = transcript.text.chars().count();
    morph_tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            let class =
                japanese_morph_boundary_class(token, morph_tokens.get(index + 1), text_len)?;
            let char_end = token.char_range.end;
            let sample_end = sample_end_for_char_end(transcript, char_end, audio_len)?;
            let audio_window = audio_window_for_boundary(audio_len, vad_results, sample_end);
            Some(TurnBoundaryCandidate {
                char_end,
                sample_end,
                prefix_audio_end: audio_window.prefix_audio_end,
                suffix_audio_start: audio_window.suffix_audio_start,
                class,
            })
        })
        .collect()
}

fn japanese_morph_boundary_class(
    token: &JapaneseMorphToken,
    next: Option<&JapaneseMorphToken>,
    text_len: usize,
) -> Option<GrammarBoundaryClass> {
    let feature = token.feature.as_str();
    let is_terminal_token = token.char_range.end >= text_len;

    if matches!(token.surface.as_str(), "。" | "！" | "？" | "!" | "?") {
        return Some(GrammarBoundaryClass::StrongEnd);
    }
    if has_pos(feature, "補助記号", "句点") {
        return Some(GrammarBoundaryClass::StrongEnd);
    }
    if has_pos(feature, "補助記号", "読点") {
        return is_terminal_token.then_some(GrammarBoundaryClass::ClauseWeak);
    }
    if has_pos(feature, "助詞", "終助詞") {
        return Some(GrammarBoundaryClass::StrongEnd);
    }
    if has_any_pos2(feature, "助詞", &["格助詞", "係助詞", "副助詞", "準体助詞"]) {
        return is_terminal_token.then_some(GrammarBoundaryClass::Reject);
    }
    if has_pos(feature, "助詞", "接続助詞") {
        return is_terminal_token.then_some(GrammarBoundaryClass::ClauseWeak);
    }
    if has_any_pos1(feature, &["動詞", "形容詞", "助動詞"]) {
        if has_any_cform(feature, &["未然形", "連用形", "仮定形"]) {
            return is_terminal_token.then_some(GrammarBoundaryClass::Reject);
        }
        if has_cform(feature, "連体形") {
            return is_terminal_token.then_some(GrammarBoundaryClass::Reject);
        }
        if has_any_cform(feature, &["終止形", "命令形", "意志推量形"]) {
            return if is_terminal_token || !next.is_some_and(token_can_continue_after_predicate) {
                Some(GrammarBoundaryClass::PredicateEnd)
            } else {
                None
            };
        }
    }
    if has_any_pos1(feature, &["名詞", "代名詞"]) {
        return is_terminal_token.then_some(GrammarBoundaryClass::NormalEnd);
    }
    if has_pos1(feature, "接尾辞") && is_nominal_suffix(feature) {
        return is_terminal_token.then_some(GrammarBoundaryClass::NormalEnd);
    }
    if has_pos1(feature, "形状詞") {
        return is_terminal_token.then_some(GrammarBoundaryClass::NormalEnd);
    }
    if has_pos1(feature, "感動詞") {
        return is_terminal_token.then_some(GrammarBoundaryClass::NormalEnd);
    }
    if has_any_pos1(feature, &["接頭辞", "連体詞"]) {
        return is_terminal_token.then_some(GrammarBoundaryClass::Reject);
    }
    None
}

fn token_can_continue_after_predicate(token: &JapaneseMorphToken) -> bool {
    let feature = token.feature.as_str();
    matches!(
        token.surface.as_str(),
        "、" | "," | "ので" | "けど" | "から" | "し"
    ) || has_pos(feature, "補助記号", "読点")
        || has_pos(feature, "助詞", "接続助詞")
        || has_pos(feature, "助詞", "終助詞")
        || has_any_pos1(feature, &["名詞", "代名詞", "接尾辞"])
}

fn has_pos(feature: &str, pos1: &str, pos2: &str) -> bool {
    if let Some(codes) = numeric_morph_feature(feature) {
        return pos1_code(pos1) == Some(codes.pos1) && pos2_code(pos1, pos2) == Some(codes.pos2);
    }
    feature_pos1(feature).is_some_and(|field| field == pos1)
        && feature_pos2(feature).is_some_and(|field| field == pos2)
}

pub(super) fn has_pos1(feature: &str, pos1: &str) -> bool {
    if let Some(codes) = numeric_morph_feature(feature) {
        return pos1_code(pos1) == Some(codes.pos1);
    }
    feature_pos1(feature).is_some_and(|field| field == pos1)
}

fn has_any_pos1(feature: &str, pos1_values: &[&str]) -> bool {
    pos1_values.iter().any(|pos1| has_pos1(feature, pos1))
}

fn has_any_pos2(feature: &str, pos1: &str, pos2_values: &[&str]) -> bool {
    pos2_values.iter().any(|pos2| has_pos(feature, pos1, pos2))
}

fn has_cform(feature: &str, cform: &str) -> bool {
    if let Some(codes) = numeric_morph_feature(feature) {
        return cform_code(cform) == Some(codes.cform);
    }
    feature.contains(cform)
}

fn has_any_cform(feature: &str, cforms: &[&str]) -> bool {
    cforms.iter().any(|cform| has_cform(feature, cform))
}

pub(super) fn is_nominal_suffix(feature: &str) -> bool {
    if let Some(codes) = numeric_morph_feature(feature) {
        return codes.pos1 == 8 && codes.pos2 == 9;
    }
    feature_pos2(feature).is_some_and(|field| field.starts_with("名詞的") || field.contains("名詞"))
}

#[derive(Clone, Copy)]
struct NumericMorphFeature {
    pos1: u8,
    pos2: u8,
    cform: u8,
}

fn numeric_morph_feature(feature: &str) -> Option<NumericMorphFeature> {
    let bytes = feature.as_bytes();
    if bytes.len() != 4 || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(NumericMorphFeature {
        pos1: (bytes[0] - b'0') * 10 + (bytes[1] - b'0'),
        pos2: bytes[2] - b'0',
        cform: bytes[3] - b'0',
    })
}

fn pos1_code(pos1: &str) -> Option<u8> {
    match pos1 {
        "補助記号" => Some(1),
        "助詞" => Some(2),
        "動詞" => Some(3),
        "形容詞" => Some(4),
        "助動詞" => Some(5),
        "名詞" => Some(6),
        "代名詞" => Some(7),
        "接尾辞" => Some(8),
        "形状詞" => Some(9),
        "感動詞" => Some(10),
        "接頭辞" => Some(11),
        "連体詞" => Some(12),
        _ => None,
    }
}

fn pos2_code(pos1: &str, pos2: &str) -> Option<u8> {
    match (pos1, pos2) {
        ("補助記号", "句点") => Some(1),
        ("補助記号", "読点") => Some(2),
        ("助詞", "終助詞") => Some(3),
        ("助詞", "格助詞") => Some(4),
        ("助詞", "係助詞") => Some(5),
        ("助詞", "副助詞") => Some(6),
        ("助詞", "準体助詞") => Some(7),
        ("助詞", "接続助詞") => Some(8),
        ("接尾辞", pos2) if pos2.starts_with("名詞的") || pos2.contains("名詞") => Some(9),
        _ => None,
    }
}

fn cform_code(cform: &str) -> Option<u8> {
    match cform {
        "未然形" => Some(1),
        "連用形" => Some(2),
        "仮定形" => Some(3),
        "連体形" => Some(4),
        "終止形" => Some(5),
        "命令形" => Some(6),
        "意志推量形" => Some(7),
        _ => None,
    }
}

fn feature_pos1(feature: &str) -> Option<&str> {
    let field = feature.split(',').next()?.trim();
    field.split_once('-').map_or(Some(field), |(pos1, _)| {
        let pos1 = pos1.trim();
        (!pos1.is_empty()).then_some(pos1)
    })
}

fn feature_pos2(feature: &str) -> Option<&str> {
    let mut fields = feature.split(',').map(str::trim);
    let first = fields.next()?;
    if let Some((_, pos2)) = first.split_once('-') {
        let pos2 = pos2.trim();
        if !pos2.is_empty() {
            return Some(pos2);
        }
    }
    if let Some(second) = fields.next()
        && !second.is_empty()
    {
        return Some(second);
    }
    None
}
