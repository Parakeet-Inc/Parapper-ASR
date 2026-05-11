use std::collections::HashSet;

use crate::config::{
    AsrLanguage, AsrModel, SpeechBackend, SpeechMapping, SpeechSourceKind, TranslationMapping,
};

#[derive(Clone, Copy)]
pub(crate) enum SpeechTextSource<'a> {
    Recognition,
    Translation { target_lang: &'a str },
}

pub(crate) fn speech_mapping_matches(
    mapping: &SpeechMapping,
    source: SpeechTextSource<'_>,
    source_asr_model: AsrModel,
) -> bool {
    if mapping.muted {
        return false;
    }
    match mapping.backend {
        SpeechBackend::Ync if mapping.talker.trim().is_empty() => return false,
        SpeechBackend::LocalTts if mapping.local_tts_voice.is_none() => return false,
        _ => {}
    }
    let source_kind_matches = match source {
        SpeechTextSource::Recognition => {
            mapping.source_kind == SpeechSourceKind::Recognition
                && mapping
                    .source_asr_model
                    .is_none_or(|model| model == source_asr_model)
        }
        SpeechTextSource::Translation { target_lang } => {
            mapping.source_kind == SpeechSourceKind::Translation
                && mapping
                    .target_lang
                    .as_deref()
                    .is_some_and(|mapping_target| mapping_target == target_lang)
        }
    };
    if !source_kind_matches {
        return false;
    }
    true
}

pub(crate) fn translation_targets_for_mappings(
    mappings: &[TranslationMapping],
    source_asr_model: AsrModel,
    source_language: AsrLanguage,
) -> Vec<String> {
    let mut seen = HashSet::new();
    mappings
        .iter()
        .filter(|mapping| {
            mapping
                .source_asr_model
                .is_none_or(|model| model == source_asr_model)
        })
        .map(|mapping| mapping.target_lang.trim())
        .filter(|target| !target.is_empty())
        .filter(|target| !translation_target_matches_source_language(target, source_language))
        .filter(|target| seen.insert((*target).to_string()))
        .map(ToString::to_string)
        .collect()
}

fn translation_target_matches_source_language(target: &str, source_language: AsrLanguage) -> bool {
    let normalized = target.to_ascii_lowercase();
    match source_language {
        AsrLanguage::Japanese => normalized.starts_with("ja"),
        AsrLanguage::English => normalized.starts_with("en"),
        AsrLanguage::EuropeanMultilingual => false,
    }
}
