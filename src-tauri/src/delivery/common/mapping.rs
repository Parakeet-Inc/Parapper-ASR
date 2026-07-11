use std::collections::HashSet;

use crate::config::{
    AsrLanguage, AsrModel, LocalTranslationModel, SpeechBackend, SpeechMapping, SpeechSourceKind,
    TranslationBackend, TranslationLanguage, TranslationMapping,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TranslationTarget {
    pub(crate) provider_id: TranslationProviderId,
    pub(crate) source_lang: TranslationLanguage,
    pub(crate) target_lang: TranslationLanguage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TranslationProviderId {
    Ync,
    Local(LocalTranslationModel),
}

impl TranslationTarget {
    pub(crate) fn target_lang_code(self) -> &'static str {
        self.target_lang.as_code()
    }
}

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
    detected_language: Option<&str>,
) -> Vec<TranslationTarget> {
    let Some(source_lang) = detected_language
        .and_then(TranslationLanguage::from_code)
        .or_else(|| TranslationLanguage::from_asr_language(source_language))
    else {
        return Vec::new();
    };

    let mut seen = HashSet::new();
    mappings
        .iter()
        .filter(|mapping| {
            mapping
                .source_asr_model
                .is_none_or(|model| model == source_asr_model)
        })
        .filter(|mapping| mapping.source_lang == source_lang)
        .filter(|mapping| mapping.target_lang != source_lang)
        .map(|mapping| TranslationTarget {
            provider_id: match mapping.backend {
                TranslationBackend::Ync => TranslationProviderId::Ync,
                TranslationBackend::Local => TranslationProviderId::Local(mapping.local_model),
            },
            source_lang: mapping.source_lang,
            target_lang: mapping.target_lang,
        })
        .filter(|target| seen.insert(target.target_lang))
        .collect()
}
