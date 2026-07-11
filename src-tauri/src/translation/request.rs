use crate::{
    config::{AsrModel, ParapperConfig, TranslationBackend},
    delivery::{
        RecognitionSourceMeta, RecognizedTextOutput,
        common::{
            TranslationTarget, text_format::trim_continuation_marker,
            timing::translation_timing_allows_output, translation_targets_for_mappings,
        },
    },
    recognition::control::events::RecognizedTextUpdateMode,
};

pub(crate) struct TranslationRequest {
    pub(super) config: ParapperConfig,
    pub(super) source_recognition_id: String,
    pub(super) source_meta: RecognitionSourceMeta,
    pub(super) source_asr_model: AsrModel,
    pub(super) source_text: String,
    pub(super) source_detected_language: Option<String>,
    pub(super) targets: Vec<TranslationTarget>,
    pub(super) is_final: bool,
    pub(super) update_mode: RecognizedTextUpdateMode,
}

impl TranslationRequest {
    pub(super) fn target_lang_codes(&self) -> Vec<&'static str> {
        self.targets
            .iter()
            .map(|target| target.target_lang_code())
            .collect()
    }
}

#[cfg(test)]
impl TranslationRequest {
    pub(crate) fn source_text(&self) -> &str {
        &self.source_text
    }

    pub(crate) fn is_final(&self) -> bool {
        self.is_final
    }
}

pub(crate) fn build_translation_request(
    config: &ParapperConfig,
    recognized_text_id: &str,
    output: &RecognizedTextOutput,
) -> Option<TranslationRequest> {
    if !config.translation.enabled {
        return None;
    }

    if !translation_timing_allows_output(config, output) {
        return None;
    }

    let text = trim_continuation_marker(output.text.trim()).to_string();
    if text.is_empty() {
        return None;
    }
    let source_meta = output.meta.source().clone();

    let mut translation_mappings = config.translation.mappings.clone();
    if !ParapperConfig::neo_http_supported() {
        let before = translation_mappings.len();
        translation_mappings.retain(|mapping| mapping.backend != TranslationBackend::Ync);
        if before != translation_mappings.len() {
            log::warn!(
                "Skipping YNC translation mappings for {recognized_text_id}: translation plugin HTTP is unsupported"
            );
        }
    }

    let targets = translation_targets_for_mappings(
        &translation_mappings,
        output.source_asr_model,
        output.source_language,
        output.detected_language.as_deref(),
    );
    if targets.is_empty() {
        log::warn!(
            "Translation is enabled for {recognized_text_id}, but no translation mappings match source_asr_model={:?}",
            output.source_asr_model
        );
        return None;
    }

    Some(TranslationRequest {
        config: config.clone(),
        source_recognition_id: recognized_text_id.to_string(),
        source_meta,
        source_asr_model: output.source_asr_model,
        source_text: text,
        source_detected_language: output.detected_language.clone(),
        targets,
        is_final: output.meta.is_final(),
        update_mode: output.meta.update_mode(),
    })
}

pub(super) fn translation_event_id(source_recognition_id: &str, target_lang: &str) -> String {
    format!("{source_recognition_id}|{target_lang}")
}
