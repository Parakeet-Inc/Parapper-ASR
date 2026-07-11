use tauri::AppHandle;

use crate::{config::SpeechSourceKind, processing::ProcessingContext};

use super::provider::{TranslationProviderRegistry, TranslationTask};
use super::request::TranslationRequest;

pub(super) fn translate_text(
    handle: Option<&AppHandle>,
    request: &TranslationRequest,
) -> anyhow::Result<Vec<(String, String)>> {
    let registry = TranslationProviderRegistry::for_request(
        handle,
        request.config.translation.ync_plugin_port,
        request.targets.iter().map(|target| target.provider_id),
    );
    request
        .targets
        .iter()
        .try_fold(Vec::new(), |mut translations, target| {
            let task = TranslationTask {
                id: request.source_recognition_id.clone(),
                context: ProcessingContext::from_source(
                    &request.source_meta,
                    SpeechSourceKind::Recognition,
                    request.source_detected_language.clone(),
                ),
                source_lang: target.source_lang,
                target_lang: target.target_lang,
                text: request.source_text.clone(),
                is_final: request.is_final,
            };
            if let Some(result) = registry.translate(target.provider_id, &task)? {
                debug_assert_eq!(result.task_id, task.id);
                debug_assert_eq!(result.context, task.context);
                translations.push((result.target_lang.as_code().to_string(), result.text));
            }
            Ok(translations)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{AsrModel, LocalTranslationModel, ParapperConfig, TranslationLanguage},
        delivery::{
            RecognitionSourceMeta,
            common::{TranslationProviderId, TranslationTarget},
        },
        recognition::control::events::RecognizedTextUpdateMode,
    };

    #[test]
    fn local_translation_interim_request_does_not_require_engine_or_app_handle() {
        let request = TranslationRequest {
            config: ParapperConfig::default(),
            source_recognition_id: "recognition-interim".to_string(),
            source_meta: RecognitionSourceMeta {
                turn_session_id: 1,
                turn_id: 1,
                turn_revision: 0,
                output_sequence: 1,
                segment_id: 1,
                previous_segment_id: None,
            },
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_text: "こんにちは".to_string(),
            source_detected_language: Some("ja".to_string()),
            targets: vec![TranslationTarget {
                provider_id: TranslationProviderId::Local(LocalTranslationModel::default()),
                source_lang: TranslationLanguage::Ja,
                target_lang: TranslationLanguage::En,
            }],
            is_final: false,
            update_mode: RecognizedTextUpdateMode::Replace,
        };

        let translations = translate_text(None, &request)
            .expect("interim local translation should be skipped before engine loading");

        assert_eq!(translations, Vec::<(String, String)>::new());
    }
}
