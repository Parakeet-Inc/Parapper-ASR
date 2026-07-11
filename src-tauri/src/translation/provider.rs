use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use tauri::AppHandle;

use crate::{
    config::{LocalTranslationModel, TranslationLanguage},
    connect::YncPluginClient,
    delivery::common::TranslationProviderId,
    processing::ProcessingContext,
};

use super::local;

#[derive(Debug, Clone)]
pub(crate) struct TranslationTask {
    pub(crate) id: String,
    pub(crate) context: ProcessingContext,
    pub(crate) source_lang: TranslationLanguage,
    pub(crate) target_lang: TranslationLanguage,
    pub(crate) text: String,
    pub(crate) is_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslationResult {
    pub(crate) task_id: String,
    pub(crate) context: ProcessingContext,
    pub(crate) target_lang: TranslationLanguage,
    pub(crate) text: String,
    pub(crate) elapsed_millis: u128,
}

pub(crate) trait TranslationProvider: Send + Sync {
    fn translate(&self, task: &TranslationTask) -> Result<Option<TranslationResult>>;
}

pub(crate) struct TranslationProviderRegistry {
    providers: HashMap<TranslationProviderId, Arc<dyn TranslationProvider>>,
}

impl TranslationProviderRegistry {
    pub(crate) fn for_request(
        handle: Option<&AppHandle>,
        ync_port: u16,
        provider_ids: impl IntoIterator<Item = TranslationProviderId>,
    ) -> Self {
        let mut registry = Self::empty();
        for provider_id in provider_ids {
            registry
                .providers
                .entry(provider_id)
                .or_insert_with(|| match provider_id {
                    TranslationProviderId::Ync => {
                        Arc::new(YncTranslationProvider { port: ync_port })
                    }
                    TranslationProviderId::Local(model) => Arc::new(InProcessTranslationProvider {
                        handle: handle.cloned(),
                        model,
                    }),
                });
        }
        registry
    }

    fn empty() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub(crate) fn translate(
        &self,
        provider_id: TranslationProviderId,
        task: &TranslationTask,
    ) -> Result<Option<TranslationResult>> {
        self.providers
            .get(&provider_id)
            .with_context(|| format!("translation provider is not registered: {provider_id:?}"))?
            .translate(task)
    }
}

struct InProcessTranslationProvider {
    handle: Option<AppHandle>,
    model: LocalTranslationModel,
}

impl TranslationProvider for InProcessTranslationProvider {
    fn translate(&self, task: &TranslationTask) -> Result<Option<TranslationResult>> {
        if !task.is_final {
            return Ok(None);
        }
        let handle = self
            .handle
            .as_ref()
            .context("local translation requires an application handle for model loading")?;
        let started_at = Instant::now();
        let text = local::translate_text(
            handle,
            self.model,
            task.source_lang,
            task.target_lang,
            &task.text,
        )?;
        Ok(Some(result(task, text, started_at)))
    }
}

struct YncTranslationProvider {
    port: u16,
}

impl TranslationProvider for YncTranslationProvider {
    fn translate(&self, task: &TranslationTask) -> Result<Option<TranslationResult>> {
        let started_at = Instant::now();
        let mut client = YncPluginClient::for_command(self.port)?;
        let response = client.translate(&task.id, task.target_lang.as_code(), &task.text)?;
        Ok(Some(result(task, response.text, started_at)))
    }
}

fn result(task: &TranslationTask, text: String, started_at: Instant) -> TranslationResult {
    TranslationResult {
        task_id: task.id.clone(),
        context: task.context.clone(),
        target_lang: task.target_lang,
        text,
        elapsed_millis: started_at.elapsed().as_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpeechSourceKind;

    fn task() -> TranslationTask {
        TranslationTask {
            id: "turn-1|en".to_string(),
            context: ProcessingContext {
                turn_session_id: 1,
                turn_id: 2,
                turn_revision: 3,
                segment_id: 4,
                source_kind: SpeechSourceKind::Recognition,
                source_language: Some("ja".to_string()),
            },
            source_lang: TranslationLanguage::Ja,
            target_lang: TranslationLanguage::En,
            text: "こんにちは".to_string(),
            is_final: true,
        }
    }

    #[test]
    fn unknown_translation_provider_is_an_error_without_fallback() {
        let error = TranslationProviderRegistry::empty()
            .translate(TranslationProviderId::Ync, &task())
            .expect_err("an unregistered provider must not fall back");

        assert!(error.to_string().contains("not registered"));
    }
}
