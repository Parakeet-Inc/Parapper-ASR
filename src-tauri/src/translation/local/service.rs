use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::Result;
use tauri::AppHandle;

use crate::{
    config::{LocalTranslationModel, TranslationLanguage},
    model::local_translation_model_dir,
};

use super::cache::{TranslationCacheKey, TranslationResultCache};
use super::engine::LocalTranslationEngine;

const TRANSLATION_RESULT_CACHE_CAPACITY: usize = 10;

trait LocalTranslationEngineBackend: Send {
    fn translate(
        &mut self,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        source_text: &str,
    ) -> Result<String>;
}

impl LocalTranslationEngineBackend for LocalTranslationEngine {
    fn translate(
        &mut self,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        source_text: &str,
    ) -> Result<String> {
        LocalTranslationEngine::translate(self, source_lang, target_lang, source_text)
    }
}

type EngineLoader = Box<
    dyn FnMut(PathBuf, LocalTranslationModel) -> Result<Box<dyn LocalTranslationEngineBackend>>
        + Send,
>;

pub(in crate::translation) struct LocalTranslationService {
    state: Mutex<LocalTranslationServiceState>,
}

struct LocalTranslationServiceState {
    engines: HashMap<(PathBuf, LocalTranslationModel), Box<dyn LocalTranslationEngineBackend>>,
    engine_loader: EngineLoader,
    result_cache: TranslationResultCache,
}

impl LocalTranslationService {
    fn global() -> Arc<Self> {
        static SERVICE: OnceLock<Arc<LocalTranslationService>> = OnceLock::new();
        Arc::clone(SERVICE.get_or_init(|| Arc::new(Self::new())))
    }

    fn new() -> Self {
        Self::with_engine_loader(Box::new(|model_dir, local_model| {
            Ok(Box::new(LocalTranslationEngine::load(
                model_dir,
                local_model,
            )?))
        }))
    }

    fn with_engine_loader(engine_loader: EngineLoader) -> Self {
        Self {
            state: Mutex::new(LocalTranslationServiceState {
                engines: HashMap::new(),
                engine_loader,
                result_cache: TranslationResultCache::new(TRANSLATION_RESULT_CACHE_CAPACITY),
            }),
        }
    }

    fn translate(
        &self,
        model_dir: PathBuf,
        local_model: LocalTranslationModel,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        source_text: &str,
    ) -> Result<String> {
        let cache_key = TranslationCacheKey {
            source_text: source_text.to_string(),
            source_lang,
            target_lang,
            local_model,
        };
        let mut state = self
            .state
            .lock()
            .expect("local translation service lock poisoned");
        if let Some(cached) = state.result_cache.get(&cache_key) {
            return Ok(cached);
        }

        let engine_key = (model_dir, local_model);
        if !state.engines.contains_key(&engine_key) {
            let engine = (state.engine_loader)(engine_key.0.clone(), local_model)?;
            state.engines.insert(engine_key.clone(), engine);
        }
        let translated = state
            .engines
            .get_mut(&engine_key)
            .expect("local translation engine was inserted")
            .translate(source_lang, target_lang, source_text)?;
        state.result_cache.insert(cache_key, translated.clone());
        Ok(translated)
    }
}

pub(in crate::translation) fn translate_text(
    handle: &AppHandle,
    local_model: LocalTranslationModel,
    source_lang: TranslationLanguage,
    target_lang: TranslationLanguage,
    source_text: &str,
) -> Result<String> {
    let model_dir = local_translation_model_dir(handle, local_model)?;
    LocalTranslationService::global().translate(
        model_dir,
        local_model,
        source_lang,
        target_lang,
        source_text,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct FakeEngine {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl LocalTranslationEngineBackend for FakeEngine {
        fn translate(
            &mut self,
            source_lang: TranslationLanguage,
            target_lang: TranslationLanguage,
            source_text: &str,
        ) -> Result<String> {
            self.calls
                .lock()
                .expect("fake engine calls lock poisoned")
                .push(format!(
                    "{}:{}:{source_text}",
                    source_lang.as_code(),
                    target_lang.as_code()
                ));
            Ok(format!("translated:{source_text}"))
        }
    }

    #[test]
    fn local_translation_service_cache_hit_does_not_call_engine_twice() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let loader_calls = Arc::new(Mutex::new(Vec::new()));
        let service = {
            let calls = Arc::clone(&calls);
            let loader_calls = Arc::clone(&loader_calls);
            LocalTranslationService::with_engine_loader(Box::new(move |model_dir, local_model| {
                loader_calls
                    .lock()
                    .expect("loader calls lock poisoned")
                    .push((model_dir, local_model));
                Ok(Box::new(FakeEngine {
                    calls: Arc::clone(&calls),
                }))
            }))
        };
        let model_dir = PathBuf::from("models/lfm2-q4-k-quant");

        let first = service
            .translate(
                model_dir.clone(),
                LocalTranslationModel::Lfm2Q4,
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "こんにちは",
            )
            .expect("first translation should call fake engine");
        let second = service
            .translate(
                model_dir,
                LocalTranslationModel::Lfm2Q4,
                TranslationLanguage::Ja,
                TranslationLanguage::En,
                "こんにちは",
            )
            .expect("second translation should hit cache");

        assert_eq!(first, "translated:こんにちは");
        assert_eq!(second, "translated:こんにちは");
        assert_eq!(
            calls
                .lock()
                .expect("fake engine calls lock poisoned")
                .as_slice(),
            ["ja:en:こんにちは"]
        );
        assert_eq!(
            loader_calls
                .lock()
                .expect("loader calls lock poisoned")
                .len(),
            1
        );
    }
}
