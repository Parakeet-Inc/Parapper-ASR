use std::collections::VecDeque;

use crate::config::{LocalTranslationModel, TranslationLanguage};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TranslationCacheKey {
    pub(super) source_text: String,
    pub(super) source_lang: TranslationLanguage,
    pub(super) target_lang: TranslationLanguage,
    pub(super) local_model: LocalTranslationModel,
}

#[derive(Debug)]
pub(super) struct TranslationResultCache {
    capacity: usize,
    entries: VecDeque<(TranslationCacheKey, String)>,
}

impl TranslationResultCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
        }
    }

    pub(super) fn get(&mut self, key: &TranslationCacheKey) -> Option<String> {
        let index = self
            .entries
            .iter()
            .position(|(entry_key, _)| entry_key == key)?;
        let (entry_key, value) = self
            .entries
            .remove(index)
            .expect("cache entry index was found");
        self.entries.push_back((entry_key, value.clone()));
        Some(value)
    }

    pub(super) fn insert(&mut self, key: TranslationCacheKey, value: String) {
        if self.capacity == 0 {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|(entry_key, _)| entry_key == &key)
        {
            self.entries.remove(index);
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((key, value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_key(text: &str, target_lang: TranslationLanguage) -> TranslationCacheKey {
        TranslationCacheKey {
            source_text: text.to_string(),
            source_lang: target_lang.other(),
            target_lang,
            local_model: LocalTranslationModel::Lfm2Q4,
        }
    }

    #[test]
    fn translation_cache_reuses_same_text_language_and_model_without_request_id() {
        let key = cache_key("こんにちは", TranslationLanguage::En);
        let mut cache = TranslationResultCache::new(10);

        cache.insert(key.clone(), "Hello.".to_string());

        assert_eq!(cache.get(&key), Some("Hello.".to_string()));
    }

    #[test]
    fn translation_cache_eviction_removes_oldest_entry_after_capacity() {
        let mut cache = TranslationResultCache::new(2);
        let first = cache_key("一", TranslationLanguage::En);
        let second = cache_key("二", TranslationLanguage::En);
        let third = cache_key("三", TranslationLanguage::En);

        cache.insert(first.clone(), "one".to_string());
        cache.insert(second.clone(), "two".to_string());
        cache.insert(third.clone(), "three".to_string());

        assert_eq!(cache.get(&first), None);
        assert_eq!(cache.get(&second), Some("two".to_string()));
        assert_eq!(cache.get(&third), Some("three".to_string()));
    }
}
