use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use tauri::AppHandle;

use crate::{config::ParapperConfig, processing::ProcessingContext};

use super::{
    clients::send_ync_speech_request, local::enqueue_local_tts_request,
    request::QueuedSpeechRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpeechOutputProviderId {
    Ync,
    Local,
}

#[derive(Debug, Clone)]
pub(crate) struct SpeechTask {
    pub(crate) id: String,
    pub(crate) context: ProcessingContext,
    pub(crate) text: String,
    pub(crate) language: Option<String>,
    pub(crate) volume: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeechOutcome {
    Accepted { elapsed_millis: u128 },
    Deferred,
}

pub(crate) trait SpeechOutputProvider: Send + Sync {
    fn submit(
        &self,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome>;
}

pub(crate) struct SpeechOutputProviderRegistry {
    providers: HashMap<SpeechOutputProviderId, Arc<dyn SpeechOutputProvider>>,
}

impl SpeechOutputProviderRegistry {
    pub(crate) fn standard() -> Self {
        let mut providers: HashMap<SpeechOutputProviderId, Arc<dyn SpeechOutputProvider>> =
            HashMap::new();
        providers.insert(
            SpeechOutputProviderId::Local,
            Arc::new(InProcessSpeechOutputProvider),
        );
        providers.insert(
            SpeechOutputProviderId::Ync,
            Arc::new(YncSpeechOutputProvider),
        );
        Self { providers }
    }

    #[cfg(test)]
    fn empty() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub(crate) fn submit(
        &self,
        provider_id: SpeechOutputProviderId,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        self.providers
            .get(&provider_id)
            .with_context(|| format!("speech output provider is not registered: {provider_id:?}"))?
            .submit(handle, task, request)
    }
}

struct InProcessSpeechOutputProvider;

impl SpeechOutputProvider for InProcessSpeechOutputProvider {
    fn submit(
        &self,
        handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        debug_assert_eq!(task.id, request.id);
        debug_assert_eq!(task.context.source_kind, request.source_kind);
        debug_assert_eq!(task.text, request.text);
        debug_assert!((task.volume - request.volume).abs() < f32::EPSILON);
        enqueue_local_tts_request(handle, request);
        Ok(SpeechOutcome::Deferred)
    }
}

struct YncSpeechOutputProvider;

impl SpeechOutputProvider for YncSpeechOutputProvider {
    fn submit(
        &self,
        _handle: Option<&AppHandle>,
        task: &SpeechTask,
        request: QueuedSpeechRequest,
    ) -> Result<SpeechOutcome> {
        if !ParapperConfig::neo_http_supported() {
            anyhow::bail!("translation/speech plugin HTTP is unsupported");
        }
        debug_assert_eq!(task.id, request.id);
        debug_assert_eq!(task.context.source_kind, request.source_kind);
        debug_assert_eq!(task.text, request.text);
        debug_assert_eq!(task.language, request.target_lang);
        debug_assert!((task.volume - request.volume).abs() < f32::EPSILON);
        let started_at = Instant::now();
        let elapsed_millis = send_ync_speech_request(&request, started_at)?;
        Ok(SpeechOutcome::Accepted { elapsed_millis })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SpeechBackend, SpeechSourceKind};

    fn task() -> SpeechTask {
        SpeechTask {
            id: "speech-1".to_string(),
            context: ProcessingContext {
                turn_session_id: 1,
                turn_id: 2,
                turn_revision: 0,
                segment_id: 3,
                source_kind: SpeechSourceKind::Recognition,
                source_language: Some("ja".to_string()),
            },
            text: "hello".to_string(),
            language: Some("en".to_string()),
            volume: 1.0,
        }
    }

    fn request() -> QueuedSpeechRequest {
        QueuedSpeechRequest {
            port: 8080,
            id: "speech-1".to_string(),
            source_event_id: "recognition-1".to_string(),
            source_meta: crate::delivery::RecognitionSourceMeta {
                turn_session_id: 1,
                turn_id: 2,
                turn_revision: 0,
                output_sequence: 1,
                segment_id: 3,
                previous_segment_id: None,
            },
            source_kind: SpeechSourceKind::Recognition,
            target_lang: Some("en".to_string()),
            text: "hello".to_string(),
            backend: SpeechBackend::Ync,
            talker: "voice".to_string(),
            local_tts_voice: None,
            local_tts_language: None,
            local_tts_speaker_id: None,
            output_device_host: None,
            output_device_id: None,
            volume: 1.0,
        }
    }

    #[test]
    fn unknown_speech_provider_is_an_error_without_fallback() {
        let error = SpeechOutputProviderRegistry::empty()
            .submit(SpeechOutputProviderId::Ync, None, &task(), request())
            .expect_err("an unregistered provider must not fall back");

        assert!(error.to_string().contains("not registered"));
    }
}
