use std::time::Instant;

use tauri::AppHandle;

use crate::{
    config::LocalTtsVoice,
    model::local_tts_model_dir,
    synthesis::{
        engines::{SherpaOnnxTtsEngine, SupertonicOnnxTtsEngine},
        request::QueuedSpeechRequest,
    },
};

use super::{audio::GeneratedLocalTtsAudio, key::LocalTtsQueueKey};

pub(super) enum LocalTtsEngine {
    Sherpa(Box<SherpaOnnxTtsEngine>),
    Supertonic(Box<SupertonicOnnxTtsEngine>),
}

impl LocalTtsEngine {
    fn new(handle: &AppHandle, voice: LocalTtsVoice) -> anyhow::Result<Self> {
        let model_dir = local_tts_model_dir(handle, voice)?;
        if matches!(
            voice,
            LocalTtsVoice::Supertonic2Onnx | LocalTtsVoice::Supertonic3Onnx
        ) {
            let supported_languages = voice
                .supported_language_codes()
                .ok_or_else(|| anyhow::anyhow!("Supertonic TTS languages are not configured"))?;
            return Ok(Self::Supertonic(Box::new(SupertonicOnnxTtsEngine::new(
                &model_dir,
                None,
                supported_languages,
            )?)));
        }
        Ok(Self::Sherpa(Box::new(SherpaOnnxTtsEngine::new(
            &model_dir, voice, 2,
        )?)))
    }

    fn synthesize(
        &mut self,
        request: &QueuedSpeechRequest,
    ) -> anyhow::Result<GeneratedLocalTtsAudio> {
        match self {
            Self::Sherpa(engine) => {
                let audio =
                    engine.synthesize(&request.text, request.local_tts_language.as_deref())?;
                Ok(GeneratedLocalTtsAudio {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                })
            }
            Self::Supertonic(engine) => {
                let samples = engine.synthesize(
                    &request.text,
                    request.local_tts_speaker_id,
                    request.local_tts_language.as_deref(),
                )?;
                Ok(GeneratedLocalTtsAudio {
                    samples,
                    sample_rate: engine.sample_rate,
                })
            }
        }
    }
}

pub(super) fn ensure_local_tts_engine(
    engine: &mut Option<LocalTtsEngine>,
    handle: &AppHandle,
    queue_key: LocalTtsQueueKey,
) -> anyhow::Result<()> {
    if engine.is_some() {
        return Ok(());
    }
    let voice = queue_key
        .voice
        .ok_or_else(|| anyhow::anyhow!("Local TTS queue has no voice"))?;
    *engine = Some(LocalTtsEngine::new(handle, voice)?);
    Ok(())
}

pub(super) fn synthesize_cached_local_tts_request(
    engine: &mut Option<LocalTtsEngine>,
    queue_key: LocalTtsQueueKey,
    handle: Option<&AppHandle>,
    request: &QueuedSpeechRequest,
    started_at: Instant,
) -> anyhow::Result<GeneratedLocalTtsAudio> {
    let handle = handle.ok_or_else(|| anyhow::anyhow!("AppHandle is required for local TTS"))?;
    let voice = request
        .local_tts_voice
        .ok_or_else(|| anyhow::anyhow!("Sherpa ONNX TTS voice is not configured"))?;
    if queue_key.voice != Some(voice) {
        anyhow::bail!(
            "Local TTS queue voice mismatch: queue={:?}, request={}",
            queue_key,
            voice.dir_name()
        );
    }
    ensure_local_tts_engine(engine, handle, queue_key)?;
    log::info!(
        "Local TTS synth start id={} voice={} text_chars={}",
        request.id,
        voice.dir_name(),
        request.text.chars().count()
    );
    let audio = engine
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Local TTS engine is not initialized"))?
        .synthesize(request)?;
    log::info!(
        "Local TTS synth finished id={} voice={} elapsed_ms={}",
        request.id,
        voice.dir_name(),
        started_at.elapsed().as_millis()
    );
    Ok(audio)
}
