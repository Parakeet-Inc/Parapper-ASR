use crate::{
    playback::{PlaybackEvent, PlaybackManager},
    recognition::control::events::SpeechRequestStatus,
    synthesis::{artifact::TtsArtifact, manager::emit_speech_request_event},
};

use super::audio::GeneratedLocalTtsItem;

pub(super) fn submit_generated_local_tts_for_playback(item: GeneratedLocalTtsItem) {
    log::info!(
        "Local TTS playback queue id={} text_chars={}",
        item.request.id,
        item.request.text.chars().count()
    );
    let artifact = TtsArtifact {
        request_id: item.request.id.clone(),
        samples: item.audio.samples,
        sample_rate: item.audio.sample_rate,
        volume: item.request.volume,
        output_device_host: item.request.output_device_host.clone(),
        output_device_id: item.request.output_device_id.clone(),
    };
    PlaybackManager::global().submit(artifact.into_playback_request(Box::new(move |event| {
        match event {
            PlaybackEvent::Finished {
                request_id,
                elapsed_millis,
            } => {
                log::info!(
                    "Local TTS playback finished id={request_id} elapsed_ms={elapsed_millis}"
                );
                emit_speech_request_event(
                    item.handle.as_ref(),
                    &item.request,
                    elapsed_millis,
                    SpeechRequestStatus::Accepted,
                    None,
                );
            }
            PlaybackEvent::Failed {
                request_id,
                elapsed_millis,
                error,
            } => {
                log::warn!("Local TTS playback failed for {request_id}: {error}");
                emit_speech_request_event(
                    item.handle.as_ref(),
                    &item.request,
                    elapsed_millis,
                    SpeechRequestStatus::Failure,
                    Some(error),
                );
            }
        }
    })));
}
