use std::time::Instant;

use crate::connect::{SpeechRequest, YncPluginClient};

use super::super::request::QueuedSpeechRequest;

pub(in crate::synthesis) fn send_ync_speech_request(
    request: &QueuedSpeechRequest,
    started_at: Instant,
) -> anyhow::Result<u128> {
    log::info!(
        "Speech request send start id={} port={} talker={} text_chars={}",
        request.id,
        request.port,
        request.talker,
        request.text.chars().count()
    );
    let mut client = YncPluginClient::for_speech(request.port)?;
    let response = client.speech(SpeechRequest {
        id: &request.id,
        text: &request.text,
        talker: &request.talker,
        volume: request.volume,
    })?;
    if response.id != request.id {
        log::warn!(
            "YNC speech response id differs: request={}, response={}",
            request.id,
            response.id
        );
    }
    log::info!(
        "Speech request accepted id={} response_id={} elapsed_ms={}",
        request.id,
        response.id,
        started_at.elapsed().as_millis()
    );
    Ok(started_at.elapsed().as_millis())
}
