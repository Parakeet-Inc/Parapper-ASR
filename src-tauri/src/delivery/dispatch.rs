use std::{
    thread::JoinHandle,
    time::{SystemTime, UNIX_EPOCH},
};

use tauri::AppHandle;

use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    delivery::{
        RecognizedTextOutput,
        sinks::{DispatchContext, registered_recognized_text_sinks},
    },
};

pub(crate) fn dispatch_recognized_text(
    handle: &AppHandle,
    config: &ParapperConfig,
    mute_check: Option<JoinHandle<bool>>,
    output: &RecognizedTextOutput,
) {
    let elapsed_millis = output.elapsed_millis;
    #[expect(clippy::cast_precision_loss)]
    let audio_seconds = output.phrase.len() as f64 / f64::from(ASR_SAMPLE_RATE);
    let recognized_at_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default();
    let recognized_text_id = output.meta.id.clone();
    let is_final = is_final_for_ync_delivery(output);
    log::info!(
        "Recognized text dispatch id={recognized_text_id} final={} update={:?} elapsed_ms={} audio_seconds={audio_seconds:.3}",
        output.meta.is_final,
        output.meta.update_mode,
        elapsed_millis
    );
    let metadata = crate::delivery::sinks::DispatchMetadata {
        recognized_text_id: &recognized_text_id,
        recognized_at_millis,
        audio_seconds,
        elapsed_millis,
        is_final_for_ync_delivery: is_final,
    };
    let ctx = DispatchContext::from_metadata(handle, config, &metadata, mute_check);
    for sink in registered_recognized_text_sinks() {
        log::trace!("Delivering recognized text to {}", sink.name());
        sink.deliver(&ctx, output);
    }
}

fn is_final_for_ync_delivery(output: &RecognizedTextOutput) -> bool {
    output.meta.is_final
}

#[cfg(test)]
mod tests {
    use super::is_final_for_ync_delivery;
    use crate::{
        config::{AsrLanguage, AsrModel},
        delivery::{RecognitionSourceMeta, RecognizedTextMeta, RecognizedTextOutput},
    };

    #[test]
    fn ync_delivery_final_flag_does_not_promote_partial_turns() {
        let output = RecognizedTextOutput {
            phrase: Vec::new(),
            text: "今日は...".to_string(),
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            detected_language: None,
            meta: RecognizedTextMeta::replace_turn(
                "turn-1".to_string(),
                RecognitionSourceMeta {
                    turn_session_id: 1,
                    turn_id: 1,
                    turn_revision: 0,
                    output_sequence: 1,
                    segment_id: 1,
                    previous_segment_id: None,
                },
                false,
            ),
            elapsed_millis: 0,
        };

        assert!(!is_final_for_ync_delivery(&output));
    }
}
