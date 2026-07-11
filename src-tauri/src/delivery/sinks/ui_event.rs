use tauri::{AppHandle, Emitter};

use crate::{
    audio::ASR_SAMPLE_RATE,
    config::ParapperConfig,
    delivery::RecognizedTextOutput,
    recognition::control::events::{ConnectionStateEvent, ConnectionTarget, RecognizedTextEvent},
};

use super::{DispatchContext, RecognizedTextSink};

pub(crate) static SINK: UiEventSink = UiEventSink;

pub(crate) struct UiEventSink;

impl RecognizedTextSink for UiEventSink {
    fn name(&self) -> &'static str {
        "ui_event"
    }

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
        emit_recognized_text_to_ui(ctx, output);
    }
}

fn emit_recognized_text_to_ui(ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
    emit_recognized_text_event(
        ctx.handle,
        ctx.config,
        output,
        ctx.recognized_text_id.to_string(),
        ctx.recognized_at_millis,
        ctx.audio_seconds,
        ctx.elapsed_millis,
    );
}

pub(crate) fn emit_recognized_text_event(
    handle: &AppHandle,
    config: &ParapperConfig,
    output: &RecognizedTextOutput,
    id: String,
    recognized_at_millis: u64,
    audio_seconds: f64,
    elapsed_millis: u128,
) {
    let _ = handle.emit(
        "parapper://recognized-text",
        RecognizedTextEvent {
            id,
            source: output.meta.source().clone(),
            is_final: output.meta.is_final,
            update_mode: output.meta.update_mode,
            text: output.text.clone(),
            source_asr_model: output.source_asr_model,
            source_language: output.source_language,
            detected_language: output.detected_language.clone(),
            recognized_at_millis,
            audio_seconds,
            elapsed_millis,
            audio_frames: output.phrase.len(),
            debug_asr_audio_sample_rate: config.debug.asr_audio_playback.then_some(ASR_SAMPLE_RATE),
            debug_asr_audio_samples: config
                .debug
                .asr_audio_playback
                .then(|| output.phrase.to_vec()),
        },
    );
}

pub(crate) fn emit_connection_state(
    handle: &AppHandle,
    target: ConnectionTarget,
    found: bool,
    detail: Option<String>,
) {
    let _ = handle.emit(
        "parapper://connection-state",
        ConnectionStateEvent {
            target,
            found,
            detail,
        },
    );
}
