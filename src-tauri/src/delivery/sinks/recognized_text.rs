use std::{cell::RefCell, thread::JoinHandle};

use tauri::AppHandle;

use crate::{config::ParapperConfig, delivery::RecognizedTextOutput, synthesis, translation};

use super::{developer_http, ui_event, ync_text};

pub(crate) struct DispatchContext<'a> {
    pub(crate) handle: &'a AppHandle,
    pub(crate) config: &'a ParapperConfig,
    pub(crate) recognized_text_id: &'a str,
    pub(crate) recognized_at_millis: u64,
    pub(crate) audio_seconds: f64,
    pub(crate) elapsed_millis: u128,
    pub(crate) is_final_for_ync_delivery: bool,
    mute_check: RefCell<Option<JoinHandle<bool>>>,
}

pub(crate) struct DispatchMetadata<'a> {
    pub(crate) recognized_text_id: &'a str,
    pub(crate) recognized_at_millis: u64,
    pub(crate) audio_seconds: f64,
    pub(crate) elapsed_millis: u128,
    pub(crate) is_final_for_ync_delivery: bool,
}

impl<'a> DispatchContext<'a> {
    pub(crate) fn from_metadata(
        handle: &'a AppHandle,
        config: &'a ParapperConfig,
        metadata: &DispatchMetadata<'a>,
        mute_check: Option<JoinHandle<bool>>,
    ) -> Self {
        Self {
            handle,
            config,
            recognized_text_id: metadata.recognized_text_id,
            recognized_at_millis: metadata.recognized_at_millis,
            audio_seconds: metadata.audio_seconds,
            elapsed_millis: metadata.elapsed_millis,
            is_final_for_ync_delivery: metadata.is_final_for_ync_delivery,
            mute_check: RefCell::new(mute_check),
        }
    }

    pub(super) fn take_vrchat_mute_check(&self) -> Option<JoinHandle<bool>> {
        self.mute_check.borrow_mut().take()
    }
}

pub(crate) trait RecognizedTextSink: Send + Sync {
    fn name(&self) -> &'static str;

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput);
}

pub(crate) fn registered_recognized_text_sinks() -> [&'static dyn RecognizedTextSink; 5] {
    [
        &TRANSLATION_SINK,
        &SYNTHESIS_SINK,
        &ui_event::SINK,
        &ync_text::SINK,
        &developer_http::SINK,
    ]
}

static TRANSLATION_SINK: TranslationSink = TranslationSink;
static SYNTHESIS_SINK: SynthesisSink = SynthesisSink;

struct TranslationSink;
struct SynthesisSink;

impl RecognizedTextSink for TranslationSink {
    fn name(&self) -> &'static str {
        "translation"
    }

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
        submit_recognized_text_to_translation(ctx, output);
    }
}

impl RecognizedTextSink for SynthesisSink {
    fn name(&self) -> &'static str {
        "synthesis"
    }

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
        submit_recognized_text_to_synthesis(ctx, output);
    }
}

fn submit_recognized_text_to_translation(ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
    translation::submit_recognized_text(ctx.handle, ctx.config, ctx.recognized_text_id, output);
}

fn submit_recognized_text_to_synthesis(ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
    synthesis::submit_recognized_text(ctx.handle, ctx.config, ctx.recognized_text_id, output);
}
