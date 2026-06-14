use std::{
    sync::{OnceLock, mpsc},
    thread::{self, JoinHandle},
};

use tauri::AppHandle;

use super::vrchat_mute::is_vrchat_muted_before_send;
use crate::{
    config::{NeoSendTiming, ParapperConfig},
    connect::{TextTransport, YncTextInputTransport},
    delivery::{RecognizedTextOutput, sinks::ui_event::emit_connection_state},
    recognition::control::events::ConnectionTarget,
};

use super::{DispatchContext, RecognizedTextSink};

static TEXT_DELIVERY_QUEUE: OnceLock<mpsc::Sender<TextDeliveryRequest>> = OnceLock::new();

pub(crate) static SINK: YncTextSink = YncTextSink;

pub(crate) struct YncTextSink;

impl RecognizedTextSink for YncTextSink {
    fn name(&self) -> &'static str {
        "ync_text"
    }

    fn deliver(&self, ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
        enqueue_recognized_text_for_ync_text(ctx, output);
    }
}

struct TextDeliveryRequest {
    handle: AppHandle,
    config: ParapperConfig,
    mute_check: Option<JoinHandle<bool>>,
    text: String,
}

fn enqueue_recognized_text_for_ync_text(ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
    enqueue_text_to_neo_if_needed(
        ctx.handle,
        ctx.config,
        ctx.take_vrchat_mute_check(),
        output.text.clone(),
        ctx.is_final_for_ync_delivery,
    );
}

pub(crate) fn enqueue_text_to_neo_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
    mute_check: Option<JoinHandle<bool>>,
    text: String,
    is_final: bool,
) {
    if !should_send_to_neo(config, is_final) {
        return;
    }

    let request = TextDeliveryRequest {
        handle: handle.clone(),
        config: config.clone(),
        mute_check,
        text,
    };
    if let Err(err) = text_delivery_queue().send(request) {
        log::warn!("Failed to queue YNC text input request: {err}");
    }
}

fn text_delivery_queue() -> &'static mpsc::Sender<TextDeliveryRequest> {
    TEXT_DELIVERY_QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("parapper-ync-text-delivery".to_string())
            .spawn(move || run_text_delivery_queue(&receiver))
            .expect("failed to spawn YNC text delivery worker");
        sender
    })
}

fn run_text_delivery_queue(receiver: &mpsc::Receiver<TextDeliveryRequest>) {
    let mut text_transport = None;
    while let Ok(request) = receiver.recv() {
        let request = latest_text_delivery_request(request, receiver);
        let transport = text_transport_for_port(&mut text_transport, request.config.neo.http_port);
        send_text_to_neo_if_needed(
            &request.handle,
            &request.config,
            transport,
            request.mute_check,
            &request.text,
        );
    }
}

fn latest_text_delivery_request(
    mut request: TextDeliveryRequest,
    receiver: &mpsc::Receiver<TextDeliveryRequest>,
) -> TextDeliveryRequest {
    while let Ok(next_request) = receiver.try_recv() {
        request = next_request;
    }
    request
}

fn text_transport_for_port(
    text_transport: &mut Option<(u16, YncTextInputTransport)>,
    port: u16,
) -> &mut YncTextInputTransport {
    if text_transport
        .as_ref()
        .is_none_or(|(current_port, _)| *current_port != port)
    {
        *text_transport = Some((port, YncTextInputTransport::localhost(port)));
    }
    &mut text_transport.as_mut().expect("transport should exist").1
}

fn send_text_to_neo_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
    text_transport: &mut dyn TextTransport,
    mute_check: Option<JoinHandle<bool>>,
    text: &str,
) {
    let vrchat_muted = config.vrc.osc_micmute
        && ParapperConfig::vrc_osc_supported()
        && is_vrchat_muted_before_send(handle, mute_check);
    if vrchat_muted {
        let started_at = std::time::Instant::now();
        if let Err(err) = text_transport.send_text("") {
            log::warn!("Failed to clear NEO text while VRChat is muted: {err}");
            emit_connection_state(handle, ConnectionTarget::Neo, false, Some(err.to_string()));
        } else {
            log::info!(
                "NEO text clear sent elapsed_ms={}",
                started_at.elapsed().as_millis()
            );
            emit_connection_state(handle, ConnectionTarget::Neo, true, None);
        }
    } else {
        let started_at = std::time::Instant::now();
        log::info!("NEO text send start text_chars={}", text.chars().count());
        if let Err(err) = text_transport.send_text(text) {
            log::warn!("Failed to send text to NEO API: {err}");
            emit_connection_state(handle, ConnectionTarget::Neo, false, Some(err.to_string()));
        } else {
            log::info!(
                "NEO text send success elapsed_ms={}",
                started_at.elapsed().as_millis()
            );
            emit_connection_state(handle, ConnectionTarget::Neo, true, None);
        }
    }
}

pub(crate) fn should_send_to_neo(config: &ParapperConfig, is_final: bool) -> bool {
    config.neo.http_enabled
        && ParapperConfig::neo_http_supported()
        && (config.neo.send_timing == NeoSendTiming::Interim || is_final)
}
