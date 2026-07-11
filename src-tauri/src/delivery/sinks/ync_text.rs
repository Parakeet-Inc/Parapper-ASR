use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, OnceLock, mpsc},
    thread::{self, JoinHandle},
};

use tauri::AppHandle;
use uuid::Uuid;

use super::vrchat_mute::is_vrchat_muted_before_send;
use crate::{
    config::ParapperConfig,
    connect::{TextInputPayload, TextTransport, YncTextInputTransport},
    delivery::{RecognizedTextOutput, sinks::ui_event::emit_connection_state},
    recognition::control::events::ConnectionTarget,
};

use super::{DispatchContext, RecognizedTextSink};

static TEXT_DELIVERY_QUEUE: OnceLock<mpsc::Sender<TextDeliveryRequest>> = OnceLock::new();
static TEXT_ID_SESSION_CACHE: Mutex<Option<(u64, String)>> = Mutex::new(None);

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
    text_id: String,
    is_final: bool,
}

fn enqueue_recognized_text_for_ync_text(ctx: &DispatchContext<'_>, output: &RecognizedTextOutput) {
    let source = output.meta.source();
    let text_id = {
        let mut cache = TEXT_ID_SESSION_CACHE
            .lock()
            .expect("YNC text id cache lock poisoned");
        text_id_for_turn(&mut cache, source.turn_session_id, source.turn_id)
    };
    enqueue_text_to_neo_if_needed(
        ctx.handle,
        ctx.config,
        ctx.take_vrchat_mute_check(),
        output.text.clone(),
        text_id,
        ctx.is_final_for_ync_delivery,
    );
}

fn text_id_for_turn(
    cache: &mut Option<(u64, String)>,
    turn_session_id: u64,
    turn_id: u64,
) -> String {
    let session_uuid = match cache {
        Some((session, uuid)) if *session == turn_session_id => uuid.clone(),
        _ => {
            let uuid = Uuid::new_v4().to_string();
            *cache = Some((turn_session_id, uuid.clone()));
            uuid
        }
    };
    format!("{session_uuid}-{turn_id}")
}

pub(crate) fn enqueue_text_to_neo_if_needed(
    handle: &AppHandle,
    config: &ParapperConfig,
    mute_check: Option<JoinHandle<bool>>,
    text: String,
    text_id: String,
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
        text_id,
        is_final,
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
    let mut last_sent_text_id: Option<String> = None;
    while let Ok(request) = receiver.recv() {
        let requests = drain_text_delivery_requests(request, receiver);
        let selected: HashSet<usize> = {
            let items = requests
                .iter()
                .map(|request| (request.text_id.as_str(), request.is_final))
                .collect::<Vec<_>>();
            select_indices_to_send(&items).into_iter().collect()
        };
        for (index, request) in requests.into_iter().enumerate() {
            if !selected.contains(&index) {
                continue;
            }
            let transport =
                text_transport_for_port(&mut text_transport, request.config.neo.http_port);
            send_text_to_neo_if_needed(request, transport, &mut last_sent_text_id);
        }
    }
}

fn drain_text_delivery_requests(
    first: TextDeliveryRequest,
    receiver: &mpsc::Receiver<TextDeliveryRequest>,
) -> Vec<TextDeliveryRequest> {
    let mut requests = vec![first];
    while let Ok(next) = receiver.try_recv() {
        requests.push(next);
    }
    requests
}

/// Select which drained requests to send, preserving arrival order and the YNC spec
/// guarantee that every textID eventually receives a `fixedText: true` message.
///
/// The newest request (last element) is always kept last, and any earlier requests
/// sharing its textID are dropped because it supersedes them. For every other textID,
/// only the last `is_final` request is kept so a superseded turn still gets finalized.
fn select_indices_to_send(items: &[(&str, bool)]) -> Vec<usize> {
    let Some(newest_index) = items.len().checked_sub(1) else {
        return Vec::new();
    };
    let newest_id = items[newest_index].0;

    let mut last_final_for_id: HashMap<&str, usize> = HashMap::new();
    for (index, (text_id, is_final)) in items[..newest_index].iter().enumerate() {
        if *is_final && *text_id != newest_id {
            last_final_for_id.insert(*text_id, index);
        }
    }

    let kept: HashSet<usize> = last_final_for_id.into_values().collect();
    let mut selected: Vec<usize> = (0..newest_index)
        .filter(|index| kept.contains(index))
        .collect();
    selected.push(newest_index);
    selected
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
    request: TextDeliveryRequest,
    text_transport: &mut dyn TextTransport,
    last_sent_text_id: &mut Option<String>,
) {
    let TextDeliveryRequest {
        handle,
        config,
        mute_check,
        text,
        text_id,
        is_final,
    } = request;
    let vrchat_muted = config.vrc.osc_micmute
        && ParapperConfig::vrc_osc_supported()
        && is_vrchat_muted_before_send(&handle, mute_check);
    if vrchat_muted {
        let clear_text_id = last_sent_text_id.clone().unwrap_or(text_id);
        let started_at = std::time::Instant::now();
        let payload = TextInputPayload {
            text: "",
            is_final: true,
            text_id: &clear_text_id,
        };
        if let Err(err) = text_transport.send_text(payload) {
            log::warn!("Failed to clear NEO text while VRChat is muted: {err}");
            emit_connection_state(&handle, ConnectionTarget::Neo, false, Some(err.to_string()));
        } else {
            log::info!(
                "NEO text clear sent text_id={clear_text_id} final=true elapsed_ms={}",
                started_at.elapsed().as_millis()
            );
            emit_connection_state(&handle, ConnectionTarget::Neo, true, None);
            *last_sent_text_id = None;
        }
    } else {
        let started_at = std::time::Instant::now();
        log::info!(
            "NEO text send start text_chars={} text_id={text_id} final={is_final}",
            text.chars().count()
        );
        let payload = TextInputPayload {
            text: &text,
            is_final,
            text_id: &text_id,
        };
        if let Err(err) = text_transport.send_text(payload) {
            log::warn!("Failed to send text to NEO API: {err}");
            emit_connection_state(&handle, ConnectionTarget::Neo, false, Some(err.to_string()));
        } else {
            log::info!(
                "NEO text send success text_id={text_id} final={is_final} elapsed_ms={}",
                started_at.elapsed().as_millis()
            );
            emit_connection_state(&handle, ConnectionTarget::Neo, true, None);
            *last_sent_text_id = Some(text_id);
        }
    }
}

pub(crate) fn should_send_to_neo(config: &ParapperConfig, _is_final: bool) -> bool {
    config.neo.http_enabled && ParapperConfig::neo_http_supported()
}

#[cfg(test)]
mod tests {
    use super::{select_indices_to_send, text_id_for_turn};

    #[test]
    fn text_id_is_stable_across_interim_and_final_of_same_turn() {
        let mut cache = None;
        let interim = text_id_for_turn(&mut cache, 7, 3);
        let final_id = text_id_for_turn(&mut cache, 7, 3);

        assert_eq!(interim, final_id);
        assert!(interim.ends_with("-3"));
    }

    #[test]
    fn text_id_shares_session_uuid_across_turns_but_differs_by_turn_id() {
        let mut cache = None;
        let turn_one = text_id_for_turn(&mut cache, 7, 1);
        let turn_two = text_id_for_turn(&mut cache, 7, 2);

        let prefix_one = turn_one
            .rsplit_once('-')
            .expect("text id has a turn suffix")
            .0;
        let prefix_two = turn_two
            .rsplit_once('-')
            .expect("text id has a turn suffix")
            .0;

        assert_eq!(prefix_one, prefix_two);
        assert_ne!(turn_one, turn_two);
        assert!(turn_one.ends_with("-1"));
        assert!(turn_two.ends_with("-2"));
    }

    #[test]
    fn new_session_generates_a_different_uuid_prefix() {
        let mut cache = None;
        let first_session = text_id_for_turn(&mut cache, 7, 1);
        let second_session = text_id_for_turn(&mut cache, 8, 1);

        let prefix_one = first_session
            .rsplit_once('-')
            .expect("text id has a turn suffix")
            .0;
        let prefix_two = second_session
            .rsplit_once('-')
            .expect("text id has a turn suffix")
            .0;

        assert_ne!(prefix_one, prefix_two);
    }

    #[test]
    fn selection_keeps_previous_turn_final_then_newest_interim() {
        let items = [("A", false), ("A", true), ("B", false)];

        assert_eq!(select_indices_to_send(&items), vec![1, 2]);
    }

    #[test]
    fn selection_drops_superseded_requests_sharing_the_newest_text_id() {
        let items = [("A", false), ("B", false), ("B", true)];

        assert_eq!(select_indices_to_send(&items), vec![2]);
    }

    #[test]
    fn selection_keeps_last_final_per_other_id_and_newest_last() {
        let items = [("A", true), ("A", false), ("B", true), ("C", false)];

        assert_eq!(select_indices_to_send(&items), vec![0, 2, 3]);
    }

    #[test]
    fn selection_of_single_request_keeps_it() {
        let items = [("A", false)];

        assert_eq!(select_indices_to_send(&items), vec![0]);
    }

    #[test]
    fn selection_of_empty_batch_is_empty() {
        assert!(select_indices_to_send(&[]).is_empty());
    }
}
