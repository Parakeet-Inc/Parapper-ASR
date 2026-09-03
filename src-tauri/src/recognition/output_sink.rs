use crate::{
    config::{DeliveryRouteSnapshot, ParapperConfig},
    delivery::{RecognizedTextOutput, dispatch_recognized_text, spawn_mute_check_if_needed},
    recognition::RecognitionStreamEvent,
    state::AppState,
};
use std::sync::mpsc::Sender;
use tauri::{AppHandle, Manager};

pub(crate) use parapper_stt_engine::ports::RecognitionOutputSink as TurnOutputSink;

pub(crate) struct WebSocketTurnOutputSink {
    sender: Sender<RecognitionStreamEvent>,
}

impl WebSocketTurnOutputSink {
    pub(crate) fn new(sender: Sender<RecognitionStreamEvent>) -> Self {
        Self { sender }
    }
}

impl parapper_stt_engine::ports::RecognitionOutputSink for WebSocketTurnOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        if self
            .sender
            .send(RecognitionStreamEvent::Output(Box::new(output)))
            .is_err()
        {
            log::debug!("WebSocket recognition output receiver is gone");
        }
    }
}

pub(crate) struct CompositeTurnOutputSink {
    sinks: Vec<Box<dyn TurnOutputSink>>,
}

impl CompositeTurnOutputSink {
    pub(crate) fn new(sinks: Vec<Box<dyn TurnOutputSink>>) -> Self {
        Self { sinks }
    }
}

impl parapper_stt_engine::ports::RecognitionOutputSink for CompositeTurnOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        let Some((last, preceding)) = self.sinks.split_last_mut() else {
            return;
        };
        for sink in preceding {
            sink.emit(output.clone());
        }
        last.emit(output);
    }
}

#[cfg(test)]
pub(crate) struct NoopTurnOutputSink;

#[cfg(test)]
impl parapper_stt_engine::ports::RecognitionOutputSink for NoopTurnOutputSink {
    fn emit(&mut self, _output: RecognizedTextOutput) {}
}

pub(crate) struct DeliveryTurnOutputSink {
    handle: AppHandle,
    config: ParapperConfig,
    route: DeliveryRouteSnapshot,
}

impl DeliveryTurnOutputSink {
    pub(crate) fn new(handle: AppHandle, config: &ParapperConfig) -> Self {
        Self {
            handle,
            config: config.clone(),
            route: config.legacy_delivery_route(),
        }
    }

    pub(crate) fn new_for_source(
        handle: AppHandle,
        config: &ParapperConfig,
        source_id: &str,
    ) -> anyhow::Result<Self> {
        let route = config.resolved_delivery_route_for_source(source_id)?;
        Ok(Self {
            handle,
            config: config.clone(),
            route,
        })
    }

    /// Constructs a profile-owned sink with the delivery route resolved at
    /// profile startup. Subsequent live configuration updates may change
    /// non-route delivery details, but cannot redirect an in-flight profile.
    pub(crate) fn new_for_stt_profile(
        handle: AppHandle,
        config: &ParapperConfig,
        profile_id: &str,
    ) -> anyhow::Result<Self> {
        let route = config.resolved_delivery_route_for_stt_profile(profile_id)?;
        Ok(Self {
            handle,
            config: config.clone(),
            route,
        })
    }

    #[cfg(test)]
    pub(crate) fn update_config(&mut self, config: &ParapperConfig) {
        self.config = config.clone();
    }
}

impl parapper_stt_engine::ports::RecognitionOutputSink for DeliveryTurnOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        let config = self
            .handle
            .try_state::<AppState>()
            .and_then(|state| state.runtime_config_snapshot().ok())
            .unwrap_or_else(|| self.config.clone());
        let mute_check = spawn_mute_check_if_needed(&self.handle, &config);
        dispatch_recognized_text(&self.handle, &config, mute_check, &output, &self.route);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    use tauri::Listener;

    use crate::{
        config::{AsrLanguage, AsrModel, DeliveryProfileConfig, RecognitionSourceConfig},
        delivery::{RecognitionSourceMeta, RecognizedTextMeta},
        recognition::events::RecognizedTextEvent,
    };

    #[test]
    fn websocket_only_sink_emits_the_complete_structured_output_once() {
        let (sender, receiver) = mpsc::channel();
        let expected = recognized_output("ws-only", "構造化出力。");
        let mut sink = WebSocketTurnOutputSink::new(sender);

        sink.emit(expected.clone());

        assert_eq!(
            receiver.try_iter().collect::<Vec<_>>(),
            vec![RecognitionStreamEvent::Output(Box::new(expected))]
        );
    }

    #[test]
    fn composite_sink_emits_the_same_output_once_to_each_sink() {
        let (first_sender, first_receiver) = mpsc::channel();
        let (second_sender, second_receiver) = mpsc::channel();
        let expected = recognized_output("composite", "複合出力。");
        let mut sink = CompositeTurnOutputSink::new(vec![
            Box::new(WebSocketTurnOutputSink::new(first_sender)),
            Box::new(WebSocketTurnOutputSink::new(second_sender)),
        ]);

        sink.emit(expected.clone());

        assert_eq!(
            first_receiver.try_iter().collect::<Vec<_>>(),
            vec![RecognitionStreamEvent::Output(Box::new(expected.clone()))]
        );
        assert_eq!(
            second_receiver.try_iter().collect::<Vec<_>>(),
            vec![RecognitionStreamEvent::Output(Box::new(expected))]
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn delivery_turn_output_sink_emit_dispatches_recognized_text_with_updated_config() {
        let handle = tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<RecognizedTextEvent>();
        let _event_id = handle.listen("parapper://recognized-text", move |event| {
            let payload = serde_json::from_str::<RecognizedTextEvent>(event.payload())
                .expect("recognized text event payload should decode");
            sender
                .send(payload)
                .expect("recognized text event should be recorded");
        });
        let initial_config = parapper_config! {
            neo_http_enabled: false,
            debug_asr_audio_playback: false,
            ..ParapperConfig::default()
        };
        let updated_config = parapper_config! {
            neo_http_enabled: false,
            debug_asr_audio_playback: true,
            ..ParapperConfig::default()
        };
        let mut sink = DeliveryTurnOutputSink::new(handle, &initial_config);

        sink.update_config(&updated_config);
        sink.emit(recognized_output("turn-output-sink", "配信テスト。"));

        let event = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("DeliveryTurnOutputSink should dispatch a recognized-text UI event");
        assert_eq!(event.id, "turn-output-sink");
        assert_eq!(event.text, "配信テスト。");
        assert!(event.is_final);
        assert_eq!(
            event.source.identity.source_id.as_str(),
            parapper_stt_engine::SourceId::LEGACY_SINGLE_SOURCE
        );
        assert_eq!(event.source.identity.channel_index, None);
        assert_eq!(event.source.turn_id, 1);
        assert_eq!(event.audio_frames, 2);
        assert_eq!(event.debug_asr_audio_samples, Some(vec![0.5, -0.5]));
        assert_eq!(
            event.debug_asr_audio_sample_rate,
            Some(crate::audio::ASR_SAMPLE_RATE),
            "emit must use the latest config passed through update_config"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn source_delivery_route_remains_the_startup_snapshot_after_live_config_update() {
        let handle = tauri_test_handle();
        let (sender, receiver) = mpsc::channel::<RecognizedTextEvent>();
        let _event_id = handle.listen("parapper://recognized-text", move |event| {
            let payload = serde_json::from_str::<RecognizedTextEvent>(event.payload())
                .expect("recognized text event payload should decode");
            sender
                .send(payload)
                .expect("recognized text event should be recorded");
        });
        let mut startup_config = ParapperConfig::default();
        startup_config.input.recognition_sources = vec![RecognitionSourceConfig {
            source_id: "channel-1".to_owned(),
            speaker_label: "Speaker 1".to_owned(),
            capture_endpoint_id: "interface-1".to_owned(),
            channel_index: 0,
            delivery_profile_id: Some("startup-route".to_owned()),
            asr_route_policy: None,
        }];
        startup_config.delivery_profiles = vec![DeliveryProfileConfig {
            id: "startup-route".to_owned(),
            gui_enabled: false,
            translation_mapping_ids: Vec::new(),
            speech_mapping_ids: Vec::new(),
            http_profile_ids: Vec::new(),
            neo_text_enabled: false,
        }];
        let mut updated_config = startup_config.clone();
        updated_config.delivery_profiles[0].gui_enabled = true;
        let mut sink = DeliveryTurnOutputSink::new_for_source(handle, &startup_config, "channel-1")
            .expect("explicit source route should resolve at startup");

        sink.update_config(&updated_config);
        sink.emit(recognized_output("route-snapshot", "起動時の経路。"));

        assert!(
            receiver.recv_timeout(Duration::from_millis(100)).is_err(),
            "a live config update must not enable a GUI route excluded by the source startup snapshot"
        );
    }

    fn recognized_output(id: &str, text: &str) -> RecognizedTextOutput {
        RecognizedTextOutput {
            phrase: vec![0.5, -0.5].into(),
            text: text.to_string(),
            source_asr_model: AsrModel::ReazonSpeechK2V2,
            source_language: AsrLanguage::Japanese,
            detected_language: None,
            meta: RecognizedTextMeta::replace_turn(
                id.to_string(),
                RecognitionSourceMeta {
                    identity: parapper_stt_engine::SourceIdentitySnapshot::legacy_single_source(),
                    turn_session_id: 1,
                    turn_id: 1,
                    turn_revision: 0,
                    output_sequence: 1,
                    segment_id: 1,
                    previous_segment_id: None,
                },
                true,
            ),
            elapsed_millis: 37,
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn tauri_test_handle() -> tauri::AppHandle {
        let builder = tauri::Builder::default();
        #[cfg(any(windows, target_os = "linux"))]
        let builder = builder.any_thread();
        let app = builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("test app should build");
        app.handle().clone()
    }
}
