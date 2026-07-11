use crate::{
    config::ParapperConfig,
    delivery::{RecognizedTextOutput, dispatch_recognized_text, spawn_mute_check_if_needed},
    recognition::RecognitionStreamEvent,
};
use std::sync::mpsc::Sender;
use tauri::AppHandle;

pub(crate) trait TurnOutputSink: Send {
    fn update_config(&mut self, _config: &ParapperConfig) {}
    fn emit(&mut self, output: RecognizedTextOutput);
}

pub(crate) struct WebSocketTurnOutputSink {
    sender: Sender<RecognitionStreamEvent>,
}

impl WebSocketTurnOutputSink {
    pub(crate) fn new(sender: Sender<RecognitionStreamEvent>) -> Self {
        Self { sender }
    }
}

impl TurnOutputSink for WebSocketTurnOutputSink {
    fn emit(&mut self, output: RecognizedTextOutput) {
        if self
            .sender
            .send(RecognitionStreamEvent::Output(output))
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

impl TurnOutputSink for CompositeTurnOutputSink {
    fn update_config(&mut self, config: &ParapperConfig) {
        for sink in &mut self.sinks {
            sink.update_config(config);
        }
    }

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
impl TurnOutputSink for NoopTurnOutputSink {
    fn emit(&mut self, _output: RecognizedTextOutput) {}
}

pub(crate) struct DeliveryTurnOutputSink {
    handle: AppHandle,
    config: ParapperConfig,
}

impl DeliveryTurnOutputSink {
    pub(crate) fn new(handle: AppHandle, config: &ParapperConfig) -> Self {
        Self {
            handle,
            config: config.clone(),
        }
    }
}

impl TurnOutputSink for DeliveryTurnOutputSink {
    fn update_config(&mut self, config: &ParapperConfig) {
        self.config = config.clone();
    }

    fn emit(&mut self, output: RecognizedTextOutput) {
        let mute_check = spawn_mute_check_if_needed(&self.handle, &self.config);
        dispatch_recognized_text(&self.handle, &self.config, mute_check, &output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    use tauri::Listener;

    use crate::{
        config::{AsrLanguage, AsrModel},
        delivery::{RecognitionSourceMeta, RecognizedTextMeta},
        recognition::control::events::RecognizedTextEvent,
    };

    #[test]
    fn websocket_only_sink_emits_the_complete_structured_output_once() {
        let (sender, receiver) = mpsc::channel();
        let expected = recognized_output("ws-only", "構造化出力。");
        let mut sink = WebSocketTurnOutputSink::new(sender);

        sink.emit(expected.clone());

        assert_eq!(
            receiver.try_iter().collect::<Vec<_>>(),
            vec![RecognitionStreamEvent::Output(expected)]
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
            vec![RecognitionStreamEvent::Output(expected.clone())]
        );
        assert_eq!(
            second_receiver.try_iter().collect::<Vec<_>>(),
            vec![RecognitionStreamEvent::Output(expected)]
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
        assert_eq!(event.source.turn_id, 1);
        assert_eq!(event.audio_frames, 2);
        assert_eq!(event.debug_asr_audio_samples, Some(vec![0.5, -0.5]));
        assert_eq!(
            event.debug_asr_audio_sample_rate,
            Some(crate::audio::ASR_SAMPLE_RATE),
            "emit must use the latest config passed through update_config"
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
