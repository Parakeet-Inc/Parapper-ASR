use std::{
    net::SocketAddr,
    sync::{Arc, mpsc},
};

use anyhow::Result;
use parapper_stt_server::{
    ActiveRecognitionSession, AudioInput, BackendStartError, InputSendError, RecognitionBackend,
    RecognitionBackendConfig, StartedRecognitionSession,
};
use tauri::{AppHandle, Emitter, Manager};

use crate::{
    recognition::{
        BoundedInputSendError, BoundedInputSender, CompositeTurnOutputSink, DeliveryTurnOutputSink,
        RecognitionShutdownResult, RecognitionStartError, RunningInputSource, TurnOutputSink,
        WebSocketTurnOutputSink,
    },
    state::AppState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkOutputMode {
    WebSocketOnly,
    WebSocketAndDesktop,
}

pub(super) struct AppRecognitionBackend {
    handle: AppHandle,
    output_mode: NetworkOutputMode,
}

impl AppRecognitionBackend {
    pub(super) fn new(handle: AppHandle, output_mode: NetworkOutputMode) -> Arc<Self> {
        Arc::new(Self {
            handle,
            output_mode,
        })
    }
}

impl RecognitionBackend for AppRecognitionBackend {
    fn start(
        &self,
        session_id: &str,
        audio: &parapper_stt_server::protocol::AudioFormat,
        _backend_config: RecognitionBackendConfig,
    ) -> Result<StartedRecognitionSession, BackendStartError> {
        let (input_sender, source) =
            RunningInputSource::bounded_channel(audio.sample_rate, audio.sample_rate as usize * 2);
        let (event_sender, event_receiver) = mpsc::channel();
        let state = self.handle.state::<AppState>();
        let config = tauri::async_runtime::block_on(state.get_config());
        let websocket_sink: Box<dyn TurnOutputSink> =
            Box::new(WebSocketTurnOutputSink::new(event_sender.clone()));
        let output_sink: Box<dyn TurnOutputSink> = match self.output_mode {
            NetworkOutputMode::WebSocketOnly => websocket_sink,
            NetworkOutputMode::WebSocketAndDesktop => Box::new(CompositeTurnOutputSink::new(vec![
                websocket_sink,
                Box::new(DeliveryTurnOutputSink::new(self.handle.clone(), &config)),
            ])),
        };
        let start = state.start_network_input(
            self.handle.clone(),
            session_id.to_string(),
            source,
            output_sink,
            event_sender,
        );
        tauri::async_runtime::block_on(start).map_err(|error| map_start_error(&error))?;
        let _ = self.handle.emit(
            "parapper://status",
            crate::recognition::RecognitionStatus::Listening,
        );

        Ok(StartedRecognitionSession::new(
            Box::new(AppAudioInput { input_sender }),
            Box::new(AppActiveRecognitionSession {
                handle: self.handle.clone(),
                session_id: session_id.to_string(),
                finished: false,
            }),
            event_receiver,
        ))
    }
}

struct AppAudioInput {
    input_sender: BoundedInputSender,
}

impl AudioInput for AppAudioInput {
    fn try_send(&self, samples: Vec<f32>) -> Result<(), InputSendError> {
        self.input_sender
            .try_send(samples)
            .map_err(|error| match error {
                BoundedInputSendError::Overrun => InputSendError::Overrun,
                BoundedInputSendError::Disconnected => InputSendError::Disconnected,
            })
    }
}

fn map_start_error(error: &RecognitionStartError) -> BackendStartError {
    match error {
        RecognitionStartError::Busy => BackendStartError::Busy,
        RecognitionStartError::AudioInput(_) | RecognitionStartError::Asr(_) => {
            BackendStartError::ModelUnavailable
        }
    }
}

struct AppActiveRecognitionSession {
    handle: AppHandle,
    session_id: String,
    finished: bool,
}

impl AppActiveRecognitionSession {
    fn finish(&mut self, cancel: bool) -> RecognitionShutdownResult {
        if self.finished {
            return RecognitionShutdownResult::Cancelled;
        }
        let state = self.handle.state::<AppState>();
        if !cancel {
            let draining = tauri::async_runtime::block_on(
                state.set_recognition_status(crate::recognition::RecognitionStatus::Draining),
            );
            let _ = self.handle.emit("parapper://status", draining);
        }
        let (status, result) =
            tauri::async_runtime::block_on(state.stop_network_input(&self.session_id, cancel));
        let _ = self.handle.emit("parapper://status", status);
        self.finished = true;
        result
    }
}

impl ActiveRecognitionSession for AppActiveRecognitionSession {
    fn stop(&mut self) -> RecognitionShutdownResult {
        self.finish(false)
    }

    fn cancel(&mut self) {
        let _ = self.finish(true);
    }
}

impl Drop for AppActiveRecognitionSession {
    fn drop(&mut self) {
        let _ = self.finish(true);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StreamingRecognitionServerConfig {
    pub(crate) bind_addr: SocketAddr,
    pub(crate) api_key: Option<String>,
    pub(crate) output_mode: NetworkOutputMode,
}

pub(crate) struct StreamingRecognitionServer {
    inner: parapper_stt_server::StreamingRecognitionServer,
}

impl StreamingRecognitionServer {
    pub(crate) fn start(
        handle: AppHandle,
        config: StreamingRecognitionServerConfig,
    ) -> Result<Self> {
        let backend = AppRecognitionBackend::new(handle, config.output_mode);
        let inner = parapper_stt_server::StreamingRecognitionServer::start_with_backend(
            parapper_stt_server::StreamingRecognitionServerConfig {
                bind_addr: config.bind_addr,
                api_key: config.api_key,
                backend_config: RecognitionBackendConfig::default(),
            },
            backend,
        )?;
        Ok(Self { inner })
    }

    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.inner.local_addr()
    }

    pub(crate) fn stop(self) {
        self.inner.stop();
    }

    #[cfg(feature = "smoke-server")]
    pub(crate) fn start_smoke(bind_addr: SocketAddr, api_key: Option<String>) -> Result<Self> {
        parapper_stt_server::StreamingRecognitionServer::start_smoke(bind_addr, api_key)
            .map(|inner| Self { inner })
    }
}
