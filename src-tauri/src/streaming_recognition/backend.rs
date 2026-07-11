use std::sync::{Arc, mpsc};

use tauri::{AppHandle, Emitter, Manager};

use crate::{
    recognition::{
        CompositeTurnOutputSink, DeliveryTurnOutputSink, RecognitionShutdownResult,
        RecognitionStartError, RecognitionStreamEvent, RunningInputSource, TurnOutputSink,
        WebSocketTurnOutputSink,
    },
    state::AppState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkOutputMode {
    WebSocketOnly,
    WebSocketAndDesktop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackendStartError {
    Busy,
    ModelUnavailable,
}

pub(super) trait ActiveRecognitionSession: Send {
    fn stop(&mut self) -> RecognitionShutdownResult;
    fn cancel(&mut self);
}

pub(super) struct StartedRecognitionSession {
    pub(super) active: Option<Box<dyn ActiveRecognitionSession>>,
    pub(super) event_receiver: mpsc::Receiver<RecognitionStreamEvent>,
}

pub(super) trait RecognitionBackend: Send + Sync {
    fn start(
        &self,
        session_id: &str,
        source: RunningInputSource,
        output_mode: NetworkOutputMode,
    ) -> Result<StartedRecognitionSession, BackendStartError>;
}

pub(super) struct AppRecognitionBackend {
    handle: AppHandle,
}

impl AppRecognitionBackend {
    pub(super) fn new(handle: AppHandle) -> Arc<Self> {
        Arc::new(Self { handle })
    }
}

impl RecognitionBackend for AppRecognitionBackend {
    fn start(
        &self,
        session_id: &str,
        source: RunningInputSource,
        output_mode: NetworkOutputMode,
    ) -> Result<StartedRecognitionSession, BackendStartError> {
        let (event_sender, event_receiver) = mpsc::channel();
        let state = self.handle.state::<AppState>();
        let config = tauri::async_runtime::block_on(state.get_config());
        let websocket_sink: Box<dyn TurnOutputSink> =
            Box::new(WebSocketTurnOutputSink::new(event_sender.clone()));
        let output_sink: Box<dyn TurnOutputSink> = match output_mode {
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
        tauri::async_runtime::block_on(start).map_err(map_start_error)?;
        let _ = self.handle.emit(
            "parapper://status",
            crate::recognition::RecognitionStatus::Listening,
        );

        Ok(StartedRecognitionSession {
            active: Some(Box::new(AppActiveRecognitionSession {
                handle: self.handle.clone(),
                session_id: session_id.to_string(),
                finished: false,
            })),
            event_receiver,
        })
    }
}

fn map_start_error(error: RecognitionStartError) -> BackendStartError {
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
