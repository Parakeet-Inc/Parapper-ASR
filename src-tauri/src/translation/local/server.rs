use std::{
    net::{SocketAddr, TcpListener},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::AppHandle;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::config::{LocalTranslationModel, TranslationLanguage};
use crate::model::local_translation_model_is_installed;

use super::{openai_adapter, service::translate_text, ync_adapter};

trait LocalServerTranslator: Send + Sync {
    fn translate(
        &self,
        local_model: LocalTranslationModel,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        source_text: &str,
    ) -> Result<String>;
}

struct AppLocalServerTranslator {
    handle: AppHandle,
}

impl LocalServerTranslator for AppLocalServerTranslator {
    fn translate(
        &self,
        local_model: LocalTranslationModel,
        source_lang: TranslationLanguage,
        target_lang: TranslationLanguage,
        source_text: &str,
    ) -> Result<String> {
        translate_text(
            &self.handle,
            local_model,
            source_lang,
            target_lang,
            source_text,
        )
    }
}

#[derive(Clone)]
struct LocalTranslationServerState {
    local_model: LocalTranslationModel,
    translator: Arc<dyn LocalServerTranslator>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatCompletionRequest {
    model: Option<String>,
    messages: Vec<OpenAiMessage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: &'static str,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiChoice {
    index: u32,
    message: OpenAiResponseMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Debug, Serialize)]
struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

// Observed YukaCone NEO custom-JSON translation request (POST /):
// {"text":"...","target_language":"ja","source_language":"en"}
// NEO renders the `translatedText` field of the response.
#[derive(Debug, Deserialize)]
struct NeoJsonTranslationRequest {
    text: String,
    target_language: String,
    source_language: Option<String>,
}

#[derive(Debug, Serialize)]
struct NeoJsonTranslationResponse {
    #[serde(rename = "translatedText")]
    translated_text: String,
}

#[derive(Debug, Serialize)]
struct NeoJsonErrorResponse {
    error: String,
}

#[derive(Debug)]
struct NeoJsonTranslationError {
    status: StatusCode,
    message: String,
}

impl NeoJsonTranslationError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode(400),
            message: message.into(),
        }
    }

    fn translation(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode(500),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for NeoJsonTranslationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[derive(Debug)]
pub(crate) struct TranslationHttpListener {
    local_addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl TranslationHttpListener {
    pub(crate) fn start(
        handle: AppHandle,
        port: u16,
        local_model: LocalTranslationModel,
    ) -> Result<Self> {
        if port == 0 {
            anyhow::bail!("translation HTTP listener port must be between 1 and 65535");
        }
        if !local_model.is_available() {
            anyhow::bail!("local translation model is not available: {local_model:?}");
        }
        if !local_translation_model_is_installed(&handle, local_model)? {
            anyhow::bail!("local translation model is not installed: {local_model:?}");
        }
        Self::start_with_state(
            port,
            LocalTranslationServerState {
                local_model,
                translator: Arc::new(AppLocalServerTranslator { handle }),
            },
        )
    }

    fn start_with_state(port: u16, state: LocalTranslationServerState) -> Result<Self> {
        let server = Server::http(("127.0.0.1", port))
            .map_err(|err| anyhow!("{err}"))
            .with_context(|| format!("Failed to bind local translation server port {port}"))?;
        let local_addr = server
            .server_addr()
            .to_ip()
            .context("translation HTTP listener did not bind an IP address")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = thread::Builder::new()
            .name("parapper-local-translation-server".to_string())
            .spawn(move || run_server(server, state, &worker_shutdown))
            .context("Failed to spawn local translation server thread")?;
        log::info!("Local translation server listening on {local_addr}");
        Ok(Self {
            local_addr,
            shutdown,
            worker: Some(worker),
        })
    }

    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub(crate) fn stop(mut self) -> Result<()> {
        self.stop_inner()
    }

    fn stop_inner(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow!("translation HTTP listener thread panicked"))?;
            wait_for_listener_release(self.local_addr)?;
        }
        Ok(())
    }
}

fn wait_for_listener_release(local_addr: SocketAddr) -> Result<()> {
    const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);
    const POLL_INTERVAL: Duration = Duration::from_millis(10);

    // tiny_http drops its listener on a separate accept thread. Joining our
    // request worker therefore does not guarantee that Windows has released
    // the port yet.
    let deadline = Instant::now() + RELEASE_TIMEOUT;
    loop {
        match TcpListener::bind(local_addr) {
            Ok(probe) => {
                drop(probe);
                return Ok(());
            }
            Err(_) if Instant::now() < deadline => thread::sleep(POLL_INTERVAL),
            Err(error) => {
                return Err(anyhow!(error)).with_context(|| {
                    format!("Timed out waiting for translation HTTP listener {local_addr} to stop")
                });
            }
        }
    }
}

impl Drop for TranslationHttpListener {
    fn drop(&mut self) {
        if let Err(err) = self.stop_inner() {
            log::warn!("Failed to stop translation HTTP listener: {err:#}");
        }
    }
}

fn run_server(server: Server, state: LocalTranslationServerState, shutdown: &AtomicBool) {
    while !shutdown.load(Ordering::Acquire) {
        match server.recv_timeout(Duration::from_millis(50)) {
            Ok(Some(request)) => handle_http_request(request, &state),
            Ok(None) => {}
            Err(err) => {
                log::warn!("Translation HTTP listener receive failed: {err}");
                break;
            }
        }
    }
}

fn handle_http_request(mut request: Request, state: &LocalTranslationServerState) {
    let method = request.method().clone();
    let path = request.url().to_string();
    let remote = request
        .remote_addr()
        .map(std::string::ToString::to_string)
        .unwrap_or_else(|| "unknown".to_string());
    let response = if !accepts_local_translation_request(&method, &path) {
        log::warn!(
            "Local translation server rejected request remote={} method={} path={}",
            remote,
            method.as_str(),
            path
        );
        openai_error_response(
            StatusCode(404),
            "local translation server only accepts POST / or POST /v1/chat/completions".to_string(),
        )
    } else {
        let mut body = String::new();
        match request.as_reader().read_to_string(&mut body) {
            Ok(_) => translation_response_for_path(&path, &body, state, &remote),
            Err(err) => {
                log::warn!(
                    "Local translation server failed to read request body remote={} method={} path={} error={}",
                    remote,
                    method.as_str(),
                    path,
                    err
                );
                openai_error_response(
                    StatusCode(400),
                    format!("failed to read request body: {err}"),
                )
            }
        }
    };
    if let Err(err) = request.respond(response) {
        log::warn!("Failed to send local translation server response: {err}");
    }
}

fn accepts_local_translation_request(method: &Method, path: &str) -> bool {
    ync_adapter::accepts(method, path) || openai_adapter::accepts(method, path)
}

fn translation_response_for_path(
    path: &str,
    body: &str,
    state: &LocalTranslationServerState,
    remote: &str,
) -> Response<std::io::Cursor<Vec<u8>>> {
    match path {
        openai_adapter::PATH => chat_completion_response(body, state, remote, path),
        ync_adapter::PATH => neo_json_response(body, state, remote, path),
        _ => unreachable!("request path is validated before dispatch"),
    }
}

fn chat_completion_response(
    body: &str,
    state: &LocalTranslationServerState,
    remote: &str,
    path: &str,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let started = Instant::now();
    log::info!(
        "Local translation server request received remote={} path={} body_bytes={} {}",
        remote,
        path,
        body.len(),
        openai_request_log(body)
    );
    match handle_openai_chat_completion_body(body, state) {
        Ok(body) => {
            log::info!(
                "Local translation server request success remote={} path={} elapsed_ms={} {}",
                remote,
                path,
                started.elapsed().as_millis(),
                openai_response_log(&body)
            );
            json_response(StatusCode(200), &body)
        }
        Err(err) => {
            log::warn!(
                "Local translation server request failure remote={} path={} elapsed_ms={} error={}",
                remote,
                path,
                started.elapsed().as_millis(),
                err
            );
            openai_error_response(StatusCode(400), err)
        }
    }
}

fn neo_json_response(
    body: &str,
    state: &LocalTranslationServerState,
    remote: &str,
    path: &str,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let started = Instant::now();
    log::info!(
        "Local translation server request received remote={} path={} body_bytes={} {}",
        remote,
        path,
        body.len(),
        neo_json_request_log(body)
    );
    match handle_neo_json_body(body, state) {
        Ok(translated) => {
            log::info!(
                "Local translation server request success remote={} path={} elapsed_ms={} operation=neo_json status=success text_chars={} error=-",
                remote,
                path,
                started.elapsed().as_millis(),
                translated.chars().count()
            );
            json_response(
                StatusCode(200),
                &NeoJsonTranslationResponse {
                    translated_text: translated,
                },
            )
        }
        Err(err) => {
            log::warn!(
                "Local translation server request failure remote={} path={} elapsed_ms={} status={} error={}",
                remote,
                path,
                started.elapsed().as_millis(),
                err.status.0,
                err,
            );
            json_response(err.status, &NeoJsonErrorResponse { error: err.message })
        }
    }
}

fn handle_neo_json_body(
    body: &str,
    state: &LocalTranslationServerState,
) -> Result<String, NeoJsonTranslationError> {
    let request = serde_json::from_str::<NeoJsonTranslationRequest>(body).map_err(|err| {
        NeoJsonTranslationError::bad_request(format!("invalid translation request: {err}"))
    })?;
    let source_text =
        non_empty_text(&request.text).map_err(NeoJsonTranslationError::bad_request)?;
    let target_lang =
        TranslationLanguage::from_code(&request.target_language).ok_or_else(|| {
            NeoJsonTranslationError::bad_request(format!(
                "unsupported target_language: {}",
                request.target_language
            ))
        })?;
    // NEO's source_language is unreliable (English text arrives tagged fr/de),
    // so trust it only when it maps to a supported language different from the
    // target; otherwise infer from the text itself.
    let source_lang = request
        .source_language
        .as_deref()
        .and_then(TranslationLanguage::from_code)
        .filter(|source_lang| *source_lang != target_lang)
        .unwrap_or_else(|| infer_source_lang_from_text(source_text));
    if source_lang == target_lang {
        return Ok(source_text.to_string());
    }
    state
        .translator
        .translate(state.local_model, source_lang, target_lang, source_text)
        .map_err(|err| NeoJsonTranslationError::translation(err.to_string()))
}

fn infer_source_lang_from_text(text: &str) -> TranslationLanguage {
    if contains_japanese_text(text) {
        TranslationLanguage::Ja
    } else {
        TranslationLanguage::En
    }
}

fn neo_json_request_log(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return "format=neo_json parse=invalid_json".to_string();
    };
    let target = value
        .get("target_language")
        .and_then(Value::as_str)
        .unwrap_or("-");
    let source = value
        .get("source_language")
        .and_then(Value::as_str)
        .unwrap_or("-");
    let text_chars = value
        .get("text")
        .and_then(Value::as_str)
        .map(|text| text.chars().count());
    format!(
        "format=neo_json target={} source={} text_chars={}",
        target,
        source,
        text_chars
            .map(|count| count.to_string())
            .unwrap_or_else(|| "-".to_string())
    )
}

fn handle_openai_chat_completion_body(
    body: &str,
    state: &LocalTranslationServerState,
) -> Result<OpenAiChatCompletionResponse, String> {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(body)
        .map_err(|err| format!("invalid OpenAI chat completions request: {err}"))?;
    let source_text = openai_source_text(&request)
        .and_then(|text| non_empty_text(&text).ok().map(str::to_string))
        .ok_or_else(|| "user message text must not be empty".to_string())?;
    let target_lang = infer_target_lang_from_messages(&request.messages).unwrap_or_else(|| {
        if contains_japanese_text(&source_text) {
            TranslationLanguage::En
        } else {
            TranslationLanguage::Ja
        }
    });
    let translated = state
        .translator
        .translate(
            state.local_model,
            target_lang.other(),
            target_lang,
            &source_text,
        )
        .map_err(|err| err.to_string())?;
    Ok(OpenAiChatCompletionResponse {
        id: "chatcmpl-parapper-local-translation",
        object: "chat.completion",
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
        model: request
            .model
            .unwrap_or_else(|| "parapper-local-translation".to_string()),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiResponseMessage {
                role: "assistant",
                content: translated,
            },
            finish_reason: "stop",
        }],
        usage: OpenAiUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}

fn non_empty_text(text: &str) -> Result<&str, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("text must not be empty".to_string());
    }
    Ok(text)
}

fn openai_source_text(request: &OpenAiChatCompletionRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .or_else(|| request.messages.last())
        .map(|message| message_content_text(&message.content))
        .map(|text| strip_translation_prompt_text(&text).to_string())
}

fn infer_target_lang_from_messages(messages: &[OpenAiMessage]) -> Option<TranslationLanguage> {
    messages
        .iter()
        .map(|message| message_content_text(&message.content))
        .find_map(|text| infer_target_lang_from_text(&text))
}

fn infer_target_lang_from_text(text: &str) -> Option<TranslationLanguage> {
    let lower = text.to_ascii_lowercase();
    if text.contains("英語") || lower.contains("english") || lower.contains(" en ") {
        return Some(TranslationLanguage::En);
    }
    if text.contains("日本語") || lower.contains("japanese") || lower.contains(" ja ") {
        return Some(TranslationLanguage::Ja);
    }
    None
}

fn message_content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn strip_translation_prompt_text(text: &str) -> &str {
    let trimmed = text.trim();
    for separator in ["\n\n", "```"] {
        if let Some((_, candidate)) = trimmed.rsplit_once(separator) {
            let candidate = strip_code_fence(candidate.trim());
            if !candidate.is_empty() {
                return candidate;
            }
        }
    }
    strip_code_fence(trimmed)
}

fn strip_code_fence(text: &str) -> &str {
    text.strip_prefix("```")
        .and_then(|text| text.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(text)
}

fn contains_japanese_text(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(
            ch as u32,
            0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff
        )
    })
}

fn openai_request_log(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return "format=openai_chat_completion operation=chat.completions parse=invalid_json"
            .to_string();
    };
    let model = value.get("model").and_then(Value::as_str).unwrap_or("-");
    let stream = value
        .get("stream")
        .and_then(Value::as_bool)
        .map(|stream| stream.to_string())
        .unwrap_or_else(|| "-".to_string());
    let text_chars = value
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("content"))
        .map(message_content_text)
        .map(|text| text.chars().count());
    format!(
        "format=openai_chat_completion operation=chat.completions model={} stream={} text_chars={}",
        model,
        stream,
        text_chars
            .map(|count| count.to_string())
            .unwrap_or_else(|| "-".to_string())
    )
}

fn openai_response_log(response: &OpenAiChatCompletionResponse) -> String {
    let text_chars = response
        .choices
        .first()
        .map(|choice| choice.message.content.chars().count())
        .unwrap_or(0);
    format!(
        "operation=openai_chat_completion status=success id={} text_chars={} error=-",
        response.id, text_chars
    )
}

fn openai_error_response(
    status: StatusCode,
    message: String,
) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(
        status,
        &OpenAiErrorResponse {
            error: OpenAiError {
                message,
                kind: "invalid_request_error",
            },
        },
    )
}

fn json_response<T>(status: StatusCode, body: &T) -> Response<std::io::Cursor<Vec<u8>>>
where
    T: Serialize,
{
    let body = serde_json::to_string(body).expect("local translation response should serialize");
    Response::from_string(body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "application/json; charset=utf-8")
                .expect("static content-type header is valid"),
        )
}

#[cfg(test)]
mod tests {
    use std::{net::TcpListener, sync::Mutex};

    use super::*;

    struct FakeTranslator {
        calls: Mutex<Vec<String>>,
    }

    impl FakeTranslator {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().expect("calls lock poisoned").clone()
        }
    }

    impl LocalServerTranslator for FakeTranslator {
        fn translate(
            &self,
            local_model: LocalTranslationModel,
            source_lang: TranslationLanguage,
            target_lang: TranslationLanguage,
            source_text: &str,
        ) -> Result<String> {
            self.calls
                .lock()
                .expect("calls lock poisoned")
                .push(format!(
                    "{local_model:?}:{}:{}:{source_text}",
                    source_lang.as_code(),
                    target_lang.as_code()
                ));
            Ok(format!("{}:{source_text}", target_lang.as_code()))
        }
    }

    fn listener_state() -> LocalTranslationServerState {
        LocalTranslationServerState {
            local_model: LocalTranslationModel::Lfm2Q4,
            translator: Arc::new(FakeTranslator::new()),
        }
    }

    #[test]
    fn explicit_listener_stop_allows_rebinding_the_same_port() {
        let listener = TranslationHttpListener::start_with_state(0, listener_state())
            .expect("first listener should bind");
        let port = listener.local_addr().port();
        listener
            .stop()
            .expect("first listener should stop and join");

        let rebound = TranslationHttpListener::start_with_state(port, listener_state())
            .expect("stopped listener port should be reusable");
        assert_eq!(rebound.local_addr().port(), port);
        rebound.stop().expect("rebound listener should stop");
    }

    #[test]
    fn occupied_listener_port_returns_bind_error_without_fallback() {
        let occupied = TcpListener::bind(("127.0.0.1", 0)).expect("test port should bind");
        let port = occupied.local_addr().expect("test address").port();

        let error = TranslationHttpListener::start_with_state(port, listener_state())
            .expect_err("listener must not fall back to another port");

        assert!(error.to_string().contains(&port.to_string()));
    }

    struct FailingTranslator;

    impl LocalServerTranslator for FailingTranslator {
        fn translate(
            &self,
            _local_model: LocalTranslationModel,
            _source_lang: TranslationLanguage,
            _target_lang: TranslationLanguage,
            _source_text: &str,
        ) -> Result<String> {
            Err(anyhow!("model execution failed"))
        }
    }

    fn test_state(translator: Arc<FakeTranslator>) -> LocalTranslationServerState {
        LocalTranslationServerState {
            local_model: LocalTranslationModel::Lfm2Q4,
            translator,
        }
    }

    fn response_value(
        body: &str,
        state: &LocalTranslationServerState,
    ) -> OpenAiChatCompletionResponse {
        handle_openai_chat_completion_body(body, state).expect("request should succeed")
    }

    fn failure_message(body: &str, state: &LocalTranslationServerState) -> String {
        handle_openai_chat_completion_body(body, state).expect_err("request should fail")
    }

    #[test]
    fn local_server_openai_chat_completion_translates_last_user_message() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let response = response_value(
            r#"{"model":"LFM2-350M-ENJP-MT-ONNX","messages":[{"role":"system","content":"Translate into English."},{"role":"user","content":"こんにちは"}]}"#,
            &state,
        );

        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.model, "LFM2-350M-ENJP-MT-ONNX");
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(response.choices[0].message.content, "en:こんにちは");
        assert_eq!(
            translator.calls(),
            vec!["Lfm2Q4:ja:en:こんにちは".to_string()]
        );
    }

    #[test]
    fn local_server_openai_chat_completion_strips_prompt_wrapper() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let response = response_value(
            r#"{"messages":[{"role":"user","content":"Translate into English.\n\n```こんにちは```"}]}"#,
            &state,
        );

        assert_eq!(response.choices[0].message.content, "en:こんにちは");
        assert_eq!(
            translator.calls(),
            vec!["Lfm2Q4:ja:en:こんにちは".to_string()]
        );
    }

    #[test]
    fn local_server_openai_chat_completion_empty_text_fails_without_translating() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let error = failure_message(
            r#"{"model":"LFM2-350M-ENJP-MT-ONNX","messages":[{"role":"user","content":"   "} ]}"#,
            &state,
        );

        assert!(error.to_string().contains("empty"));
        assert!(translator.calls().is_empty());
    }

    fn neo_translated_text(body: &str, state: &LocalTranslationServerState) -> String {
        handle_neo_json_body(body, state).expect("NEO JSON request should succeed")
    }

    fn post_json(path: &str, body: &str, state: LocalTranslationServerState) -> (u16, Value) {
        let server = Server::http(("127.0.0.1", 0)).expect("test server should bind");
        let address = server
            .server_addr()
            .to_ip()
            .expect("test server should use an IP address");
        let server_thread = thread::spawn(move || {
            let request = server.recv().expect("test server should receive request");
            handle_http_request(request, &state);
        });
        let response = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .expect("test client should build")
            .post(format!("http://{address}{path}"))
            .header("Content-Type", "application/json; charset=utf-8")
            .body(body.to_string())
            .send()
            .expect("test request should complete");
        let status = response.status().as_u16();
        let response_body = response.json().expect("response should be JSON");
        server_thread.join().expect("test server should stop");
        (status, response_body)
    }

    #[test]
    fn local_server_root_path_does_not_fallback_to_openai_format() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let (status, response) = post_json(
            "/",
            r#"{"model":"m","messages":[{"role":"user","content":"Hello."}]}"#,
            state,
        );

        assert_eq!(status, 400);
        assert!(response.get("translatedText").is_none());
        assert!(translator.calls().is_empty());
    }

    #[test]
    fn local_server_neo_json_translates_with_observed_request_shape() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let translated = neo_translated_text(
            r#"{"text":"こんにちは","target_language":"en","source_language":"ja"}"#,
            &state,
        );

        assert_eq!(translated, "en:こんにちは");
        assert_eq!(
            translator.calls(),
            vec!["Lfm2Q4:ja:en:こんにちは".to_string()]
        );
    }

    #[test]
    fn local_server_neo_json_ignores_misdetected_source_language() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        // Observed: NEO tags English text as fr/de. The text itself decides.
        let translated = neo_translated_text(
            r#"{"text":"Why don't you join us?","target_language":"ja","source_language":"fr"}"#,
            &state,
        );

        assert_eq!(translated, "ja:Why don't you join us?");
        assert_eq!(
            translator.calls(),
            vec!["Lfm2Q4:en:ja:Why don't you join us?".to_string()]
        );
    }

    #[test]
    fn local_server_neo_json_unsupported_target_language_returns_http_error() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let (status, response) = post_json(
            "/",
            r#"{"text":"Okay.","target_language":"fr","source_language":"en"}"#,
            state,
        );

        assert_eq!(status, 400);
        assert!(response.get("translatedText").is_none());
        assert!(translator.calls().is_empty());
    }

    #[test]
    fn local_server_neo_json_missing_target_language_returns_http_error() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let (status, response) =
            post_json("/", r#"{"text":"Okay.","source_language":"en"}"#, state);

        assert_eq!(status, 400);
        assert!(response.get("translatedText").is_none());
        assert!(translator.calls().is_empty());
    }

    #[test]
    fn local_server_neo_json_translation_failure_returns_server_error() {
        let state = LocalTranslationServerState {
            local_model: LocalTranslationModel::Lfm2Q4,
            translator: Arc::new(FailingTranslator),
        };

        let (status, response) = post_json(
            "/",
            r#"{"text":"Hello.","target_language":"ja","source_language":"en"}"#,
            state,
        );

        assert_eq!(status, 500);
        assert!(response.get("translatedText").is_none());
    }

    #[test]
    fn local_server_neo_json_same_source_and_target_passes_text_through() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let translated = neo_translated_text(
            r#"{"text":"Why don't you join us?","target_language":"en","source_language":"fr"}"#,
            &state,
        );

        assert_eq!(translated, "Why don't you join us?");
        assert!(translator.calls().is_empty());
    }

    #[test]
    fn local_server_neo_json_empty_text_fails_without_translating() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let error = handle_neo_json_body(
            r#"{"text":"   ","target_language":"ja","source_language":"en"}"#,
            &state,
        )
        .expect_err("empty text should fail");

        assert!(error.to_string().contains("empty"));
        assert!(translator.calls().is_empty());
    }

    #[test]
    fn local_server_neo_json_http_contract_uses_observed_request_and_response_fields() {
        let translator = Arc::new(FakeTranslator::new());
        let state = test_state(Arc::clone(&translator));

        let (status, response) = post_json(
            "/",
            r#"{"text":"Hello.","target_language":"ja","source_language":"en"}"#,
            state,
        );

        assert_eq!(status, 200);
        assert_eq!(response, serde_json::json!({"translatedText":"ja:Hello."}));
        assert_eq!(translator.calls(), vec!["Lfm2Q4:en:ja:Hello.".to_string()]);
    }

    #[test]
    fn local_server_accepts_openai_chat_completion_endpoints_only() {
        assert!(accepts_local_translation_request(
            &Method::Post,
            "/v1/chat/completions"
        ));
        assert!(accepts_local_translation_request(&Method::Post, "/"));
        assert!(!accepts_local_translation_request(
            &Method::Post,
            "/api/input"
        ));
        assert!(!accepts_local_translation_request(
            &Method::Get,
            "/v1/chat/completions"
        ));
    }
}
