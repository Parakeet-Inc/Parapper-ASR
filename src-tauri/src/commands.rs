use std::{fs, process::Command};

use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;

use crate::{
    audio::{DeviceInfo, collect_input_devices, collect_output_devices},
    config::ParapperConfig,
    config_preset::ConfigPreset,
    connect::{
        SpeechRequest, YncPluginClient, detect_ync_plugin_http_port,
        detect_ync_text_input_http_port, query_current_mute_state, ync_text_input_http_available,
    },
    error_event::{ErrorSeverity, ParapperErrorPayload, ParapperErrorType, parapper_error_payload},
    model::{ModelStatus, ensure_models_downloaded},
    recognition::RecognitionStatus,
    state::AppState,
};

type CommandResult<T> = Result<T, ParapperErrorPayload>;

fn command_error(error_type: ParapperErrorType, detail: String) -> ParapperErrorPayload {
    parapper_error_payload(error_type, ErrorSeverity::Fatal, detail)
}

#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri::command が JS invoke payload から所有 String を受け取るため"
)]
pub fn open_external_url(url: String) -> CommandResult<()> {
    if !is_safe_external_url(&url) {
        return Err(command_error(
            ParapperErrorType::Unknown,
            format!("Unsupported external URL: {url}"),
        ));
    }

    open_url_with_platform(&url)
        .map(|_| ())
        .map_err(|err| command_error(ParapperErrorType::Unknown, err.to_string()))
}

fn is_safe_external_url(url: &str) -> bool {
    (url.starts_with("https://") || url.starts_with("http://"))
        && !url.chars().any(char::is_control)
}

#[cfg(target_os = "windows")]
fn open_url_with_platform(url: &str) -> std::io::Result<std::process::Child> {
    Command::new("rundll32")
        .arg("url.dll,FileProtocolHandler")
        .arg(url)
        .spawn()
}

#[cfg(target_os = "macos")]
fn open_url_with_platform(url: &str) -> std::io::Result<std::process::Child> {
    Command::new("open").arg(url).spawn()
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_url_with_platform(url: &str) -> std::io::Result<std::process::Child> {
    Command::new("xdg-open").arg(url).spawn()
}

#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> CommandResult<ParapperConfig> {
    Ok(state.get_config().await)
}

#[tauri::command]
pub async fn save_config(
    state: State<'_, AppState>,
    config: ParapperConfig,
) -> CommandResult<ParapperConfig> {
    state
        .set_config(config)
        .await
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))
}

#[tauri::command]
pub async fn reset_config(state: State<'_, AppState>) -> CommandResult<ParapperConfig> {
    state
        .set_config(ParapperConfig::default())
        .await
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))
}

// 以下 3 つは sync コマンドで `State<'_, AppState>` を値で受け取るため、
// clippy::needless_pass_by_value の対象になる。tauri::command が要求する正規のシグネチャなので
// 偽陽性として #[expect] でマークする（必要なくなれば lint がそれ自身を warn してくれる）。
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri::command が要求する State<'_, T> の値渡しによる偽陽性"
)]
pub(crate) fn get_config_presets(state: State<'_, AppState>) -> CommandResult<Vec<ConfigPreset>> {
    state
        .config_presets()
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))
}

#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri::command が要求する State<'_, T> の値渡しによる偽陽性"
)]
pub(crate) fn save_config_preset(
    state: State<'_, AppState>,
    name: String,
    config: ParapperConfig,
) -> CommandResult<Vec<ConfigPreset>> {
    state
        .save_config_preset(name, config)
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))
}

#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "tauri::command が要求する State<'_, T> の値渡しによる偽陽性"
)]
pub(crate) fn delete_config_preset(
    state: State<'_, AppState>,
    name: String,
) -> CommandResult<Vec<ConfigPreset>> {
    state
        .delete_config_preset(name)
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))
}

#[tauri::command]
pub fn get_audio_devices() -> Vec<DeviceInfo> {
    collect_input_devices()
}

#[tauri::command]
pub fn get_output_audio_devices() -> Vec<DeviceInfo> {
    collect_output_devices()
}

#[tauri::command]
pub fn find_neo_http_port() -> Option<u16> {
    if !ParapperConfig::neo_http_supported() {
        return None;
    }
    detect_ync_text_input_http_port()
}

#[tauri::command]
pub fn find_ync_plugin_http_port() -> Option<u16> {
    if !ParapperConfig::neo_http_supported() {
        return None;
    }
    detect_ync_plugin_http_port()
}

#[tauri::command]
pub fn check_neo_http_available(neo_http_enabled: bool, neo_http_port: u16) -> bool {
    if !ParapperConfig::neo_http_supported() {
        return true;
    }
    if !neo_http_enabled {
        return true;
    }
    ync_text_input_http_available(neo_http_port)
}

#[tauri::command]
pub fn check_vrchat_oscquery_available(vrc_osc_micmute: bool) -> bool {
    if !ParapperConfig::vrc_osc_supported() {
        return true;
    }
    if !vrc_osc_micmute {
        return true;
    }
    query_current_mute_state().is_ok()
}

#[tauri::command]
pub async fn fetch_neo_voice_list(port: u16) -> CommandResult<Vec<String>> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut client = YncPluginClient::for_command(port)
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?;
        client
            .voice_list("voice-list")
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))
    })
    .await
    .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?
}

#[tauri::command]
pub async fn neo_speech_stop(port: u16) -> CommandResult<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut client = YncPluginClient::for_speech(port)
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?;
        client
            .speech_stop("speech-stop")
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))
    })
    .await
    .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?
}

#[tauri::command]
pub async fn neo_speech_test(port: u16, talker: String, text: String) -> CommandResult<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut client = YncPluginClient::for_command(port)
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?;
        let response = client
            .speech(SpeechRequest {
                id: "speech-test",
                text: &text,
                talker: &talker,
                volume: 1.0,
            })
            .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?;
        if response.id != "speech-test" {
            log::warn!(
                "YNC speech test response id differs: request=speech-test, response={}",
                response.id
            );
        }
        Ok(())
    })
    .await
    .map_err(|err| command_error(ParapperErrorType::NeoHttp, err.to_string()))?
}

#[tauri::command]
pub async fn get_model_status(state: State<'_, AppState>) -> CommandResult<ModelStatus> {
    let config = state
        .runtime_config_snapshot()
        .map_err(|err| command_error(ParapperErrorType::Config, err.to_string()))?;
    Ok(state.model_status(&config))
}

#[tauri::command]
pub async fn has_any_model_installed(state: State<'_, AppState>) -> CommandResult<bool> {
    state
        .has_any_model_installed()
        .map_err(|err| command_error(ParapperErrorType::ModelDownload, err.to_string()))
}

#[tauri::command]
pub async fn download_models(
    handle: AppHandle,
    config: ParapperConfig,
) -> CommandResult<ModelStatus> {
    ensure_models_downloaded(&handle, &config)
        .await
        .map_err(|err| command_error(ParapperErrorType::ModelDownload, err.to_string()))
}

#[tauri::command]
pub async fn save_recognition_csv(
    handle: AppHandle,
    default_file_name: String,
    content: String,
) -> CommandResult<Option<String>> {
    let Some(path) = handle
        .dialog()
        .file()
        .set_title("認識ログをCSVで保存")
        .set_file_name(default_file_name)
        .add_filter("CSV", &["csv"])
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let path = path.into_path().map_err(|err| {
        command_error(
            ParapperErrorType::FileSave,
            format!("Invalid save path: {err}"),
        )
    })?;
    fs::write(&path, content).map_err(|err| {
        command_error(
            ParapperErrorType::FileSave,
            format!("Failed to write file {}: {err}", path.display()),
        )
    })?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command]
pub async fn save_asr_input_wav(
    handle: AppHandle,
    default_file_name: String,
    content: Vec<u8>,
) -> CommandResult<Option<String>> {
    let Some(path) = handle
        .dialog()
        .file()
        .set_title("ASR入力音声をWAVで保存")
        .set_file_name(default_file_name)
        .add_filter("WAV", &["wav"])
        .blocking_save_file()
    else {
        return Ok(None);
    };
    let path = path.into_path().map_err(|err| {
        command_error(
            ParapperErrorType::FileSave,
            format!("Invalid save path: {err}"),
        )
    })?;
    fs::write(&path, content).map_err(|err| {
        command_error(
            ParapperErrorType::FileSave,
            format!("Failed to write file {}: {err}", path.display()),
        )
    })?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command]
pub async fn get_recognition_status(
    state: State<'_, AppState>,
) -> CommandResult<RecognitionStatus> {
    Ok(state.get_recognition_status().await)
}

#[tauri::command]
pub async fn start_recognition(
    handle: AppHandle,
    state: State<'_, AppState>,
) -> CommandResult<RecognitionStatus> {
    let status = state
        .start_audio_input(handle.clone())
        .await
        .map_err(|err| command_error(ParapperErrorType::AudioInput, err.to_string()))?;
    handle
        .emit("parapper://status", status)
        .map_err(|err| command_error(ParapperErrorType::Unknown, err.to_string()))?;
    Ok(status)
}

#[tauri::command]
pub async fn stop_recognition(
    handle: AppHandle,
    state: State<'_, AppState>,
) -> CommandResult<RecognitionStatus> {
    let status = state.stop_audio_input().await;
    handle
        .emit("parapper://status", status)
        .map_err(|err| command_error(ParapperErrorType::Unknown, err.to_string()))?;
    Ok(status)
}
