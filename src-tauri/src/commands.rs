use std::fs;

use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;

use crate::{
    audio::{DeviceInfo, collect_input_devices},
    config::ParapperConfig,
    connect::{detect_neo_http_port, neo_http_available, query_current_mute_state},
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

#[tauri::command]
pub fn get_audio_devices() -> Vec<DeviceInfo> {
    collect_input_devices()
}

#[tauri::command]
pub fn find_neo_http_port() -> Option<u16> {
    if !ParapperConfig::neo_http_supported() {
        return None;
    }
    detect_neo_http_port()
}

#[tauri::command]
pub fn check_neo_http_available(neo_http_enabled: bool, neo_http_port: u16) -> bool {
    if !ParapperConfig::neo_http_supported() {
        return true;
    }
    if !neo_http_enabled {
        return true;
    }
    neo_http_available(neo_http_port) || detect_neo_http_port().is_some_and(neo_http_available)
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
