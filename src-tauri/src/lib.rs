#![cfg_attr(test, allow(dead_code, unused_imports))]

#[cfg(not(test))]
use log::LevelFilter;
#[cfg(not(test))]
use tauri::{Manager, generate_handler};
#[cfg(not(test))]
use tauri_plugin_log::{Target, TargetKind};

#[cfg(test)]
macro_rules! parapper_config {
    ($($field:ident : $value:expr,)* ..$base:expr $(,)?) => {{
        let mut config = $base;
        $(parapper_config_field!(config, $field, $value);)*
        config
    }};
    ($($field:ident : $value:expr),* $(,)?) => {{
        let mut config = $crate::config::ParapperConfig::default();
        $(parapper_config_field!(config, $field, $value);)*
        config
    }};
}

#[cfg(test)]
macro_rules! parapper_config_field {
    ($config:ident, neo_http_enabled, $value:expr) => {
        $config.neo.http_enabled = $value;
    };
    ($config:ident, neo_http_port, $value:expr) => {
        $config.neo.http_port = $value;
    };
    ($config:ident, neo_send_timing, $value:expr) => {
        $config.neo.send_timing = $value;
    };
    ($config:ident, input_device_id, $value:expr) => {
        $config.input.device_id = $value;
    };
    ($config:ident, input_device_host, $value:expr) => {
        $config.input.device_host = $value;
    };
    ($config:ident, input_device_name, $value:expr) => {
        $config.input.device_name = $value;
    };
    ($config:ident, input_volume_db, $value:expr) => {
        $config.input.volume_db = $value;
    };
    ($config:ident, asr_language, $value:expr) => {
        $config.asr.language = $value;
    };
    ($config:ident, asr_model, $value:expr) => {
        $config.asr.model = $value;
    };
    ($config:ident, asr_precision, $value:expr) => {
        $config.asr.precision = $value;
    };
    ($config:ident, asr_num_threads, $value:expr) => {
        $config.asr.num_threads = $value;
    };
    ($config:ident, asr_normalize_input_audio, $value:expr) => {
        $config.asr.normalize_input_audio = $value;
    };
    ($config:ident, multilingual_asr_enabled, $value:expr) => {
        $config.asr.multilingual_enabled = $value;
    };
    ($config:ident, enabled_asr_models, $value:expr) => {
        $config.asr.enabled_models = $value;
    };
    ($config:ident, translation_enabled, $value:expr) => {
        $config.translation.enabled = $value;
    };
    ($config:ident, translation_plugin_http_port, $value:expr) => {
        $config.translation.plugin_http_port = $value;
    };
    ($config:ident, translation_send_timing, $value:expr) => {
        $config.translation.send_timing = $value;
    };
    ($config:ident, translation_mappings, $value:expr) => {
        $config.translation.mappings = $value;
    };
    ($config:ident, speech_mappings, $value:expr) => {
        $config.speech.mappings = $value;
    };
    ($config:ident, model_dir, $value:expr) => {
        $config.models.dir = $value;
    };
    ($config:ident, vad_threshold, $value:expr) => {
        $config.segmentation.vad_threshold = $value;
    };
    ($config:ident, vad_interval_ms, $value:expr) => {
        $config.segmentation.vad_interval_ms = $value;
    };
    ($config:ident, segment_start_speech_ms, $value:expr) => {
        $config.segmentation.segment_start_speech_ms = $value;
    };
    ($config:ident, turn_detector, $value:expr) => {
        $config.turn.detector = $value;
    };
    ($config:ident, interim_result_enabled, $value:expr) => {
        $config.turn.interim_result_enabled = $value;
    };
    ($config:ident, interim_result_silence_ms, $value:expr) => {
        $config.turn.interim_result_silence_ms = $value;
    };
    ($config:ident, turn_check_silence_ms, $value:expr) => {
        $config.turn.check_silence_ms = $value;
    };
    ($config:ident, namo_turn_confidence_threshold, $value:expr) => {
        $config.turn.namo_confidence_threshold = $value;
    };
    ($config:ident, namo_context_max_tokens, $value:expr) => {
        $config.turn.namo_context_max_tokens = $value;
    };
    ($config:ident, turn_rerecognize_full_on_complete, $value:expr) => {
        $config.turn.rerecognize_full_on_complete = $value;
    };
    ($config:ident, noise_cancellation_enabled, $value:expr) => {
        $config.noise_cancellation.enabled = $value;
    };
    ($config:ident, noise_cancellation_model, $value:expr) => {
        $config.noise_cancellation.model = $value;
    };
    ($config:ident, vrc_osc_micmute, $value:expr) => {
        $config.vrc.osc_micmute = $value;
    };
    ($config:ident, debug_asr_audio_playback, $value:expr) => {
        $config.debug.asr_audio_playback = $value;
    };
    ($config:ident, recognition_log_limit, $value:expr) => {
        $config.debug.recognition_log_limit = $value;
    };
    ($config:ident, debug_audio_log_limit, $value:expr) => {
        $config.debug.debug_audio_log_limit = $value;
    };
}

mod audio;
#[cfg(not(test))]
mod commands;
mod config;
mod config_preset;
mod connect;
mod delivery;
mod error_event;
mod model;
mod playback;
mod recognition;
mod state;
mod synthesis;
mod translation;

#[cfg(test)]
mod pipeline_tests;

#[cfg(not(test))]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Runs the Tauri application.
///
/// # Panics
///
/// Panics if the Tauri application cannot be built or run.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    Target::new(TargetKind::Stdout),
                    Target::new(TargetKind::LogDir { file_name: None }),
                    Target::new(TargetKind::Webview),
                ])
                .level(LevelFilter::Debug)
                .level_for("mdns_sd", LevelFilter::Info)
                .format(|c, a, r| {
                    let now = chrono::Local::now();
                    c.finish(format_args!(
                        "[{date} {time}] [{level}][{module}][{file}:{line}] {message}",
                        date = now.format("%Y-%m-%d"),
                        time = now.format("%H:%M:%S"),
                        level = r.level(),
                        module = r.target(),
                        file = r.file().unwrap_or("unknown"),
                        line = r.line().unwrap_or(0),
                        message = a
                    ));
                })
                .build(),
        )
        .setup(|app| {
            let state = state::AppState::build(app.handle())?;
            app.manage(state);
            #[cfg(debug_assertions)]
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.open_devtools();
                }
            }
            Ok(())
        })
        .invoke_handler(generate_handler![
            commands::get_config,
            commands::open_external_url,
            commands::save_config,
            commands::reset_config,
            commands::get_config_presets,
            commands::save_config_preset,
            commands::delete_config_preset,
            commands::get_audio_devices,
            commands::get_output_audio_devices,
            commands::find_neo_http_port,
            commands::find_ync_plugin_http_port,
            commands::check_neo_http_available,
            commands::check_vrchat_oscquery_available,
            commands::fetch_neo_voice_list,
            commands::neo_speech_stop,
            commands::neo_speech_test,
            commands::get_model_status,
            commands::has_any_model_installed,
            commands::download_models,
            commands::save_recognition_csv,
            commands::save_asr_input_wav,
            commands::get_recognition_status,
            commands::start_recognition,
            commands::stop_recognition,
        ])
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
