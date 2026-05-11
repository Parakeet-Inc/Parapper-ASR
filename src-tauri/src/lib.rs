#![cfg_attr(test, allow(dead_code, unused_imports))]

#[cfg(not(test))]
use log::LevelFilter;
#[cfg(not(test))]
use tauri::{Manager, generate_handler};
#[cfg(not(test))]
use tauri_plugin_log::{Target, TargetKind};

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
