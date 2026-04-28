use log::LevelFilter;
use tauri::{Manager, generate_handler};
use tauri_plugin_log::{Target, TargetKind};

mod audio;
mod commands;
mod config;
mod connect;
mod error_event;
mod model;
mod recognition;
mod state;

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
            commands::save_config,
            commands::reset_config,
            commands::get_audio_devices,
            commands::find_neo_http_port,
            commands::check_neo_http_available,
            commands::check_vrchat_oscquery_available,
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
