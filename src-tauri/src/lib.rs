//! Raptor LLM desktop shell: the composition root. It wires the four subsystem crates and the
//! gateway together, exposes typed Tauri commands, and owns the process lifecycle.
//!
//! Subsystems never depend on each other; the adapters in [`adapters`] are the only
//! places where runtime, models, context and gateway meet.

pub mod adapters;
pub mod backend_choice;
pub mod commands;
pub mod conversations;
pub mod ctxbench;
pub mod discover;
pub mod fsutil;
pub mod job;
pub mod load_progress;
pub mod logging;
pub mod monitor;
pub mod perf;
pub mod sections;
pub mod semantic;
pub mod settings;
pub mod setup;
pub mod state;
pub mod thinking;
pub mod tokens;
pub mod tuned;
pub mod tuning;
pub mod updates;

#[cfg(test)]
mod testing;

use commands::EVENT_SERVER_STATE;
use state::AppState;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager, RunEvent};

/// Bundled runtime index, relative to the resource directory (see `bundle.resources`).
const BUNDLED_RUNTIME_MANIFEST: &str = "resources/runtime-manifest.json";

/// Copies the bundled runtime index to `config/` when none is there yet. An existing file is
/// never overwritten, so user edits survive app updates. A missing bundle is logged, not fatal:
/// `runtime_ensure` then reports that no index exists.
fn seed_runtime_manifest(app: &AppHandle, state: &AppState) {
    let target = state.runtime_manifest_path();
    if target.exists() {
        return;
    }
    let source = match app
        .path()
        .resolve(BUNDLED_RUNTIME_MANIFEST, BaseDirectory::Resource)
    {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("bundled runtime index not resolvable: {e}");
            return;
        }
    };
    match std::fs::read(&source) {
        Ok(bytes) => {
            if let Err(e) = fsutil::write_atomic(&target, &bytes) {
                tracing::warn!("could not copy the runtime index to config: {e}");
            }
        }
        Err(e) => tracing::warn!("bundled runtime index unavailable: {e}"),
    }
}

/// Starts the background work: hardware detection (once, cached), the gateway per settings and
/// the llama-server state watcher. Nothing here downloads or installs anything. The watcher
/// and the gateway stop at exit, see `AppState::shutdown`.
fn start_background(app: &AppHandle) {
    let hardware_app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = hardware_app.state::<AppState>();
        if let Err(e) = state.refresh_hardware().await {
            tracing::warn!("hardware detection failed: {e}");
        }
    });

    let gateway_app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = gateway_app.state::<AppState>();
        if let Err(e) = commands::system::gateway_changed(&gateway_app, &state).await {
            tracing::warn!("gateway did not start: {e}");
        }
    });

    let (runtime, stop) = {
        let state = app.state::<AppState>();
        (state.runtime.clone(), state.background.clone())
    };
    let watch_app = app.clone();
    tauri::async_runtime::spawn(monitor::watch(
        move || runtime.status(),
        stop,
        monitor::POLL_INTERVAL,
        move |status| {
            let _ = watch_app.emit(EVENT_SERVER_STATE, status);
        },
    ));
}

pub fn run() {
    let app_state = match AppState::initialize() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Raptor LLM could not start: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = logging::init(&app_state.dirs) {
        eprintln!("Raptor LLM: file logging is off: {e}");
    }
    let built = tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(app_state)
        .setup(|app| {
            seed_runtime_manifest(app.handle(), &app.state::<AppState>());
            start_background(app.handle());
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = handle.state::<AppState>().restore_installed_runtime().await {
                    tracing::warn!(error = %e, "could not select the installed runtime");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::system::hardware_info,
            commands::system::runtime_status,
            commands::system::runtime_ensure,
            commands::system::runtime_rollback,
            commands::system::runtime_unpin,
            commands::system::runtime_recommend,
            commands::system::runtime_list,
            commands::system::setup_status,
            commands::perf::perf_history,
            commands::perf::perf_compare,
            commands::perf::perf_run_benchmark,
            commands::perf::perf_best_known,
            commands::system::gateway_status,
            commands::system::gateway_rotate_tls,
            commands::system::diagnostics_log_tail,
            commands::system::diagnostics_log_files,
            commands::system::dialog_pick_file,
            commands::system::dialog_pick_folder,
            commands::system::dialog_save_file,
            commands::system::runtime_install_preview,
            commands::system::runtime_install_progress,
            commands::system::dependencies_check,
            commands::system::dependencies_install_update,
            commands::models::models_details,
            commands::models::models_compare,
            commands::models::models_rank_for,
            commands::models::models_check_updates,
            commands::presets::presets_list,
            commands::presets::presets_get,
            commands::presets::presets_save,
            commands::presets::presets_delete,
            commands::presets::presets_set_default,
            commands::settings::settings_set_allow_cpu_fallback,
            commands::settings::settings_set_parallel_slots,
            commands::settings::settings_set_thinking_effort,
            commands::models::models_search,
            commands::models::models_repo_files,
            commands::models::models_repo_info,
            commands::models::models_list,
            commands::models::models_storage,
            commands::models::models_delete,
            commands::models::models_import,
            commands::models::models_import_progress,
            commands::models::models_scan_folder,
            commands::models::models_discover,
            commands::models::models_download_start,
            commands::models::models_download_pause,
            commands::models::models_download_resume,
            commands::models::models_download_cancel,
            commands::models::models_download_queue,
            commands::models::models_recover_downloads,
            commands::models::models_discard_partial,
            commands::models::models_load,
            commands::models::models_unload,
            commands::models::models_current,
            commands::models::models_load_report,
            commands::models::models_load_progress,
            commands::models::models_tune,
            commands::models::models_context_matrix,
            commands::models::models_context_matrix_last,
            commands::models::models_context_options,
            commands::system::dependencies_install_vulkan,
            commands::models::models_backend,
            commands::models::models_set_backend,
            commands::models::models_thinking,
            commands::models::models_tune_progress,
            commands::models::models_forget_tuning,
            commands::settings::settings_get,
            commands::settings::settings_set_storage_dir,
            commands::settings::settings_set_gateway_port,
            commands::settings::settings_set_gateway_enabled,
            commands::settings::settings_set_lan,
            commands::settings::settings_set_context_target,
            commands::settings::api_key_rotate,
            commands::settings::api_key_revoke,
            commands::settings::workspace_register,
            commands::settings::workspace_remove,
            commands::chat::chat_start,
            commands::chat::chat_cancel,
            commands::context::context_index,
            commands::context::context_index_status,
            commands::context::context_search,
            commands::context::context_build,
            commands::context::context_recover,
            commands::context::context_symbols,
            commands::context::context_impact,
            commands::context::context_set_embedding_model,
            commands::context::context_clear_embedding_model,
            commands::context::context_semantic_status,
            commands::context::context_remember_project,
            commands::context::context_forget_project_memory,
            commands::context::context_project_memory,
            commands::conversations::conversations_list,
            commands::conversations::conversations_create,
            commands::conversations::conversations_get,
            commands::conversations::conversations_rename,
            commands::conversations::conversations_delete,
            commands::conversations::conversations_append,
            commands::conversations::conversations_truncate,
            commands::conversations::conversations_replace_message,
            commands::conversations::conversations_export,
            commands::conversations::conversations_import,
            commands::conversations::conversations_search,
            commands::conversations::conversations_export_to,
        ])
        .build(tauri::generate_context!());
    let app = match built {
        Ok(app) => app,
        Err(e) => {
            eprintln!("Raptor LLM could not start: {e}");
            std::process::exit(1);
        }
    };
    app.run(|handle, event| {
        // Stop the model server and the gateway before the process exits, so no
        // llama-server is left running after the window closes.
        if let RunEvent::Exit = event {
            let state = handle.state::<AppState>();
            tauri::async_runtime::block_on(state.shutdown());
        }
    });
}
