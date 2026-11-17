//! Model discovery, downloads, the local library and load/unload.

use crate::commands::{
    parse_abs_file, parse_dir, EVENT_DOWNLOAD_FINISHED, EVENT_DOWNLOAD_PROGRESS,
};
use crate::state::{
    hardware_summary, AppState, DownloadFinished, LoadFailure, LoadModelResult, QueueEntry,
};
use raptor_core::{RaptorError, Result};
use raptor_models::download::RecoveryReport;
use raptor_models::{
    Comparison, DownloadOutcome, HfModelSummary, HfRepoInfo, ModelDetails, ModelEntry,
    RankedCandidate, RepoFile, ScanItem, SearchSort, StorageUsage, Task, UpdateCheck,
};
use raptor_runtime::EngineStatus;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

const MAX_QUERY_CHARS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportMode {
    /// Copy the file into the library (managed; deleted with the entry).
    Copy,
    /// Reference the file where it is (the library never deletes it).
    Reference,
}

fn check_repo(repo: &str) -> Result<String> {
    let r = repo.trim();
    if r.is_empty() {
        return Err(RaptorError::InvalidInput("repository id is empty".into()));
    }
    Ok(r.to_string())
}

#[tauri::command]
pub async fn models_search(
    state: State<'_, AppState>,
    query: String,
    sort: Option<SearchSort>,
    limit: Option<u32>,
) -> Result<Vec<HfModelSummary>> {
    let q = query.trim();
    if q.is_empty() || q.chars().count() > MAX_QUERY_CHARS {
        return Err(RaptorError::InvalidInput(format!(
            "query must be 1-{MAX_QUERY_CHARS} characters"
        )));
    }
    let limit = limit.unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return Err(RaptorError::InvalidInput("limit must be 1-100".into()));
    }
    state
        .models
        .search(q, sort.unwrap_or(SearchSort::Downloads), limit)
        .await
}

/// Top-level GGUF files of a repository, with size, sha256 and quant label.
#[tauri::command]
pub async fn models_repo_files(state: State<'_, AppState>, repo: String) -> Result<Vec<RepoFile>> {
    state.models.repo_files(&check_repo(&repo)?).await
}

#[tauri::command]
pub async fn models_repo_info(state: State<'_, AppState>, repo: String) -> Result<HfRepoInfo> {
    state.models.repo_info(&check_repo(&repo)?).await
}

#[tauri::command]
pub async fn models_list(state: State<'_, AppState>) -> Result<Vec<ModelEntry>> {
    state.models.library().list()
}

#[tauri::command]
pub async fn models_storage(state: State<'_, AppState>) -> Result<StorageUsage> {
    state.models.library().storage_usage()
}

/// Removes a library entry and its presets. A referenced (not copied) file stays on disk.
#[tauri::command]
pub async fn models_delete(state: State<'_, AppState>, id: String) -> Result<()> {
    state.models.delete_model(&id)
}

#[tauri::command]
pub async fn models_import(
    state: State<'_, AppState>,
    path: String,
    mode: ImportMode,
) -> Result<ModelEntry> {
    let src = parse_abs_file(&path)?;
    // Hashing a multi-GB file takes minutes; keep it off the async worker threads.
    let models = state.models.clone();
    tokio::task::spawn_blocking(move || match mode {
        ImportMode::Copy => models.library().import_copy(&src),
        ImportMode::Reference => models.library().import_reference(&src),
    })
    .await
    .map_err(|e| RaptorError::Internal(format!("import task failed: {e}")))?
}

/// `{done, total}` bytes hashed by the import running now (or the last one). Poll while
/// `models_import` runs; both are 0 before any import.
#[tauri::command]
pub async fn models_import_progress(state: State<'_, AppState>) -> Result<ImportProgress> {
    let (done, total) = state.models.library().import_progress();
    Ok(ImportProgress { done, total })
}

/// Bytes hashed so far by an import, out of the bytes it has to hash.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportProgress {
    pub done: u64,
    pub total: u64,
}

/// Lists GGUF files in a folder and whether each is already in the library.
#[tauri::command]
pub async fn models_scan_folder(state: State<'_, AppState>, dir: String) -> Result<Vec<ScanItem>> {
    let path = parse_dir(&dir)?;
    let items = state.models.library().scan_folder(&path)?;
    // A folder that was scanned is offered again later (chat window, `models_discover`).
    if !items.is_empty() {
        let _ = state.settings.update(|s| {
            if !s.model_folders.contains(&path) {
                s.model_folders.push(path.clone());
                let extra = s.model_folders.len().saturating_sub(20);
                s.model_folders.drain(..extra);
            }
            Ok(())
        });
    }
    Ok(items)
}

/// A folder where models were found that are not in the library yet.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredFolder {
    pub path: String,
    /// Which app the folder belongs to, or "Scanned by you before".
    pub source: String,
    /// Only items that can be added: new, valid GGUF files.
    pub items: Vec<ScanItem>,
}

/// Looks in the usual folders of LM Studio, Jan, GPT4All, llama.cpp and the Hugging Face cache,
/// plus folders scanned before, for GGUF models that are not in the library. Reads headers only;
/// nothing is copied or hashed. Adding one is `models_import` with mode `reference`.
#[tauri::command]
pub async fn models_discover(state: State<'_, AppState>) -> Result<Vec<DiscoveredFolder>> {
    let home = dirs::home_dir();
    let local = dirs::data_local_dir();
    let roaming = dirs::data_dir();
    let mut known = home
        .as_deref()
        .map(|h| {
            // The folder LM Studio is actually configured to use comes first.
            let mut list = crate::discover::lm_studio_configured_folders(h, roaming.as_deref());
            list.extend(crate::discover::known_model_folders(h, local.as_deref()));
            list
        })
        .unwrap_or_default();
    known.dedup_by(|a, b| a.path == b.path);
    let added = state.settings.snapshot().model_folders;
    let folders = crate::discover::candidate_folders(known, &added, |p| p.is_dir());
    let models = state.models.clone();
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        for folder in folders {
            let Ok(items) = models.library().scan_folder(&folder.path) else {
                continue;
            };
            let items: Vec<ScanItem> = items
                .into_iter()
                .filter(|i| matches!(i.status, raptor_models::library::ScanStatus::New))
                .collect();
            if !items.is_empty() {
                out.push(DiscoveredFolder {
                    path: folder.path.display().to_string(),
                    source: folder.source,
                    items,
                });
            }
        }
        out
    })
    .await
    .map_err(|e| RaptorError::Internal(format!("discovery task failed: {e}")))
}

/// Starts a verified download into the library. Progress and completion are emitted as
/// events. Returns the job key used by pause, resume and cancel.
#[tauri::command]
pub async fn models_download_start(
    app: AppHandle,
    state: State<'_, AppState>,
    repo: String,
    file_name: String,
    overwrite: Option<bool>,
) -> Result<String> {
    let repo = check_repo(&repo)?;
    let handle = state
        .models
        .start_download(&repo, &file_name, overwrite.unwrap_or(false))
        .await?;
    let key = handle.key().to_string();
    let progress = handle.progress();
    state.clear_download_outcome(&key);
    let watcher_app = app.clone();
    let watcher_key = key.clone();
    tauri::async_runtime::spawn(async move {
        let emitter = watcher_app.clone();
        let mut rx = progress;
        let forward = tauri::async_runtime::spawn(async move {
            loop {
                let snapshot = rx.borrow().clone();
                let _ = emitter.emit(EVENT_DOWNLOAD_PROGRESS, &snapshot);
                if rx.changed().await.is_err() {
                    break;
                }
            }
        });
        let outcome = handle.finish().await;
        let _ = forward.await;
        let finished = match outcome {
            Ok(DownloadOutcome::Completed { path, sha256 }) => DownloadFinished {
                key: watcher_key,
                status: "completed".into(),
                path: Some(path.display().to_string()),
                sha256: Some(sha256),
                error: None,
            },
            Ok(DownloadOutcome::Cancelled) => DownloadFinished {
                key: watcher_key,
                status: "cancelled".into(),
                path: None,
                sha256: None,
                error: None,
            },
            Err(e) => DownloadFinished {
                key: watcher_key,
                status: "failed".into(),
                path: None,
                sha256: None,
                error: Some(e.to_string()),
            },
        };
        let state = watcher_app.state::<AppState>();
        state.record_download_finished(finished.clone());
        let _ = watcher_app.emit(EVENT_DOWNLOAD_FINISHED, &finished);
    });
    Ok(key)
}

#[tauri::command]
pub async fn models_download_pause(state: State<'_, AppState>, key: String) -> Result<()> {
    state.models.downloads().pause(&key)
}

#[tauri::command]
pub async fn models_download_resume(state: State<'_, AppState>, key: String) -> Result<()> {
    state.models.downloads().resume(&key)
}

/// Cancels a download and deletes its partial file.
#[tauri::command]
pub async fn models_download_cancel(state: State<'_, AppState>, key: String) -> Result<()> {
    state.models.downloads().cancel(&key)
}

/// Downloads started in this session, with their latest progress.
#[tauri::command]
pub async fn models_download_queue(state: State<'_, AppState>) -> Result<Vec<QueueEntry>> {
    state.download_queue()
}

/// Interrupted downloads found on disk from earlier sessions. Start the same file again to
/// resume it.
#[tauri::command]
pub async fn models_recover_downloads(state: State<'_, AppState>) -> Result<RecoveryReport> {
    state.models.downloads().recover()
}

#[tauri::command]
pub async fn models_discard_partial(state: State<'_, AppState>, key: String) -> Result<()> {
    state.models.downloads().discard_partial(&key)
}

/// Loads a library model with the runtime's recommended settings (or the given context).
/// `preset` applies a saved preset of this model (its context length, unless `context` is
/// given). The GPU verdict in the result is the honest state; it is never inferred.
/// A failure is `{kind, message, backend_failure}`: `backend_failure` carries the typed GPU
/// diagnosis with its suggested actions, and is null for every other failure.
#[tauri::command]
pub async fn models_load(
    state: State<'_, AppState>,
    model_id: String,
    context: Option<u32>,
    preset: Option<String>,
    cache_type: Option<String>,
    slots: Option<u32>,
    speculation: Option<bool>,
) -> std::result::Result<LoadModelResult, LoadFailure> {
    let kv_cache = match cache_type.as_deref() {
        None => None,
        Some("f16") => Some(raptor_runtime::config::KvCacheType::F16),
        Some("q8_0") => Some(raptor_runtime::config::KvCacheType::Q8_0),
        Some("q4_0") => Some(raptor_runtime::config::KvCacheType::Q4_0),
        Some(other) => {
            return Err(LoadFailure::from(RaptorError::InvalidInput(format!(
                "unknown cache type {other}; use f16, q8_0 or q4_0"
            ))))
        }
    };
    state
        .load_model_ext(
            &model_id,
            context,
            preset.as_deref(),
            crate::state::LoadExtras {
                kv_cache,
                slots,
                speculation,
            },
        )
        .await
}

/// Full details of an installed model, including the pinned Hub revision (`revision`, null for
/// models installed before revisions were recorded) and the files of a split model.
#[tauri::command]
pub async fn models_details(state: State<'_, AppState>, id: String) -> Result<ModelDetails> {
    state.models.model_details(&id)
}

/// Side-by-side comparison of installed models at `context` tokens (default: the settings
/// context target), rated against this machine's RAM and VRAM.
#[tauri::command]
pub async fn models_compare(
    state: State<'_, AppState>,
    ids: Vec<String>,
    context: Option<u64>,
) -> Result<Comparison> {
    if ids.is_empty() || ids.len() > 8 {
        return Err(RaptorError::InvalidInput(
            "compare 1-8 installed models".into(),
        ));
    }
    let hw = hardware_summary(&state.hardware_for_use().await?);
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let ctx = context.unwrap_or(u64::from(state.settings.snapshot().context_target));
    state.models.compare(&refs, ctx, Some(hw))
}

/// Ranks the installed models for `task`. Tags are heuristic (see the models README); the
/// evidence for each tag is in the result.
#[tauri::command]
pub async fn models_rank_for(
    state: State<'_, AppState>,
    task: Task,
    context: Option<u64>,
) -> Result<Vec<RankedCandidate>> {
    let hw = hardware_summary(&state.hardware_for_use().await?);
    let ctx = context.unwrap_or(u64::from(state.settings.snapshot().context_target));
    state.models.rank_installed(task, hw, ctx)
}

/// Compares the repo head of every installed Hub model with the commit it was downloaded
/// from. Makes no download and changes nothing. Fails with `Network` when offline.
#[tauri::command]
pub async fn models_check_updates(state: State<'_, AppState>) -> Result<Vec<UpdateCheck>> {
    state.models.check_updates().await
}

#[tauri::command]
pub async fn models_unload(state: State<'_, AppState>) -> Result<EngineStatus> {
    state.unload_model().await
}

/// Returns the currently loaded model entry from the library, or None if no model is loaded.
#[tauri::command]
pub async fn models_current(state: State<'_, AppState>) -> Result<Option<ModelEntry>> {
    let id = state.current_model_id();
    match id {
        Some(model_id) => state.models.library().get(&model_id).map(Some),
        None => Ok(None),
    }
}

/// The report of the load that is serving now (result, planned context and GPU layers), so a
/// reloaded window can show what the app chose. Null when nothing was loaded by this app run.
#[tauri::command]
pub async fn models_load_report(
    state: State<'_, AppState>,
) -> Result<Option<crate::state::StoredLoadReport>> {
    Ok(state.current_load_report())
}

/// Progress of the model load running now: `{active, step, stage_count, label, elapsed_ms}`.
/// Poll while `models_load` runs. The stage comes from llama-server's log; no percentage is
/// invented because the server prints none.
#[tauri::command]
pub async fn models_load_progress(
    state: State<'_, AppState>,
) -> Result<crate::load_progress::LoadProgress> {
    Ok(state.load_progress())
}

/// Measures a model with llama-bench and saves the fastest layer count and micro-batch for it, so
/// every later load uses them. Takes a few minutes. The model must not be loaded. With `deep`
/// it also measures every setup with 16,384 tokens of context and saves a long-prompt setting for
/// windows of that size or more (several minutes longer; progress counts both passes).
#[tauri::command]
pub async fn models_tune(
    state: State<'_, AppState>,
    model_id: String,
    deep: Option<bool>,
) -> Result<crate::state::TuneReport> {
    state.tune_model(&model_id, deep.unwrap_or(false)).await
}

/// `{done, total}` steps of the tuning run in progress; `{0, 0}` when none is running.
#[tauri::command]
pub async fn models_tune_progress(state: State<'_, AppState>) -> Result<ImportProgress> {
    let (done, total) = state.tune_progress();
    Ok(ImportProgress {
        done: u64::from(done),
        total: u64::from(total),
    })
}

/// Forgets the tuning of a model, so the general speed rules apply to it again.
#[tauri::command]
pub async fn models_forget_tuning(state: State<'_, AppState>, model_id: String) -> Result<bool> {
    state.forget_tuning(&model_id).await
}

/// The context windows the GPU can hold for a model (default, large, maximum), from llama.cpp's
/// own memory projection. Pass an option's `ctx`, `kv_cache` (as `cache_type`) and `slots` to
/// `models_load` to use it.
#[tauri::command]
pub async fn models_context_options(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Vec<crate::state::ContextOption>> {
    state.context_options(&model_id).await
}

/// The backend (ROCm or Vulkan) choice for a model, what a load would use now, and which runtimes
/// are installed.
#[tauri::command]
pub async fn models_backend(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<crate::state::BackendView> {
    state.backend_view(&model_id).await
}

/// Whether the loaded model has a thinking-effort control (Low, Medium, High, Max), and which of
/// the model's own levels each one means. `supported: false` when no model is loaded or the model
/// has no such control.
#[tauri::command]
pub async fn models_thinking(state: State<'_, AppState>) -> Result<crate::thinking::ThinkingView> {
    Ok(state.thinking_view().await)
}

/// Chooses the backend for a model: `auto`, `rocm` or `vulkan`. Takes effect at the next load.
#[tauri::command]
pub async fn models_set_backend(
    state: State<'_, AppState>,
    model_id: String,
    backend: crate::backend_choice::BackendPref,
) -> Result<crate::state::BackendView> {
    state.set_backend_pref(&model_id, backend)?;
    state.backend_view(&model_id).await
}

/// Measures prefill speed, decode speed, the latency of a 512-token turn and peak GPU memory at
/// growing context lengths (0, 4K, 16K, 32K, 64K, 128K, 192K, up to `max_depth`, default 32,768), with
/// the cheapest KV cache that fits each. Takes minutes (a 128K run can take 20+). The model must
/// not be loaded. Poll `models_tune_progress` for steps done.
#[tauri::command]
pub async fn models_context_matrix(
    state: State<'_, AppState>,
    model_id: String,
    max_depth: Option<u32>,
) -> Result<crate::ctxbench::MatrixReport> {
    let depth = max_depth
        .unwrap_or(32_768)
        .clamp(4096, raptor_runtime::context_max::MAX_CONTEXT_TOKENS);
    state.context_matrix(&model_id, depth).await
}

/// The saved matrix for a model on this machine and runtime build, or null.
#[tauri::command]
pub async fn models_context_matrix_last(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Option<crate::ctxbench::MatrixReport>> {
    state.last_context_matrix(&model_id).await
}
