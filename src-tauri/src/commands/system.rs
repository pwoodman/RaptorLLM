//! Hardware, runtime, gateway and diagnostics commands.

use crate::commands::EVENT_GATEWAY_STATE;
use crate::settings::PinnedRuntime;
use crate::setup::SetupStatus;
use crate::state::{candidates_for_pin, AppState, GatewayStatus};
use raptor_core::{RaptorError, Result};
use raptor_runtime::config::Recommendation;
use raptor_runtime::hardware::HardwareInfo;
use raptor_runtime::manifest::{Backend, Manifest};
use raptor_runtime::EngineStatus;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tokio_util::sync::CancellationToken;

/// The runtime chosen by `runtime_ensure`. `degraded` is true only for a CPU build chosen
/// because no GPU build is compatible; the UI must show it. `rejected` lists candidates that
/// were not used, including entries refused for lacking a verified sha256.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSelectionView {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub degraded: bool,
    pub reason: String,
    pub gpu_arch: Option<String>,
    pub rejected: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeRollbackView {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub dir: String,
    pub binary: String,
}

/// Hardware detected at startup (cached), or probed now if nothing is cached yet.
#[tauri::command]
pub async fn hardware_info(state: State<'_, AppState>) -> Result<HardwareInfo> {
    state.hardware_for_use().await
}

/// Forces a new OS/device probe instead of returning this session's cached snapshot.
#[tauri::command]
pub async fn hardware_refresh(state: State<'_, AppState>) -> Result<HardwareInfo> {
    state.refresh_hardware().await
}

/// First-run checklist: hardware, runtime, GPU, model, storage and performance memory. It
/// reads state only and never downloads or installs anything.
#[tauri::command]
pub async fn setup_status(state: State<'_, AppState>) -> Result<SetupStatus> {
    Ok(state.setup_status().await)
}

#[tauri::command]
pub async fn runtime_status(state: State<'_, AppState>) -> Result<EngineStatus> {
    Ok(state.runtime.status())
}

/// Selects the best runtime for this machine and installs it if needed. While a runtime is
/// pinned, only the pinned version is considered, so a newer install is never promoted over
/// it. Refused while a model is running, because the running binary must not be replaced.
#[tauri::command]
pub async fn runtime_ensure(state: State<'_, AppState>) -> Result<RuntimeSelectionView> {
    let index = state.load_runtime_index()?;
    let pin = state.settings.snapshot().pinned_runtime;
    let candidates = candidates_for_pin(index.manifests, pin.as_ref())?;
    let _guard = state.runtime_lock.lock().await;
    refuse_while_serving(&state)?;
    // Drivers and GPUs may have changed since startup. Installation must select against a fresh
    // probe so an old CPU-only or wrong-vendor cache cannot silently choose the wrong build.
    state.refresh_hardware().await?;
    let selection = state
        .runtime
        .ensure_runtime(&candidates, &CancellationToken::new())
        .await?;
    let mut rejected = selection.rejected.clone();
    rejected.extend(index.refused);
    Ok(RuntimeSelectionView {
        id: selection.manifest.id.clone(),
        version: selection.manifest.version.clone(),
        backend: selection.backend,
        degraded: selection.degraded,
        reason: selection.reason.clone(),
        gpu_arch: selection.gpu_arch.clone(),
        rejected,
    })
}

/// Bytes of the runtime download written so far. 0 when no download has started. Poll while
/// `runtime_ensure` runs; the total is `download_bytes` from `runtime_install_preview`.
#[tauri::command]
pub async fn runtime_install_progress(state: State<'_, AppState>) -> Result<u64> {
    Ok(state
        .install_progress
        .load(std::sync::atomic::Ordering::Relaxed))
}

/// Makes the previous installed version of runtime `id` current again. The restored version
/// is pinned, so `runtime_ensure` keeps it until `runtime_unpin` is called.
#[tauri::command]
pub async fn runtime_rollback(
    state: State<'_, AppState>,
    id: String,
) -> Result<RuntimeRollbackView> {
    let _guard = state.runtime_lock.lock().await;
    refuse_while_serving(&state)?;
    let active = state.runtime.status().runtime.ok_or_else(|| {
        RaptorError::Conflict("no runtime is selected yet; run runtime_ensure first".into())
    })?;
    if active.id != id {
        return Err(RaptorError::InvalidInput(format!(
            "runtime {id} is not the one in use ({})",
            active.id
        )));
    }
    let restored = state.runtime.rollback()?;
    let dir = state
        .dirs
        .runtimes()
        .join(&restored.id)
        .join(&restored.version)
        .display()
        .to_string();
    let pin = PinnedRuntime {
        id: restored.id.clone(),
        version: restored.version.clone(),
    };
    if let Err(e) = state.settings.update(|s| {
        s.pinned_runtime = Some(pin);
        Ok(())
    }) {
        // The pin is what keeps the rollback across restarts. Without it the change is
        // undone, so the UI never shows a rollback that would be lost.
        let _ = state.runtime.rollback();
        return Err(e);
    }
    Ok(RuntimeRollbackView {
        id: restored.id,
        version: restored.version,
        backend: restored.backend,
        dir,
        binary: restored.binary.display().to_string(),
    })
}

/// Removes the runtime pin, so `runtime_ensure` follows the newest compatible version again.
#[tauri::command]
pub async fn runtime_unpin(state: State<'_, AppState>) -> Result<()> {
    state.settings.update(|s| {
        s.pinned_runtime = None;
        Ok(())
    })?;
    Ok(())
}

/// Recommended launch settings for a library model on this machine, without loading it.
#[tauri::command]
pub async fn runtime_recommend(
    state: State<'_, AppState>,
    model_id: String,
    context: Option<u32>,
) -> Result<Recommendation> {
    let entry = state.models.library().get(&model_id)?;
    let path = state.models.library().file_path(&model_id)?;
    let ctx = context.unwrap_or(state.settings.snapshot().context_target);
    state.recommend_for_entry(&entry, &path, ctx).await
}

#[tauri::command]
pub async fn gateway_status(state: State<'_, AppState>) -> Result<GatewayStatus> {
    Ok(state.gateway_status().await)
}

/// Replaces the gateway's self-signed certificate on the running listener and returns the
/// new status, including the new fingerprint. Clients that pinned the old one must re-pin.
#[tauri::command]
pub async fn gateway_rotate_tls(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<GatewayStatus> {
    let status = state.rotate_gateway_tls().await?;
    let _ = app.emit(EVENT_GATEWAY_STATE, &status);
    Ok(status)
}

/// Last lines of llama-server output, as JSON objects `{"stream", "text"}`. At most 2000.
#[tauri::command]
pub async fn diagnostics_log_tail(
    state: State<'_, AppState>,
    lines: Option<usize>,
) -> Result<Vec<String>> {
    let n = lines.unwrap_or(200);
    if n == 0 || n > 2000 {
        return Err(RaptorError::InvalidInput("lines must be 1-2000".into()));
    }
    Ok(state.server_log_lines(n))
}

/// The log folder and the files in it (app log and llama-server log, newest first). Files
/// older than seven days are deleted automatically.
#[derive(Debug, Clone, Serialize)]
pub struct LogFilesView {
    pub dir: String,
    pub keep_days: u32,
    pub files: Vec<LogFileView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogFileView {
    pub name: String,
    pub size_bytes: u64,
    pub modified_unix: u64,
}

#[tauri::command]
pub async fn diagnostics_log_files(state: State<'_, AppState>) -> Result<LogFilesView> {
    let dir = state.dirs.logs();
    let files = crate::logging::list_log_files(&dir)
        .into_iter()
        .map(|(name, size_bytes, modified_unix)| LogFileView {
            name,
            size_bytes,
            modified_unix,
        })
        .collect();
    Ok(LogFilesView {
        dir: dir.display().to_string(),
        keep_days: raptor_core::logfiles::KEEP_DAYS,
        files,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeEntryView {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub installed: bool,
    pub current: bool,
    pub dir: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeListView {
    pub active: Option<raptor_runtime::RuntimeSummary>,
    pub runtimes: Vec<RuntimeEntryView>,
}

/// Lists all installed runtimes on disk and available runtimes from the manifest index.
#[tauri::command]
pub async fn runtime_list(state: State<'_, AppState>) -> Result<RuntimeListView> {
    let installed = state.runtime.installed_runtimes()?;
    let active = state.runtime.status().runtime;
    let index = state.load_runtime_index().unwrap_or_default();

    let mut entries = Vec::new();
    for inst in &installed {
        let is_curr = active
            .as_ref()
            .map(|a| a.id == inst.id && a.version == inst.version)
            .unwrap_or(false);
        entries.push(RuntimeEntryView {
            id: inst.id.clone(),
            version: inst.version.clone(),
            backend: inst.backend,
            installed: true,
            current: is_curr,
            dir: Some(inst.dir.display().to_string()),
        });
    }

    for m in &index.manifests {
        if !entries
            .iter()
            .any(|e| e.id == m.id && e.version == m.version)
        {
            entries.push(RuntimeEntryView {
                id: m.id.clone(),
                version: m.version.clone(),
                backend: m.backend,
                installed: false,
                current: false,
                dir: None,
            });
        }
    }

    Ok(RuntimeListView {
        active,
        runtimes: entries,
    })
}

/// What `runtime_ensure` would install, before anything is downloaded. The source URL and
/// SHA-256 come from the runtime index. The size comes from the server's `Content-Length`
/// for that URL, and is null when the server does not report one or cannot be reached (see
/// `notes`). Nothing is downloaded or installed here.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeInstallPreview {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub degraded: bool,
    pub reason: String,
    pub gpu_arch: Option<String>,
    pub url: String,
    pub sha256: String,
    /// Download size in bytes, from the server. Null when unknown.
    pub download_bytes: Option<u64>,
    /// `content_length` when the server reported the size, otherwise `unavailable`.
    pub size_source: String,
    /// True when this exact version is already installed: nothing would be downloaded.
    pub already_installed: bool,
    pub rejected: Vec<String>,
    pub notes: Vec<String>,
}

const PREVIEW_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Previews the runtime `runtime_ensure` would install on this machine. Honors the pin.
#[tauri::command]
pub async fn runtime_install_preview(state: State<'_, AppState>) -> Result<RuntimeInstallPreview> {
    let index = state.load_runtime_index()?;
    let pin = state.settings.snapshot().pinned_runtime;
    let candidates = candidates_for_pin(index.manifests, pin.as_ref())?;
    let hw = state.refresh_hardware().await?;
    let selection = raptor_runtime::manifest::select_runtime(&hw, &candidates)?;
    let installed = state.runtime.installed_runtimes()?;
    let already_installed = installed
        .iter()
        .any(|r| r.id == selection.manifest.id && r.version == selection.manifest.version);
    let mut rejected = selection.rejected.clone();
    rejected.extend(index.refused);

    let mut notes = Vec::new();
    let (download_bytes, size_source) = if already_installed {
        (None, "unavailable".to_string())
    } else {
        match remote_size(&selection.manifest.url).await {
            Ok(Some(n)) => (Some(n), "content_length".to_string()),
            Ok(None) => {
                notes.push("The server did not report the download size.".into());
                (None, "unavailable".to_string())
            }
            Err(e) => {
                notes.push(format!("The download size could not be read: {e}"));
                (None, "unavailable".to_string())
            }
        }
    };
    Ok(RuntimeInstallPreview {
        id: selection.manifest.id.clone(),
        version: selection.manifest.version.clone(),
        backend: selection.backend,
        degraded: selection.degraded,
        reason: selection.reason.clone(),
        gpu_arch: selection.gpu_arch.clone(),
        url: selection.manifest.url.clone(),
        sha256: selection.manifest.sha256.clone(),
        download_bytes,
        size_source,
        already_installed,
        rejected,
        notes,
    })
}

/// The `Content-Length` of `url` after redirects, with a HEAD request. `None` when absent.
async fn remote_size(url: &str) -> Result<Option<u64>> {
    let client = reqwest::Client::builder()
        .timeout(PREVIEW_TIMEOUT)
        .build()
        .map_err(|e| RaptorError::Network(format!("http client: {e}")))?;
    let resp = client
        .head(url)
        .send()
        .await
        .map_err(|e| RaptorError::Network(format!("HEAD {url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(RaptorError::Network(format!(
            "HEAD {url} returned HTTP {}",
            resp.status().as_u16()
        )));
    }
    // `Response::content_length` reports the body size, which is always 0 for a HEAD response,
    // so read the header itself. A zero is treated as "not reported".
    Ok(resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0))
}

/// Opens a native file picker dialog for selecting a model file.
#[tauri::command]
pub async fn dialog_pick_file(filter_ext: Option<Vec<String>>) -> Result<Option<String>> {
    tokio::task::spawn_blocking(move || pick_file_dialog(filter_ext.as_deref()))
        .await
        .map_err(|e| RaptorError::Internal(format!("file dialog task failed: {e}")))?
}

/// Opens a native directory picker dialog for selecting a folder.
#[tauri::command]
pub async fn dialog_pick_folder() -> Result<Option<String>> {
    tokio::task::spawn_blocking(pick_folder_dialog)
        .await
        .map_err(|e| RaptorError::Internal(format!("folder dialog task failed: {e}")))?
}

#[cfg(windows)]
fn pick_file_dialog(filter_ext: Option<&[String]>) -> Result<Option<String>> {
    let filter_str = if let Some(exts) = filter_ext {
        let pattern = exts
            .iter()
            .map(|e| format!("*.{e}"))
            .collect::<Vec<_>>()
            .join(";");
        format!("Supported files ({pattern})|{pattern}|All files (*.*)|*.*")
    } else {
        "All files (*.*)|*.*".to_string()
    };
    let script = format!(
        "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
         $d = New-Object System.Windows.Forms.OpenFileDialog; \
         $d.Filter = '{filter_str}'; \
         $d.RestoreDirectory = $true; \
         if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ \
             [Console]::Write($d.FileName) \
         }}"
    );
    run_powershell_dialog(&script)
}

#[cfg(windows)]
fn pick_folder_dialog() -> Result<Option<String>> {
    let script =
        "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
         $d = New-Object System.Windows.Forms.FolderBrowserDialog; \
         $d.ShowNewFolderButton = $true; \
         if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { \
             [Console]::Write($d.SelectedPath) \
         }";
    run_powershell_dialog(script)
}

/// Opens the native save dialog with `default_name` (a plain file name) and returns the path
/// the user chose, or null when they cancel. The UI writes the file through
/// `conversations_export_to`, so the chosen path is the only place a file is written.
#[tauri::command]
pub async fn dialog_save_file(
    default_name: String,
    filter_ext: Option<String>,
) -> Result<Option<String>> {
    let name = raptor_core::paths::safe_file_name(default_name.trim())?.to_string();
    let ext = filter_ext
        .map(|e| e.trim_start_matches('.').to_string())
        .filter(|e| !e.is_empty() && e.bytes().all(|b| b.is_ascii_alphanumeric()));
    tokio::task::spawn_blocking(move || save_file_dialog(&name, ext.as_deref()))
        .await
        .map_err(|e| RaptorError::Internal(format!("save dialog task failed: {e}")))?
}

#[cfg(windows)]
fn save_file_dialog(name: &str, ext: Option<&str>) -> Result<Option<String>> {
    let filter = match ext {
        Some(e) => format!(
            "{} files (*.{e})|*.{e}|All files (*.*)|*.*",
            e.to_uppercase()
        ),
        None => "All files (*.*)|*.*".to_string(),
    };
    let script = format!(
        "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
         $d = New-Object System.Windows.Forms.SaveFileDialog; \
         $d.Filter = '{}'; \
         $d.FileName = '{}'; \
         $d.AddExtension = $true; \
         $d.OverwritePrompt = $true; \
         $d.RestoreDirectory = $true; \
         if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ \
             [Console]::Write($d.FileName) \
         }}",
        filter.replace('\'', "''"),
        name.replace('\'', "''"),
    );
    run_powershell_dialog(&script)
}

#[cfg(not(windows))]
fn save_file_dialog(_name: &str, _ext: Option<&str>) -> Result<Option<String>> {
    Ok(None)
}

#[cfg(windows)]
fn run_powershell_dialog(script: &str) -> Result<Option<String>> {
    use std::os::windows::process::CommandExt;
    let mut cmd = std::process::Command::new("powershell");
    let args = raptor_runtime::sys::powershell_encoded_args(script);
    cmd.arg("-STA");
    cmd.args(&args);
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let output = cmd
        .output()
        .map_err(|e| RaptorError::Runtime(format!("dialog process: {e}")))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let reason = if detail.is_empty() {
            format!("PowerShell exited with status {:?}", output.status.code())
        } else {
            detail.chars().take(400).collect()
        };
        return Err(RaptorError::Runtime(format!(
            "Windows file/folder dialog failed: {reason}"
        )));
    }
    let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if selected.is_empty() {
        Ok(None)
    } else {
        Ok(Some(selected))
    }
}

#[cfg(not(windows))]
fn pick_file_dialog(_filter_ext: Option<&[String]>) -> Result<Option<String>> {
    Ok(None)
}

#[cfg(not(windows))]
fn pick_folder_dialog() -> Result<Option<String>> {
    Ok(None)
}

/// Applies settings to the gateway and tells the UI the new state.
pub(crate) async fn gateway_changed(app: &AppHandle, state: &AppState) -> Result<GatewayStatus> {
    let status = state.apply_gateway().await;
    let snapshot = state.gateway_status().await;
    let _ = app.emit(EVENT_GATEWAY_STATE, &snapshot);
    status
}

fn refuse_while_serving(state: &AppState) -> Result<()> {
    if state.runtime.status().server.state != raptor_runtime::process::ServerState::Stopped {
        return Err(RaptorError::Conflict(
            "a model is loaded; unload it before changing the runtime".into(),
        ));
    }
    Ok(())
}

/// Result of checking the dependencies. A failed network check is reported in `check_error`; the
/// rest (installed versions, HIP SDK) is still returned.
#[derive(Debug, Clone, Serialize)]
pub struct DependencyReport {
    pub latest_tag: Option<String>,
    pub check_error: Option<String>,
    pub updates: Vec<crate::updates::RuntimeUpdate>,
    /// Present when an AMD GPU or an AMD runtime is in play (Windows).
    pub hip_sdk: Option<crate::updates::HipSdkStatus>,
    /// The Vulkan runtime: installed or not, and the verified download that adds or updates it.
    /// Null when the release could not be read.
    pub vulkan: Option<crate::updates::VulkanOffer>,
}

/// Looks for a newer llama.cpp build of each installed runtime and checks the AMD HIP SDK.
/// Reads release metadata only; nothing is downloaded or installed.
#[tauri::command]
pub async fn dependencies_check(state: State<'_, AppState>) -> Result<DependencyReport> {
    let installed = state.runtime.installed_runtimes()?;
    let wanted: Vec<String> = installed.iter().map(|r| r.id.clone()).collect();
    let index = state
        .load_runtime_index()
        .map(|i| i.manifests)
        .unwrap_or_default();
    let source = state
        .releases
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let vulkan_installed = installed
        .iter()
        .filter(|r| r.id == crate::updates::VULKAN_RUNTIME_ID)
        .map(|r| r.version.clone())
        .max_by_key(|v| raptor_runtime::manifest::version_key(v));
    let (latest_tag, updates, check_error, vulkan) = match source.latest().await {
        Ok(release) => (
            Some(release.tag.clone()),
            crate::updates::plan_updates(&index, &release, &wanted),
            None,
            Some(crate::updates::plan_vulkan(
                &release,
                vulkan_installed.as_deref(),
                cfg!(windows),
            )),
        ),
        Err(e) => (None, Vec::new(), Some(e.to_string()), None),
    };
    let amd_in_play = installed
        .iter()
        .any(|r| r.backend == raptor_runtime::manifest::Backend::Hip)
        || state
            .hardware_for_use()
            .await
            .map(|hw| {
                hw.gpus
                    .iter()
                    .any(|g| g.vendor == raptor_runtime::hardware::GpuVendor::Amd)
            })
            .unwrap_or(false);
    let hip_sdk = (cfg!(windows) && amd_in_play).then(|| {
        crate::updates::hip_sdk_status(&raptor_runtime::devices::HipSdkSearch::system().bin_dirs())
    });
    Ok(DependencyReport {
        latest_tag,
        check_error,
        updates,
        hip_sdk,
        vulkan,
    })
}

/// Installs (or updates) the Vulkan runtime that `dependencies_check` offered, next to ROCm. The
/// release is read again here, so nothing the UI sends decides the URL or checksum; the archive is
/// hash-checked before extraction. The runtime in use does not change: each model picks its own
/// backend at load. The build is added to the runtime index so later updates find it. Refused
/// while a model is loaded.
#[tauri::command]
pub async fn dependencies_install_vulkan(state: State<'_, AppState>) -> Result<String> {
    refuse_while_serving(&state)?;
    let installed = state.runtime.installed_runtimes()?;
    let have = installed
        .iter()
        .filter(|r| r.id == crate::updates::VULKAN_RUNTIME_ID)
        .map(|r| r.version.clone())
        .max_by_key(|v| raptor_runtime::manifest::version_key(v));
    let source = state
        .releases
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let release = source.latest().await?;
    let offer = crate::updates::plan_vulkan(&release, have.as_deref(), cfg!(windows));
    let manifest = offer.manifest.ok_or_else(|| {
        RaptorError::Conflict(format!("the Vulkan runtime is not offered: {}", offer.note))
    })?;
    manifest.validate()?;
    let _guard = state.runtime_lock.lock().await;
    state
        .runtime
        .install_runtime(&manifest, &CancellationToken::new())
        .await?;
    add_to_runtime_index(&state, &manifest)?;
    Ok(format!(
        "Vulkan runtime {} installed. Choose it per model in the Library, or leave Auto.",
        manifest.version
    ))
}

/// Adds `manifest` to the runtime index file, replacing an entry with the same id and version.
fn add_to_runtime_index(state: &AppState, manifest: &Manifest) -> Result<()> {
    let path = state.runtime_manifest_path();
    let mut root: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .map_err(|e| RaptorError::InvalidInput(format!("bad runtime index: {e}")))?;
    let list = root
        .get_mut("runtimes")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| {
            RaptorError::InvalidInput("the runtime index has no runtimes list".into())
        })?;
    list.retain(|e| {
        !(e.get("id").and_then(|v| v.as_str()) == Some(manifest.id.as_str())
            && e.get("version").and_then(|v| v.as_str()) == Some(manifest.version.as_str()))
    });
    list.push(serde_json::to_value(manifest)?);
    crate::fsutil::write_atomic(&path, &serde_json::to_vec_pretty(&root)?)?;
    Ok(())
}

/// Installs the newer build of runtime `id` that `dependencies_check` offered. The release is
/// read again here, so nothing the UI sends decides the URL or checksum. The new build is added
/// to the runtime index, any pin is dropped (the user asked for this version), and the usual
/// verified install runs. Refused while a model is loaded.
#[tauri::command]
pub async fn dependencies_install_update(
    state: State<'_, AppState>,
    id: String,
) -> Result<RuntimeSelectionView> {
    refuse_while_serving(&state)?;
    let installed = state.runtime.installed_runtimes()?;
    if !installed.iter().any(|r| r.id == id) {
        return Err(RaptorError::NotFound(format!(
            "runtime {id} is not installed"
        )));
    }
    let index = state.load_runtime_index()?;
    let source = state
        .releases
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let release = source.latest().await?;
    let update =
        crate::updates::plan_updates(&index.manifests, &release, std::slice::from_ref(&id))
            .into_iter()
            .find(|u| u.id == id && u.available)
            .ok_or_else(|| {
                RaptorError::Conflict(format!("no verified update is available for {id}"))
            })?;
    let manifest = update
        .manifest
        .ok_or_else(|| RaptorError::Internal("update has no manifest".into()))?;
    manifest.validate()?;
    add_to_runtime_index(&state, &manifest)?;

    state.settings.update(|s| {
        s.pinned_runtime = None;
        Ok(())
    })?;
    let index = state.load_runtime_index()?;
    let _guard = state.runtime_lock.lock().await;
    let selection = state
        .runtime
        .ensure_runtime(&index.manifests, &CancellationToken::new())
        .await?;
    Ok(RuntimeSelectionView {
        id: selection.manifest.id.clone(),
        version: selection.manifest.version.clone(),
        backend: selection.backend,
        degraded: selection.degraded,
        reason: selection.reason.clone(),
        gpu_arch: selection.gpu_arch.clone(),
        rejected: selection.rejected.clone(),
    })
}
