//! Settings, gateway exposure (port, LAN, API key) and workspace registration.

use crate::adapters::workspace_id;
use crate::commands::parse_dir;
use crate::commands::system::gateway_changed;
use crate::settings::{
    ApiKeyEntry, PinnedRuntime, Settings, MAX_CONTEXT_TARGET, MIN_CONTEXT_TARGET,
};
use crate::state::{AppState, GatewayStatus};
use raptor_core::{DataDirs, RaptorError, Result};
use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, State};

#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub created_at: u64,
}

/// The API key secret, returned exactly once by `api_key_rotate`.
#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyReveal {
    pub id: String,
    pub secret: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceInfo {
    pub id: String,
    pub path: String,
}

/// Settings as the UI may see them. The API key secret is never included.
#[derive(Debug, Clone, Serialize)]
pub struct SettingsView {
    pub storage_dir: Option<String>,
    pub default_storage_dir: String,
    pub active_storage_dir: String,
    /// True until the automatic app restart has reopened services on the configured path.
    pub restart_required: bool,
    pub gateway_enabled: bool,
    pub gateway_port: u16,
    pub lan_enabled: bool,
    pub api_key: Option<ApiKeyInfo>,
    pub context_target: u32,
    pub workspaces: Vec<WorkspaceInfo>,
    pub min_context_target: u32,
    pub max_context_target: u32,
    /// Runtime version kept in use by `runtime_ensure`. `None` follows the newest version.
    pub pinned_runtime: Option<PinnedRuntime>,
    /// Library id of the embedding model, or null (semantic retrieval off).
    pub embedding_model: Option<String>,
    /// Whether a load may fall back to CPU when a GPU backend has no device. Off by default.
    pub allow_cpu_fallback: bool,
    /// Whether a load may keep the KV cache in system RAM when no VRAM plan fits. Off by default.
    pub allow_ram_kv: bool,
    /// Parallel request slots for llama-server (1-8). See `Settings::parallel_slots`.
    pub parallel_slots: u32,
    /// Default thinking effort for thinking models. See `Settings::thinking_effort`.
    pub thinking_effort: raptor_gateway::Effort,
    /// Problems found at startup (settings recovery, storage fallback).
    pub notes: Vec<String>,
}

fn view(state: &AppState) -> Result<SettingsView> {
    let s = state.settings.snapshot();
    let default_root = DataDirs::default_location()?.root;
    let effective = s
        .storage_dir
        .clone()
        .unwrap_or_else(|| default_root.clone());
    let mut notes = state.startup_notes.clone();
    if let Some(r) = state.settings.recovery() {
        if !notes.contains(&r) {
            notes.push(r);
        }
    }
    Ok(SettingsView {
        storage_dir: s.storage_dir.as_ref().map(|p| display_path(p)),
        default_storage_dir: display_path(&default_root),
        active_storage_dir: display_path(&state.dirs.root),
        restart_required: effective != state.dirs.root,
        gateway_enabled: s.gateway_enabled,
        gateway_port: s.gateway_port,
        lan_enabled: s.lan_enabled,
        api_key: s.api_key.as_ref().map(|k| ApiKeyInfo {
            id: k.id().to_string(),
            created_at: k.created_at,
        }),
        context_target: s.context_target,
        workspaces: s
            .workspaces
            .iter()
            .map(|p| WorkspaceInfo {
                id: workspace_id(p),
                path: p.display().to_string(),
            })
            .collect(),
        min_context_target: MIN_CONTEXT_TARGET,
        max_context_target: MAX_CONTEXT_TARGET,
        pinned_runtime: s.pinned_runtime.clone(),
        embedding_model: s.embedding_model.clone(),
        allow_cpu_fallback: s.allow_cpu_fallback,
        allow_ram_kv: s.allow_ram_kv,
        parallel_slots: s.parallel_slots,
        thinking_effort: s.thinking_effort,
        notes,
    })
}

/// Windows canonicalize adds the extended-length prefix. Keep it internally for long-path
/// support, but show the familiar path form in editable controls and copyable status text.
fn display_path(path: &std::path::Path) -> String {
    let value = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(unc) = value.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{unc}");
        }
        if let Some(normal) = value.strip_prefix(r"\\?\") {
            return normal.to_string();
        }
    }
    value.into_owned()
}

/// Allows (or forbids) a load to continue on CPU when a GPU backend has no device. Off by
/// default. When a load does fall back, its result carries `cpu_fallback: true`.
#[tauri::command]
pub async fn settings_set_allow_cpu_fallback(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<SettingsView> {
    state.settings.update(|s| {
        s.allow_cpu_fallback = enabled;
        Ok(())
    })?;
    view(&state)
}

/// Allows (or forbids) keeping the KV cache in system RAM when no VRAM plan exists
/// (`--no-kv-offload`). Off by default. Generation is slower across the bus; only a
/// load that would otherwise not start at all uses it.
#[tauri::command]
pub async fn settings_set_allow_ram_kv(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<SettingsView> {
    state.settings.update(|s| {
        s.allow_ram_kv = enabled;
        Ok(())
    })?;
    view(&state)
}

/// Sets how many requests llama-server works on at once (1-8). Takes effect at the next load.
#[tauri::command]
pub async fn settings_set_parallel_slots(
    state: State<'_, AppState>,
    slots: u32,
) -> Result<SettingsView> {
    state.settings.update(|s| {
        s.parallel_slots = slots;
        Ok(())
    })?;
    view(&state)
}

/// Sets the default thinking effort (low, medium, high or max) for thinking models. The gateway
/// is reconfigured so API clients that send none get it too.
#[tauri::command]
pub async fn settings_set_thinking_effort(
    app: AppHandle,
    state: State<'_, AppState>,
    effort: raptor_gateway::Effort,
) -> Result<SettingsView> {
    change_and_apply(&app, &state, |s| {
        s.thinking_effort = effort;
        Ok(())
    })
    .await?;
    view(&state)
}

/// Applies a settings change to the gateway. If the gateway refuses the new configuration,
/// the settings are restored to their previous values and the gateway is restarted with them.
async fn change_and_apply<F>(app: &AppHandle, state: &AppState, change: F) -> Result<GatewayStatus>
where
    F: FnOnce(&mut Settings) -> Result<()>,
{
    let before = state.settings.snapshot();
    state.settings.update(change)?;
    match gateway_changed(app, state).await {
        Ok(status) => Ok(status),
        Err(e) => {
            state.settings.update(|s| {
                *s = before;
                Ok(())
            })?;
            let _ = gateway_changed(app, state).await;
            Err(e)
        }
    }
}

#[tauri::command]
pub async fn settings_get(state: State<'_, AppState>) -> Result<SettingsView> {
    view(&state)
}

/// Changes the storage directory (models, runtimes, downloads, conversations, logs and context
/// index) and requests an orderly app restart so all path-bound services reopen against it.
/// Existing files are left in the old folder. `null` restores the default.
#[tauri::command]
pub async fn settings_set_storage_dir(
    app: AppHandle,
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<SettingsView> {
    let dir: Option<PathBuf> = match path {
        Some(p) => Some(parse_dir(&p)?),
        None => None,
    };
    let requested = match dir.as_ref() {
        Some(path) => DataDirs::new(path.clone()),
        None => DataDirs::default_location()?,
    };
    requested.ensure()?;
    let target = std::fs::canonicalize(&requested.root).unwrap_or(requested.root);
    let active =
        std::fs::canonicalize(&state.dirs.root).unwrap_or_else(|_| state.dirs.root.clone());
    let stored = dir.map(|_| target.clone());
    if target == active {
        state.settings.update(|s| {
            s.storage_dir = stored;
            Ok(())
        })?;
        return view(&state);
    }

    state.settings.update(|s| {
        s.storage_dir = stored;
        Ok(())
    })?;
    let response = view(&state)?;
    // Path-bound services are rebuilt from settings at startup. Trigger an orderly app restart
    // automatically; callers never need to reboot Windows or remember to restart Raptor.
    app.request_restart();
    Ok(response)
}

#[tauri::command]
pub async fn settings_set_gateway_port(
    app: AppHandle,
    state: State<'_, AppState>,
    port: u16,
) -> Result<SettingsView> {
    if port == 0 {
        return Err(RaptorError::InvalidInput("port must be 1-65535".into()));
    }
    change_and_apply(&app, &state, |s| {
        s.gateway_port = port;
        Ok(())
    })
    .await?;
    view(&state)
}

#[tauri::command]
pub async fn settings_set_gateway_enabled(
    app: AppHandle,
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<GatewayStatus> {
    change_and_apply(&app, &state, |s| {
        s.gateway_enabled = enabled;
        Ok(())
    })
    .await
}

/// Exposes the gateway on the LAN (0.0.0.0). Refused with an error unless an API key exists.
#[tauri::command]
pub async fn settings_set_lan(
    app: AppHandle,
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<SettingsView> {
    change_and_apply(&app, &state, |s| {
        s.lan_enabled = enabled;
        Ok(())
    })
    .await?;
    view(&state)
}

#[tauri::command]
pub async fn settings_set_context_target(
    state: State<'_, AppState>,
    tokens: u32,
) -> Result<SettingsView> {
    state.settings.update(|s| {
        s.context_target = tokens;
        Ok(())
    })?;
    view(&state)
}

/// Generates a new API key (or replaces the current one). The secret is returned once and
/// is not retrievable afterwards.
#[tauri::command]
pub async fn api_key_rotate(app: AppHandle, state: State<'_, AppState>) -> Result<ApiKeyReveal> {
    let created = crate::fsutil::unix_now();
    let id = format!("raptor-{created}");
    // Only the digest record is stored. The secret goes to the UI in this reply and nowhere else.
    let (entry, secret) = ApiKeyEntry::create(id.clone(), created)?;
    let reveal = ApiKeyReveal { id, secret };
    change_and_apply(&app, &state, move |s| {
        s.api_key = Some(entry);
        Ok(())
    })
    .await?;
    Ok(reveal)
}

/// Removes the API key. Refused while LAN access is enabled.
#[tauri::command]
pub async fn api_key_revoke(app: AppHandle, state: State<'_, AppState>) -> Result<SettingsView> {
    change_and_apply(&app, &state, |s| {
        s.api_key = None;
        Ok(())
    })
    .await?;
    view(&state)
}

/// Registers a workspace root. The gateway's context routes serve only registered roots.
#[tauri::command]
pub async fn workspace_register(state: State<'_, AppState>, path: String) -> Result<WorkspaceInfo> {
    let root = parse_dir(&path)?;
    if !root.is_dir() {
        return Err(RaptorError::InvalidInput(format!(
            "not a directory: {}",
            root.display()
        )));
    }
    state.settings.update(|s| {
        if !s.workspaces.contains(&root) {
            s.workspaces.push(root.clone());
        }
        Ok(())
    })?;
    Ok(WorkspaceInfo {
        id: workspace_id(&root),
        path: root.display().to_string(),
    })
}

#[tauri::command]
pub async fn workspace_remove(state: State<'_, AppState>, id: String) -> Result<SettingsView> {
    state.settings.update(|s| {
        s.workspaces.retain(|p| workspace_id(p) != id);
        Ok(())
    })?;
    view(&state)
}
