//! Tauri commands, grouped by area. Every command returns `Result<T, RaptorError>`, so the
//! UI receives `{"kind": "...", "message": "..."}` with the real cause.

pub mod chat;
pub mod context;
pub mod conversations;
pub mod models;
pub mod perf;
pub mod presets;
pub mod settings;
pub mod system;

use raptor_core::{RaptorError, Result};
use std::path::PathBuf;

/// Event: `EngineStatus` whenever the llama-server state or GPU verdict changes.
pub const EVENT_SERVER_STATE: &str = "server://state";
/// Event: `DownloadProgress` for every change of a download job.
pub const EVENT_DOWNLOAD_PROGRESS: &str = "download://progress";
/// Event: `DownloadFinished` once a download completes, is cancelled or fails.
pub const EVENT_DOWNLOAD_FINISHED: &str = "download://finished";
/// Event: `GatewayStatus` after the gateway starts, stops or is reconfigured.
pub const EVENT_GATEWAY_STATE: &str = "gateway://state";

/// Event name for one chat request's stream.
pub fn chat_event_name(request_id: &str) -> String {
    format!("chat://{request_id}")
}

const MAX_PATH_CHARS: usize = 4096;

/// Parses a user-supplied directory path into an absolute path. Empty and control-character
/// input is refused; the directory itself is checked by the engine that uses it.
pub fn parse_dir(path: &str) -> Result<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(RaptorError::InvalidInput("path is empty".into()));
    }
    if trimmed.chars().count() > MAX_PATH_CHARS || trimmed.chars().any(char::is_control) {
        return Err(RaptorError::InvalidInput(
            "path is too long or contains control characters".into(),
        ));
    }
    Ok(std::path::absolute(trimmed)?)
}

/// Parses a user-supplied file path that must be absolute.
pub fn parse_abs_file(path: &str) -> Result<PathBuf> {
    let p = PathBuf::from(path.trim());
    if !p.is_absolute() {
        return Err(RaptorError::InvalidInput(
            "file path must be absolute".into(),
        ));
    }
    Ok(p)
}
