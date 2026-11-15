//! Context Intelligence commands for a workspace path. Use `workspace_register` to make a
//! root available to the gateway's context routes.

use crate::adapters::SemanticView;
use crate::commands::parse_dir;
use crate::state::AppState;
use raptor_context::{
    ContextRequest, ContextResponse, ImpactReport, IndexStatus, RecoverRequest, RecoverResponse,
    SearchRequest, SymbolHit, SymbolQuery, UpdateReport,
};
use raptor_context::{Hit, MemoryKind};
use raptor_core::{RaptorError, Result};
use tauri::State;

/// Re-indexes the workspace: only changed files are parsed.
#[tauri::command]
pub async fn context_index(state: State<'_, AppState>, path: String) -> Result<UpdateReport> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.update()
    })
    .await
}

#[tauri::command]
pub async fn context_index_status(state: State<'_, AppState>, path: String) -> Result<IndexStatus> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        Ok(eng.index_status())
    })
    .await
}

#[tauri::command]
pub async fn context_search(
    state: State<'_, AppState>,
    path: String,
    request: SearchRequest,
) -> Result<Vec<Hit>> {
    if request.query.trim().is_empty() {
        return Err(RaptorError::InvalidInput("query must not be empty".into()));
    }
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.update()?;
        eng.search(&request)
    })
    .await
}

/// Builds a token-budgeted context pack (protected items first) with its manifest.
#[tauri::command]
pub async fn context_build(
    state: State<'_, AppState>,
    path: String,
    request: ContextRequest,
) -> Result<ContextResponse> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.build_context(request)
    })
    .await
}

#[tauri::command]
pub async fn context_recover(
    state: State<'_, AppState>,
    path: String,
    request: RecoverRequest,
) -> Result<RecoverResponse> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.recover(request)
    })
    .await
}

#[tauri::command]
pub async fn context_symbols(
    state: State<'_, AppState>,
    path: String,
    query: SymbolQuery,
) -> Result<Vec<SymbolHit>> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        Ok(eng.symbols(&query))
    })
    .await
}

#[tauri::command]
pub async fn context_impact(
    state: State<'_, AppState>,
    path: String,
    file: String,
) -> Result<ImpactReport> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.impact(&file)
    })
    .await
}

/// Sets the library model that serves embeddings for semantic retrieval. It must be installed.
/// The vectors are built while that model is the loaded one; until then retrieval uses the
/// lexical and graph signals and `context_semantic_status` says why. Returns the model id.
#[tauri::command]
pub async fn context_set_embedding_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<String> {
    let entry = state.models.library().get(&model_id)?;
    state.settings.update(|s| {
        s.embedding_model = Some(entry.id.clone());
        Ok(())
    })?;
    Ok(entry.id)
}

/// Turns semantic retrieval off: no embedding model is configured.
#[tauri::command]
pub async fn context_clear_embedding_model(state: State<'_, AppState>) -> Result<()> {
    state.settings.update(|s| {
        s.embedding_model = None;
        Ok(())
    })?;
    Ok(())
}

/// Semantic retrieval for `path`: the configured model, the reason it is off, and the engine's
/// counts (vectors, chunks without a vector, last error).
#[tauri::command]
pub async fn context_semantic_status(
    state: State<'_, AppState>,
    path: String,
) -> Result<SemanticView> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || hub.semantic_status(&root)).await
}

/// Result of `context_remember_project`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProjectMemoryAdded {
    pub id: String,
    /// Entries removed to stay within the project memory cap, least valuable first.
    pub evicted: Vec<String>,
}

const MAX_MEMORY_TEXT_CHARS: usize = 4000;

/// Records a long-term project memory entry for `path` (a decision, convention, summary, ...).
/// The entry goes stale when one of the cited `paths` changes. Paths are workspace-relative.
#[tauri::command]
pub async fn context_remember_project(
    state: State<'_, AppState>,
    path: String,
    kind: MemoryKind,
    text: String,
    paths: Vec<String>,
) -> Result<ProjectMemoryAdded> {
    let text = text.trim().to_string();
    if text.is_empty() || text.chars().count() > MAX_MEMORY_TEXT_CHARS {
        return Err(RaptorError::InvalidInput(format!(
            "memory text must be 1-{MAX_MEMORY_TEXT_CHARS} characters"
        )));
    }
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        let added = eng.remember_project(kind, &text, &paths)?;
        Ok(ProjectMemoryAdded {
            id: added.id,
            evicted: added.evicted,
        })
    })
    .await
}

/// Lists the long-term project memory of `path`: decisions, conventions, summaries, completed
/// work and failed attempts, each with its cited files and whether it has gone stale.
#[tauri::command]
pub async fn context_project_memory(
    state: State<'_, AppState>,
    path: String,
) -> Result<Vec<raptor_context::MemoryEntry>> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.project_memory()
    })
    .await
}

/// Removes a project memory entry by id. This is how a decision or convention is retired.
#[tauri::command]
pub async fn context_forget_project_memory(
    state: State<'_, AppState>,
    path: String,
    id: String,
) -> Result<()> {
    let root = parse_dir(&path)?;
    let hub = state.contexts.clone();
    crate::adapters::blocking(move || {
        let engine = hub.engine(&root)?;
        let mut eng = engine.lock().unwrap_or_else(|p| p.into_inner());
        eng.forget_project_memory(&id)
    })
    .await
}
