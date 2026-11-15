//! Conversation CRUD, append, export and import. Storage is one JSON file per conversation.

use crate::commands::parse_abs_file;
use crate::conversations::{
    Conversation, ConversationList, ConversationSearch, NewMessage, SEARCH_DEFAULT_LIMIT,
};
use crate::state::AppState;
use raptor_core::Result;
use tauri::State;

#[tauri::command]
pub async fn conversations_list(state: State<'_, AppState>) -> Result<ConversationList> {
    state.conversations.list()
}

#[tauri::command]
pub async fn conversations_create(
    state: State<'_, AppState>,
    title: String,
) -> Result<Conversation> {
    state.conversations.create(&title)
}

#[tauri::command]
pub async fn conversations_get(state: State<'_, AppState>, id: String) -> Result<Conversation> {
    state.conversations.get(&id)
}

#[tauri::command]
pub async fn conversations_rename(
    state: State<'_, AppState>,
    id: String,
    title: String,
) -> Result<Conversation> {
    state.conversations.rename(&id, &title)
}

#[tauri::command]
pub async fn conversations_delete(state: State<'_, AppState>, id: String) -> Result<()> {
    state.conversations.delete(&id)
}

/// Appends messages in order. The UI calls this after each completed exchange.
#[tauri::command]
pub async fn conversations_append(
    state: State<'_, AppState>,
    id: String,
    messages: Vec<NewMessage>,
) -> Result<Conversation> {
    state.conversations.append(&id, messages)
}

/// Truncates the conversation to keep the first `count` messages.
#[tauri::command]
pub async fn conversations_truncate(
    state: State<'_, AppState>,
    id: String,
    count: usize,
) -> Result<Conversation> {
    state.conversations.truncate(&id, count)
}

/// Replaces the content of a specific message in the conversation.
#[tauri::command]
pub async fn conversations_replace_message(
    state: State<'_, AppState>,
    id: String,
    message_id: String,
    content: String,
) -> Result<Conversation> {
    state
        .conversations
        .replace_message(&id, &message_id, &content)
}

/// Returns the conversation as JSON text for the UI to save to a file of the user's choice.
#[tauri::command]
pub async fn conversations_export(state: State<'_, AppState>, id: String) -> Result<String> {
    state.conversations.export(&id)
}

/// Imports a conversation from JSON text. It is stored under a new id.
#[tauri::command]
pub async fn conversations_import(
    state: State<'_, AppState>,
    json: String,
) -> Result<Conversation> {
    state.conversations.import(&json)
}

/// Case-insensitive search across every conversation. `limit` defaults to 50 (max 200).
#[tauri::command]
pub async fn conversations_search(
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
) -> Result<ConversationSearch> {
    let limit = limit.unwrap_or(SEARCH_DEFAULT_LIMIT);
    let store = state.conversations.clone();
    crate::adapters::blocking(move || store.search(&query, limit)).await
}

/// Writes conversation `id` as JSON to `path`, which the user picked in the save dialog
/// (`dialog_save_file`). The path must be absolute and end in `.json`. Returns the file path.
#[tauri::command]
pub async fn conversations_export_to(
    state: State<'_, AppState>,
    id: String,
    path: String,
) -> Result<ExportedFile> {
    let target = parse_abs_file(&path)?;
    let store = state.conversations.clone();
    let written = crate::adapters::blocking({
        let target = target.clone();
        move || store.export_to(&id, &target)
    })
    .await?;
    Ok(ExportedFile {
        path: target.display().to_string(),
        bytes: written,
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExportedFile {
    pub path: String,
    pub bytes: u64,
}
