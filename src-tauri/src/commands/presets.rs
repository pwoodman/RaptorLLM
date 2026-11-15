//! Per-model presets: named sampling and context settings, with at most one default per model.
//! Validation (ranges, names, prompt size) is done by the model manager on every write.

use crate::state::AppState;
use raptor_core::Result;
use raptor_models::Preset;
use serde::Serialize;
use tauri::State;

/// The presets of one model and which one is the default.
#[derive(Debug, Clone, Serialize)]
pub struct PresetList {
    pub model_id: String,
    pub presets: Vec<Preset>,
    /// Name of the default preset, or null.
    pub default: Option<String>,
}

/// Refuses a model id that is not in the library, so presets never attach to nothing.
fn require_installed(state: &AppState, model_id: &str) -> Result<()> {
    state.models.library().get(model_id).map(|_| ())
}

fn list_view(state: &AppState, model_id: &str) -> Result<PresetList> {
    let presets = state.models.presets();
    Ok(PresetList {
        model_id: model_id.to_string(),
        presets: presets.list_for_model(model_id)?,
        default: presets.default_for_model(model_id)?.map(|p| p.name),
    })
}

/// Presets of an installed model, with its default.
#[tauri::command]
pub async fn presets_list(state: State<'_, AppState>, model_id: String) -> Result<PresetList> {
    require_installed(&state, &model_id)?;
    list_view(&state, &model_id)
}

#[tauri::command]
pub async fn presets_get(
    state: State<'_, AppState>,
    model_id: String,
    name: String,
) -> Result<Preset> {
    state.models.presets().get(&model_id, &name)
}

/// Creates or replaces a preset. The preset's `model_id` must be an installed model.
#[tauri::command]
pub async fn presets_save(state: State<'_, AppState>, preset: Preset) -> Result<Preset> {
    require_installed(&state, &preset.model_id)?;
    state.models.presets().save(preset.clone())?;
    Ok(preset)
}

/// Deletes a preset. Deleting the default clears the default.
#[tauri::command]
pub async fn presets_delete(
    state: State<'_, AppState>,
    model_id: String,
    name: String,
) -> Result<PresetList> {
    state.models.presets().delete(&model_id, &name)?;
    list_view(&state, &model_id)
}

/// Makes `name` the default preset of the model. `name` null clears the default.
#[tauri::command]
pub async fn presets_set_default(
    state: State<'_, AppState>,
    model_id: String,
    name: Option<String>,
) -> Result<PresetList> {
    require_installed(&state, &model_id)?;
    match name {
        Some(n) => state.models.presets().set_default(&model_id, &n)?,
        None => {
            state.models.presets().clear_default(&model_id)?;
        }
    }
    list_view(&state, &model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_core::RaptorError;

    #[test]
    fn list_view_reports_the_default_name_for_each_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = raptor_models::PresetStore::open(dir.path().join("presets.json")).unwrap();
        store
            .save(Preset::with_defaults("org/repo/a.gguf", "Coding"))
            .unwrap();
        store
            .save(Preset::with_defaults("org/repo/a.gguf", "Chat"))
            .unwrap();
        store.set_default("org/repo/a.gguf", "Chat").unwrap();
        let presets = store.list_for_model("org/repo/a.gguf").unwrap();
        assert_eq!(presets.len(), 2);
        assert_eq!(
            store
                .default_for_model("org/repo/a.gguf")
                .unwrap()
                .map(|p| p.name),
            Some("Chat".to_string())
        );
        store.delete("org/repo/a.gguf", "Chat").unwrap();
        assert!(store
            .default_for_model("org/repo/a.gguf")
            .unwrap()
            .is_none());
        assert!(matches!(
            store.set_default("org/repo/a.gguf", "Missing"),
            Err(RaptorError::NotFound(_))
        ));
    }
}
