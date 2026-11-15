//! Per-model presets (system prompt plus sampling and context settings), stored as JSON.
//!
//! A preset is keyed by `(model_id, name)`. `create` refuses an existing key and `update`
//! refuses a missing one, so a plain write never silently replaces another preset; `save`
//! is the explicit create-or-replace. Each model may name one default preset. Every write is
//! validated and saved atomically before memory is changed.

use crate::util::{lock, write_atomic};
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

const FORMAT_VERSION: u32 = 1;
const MAX_NAME_CHARS: usize = 64;
const MAX_MODEL_ID_BYTES: usize = 256;
const MAX_SYSTEM_PROMPT_BYTES: usize = 32 * 1024;
/// Smallest context a preset may request.
pub const MIN_CTX: u32 = 256;
/// Largest context a preset may request.
pub const MAX_CTX: u32 = 1_048_576;
/// Largest `max_tokens` a preset may request.
pub const MAX_TOKENS_LIMIT: u32 = 1_000_000;
/// Allowed `temperature` range, inclusive.
pub const TEMPERATURE_RANGE: (f32, f32) = (0.0, 2.0);
/// Allowed `repeat_penalty` range, inclusive.
pub const REPEAT_PENALTY_RANGE: (f32, f32) = (0.5, 2.0);
/// Largest `top_k` a preset may request (0 disables it).
pub const MAX_TOP_K: u32 = 1000;

/// One named configuration for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub model_id: String,
    pub name: String,
    pub system_prompt: Option<String>,
    pub n_ctx: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    /// `None` leaves generation length to the runtime.
    pub max_tokens: Option<u32>,
}

impl Preset {
    /// A preset with conservative defaults for `model_id`.
    pub fn with_defaults(model_id: &str, name: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            name: name.to_string(),
            system_prompt: None,
            n_ctx: 4096,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: None,
        }
    }
}

/// Checks every field against its range. Called on every write.
///
/// Ranges: `n_ctx` 256 to 1,048,576; `temperature` 0 to 2; `top_p` in (0, 1]; `top_k` 0 to 1000
/// (0 disables it); `repeat_penalty` 0.5 to 2; `max_tokens` 1 to 1,000,000 or unset; names are
/// 1 to 64 printable characters without edge spaces; system prompts at most 32 KiB.
pub fn validate(p: &Preset) -> Result<()> {
    let bad = |msg: String| Err(RaptorError::InvalidInput(msg));
    if p.model_id.trim().is_empty() || p.model_id.len() > MAX_MODEL_ID_BYTES {
        return bad("preset model id is empty or too long".into());
    }
    if p.name.trim().is_empty()
        || p.name != p.name.trim()
        || p.name.chars().count() > MAX_NAME_CHARS
        || p.name.chars().any(char::is_control)
    {
        return bad(format!(
            "preset name must be 1-{MAX_NAME_CHARS} printable characters, without edge spaces"
        ));
    }
    if let Some(sp) = &p.system_prompt {
        if sp.len() > MAX_SYSTEM_PROMPT_BYTES {
            return bad(format!(
                "system prompt exceeds {MAX_SYSTEM_PROMPT_BYTES} bytes"
            ));
        }
    }
    if !(MIN_CTX..=MAX_CTX).contains(&p.n_ctx) {
        return bad(format!("n_ctx must be between {MIN_CTX} and {MAX_CTX}"));
    }
    let (t_lo, t_hi) = TEMPERATURE_RANGE;
    if !(p.temperature.is_finite() && (t_lo..=t_hi).contains(&p.temperature)) {
        return bad(format!("temperature must be between {t_lo} and {t_hi}"));
    }
    if !(p.top_p.is_finite() && p.top_p > 0.0 && p.top_p <= 1.0) {
        return bad("top_p must be in (0, 1]".into());
    }
    if p.top_k > MAX_TOP_K {
        return bad(format!("top_k must be at most {MAX_TOP_K} (0 disables it)"));
    }
    let (r_lo, r_hi) = REPEAT_PENALTY_RANGE;
    if !(p.repeat_penalty.is_finite() && (r_lo..=r_hi).contains(&p.repeat_penalty)) {
        return bad(format!("repeat_penalty must be between {r_lo} and {r_hi}"));
    }
    if let Some(t) = p.max_tokens {
        if t == 0 || t > MAX_TOKENS_LIMIT {
            return bad(format!(
                "max_tokens must be between 1 and {MAX_TOKENS_LIMIT}"
            ));
        }
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct PresetFile {
    version: u32,
    presets: Vec<Preset>,
    /// Default preset name per model id. Missing in files written before defaults existed.
    #[serde(default)]
    defaults: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
struct State {
    presets: Vec<Preset>,
    defaults: BTreeMap<String, String>,
}

/// JSON-backed preset store.
pub struct PresetStore {
    path: PathBuf,
    state: Mutex<State>,
}

impl PresetStore {
    /// Opens (or starts empty) the store at `path`. A corrupt or wrong-version file is an error.
    /// A default that points at a preset which no longer exists is dropped on load.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let mut state = match fs::read(&path) {
            Ok(bytes) => {
                let file: PresetFile = serde_json::from_slice(&bytes)
                    .map_err(|e| RaptorError::Integrity(format!("preset file is corrupt: {e}")))?;
                if file.version != FORMAT_VERSION {
                    return Err(RaptorError::Unsupported(format!(
                        "preset file version {} is not supported",
                        file.version
                    )));
                }
                State {
                    presets: file.presets,
                    defaults: file.defaults,
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => State::default(),
            Err(e) => return Err(e.into()),
        };
        let before = state.defaults.len();
        let presets = state.presets.clone();
        state.defaults.retain(|model, name| {
            presets
                .iter()
                .any(|p| p.model_id == *model && p.name == *name)
        });
        if state.defaults.len() != before {
            tracing::warn!("dropped default presets that no longer exist");
        }
        Ok(Self {
            path,
            state: Mutex::new(state),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        lock(&self.state)
    }

    /// Presets for one model, in creation order.
    pub fn list_for_model(&self, model_id: &str) -> Result<Vec<Preset>> {
        let state = self.lock()?;
        Ok(state
            .presets
            .iter()
            .filter(|p| p.model_id == model_id)
            .cloned()
            .collect())
    }

    /// One preset by model and name.
    pub fn get(&self, model_id: &str, name: &str) -> Result<Preset> {
        let state = self.lock()?;
        state
            .presets
            .iter()
            .find(|p| p.model_id == model_id && p.name == name)
            .cloned()
            .ok_or_else(|| RaptorError::NotFound(format!("no such preset: {name}")))
    }

    /// The default preset of a model, if one is set.
    pub fn default_for_model(&self, model_id: &str) -> Result<Option<Preset>> {
        let state = self.lock()?;
        Ok(state
            .defaults
            .get(model_id)
            .and_then(|name| {
                state
                    .presets
                    .iter()
                    .find(|p| p.model_id == model_id && p.name == *name)
            })
            .cloned())
    }

    /// Creates a preset. Fails with `Conflict` if `(model_id, name)` already exists.
    pub fn create(&self, preset: Preset) -> Result<()> {
        validate(&preset)?;
        let mut state = self.lock()?;
        if state
            .presets
            .iter()
            .any(|p| p.model_id == preset.model_id && p.name == preset.name)
        {
            return Err(RaptorError::Conflict(format!(
                "preset already exists: {}",
                preset.name
            )));
        }
        let mut next = state.clone();
        next.presets.push(preset);
        self.persist(&next)?;
        *state = next;
        Ok(())
    }

    /// Replaces an existing preset. Fails with `NotFound` if it does not exist.
    pub fn update(&self, preset: Preset) -> Result<()> {
        validate(&preset)?;
        let mut state = self.lock()?;
        let pos = state
            .presets
            .iter()
            .position(|p| p.model_id == preset.model_id && p.name == preset.name)
            .ok_or_else(|| RaptorError::NotFound(format!("no such preset: {}", preset.name)))?;
        let mut next = state.clone();
        next.presets[pos] = preset;
        self.persist(&next)?;
        *state = next;
        Ok(())
    }

    /// Creates or replaces a preset. Returns `true` when it was created.
    pub fn save(&self, preset: Preset) -> Result<bool> {
        validate(&preset)?;
        let mut state = self.lock()?;
        let mut next = state.clone();
        let created = match next
            .presets
            .iter()
            .position(|p| p.model_id == preset.model_id && p.name == preset.name)
        {
            Some(pos) => {
                next.presets[pos] = preset;
                false
            }
            None => {
                next.presets.push(preset);
                true
            }
        };
        self.persist(&next)?;
        *state = next;
        Ok(created)
    }

    /// Deletes one preset. Clears the model's default if it pointed at this preset.
    pub fn delete(&self, model_id: &str, name: &str) -> Result<()> {
        let mut state = self.lock()?;
        let pos = state
            .presets
            .iter()
            .position(|p| p.model_id == model_id && p.name == name)
            .ok_or_else(|| RaptorError::NotFound(format!("no such preset: {name}")))?;
        let mut next = state.clone();
        next.presets.remove(pos);
        if next.defaults.get(model_id).map(String::as_str) == Some(name) {
            next.defaults.remove(model_id);
        }
        self.persist(&next)?;
        *state = next;
        Ok(())
    }

    /// Makes `name` the default for `model_id`. Fails with `NotFound` if that preset is missing.
    pub fn set_default(&self, model_id: &str, name: &str) -> Result<()> {
        let mut state = self.lock()?;
        if !state
            .presets
            .iter()
            .any(|p| p.model_id == model_id && p.name == name)
        {
            return Err(RaptorError::NotFound(format!("no such preset: {name}")));
        }
        let mut next = state.clone();
        next.defaults.insert(model_id.to_string(), name.to_string());
        self.persist(&next)?;
        *state = next;
        Ok(())
    }

    /// Removes a model's default. Presets themselves are kept. Returns whether one was set.
    pub fn clear_default(&self, model_id: &str) -> Result<bool> {
        let mut state = self.lock()?;
        if !state.defaults.contains_key(model_id) {
            return Ok(false);
        }
        let mut next = state.clone();
        next.defaults.remove(model_id);
        self.persist(&next)?;
        *state = next;
        Ok(true)
    }

    /// Deletes every preset for a model and its default (used when the model is removed).
    /// Returns the number of presets removed.
    pub fn delete_for_model(&self, model_id: &str) -> Result<usize> {
        let mut state = self.lock()?;
        let next_presets: Vec<Preset> = state
            .presets
            .iter()
            .filter(|p| p.model_id != model_id)
            .cloned()
            .collect();
        let removed = state.presets.len() - next_presets.len();
        if removed > 0 || state.defaults.contains_key(model_id) {
            let mut next = State {
                presets: next_presets,
                defaults: state.defaults.clone(),
            };
            next.defaults.remove(model_id);
            self.persist(&next)?;
            *state = next;
        }
        Ok(removed)
    }

    /// Path of the backing JSON file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn persist(&self, state: &State) -> Result<()> {
        let file = PresetFile {
            version: FORMAT_VERSION,
            presets: state.presets.clone(),
            defaults: state.defaults.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file)?;
        write_atomic(&self.path, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, PathBuf, PresetStore) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config").join("presets.json");
        let s = PresetStore::open(&path).unwrap();
        (tmp, path, s)
    }

    #[test]
    fn create_and_list_per_model() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "coder")).unwrap();
        s.create(Preset::with_defaults("m1", "writer")).unwrap();
        s.create(Preset::with_defaults("m2", "coder")).unwrap();
        assert_eq!(s.list_for_model("m1").unwrap().len(), 2);
        assert_eq!(s.list_for_model("m2").unwrap().len(), 1);
        assert!(s.list_for_model("nope").unwrap().is_empty());
    }

    #[test]
    fn get_returns_one_preset_or_not_found() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        assert_eq!(s.get("m1", "a").unwrap().name, "a");
        assert!(matches!(s.get("m1", "b"), Err(RaptorError::NotFound(_))));
        assert!(matches!(s.get("m2", "a"), Err(RaptorError::NotFound(_))));
    }

    #[test]
    fn duplicate_create_is_conflict() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        assert!(matches!(
            s.create(Preset::with_defaults("m1", "a")),
            Err(RaptorError::Conflict(_))
        ));
    }

    #[test]
    fn update_requires_existing_preset() {
        let (_t, _p, s) = store();
        assert!(matches!(
            s.update(Preset::with_defaults("m1", "ghost")),
            Err(RaptorError::NotFound(_))
        ));
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        let mut changed = Preset::with_defaults("m1", "a");
        changed.system_prompt = Some("Be terse.".into());
        s.update(changed).unwrap();
        assert_eq!(
            s.list_for_model("m1").unwrap()[0].system_prompt.as_deref(),
            Some("Be terse.")
        );
    }

    #[test]
    fn save_creates_then_replaces() {
        let (_t, _p, s) = store();
        assert!(s.save(Preset::with_defaults("m1", "a")).unwrap());
        let mut again = Preset::with_defaults("m1", "a");
        again.top_k = 7;
        assert!(!s.save(again).unwrap());
        assert_eq!(s.get("m1", "a").unwrap().top_k, 7);
        assert_eq!(s.list_for_model("m1").unwrap().len(), 1);
    }

    #[test]
    fn delete_and_delete_for_model() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        s.create(Preset::with_defaults("m1", "b")).unwrap();
        s.create(Preset::with_defaults("m2", "a")).unwrap();
        s.delete("m1", "a").unwrap();
        assert!(matches!(s.delete("m1", "a"), Err(RaptorError::NotFound(_))));
        assert_eq!(s.delete_for_model("m1").unwrap(), 1);
        assert_eq!(s.delete_for_model("m1").unwrap(), 0);
        assert_eq!(s.list_for_model("m2").unwrap().len(), 1);
    }

    #[test]
    fn default_is_per_model_and_must_exist() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        s.create(Preset::with_defaults("m1", "b")).unwrap();
        s.create(Preset::with_defaults("m2", "a")).unwrap();
        assert!(s.default_for_model("m1").unwrap().is_none());
        assert!(matches!(
            s.set_default("m1", "ghost"),
            Err(RaptorError::NotFound(_))
        ));
        s.set_default("m1", "b").unwrap();
        s.set_default("m2", "a").unwrap();
        assert_eq!(s.default_for_model("m1").unwrap().unwrap().name, "b");
        assert_eq!(s.default_for_model("m2").unwrap().unwrap().model_id, "m2");
        s.set_default("m1", "a").unwrap();
        assert_eq!(s.default_for_model("m1").unwrap().unwrap().name, "a");
    }

    #[test]
    fn deleting_the_default_clears_it_but_keeps_other_presets() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        s.create(Preset::with_defaults("m1", "b")).unwrap();
        s.set_default("m1", "a").unwrap();
        s.delete("m1", "a").unwrap();
        assert!(s.default_for_model("m1").unwrap().is_none());
        assert_eq!(s.list_for_model("m1").unwrap().len(), 1);
    }

    #[test]
    fn clear_default_and_delete_for_model_drop_the_pointer() {
        let (_t, _p, s) = store();
        s.create(Preset::with_defaults("m1", "a")).unwrap();
        s.set_default("m1", "a").unwrap();
        assert!(s.clear_default("m1").unwrap());
        assert!(!s.clear_default("m1").unwrap());
        assert!(s.list_for_model("m1").unwrap().len() == 1);
        s.set_default("m1", "a").unwrap();
        s.delete_for_model("m1").unwrap();
        assert!(s.default_for_model("m1").unwrap().is_none());
    }

    #[test]
    fn defaults_persist_and_dangling_ones_are_dropped_on_open() {
        let (_t, path, s) = store();
        s.create(Preset::with_defaults("m1", "keep")).unwrap();
        s.set_default("m1", "keep").unwrap();
        let reopened = PresetStore::open(&path).unwrap();
        assert_eq!(
            reopened.default_for_model("m1").unwrap().unwrap().name,
            "keep"
        );
        // A file whose default names a preset that does not exist loads without that default.
        fs::write(
            &path,
            br#"{"version":1,"presets":[{"model_id":"m1","name":"keep","system_prompt":null,"n_ctx":4096,"temperature":0.7,"top_p":0.95,"top_k":40,"repeat_penalty":1.1,"max_tokens":null}],"defaults":{"m1":"nothing"}}"#,
        )
        .unwrap();
        let dangling = PresetStore::open(&path).unwrap();
        assert!(dangling.default_for_model("m1").unwrap().is_none());
        assert_eq!(dangling.list_for_model("m1").unwrap().len(), 1);
    }

    #[test]
    fn files_without_defaults_still_open() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("p.json");
        fs::write(
            &path,
            br#"{"version":1,"presets":[{"model_id":"m","name":"n","system_prompt":null,"n_ctx":4096,"temperature":0.7,"top_p":0.95,"top_k":40,"repeat_penalty":1.1,"max_tokens":null}]}"#,
        )
        .unwrap();
        let s = PresetStore::open(&path).unwrap();
        assert_eq!(s.list_for_model("m").unwrap().len(), 1);
        assert!(s.default_for_model("m").unwrap().is_none());
    }

    #[test]
    fn persists_across_reopen() {
        let (_t, path, s) = store();
        s.create(Preset::with_defaults("m1", "keep")).unwrap();
        let reopened = PresetStore::open(&path).unwrap();
        assert_eq!(reopened.list_for_model("m1").unwrap()[0].name, "keep");
    }

    #[test]
    fn corrupt_file_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("p.json");
        fs::write(&path, b"[1,2,").unwrap();
        assert!(matches!(
            PresetStore::open(&path),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn validation_rejects_out_of_range_values() {
        let base = Preset::with_defaults("m1", "ok");
        assert!(validate(&base).is_ok());

        let mut p = base.clone();
        p.n_ctx = 10;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.temperature = f32::NAN;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.temperature = 5.0;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.top_p = 0.0;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.top_p = 1.5;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.top_k = MAX_TOP_K + 1;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.max_tokens = Some(0);
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.repeat_penalty = 9.0;
        assert!(validate(&p).is_err());

        let mut p = base.clone();
        p.repeat_penalty = f32::INFINITY;
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validation_accepts_the_range_edges() {
        let mut p = Preset::with_defaults("m1", "edges");
        p.n_ctx = MIN_CTX;
        p.temperature = 0.0;
        p.top_p = 1.0;
        p.top_k = 0;
        p.repeat_penalty = 0.5;
        p.max_tokens = Some(MAX_TOKENS_LIMIT);
        assert!(validate(&p).is_ok());
        p.n_ctx = MAX_CTX;
        p.temperature = 2.0;
        p.repeat_penalty = 2.0;
        assert!(validate(&p).is_ok());
    }

    #[test]
    fn validation_rejects_bad_names_and_big_prompts() {
        let mut p = Preset::with_defaults("m1", " padded ");
        assert!(validate(&p).is_err());
        p.name = "bad\nname".into();
        assert!(validate(&p).is_err());
        p.name = "x".repeat(MAX_NAME_CHARS + 1);
        assert!(validate(&p).is_err());
        p.name = "fine".into();
        p.system_prompt = Some("a".repeat(MAX_SYSTEM_PROMPT_BYTES + 1));
        assert!(validate(&p).is_err());
        p.system_prompt = None;
        p.model_id = "  ".into();
        assert!(validate(&p).is_err());
    }

    #[test]
    fn invalid_preset_is_never_stored() {
        let (_t, _p, s) = store();
        let mut bad = Preset::with_defaults("m1", "bad");
        bad.n_ctx = 1;
        assert!(s.create(bad.clone()).is_err());
        assert!(s.save(bad).is_err());
        assert!(s.list_for_model("m1").unwrap().is_empty());
    }

    #[test]
    fn failed_write_leaves_memory_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("presets.json");
        let s = PresetStore::open(&path).unwrap();
        // A directory where the file should go makes the atomic rename fail.
        fs::create_dir(&path).unwrap();
        assert!(s.create(Preset::with_defaults("m1", "a")).is_err());
        assert!(s.list_for_model("m1").unwrap().is_empty());
    }
}
