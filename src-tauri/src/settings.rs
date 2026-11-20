//! Persistent application settings with crash-safe load and atomic save.
//!
//! The settings file always lives in the default data location, even when a custom
//! storage directory is configured, so the app can always find its own configuration.
//! A corrupt file is moved aside to `settings.json.corrupt-<unix>` and defaults are used;
//! the reason is kept in [`SettingsStore::recovery`] so the UI can show it.
//!
//! The API key is stored as the gateway's salted digest record (`key_id`, `salt`,
//! `digest`). The secret itself is never written to disk. Settings files from version 1
//! carried the plaintext secret; those are converted on load and rewritten at once.

use crate::fsutil::{unix_now, write_atomic};
use raptor_core::{RaptorError, Result};
use raptor_gateway::{ApiKeyRecord, StoredApiKey as KeyFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const SETTINGS_VERSION: u32 = 3;
/// What a fresh install targets for chat: 120K tokens when the hardware can hold
/// it, at least `MIN_CONTEXT_TARGET` otherwise. The recommender shrinks the window
/// to what fits (never below the floor) per model and machine.
pub const IDEAL_CONTEXT_TARGET: u32 = 122_880;
/// Models with native long context reserve at least this many tokens for chat.
pub const DEFAULT_CONTEXT_TARGET: u32 = IDEAL_CONTEXT_TARGET;
pub const MIN_CONTEXT_TARGET: u32 = raptor_runtime::config::LONG_CHAT_CONTEXT;
pub const MAX_CONTEXT_TARGET: u32 = 200_000;
pub const MAX_EMBEDDING_MODEL_ID_BYTES: usize = 512;

/// The gateway API key as persisted: the key id, creation time, and the digest record.
/// Holds no secret, so a settings file cannot authenticate on its own.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "KeyFile", into = "KeyFile")]
pub struct ApiKeyEntry {
    pub record: ApiKeyRecord,
    pub created_at: u64,
}

/// Wire shape of [`ApiKeyEntry`]: the gateway's `{key_id, salt, digest}` plus `created_at`.
#[derive(Serialize, Deserialize)]
struct KeyFile {
    #[serde(flatten)]
    fields: KeyFields,
    created_at: u64,
}

impl From<ApiKeyEntry> for KeyFile {
    fn from(e: ApiKeyEntry) -> Self {
        Self {
            fields: e.record.to_stored(),
            created_at: e.created_at,
        }
    }
}

impl TryFrom<KeyFile> for ApiKeyEntry {
    type Error = RaptorError;
    fn try_from(k: KeyFile) -> Result<Self> {
        Ok(Self {
            record: ApiKeyRecord::from_stored(k.fields)?,
            created_at: k.created_at,
        })
    }
}

impl ApiKeyEntry {
    /// Creates a key with a fresh secret. Returns the entry (digest only) and the secret,
    /// which the caller must show to the user once and must not persist.
    pub fn create(id: String, created_at: u64) -> Result<(Self, String)> {
        let (record, secret) = ApiKeyRecord::generate(id)?;
        Ok((Self { record, created_at }, secret))
    }

    /// The non-secret label of the key.
    pub fn id(&self) -> &str {
        self.record.id()
    }
}

impl std::fmt::Debug for ApiKeyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyEntry")
            .field("record", &self.record)
            .field("created_at", &self.created_at)
            .finish()
    }
}

/// The runtime version the user pinned, usually by rolling back. While set, `runtime_ensure`
/// installs only this version and never promotes a newer one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinnedRuntime {
    pub id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    /// Root for models, runtimes, conversations and the context index. `None` uses the
    /// platform default. Takes effect on the next start.
    pub storage_dir: Option<PathBuf>,
    /// Start the gateway when the app starts.
    pub gateway_enabled: bool,
    pub gateway_port: u16,
    /// Bind 0.0.0.0 instead of 127.0.0.1. Only valid with an API key.
    pub lan_enabled: bool,
    pub api_key: Option<ApiKeyEntry>,
    /// Model context window used when loading a model and for context building.
    pub context_target: u32,
    /// Registered workspace roots (absolute paths). The gateway only serves these.
    pub workspaces: Vec<PathBuf>,
    /// Runtime version kept in use regardless of newer installs. `None` means follow the
    /// newest compatible version.
    pub pinned_runtime: Option<PinnedRuntime>,
    /// Library id of the model that serves embeddings for semantic retrieval. `None` disables
    /// it; the context status then says so. The model must be loaded for it to be used.
    pub embedding_model: Option<String>,
    /// Lets a load continue on CPU when a GPU backend has no device. Off by default. A load
    /// that does fall back says so in its result (`cpu_fallback`).
    pub allow_cpu_fallback: bool,
    /// Folders the user scanned for models, offered again in the chat window and Models page.
    pub model_folders: Vec<PathBuf>,
    /// Requests llama-server works on at once (`-np`). Each slot keeps its own recurrent state and
    /// prompt checkpoints (Qwen3.8-27B: about 450 MiB more at 4 slots than at 1), and generation
    /// speed does not depend on it, so fewer slots mean more room for context. Default 2: one
    /// background request next to the one being watched.
    pub parallel_slots: u32,
    /// Backend the user chose for a model, by library id. A model without an entry is `Auto`.
    pub backend_prefs: std::collections::BTreeMap<String, crate::backend_choice::BackendPref>,
    /// Drop reasoning (`<think>` blocks, `reasoning_content`) from earlier assistant turns of API
    /// chat requests before they reach the model, so agent sessions do not fill the context with
    /// old thinking. On by default; set false in settings.json to forward history untouched.
    pub strip_history_reasoning: bool,
    /// How hard thinking models think when a chat does not choose: Low, Medium, High or Max. The
    /// app's chat and the API gateway both use it, and only for a model whose chat template has
    /// an effort control. Medium by default; an old settings file without the field gets Medium.
    pub thinking_effort: raptor_gateway::Effort,
}

/// Default for [`Settings::parallel_slots`].
pub const DEFAULT_PARALLEL_SLOTS: u32 = 2;

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            storage_dir: None,
            gateway_enabled: true,
            gateway_port: raptor_gateway::GatewayConfig::default().port,
            lan_enabled: false,
            api_key: None,
            context_target: DEFAULT_CONTEXT_TARGET,
            workspaces: Vec::new(),
            pinned_runtime: None,
            embedding_model: None,
            allow_cpu_fallback: false,
            model_folders: Vec::new(),
            parallel_slots: DEFAULT_PARALLEL_SLOTS,
            backend_prefs: Default::default(),
            strip_history_reasoning: true,
            thinking_effort: raptor_gateway::Effort::default(),
        }
    }
}

impl Settings {
    /// Checks field ranges and cross-field rules. LAN mode without an API key is refused.
    pub fn validate(&self) -> Result<()> {
        if self.version > SETTINGS_VERSION {
            return Err(RaptorError::Unsupported(format!(
                "settings version {} is newer than this app supports ({SETTINGS_VERSION})",
                self.version
            )));
        }
        if !(1..=8).contains(&self.parallel_slots) {
            return Err(RaptorError::InvalidInput(
                "parallel requests must be 1-8".into(),
            ));
        }
        if self.gateway_port == 0 {
            return Err(RaptorError::InvalidInput(
                "gateway port must be 1-65535".into(),
            ));
        }
        if !(MIN_CONTEXT_TARGET..=MAX_CONTEXT_TARGET).contains(&self.context_target) {
            return Err(RaptorError::InvalidInput(format!(
                "context target must be {MIN_CONTEXT_TARGET}-{MAX_CONTEXT_TARGET} tokens"
            )));
        }
        if let Some(dir) = &self.storage_dir {
            if !dir.is_absolute() {
                return Err(RaptorError::InvalidInput(
                    "storage directory must be an absolute path".into(),
                ));
            }
        }
        if let Some(model) = &self.embedding_model {
            if model.trim().is_empty() || model.len() > MAX_EMBEDDING_MODEL_ID_BYTES {
                return Err(RaptorError::InvalidInput(format!(
                    "embedding model id must be 1-{MAX_EMBEDDING_MODEL_ID_BYTES} bytes"
                )));
            }
        }
        if self.lan_enabled && self.api_key.is_none() {
            return Err(RaptorError::Unauthorized(
                "LAN access requires an API key; generate one first".into(),
            ));
        }
        Ok(())
    }
}

/// Result of loading the settings file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    pub settings: Settings,
    /// Set when the file was corrupt or inconsistent and defaults were used, and when a
    /// legacy key was migrated.
    pub recovery: Option<String>,
    /// True when the file held a legacy plaintext key that must be rewritten.
    pub migrated: bool,
}

/// Version 1 stored the key as `{"id", "secret", "created_at"}`.
#[derive(Deserialize)]
struct LegacyKey {
    id: String,
    secret: String,
    created_at: u64,
}

/// Converts a legacy plaintext key in `root` to the digest record in place. Returns true
/// when a conversion happened. An unusable legacy key is dropped, together with LAN mode,
/// which cannot work without a key. The secret never appears in the returned messages.
fn migrate_legacy_key(root: &mut Value) -> std::result::Result<Option<String>, String> {
    let Some(obj) = root.get_mut("api_key") else {
        return Ok(None);
    };
    if obj.is_null() || obj.get("secret").is_none() {
        return Ok(None);
    }
    let converted = serde_json::from_value::<LegacyKey>(obj.clone())
        .map_err(|_| "stored API key is malformed".to_string())
        .and_then(|legacy| {
            ApiKeyEntry::from_legacy(legacy.id, &legacy.secret, legacy.created_at)
                .map_err(|e| e.to_string())
        });
    match converted {
        Ok(entry) => {
            *obj = serde_json::to_value(KeyFile::from(entry)).map_err(|e| e.to_string())?;
            Ok(Some(
                "the stored API key was converted to a hash-only record; the key still works"
                    .to_string(),
            ))
        }
        Err(reason) => {
            *obj = Value::Null;
            if let Some(map) = root.as_object_mut() {
                map.insert("lan_enabled".into(), Value::Bool(false));
            }
            Ok(Some(format!(
                "the stored API key could not be converted ({reason}); it was removed and LAN access was turned off. Generate a new key"
            )))
        }
    }
}

impl ApiKeyEntry {
    fn from_legacy(id: String, secret: &str, created_at: u64) -> Result<Self> {
        Ok(Self {
            record: ApiKeyRecord::new(id, secret)?,
            created_at,
        })
    }
}

/// Reads settings from `path`. Never fails: a missing file yields defaults; a corrupt
/// file is moved aside and reported.
pub fn load(path: &Path) -> Loaded {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Loaded {
                settings: Settings::default(),
                recovery: None,
                migrated: false,
            };
        }
        Err(e) => {
            return Loaded {
                settings: Settings::default(),
                recovery: Some(format!(
                    "settings could not be read ({e}); defaults are in use and the file was left untouched"
                )),
                migrated: false,
            };
        }
    };
    let parsed = parse_settings(&bytes);
    match parsed {
        Ok((settings, note)) => {
            let migrated = note.is_some();
            Loaded {
                settings,
                recovery: note,
                migrated,
            }
        }
        Err(reason) => {
            let backup = backup_path(path);
            let note = match std::fs::rename(path, &backup) {
                Ok(()) => format!("moved to {}", backup.display()),
                Err(e) => format!("could not be moved aside ({e})"),
            };
            Loaded {
                settings: Settings::default(),
                recovery: Some(format!(
                    "settings file was invalid ({reason}); {note} and defaults were restored"
                )),
                migrated: false,
            }
        }
    }
}

/// Parses settings bytes, converting a legacy key first. Returns the settings and an
/// optional note for the UI (set when a migration happened).
fn parse_settings(bytes: &[u8]) -> std::result::Result<(Settings, Option<String>), String> {
    let mut root: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let mut notes = Vec::new();
    if let Some(note) = migrate_legacy_key(&mut root)? {
        notes.push(note);
    }
    let mut settings: Settings = serde_json::from_value(root).map_err(|e| e.to_string())?;
    if settings.version < SETTINGS_VERSION {
        notes.push(format!(
            "settings were migrated to version {SETTINGS_VERSION}"
        ));
    }
    if !(MIN_CONTEXT_TARGET..=MAX_CONTEXT_TARGET).contains(&settings.context_target) {
        let previous = settings.context_target;
        settings.context_target = previous.clamp(MIN_CONTEXT_TARGET, MAX_CONTEXT_TARGET);
        notes.push(format!(
            "the context target was updated from {previous} to {} tokens to match the supported {}-{} token range",
            settings.context_target, MIN_CONTEXT_TARGET, MAX_CONTEXT_TARGET
        ));
    }
    if !notes.is_empty() {
        settings.version = SETTINGS_VERSION;
    }
    settings.validate().map_err(|e| e.to_string())?;
    Ok((settings, (!notes.is_empty()).then(|| notes.join("; "))))
}

fn backup_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(format!(".corrupt-{}", unix_now()));
    path.with_file_name(name)
}

/// Validates and atomically writes `settings` to `path`.
pub fn save(path: &Path, settings: &Settings) -> Result<()> {
    settings.validate()?;
    let bytes = serde_json::to_vec_pretty(settings)?;
    write_atomic(path, &bytes)
}

/// Shared, persisted settings. Every change goes through [`SettingsStore::update`], which
/// saves before committing the new value in memory.
pub struct SettingsStore {
    path: PathBuf,
    inner: Mutex<Settings>,
    recovery: Mutex<Option<String>>,
}

impl SettingsStore {
    /// Loads (or defaults) the settings at `path`. A legacy plaintext key is rewritten
    /// immediately; if that write fails the note says so and the next save rewrites it.
    pub fn open(path: PathBuf) -> Self {
        let loaded = load(&path);
        let mut recovery = loaded.recovery;
        if loaded.migrated {
            if let Err(e) = save(&path, &loaded.settings) {
                recovery = Some(format!(
                    "{}; rewriting settings failed ({e}) and will be retried on the next change",
                    recovery.unwrap_or_default()
                ));
            }
        }
        Self {
            path,
            inner: Mutex::new(loaded.settings),
            recovery: Mutex::new(recovery),
        }
    }

    /// Store with explicit initial values, used by tests.
    pub fn with_settings(path: PathBuf, settings: Settings) -> Self {
        Self {
            path,
            inner: Mutex::new(settings),
            recovery: Mutex::new(None),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn recovery(&self) -> Option<String> {
        lock(&self.recovery).clone()
    }

    pub fn snapshot(&self) -> Settings {
        lock(&self.inner).clone()
    }

    /// Applies `change` to a copy, validates and saves it, then commits. On error nothing
    /// changes in memory or on disk.
    pub fn update<F>(&self, change: F) -> Result<Settings>
    where
        F: FnOnce(&mut Settings) -> Result<()>,
    {
        let mut guard = lock(&self.inner);
        let mut next = guard.clone();
        change(&mut next)?;
        save(&self.path, &next)?;
        *guard = next.clone();
        Ok(next)
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "nxk_0123456789abcdef0123456789abcdef";

    fn path_in(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("config").join("settings.json")
    }

    fn entry(id: &str) -> ApiKeyEntry {
        ApiKeyEntry::create(id.into(), 1).unwrap().0
    }

    #[test]
    fn missing_file_yields_localhost_defaults_without_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load(&path_in(&dir));
        assert_eq!(loaded.settings, Settings::default());
        assert!(!loaded.settings.lan_enabled);
        assert_eq!(loaded.settings.gateway_port, 8790);
        assert!(loaded.recovery.is_none());
        assert!(!loaded.migrated);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        let s = Settings {
            gateway_port: 9001,
            context_target: 65_536,
            workspaces: vec![dir.path().join("repo")],
            pinned_runtime: Some(PinnedRuntime {
                id: "llama-hip-win".into(),
                version: "b12000".into(),
            }),
            api_key: Some(entry("k1")),
            ..Settings::default()
        };
        save(&p, &s).unwrap();
        let loaded = load(&p);
        assert_eq!(loaded.settings, s);
        assert!(loaded.recovery.is_none());
    }

    #[test]
    fn the_backend_chosen_per_model_is_kept_and_an_old_file_without_it_still_loads() {
        use crate::backend_choice::BackendPref;
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        let mut s = Settings::default();
        s.backend_prefs.insert("m1".into(), BackendPref::Vulkan);
        s.backend_prefs.insert("m2".into(), BackendPref::Rocm);
        save(&p, &s).unwrap();
        assert_eq!(load(&p).settings.backend_prefs, s.backend_prefs);

        std::fs::write(&p, br#"{"version": 2, "gateway_port": 9001}"#).unwrap();
        let old = load(&p).settings;
        assert!(
            old.backend_prefs.is_empty(),
            "no choice means Auto everywhere"
        );
        assert_eq!(old.gateway_port, 9001);
    }

    #[test]
    fn corrupt_file_is_moved_aside_and_defaults_reported() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"{ not json").unwrap();
        let loaded = load(&p);
        assert_eq!(loaded.settings, Settings::default());
        let note = loaded.recovery.expect("recovery must be reported");
        assert!(note.contains("moved to"), "{note}");
        assert!(!p.exists(), "corrupt file must be moved, not left in place");
        let backups: Vec<_> = std::fs::read_dir(p.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(std::fs::read(backups[0].path()).unwrap(), b"{ not json");
    }

    #[test]
    fn lan_without_key_on_disk_is_recovered_as_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, br#"{"lan_enabled": true}"#).unwrap();
        let loaded = load(&p);
        assert!(!loaded.settings.lan_enabled);
        assert!(loaded.recovery.unwrap().contains("API key"));
    }

    #[test]
    fn out_of_range_values_are_rejected_by_save() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        let zero_port = Settings {
            gateway_port: 0,
            ..Settings::default()
        };
        assert!(save(&p, &zero_port).is_err());
        assert!(!p.exists());
        let small_ctx = Settings {
            context_target: 10,
            ..Settings::default()
        };
        assert!(save(&p, &small_ctx).is_err());
    }

    #[test]
    fn old_context_targets_are_migrated_into_the_new_supported_range() {
        let old_small = parse_settings(br#"{"version":2,"context_target":8192}"#).unwrap();
        assert_eq!(old_small.0.context_target, MIN_CONTEXT_TARGET);
        assert!(old_small.1.as_deref().is_some_and(|n| n.contains("8192")));

        let old_large = parse_settings(br#"{"version":2,"context_target":262144}"#).unwrap();
        assert_eq!(old_large.0.context_target, MAX_CONTEXT_TARGET);
        assert!(old_large.1.as_deref().is_some_and(|n| n.contains("262144")));
        assert_eq!(Settings::default().context_target, 122_880);
    }

    #[test]
    fn context_target_boundary_is_200000_tokens() {
        let mut settings = Settings {
            context_target: 200_000,
            ..Settings::default()
        };
        assert!(settings.validate().is_ok());
        settings.context_target += 1;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn store_update_persists_and_failed_update_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        let store = SettingsStore::open(p.clone());
        store
            .update(|s| {
                s.gateway_port = 9100;
                Ok(())
            })
            .unwrap();
        assert_eq!(load(&p).settings.gateway_port, 9100);

        let err = store.update(|s| {
            s.lan_enabled = true; // no key: must be refused
            Ok(())
        });
        assert!(matches!(err, Err(RaptorError::Unauthorized(_))));
        assert!(!store.snapshot().lan_enabled);
        assert!(!load(&p).settings.lan_enabled);
    }

    #[test]
    fn key_label_limits_and_creation_time_survive_a_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let record = ApiKeyRecord::new("ci-bot", SECRET)
            .unwrap()
            .with_label("Build bot")
            .unwrap();
        let limited = record
            .clone()
            .with_limits(raptor_gateway::KeyLimits {
                max_concurrent: Some(2),
                requests_per_minute: None,
            })
            .unwrap();
        let s = Settings {
            api_key: Some(ApiKeyEntry {
                record: limited,
                created_at: 1234,
            }),
            ..Settings::default()
        };
        save(&path_in(&dir), &s).unwrap();
        let back = load(&path_in(&dir)).settings;
        let key = back.api_key.expect("key kept");
        assert_eq!(key.record.label(), "Build bot");
        assert_eq!(key.record.created_at(), record.created_at());
        assert_eq!(key.record.limits().max_concurrent, Some(2));
        assert_eq!(key.record.limits().requests_per_minute, None);
        assert_eq!(key.created_at, 1234);
        assert!(key.record.verify(SECRET));
    }

    #[test]
    fn debug_output_never_contains_the_secret() {
        let (e, secret) = ApiKeyEntry::create("k".into(), 1).unwrap();
        assert!(!format!("{e:?}").contains(&secret));
        assert!(!format!(
            "{:?}",
            Settings {
                api_key: Some(e),
                ..Settings::default()
            }
        )
        .contains(&secret));
    }

    #[test]
    fn stored_key_on_disk_never_contains_the_secret() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        let store = SettingsStore::open(p.clone());
        let (e, secret) = ApiKeyEntry::create("raptor-1".into(), 1).unwrap();
        store
            .update(|s| {
                s.api_key = Some(e.clone());
                Ok(())
            })
            .unwrap();
        let on_disk = std::fs::read_to_string(&p).unwrap();
        assert!(!on_disk.contains(&secret), "secret leaked to settings.json");
        assert!(on_disk.contains("\"digest\""));
        assert!(on_disk.contains("\"salt\""));
        let back = load(&p).settings.api_key.expect("key persisted");
        assert!(
            back.record.verify(&secret),
            "the stored digest must still verify"
        );
        assert!(!back.record.verify("nxk_wrong_secret_value_0000"));
    }

    #[test]
    fn legacy_plaintext_key_is_migrated_and_rewritten_without_the_secret() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let legacy = format!(
            r#"{{"version": 1, "gateway_port": 8790, "lan_enabled": false,
                "api_key": {{"id": "old-key", "secret": "{SECRET}", "created_at": 42}}}}"#
        );
        std::fs::write(&p, legacy.as_bytes()).unwrap();

        let store = SettingsStore::open(p.clone());
        let on_disk = std::fs::read_to_string(&p).unwrap();
        assert!(
            !on_disk.contains(SECRET),
            "migration must remove the secret"
        );
        assert!(on_disk.contains("\"key_id\": \"old-key\""));
        assert!(store.recovery().unwrap().contains("hash-only"));

        let key = store.snapshot().api_key.expect("key kept");
        assert_eq!(key.id(), "old-key");
        assert_eq!(key.created_at, 42);
        assert!(
            key.record.verify(SECRET),
            "the existing client key keeps working"
        );
        assert_eq!(store.snapshot().version, SETTINGS_VERSION);

        // A second load sees the converted file and reports nothing to migrate.
        let again = load(&p);
        assert!(!again.migrated);
        assert!(again.recovery.is_none());
    }

    #[test]
    fn unusable_legacy_key_is_removed_with_lan_and_never_kept() {
        let dir = tempfile::tempdir().unwrap();
        let p = path_in(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            br#"{"lan_enabled": true, "api_key": {"id": "k", "secret": "short", "created_at": 1}}"#,
        )
        .unwrap();
        let loaded = load(&p);
        assert!(!loaded.settings.lan_enabled);
        assert!(loaded.settings.api_key.is_none());
        let note = loaded.recovery.unwrap();
        assert!(!note.contains("short"), "note must not echo the secret");
    }
}

#[cfg(test)]
mod embedding_and_fallback_tests {
    use super::*;

    #[test]
    fn embedding_model_and_cpu_fallback_default_to_off() {
        let s = Settings::default();
        assert!(s.embedding_model.is_none());
        assert!(!s.allow_cpu_fallback, "CPU fallback is never on by default");
        assert_eq!(s.parallel_slots, DEFAULT_PARALLEL_SLOTS);
        for bad in [0, 9] {
            let t = Settings {
                parallel_slots: bad,
                ..Settings::default()
            };
            assert!(t.validate().is_err(), "{bad} slots must be refused");
        }
        for ok in [1, 8] {
            let t = Settings {
                parallel_slots: ok,
                ..Settings::default()
            };
            assert!(t.validate().is_ok());
        }
        assert!(s.validate().is_ok());
    }

    #[test]
    fn a_blank_or_oversized_embedding_model_id_is_refused() {
        let blank = Settings {
            embedding_model: Some("  ".into()),
            ..Settings::default()
        };
        assert!(blank.validate().is_err());
        let long = Settings {
            embedding_model: Some("x".repeat(MAX_EMBEDDING_MODEL_ID_BYTES + 1)),
            ..Settings::default()
        };
        assert!(long.validate().is_err());
        let ok = Settings {
            embedding_model: Some("org/repo/emb.gguf".into()),
            allow_cpu_fallback: true,
            ..Settings::default()
        };
        assert!(ok.validate().is_ok());
    }
}

#[cfg(test)]
mod thinking_effort_tests {
    use super::*;
    use raptor_gateway::Effort;

    #[test]
    fn thinking_effort_defaults_to_medium() {
        assert_eq!(Settings::default().thinking_effort, Effort::Medium);
    }

    #[test]
    fn an_old_settings_file_without_the_field_loads_with_medium() {
        let s: Settings = serde_json::from_str(r#"{"gateway_port": 8790}"#).unwrap();
        assert_eq!(s.thinking_effort, Effort::Medium);
    }

    #[test]
    fn the_setting_round_trips_in_lowercase_and_rejects_unknown_names() {
        let s = Settings {
            thinking_effort: Effort::Max,
            ..Settings::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""thinking_effort":"max""#), "{json}");
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.thinking_effort, Effort::Max);
        assert!(serde_json::from_str::<Settings>(r#"{"thinking_effort": "ultra"}"#).is_err());
    }
}
