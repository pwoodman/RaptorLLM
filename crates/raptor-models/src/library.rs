//! Local model library: an index of installed GGUF files with integrity metadata.
//!
//! Managed models live under `DataDirs::models()` as `owner/repo/[subfolders/]file.gguf` (Hub
//! downloads) or `local/file.gguf` (imports). Referenced models stay where they are and are
//! indexed in place; deleting a reference only drops the index entry, never the user's file.
//!
//! A split model (`name-00001-of-00005.gguf` ... ) is ONE entry: `size_bytes` is the sum of
//! its shards, `location` and `file_name` are the first shard, and `sha256` is the set
//! fingerprint. The individual shards are listed by [`Library::details`]. A set is installed
//! only when every shard is present with its recorded size.
//!
//! A model is "installed" only when its index entry exists, the file (or every shard) is present
//! and its size matches. Partial downloads live in the downloads directory and are never indexed.
//! Revision and Hub tags are kept in a side table of the index file, so entries keep their shape.

use crate::download::{Provenance, Registrar, StagedShard};
use crate::gguf::{self, ArchInfo, GgufMeta, SplitInfo};
use crate::hf::{safe_repo_path, validate_repo_id};
use crate::shard::{parse_shard_name, ShardName};
use crate::util::{fingerprint, sha256_file, HashProgress, ProgressScope};
use raptor_core::paths::safe_file_name;
use raptor_core::{DataDirs, RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const INDEX_VERSION: u32 = 1;
const LOCAL_PUBLISHER: &str = "local";
const MAX_SCAN_DEPTH: usize = 4;
const MAX_SCAN_FILES: usize = 10_000;
const INDEX_FILE: &str = "models-index.json";

/// Where a library entry came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    HuggingFace { repo: String, file_name: String },
    LocalCopy { original_path: String },
    LocalReference { original_path: String },
}

/// One installed (or previously installed) model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Stable id: `owner/repo/[path/]file.gguf` or `local/file.gguf`. For a split model, the
    /// id names the first shard.
    pub id: String,
    /// The file name; for a split model, the first shard.
    pub file_name: String,
    /// Managed: path relative to the models directory. Referenced: absolute path.
    /// For a split model, the first shard.
    pub location: String,
    /// True when the file lives under the library and is removed on delete.
    pub managed: bool,
    /// Total size. For a split model, the sum of all shards.
    pub size_bytes: u64,
    /// SHA-256 of the file. For a split model, the fingerprint of the ordered shard digests.
    pub sha256: String,
    /// Unix seconds when the entry was added.
    pub added_at: u64,
    pub source: ModelSource,
    /// Header metadata of the first shard (or the only file).
    pub gguf: GgufMeta,
}

/// One file of an installed model. A single-file model has exactly one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    pub file_name: String,
    /// Absolute path on disk.
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

/// An installed model with what the entry alone does not carry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelDetails {
    pub entry: ModelEntry,
    /// Exact Hub commit the files were fetched from. `None` for local imports and for entries
    /// installed before revisions were recorded.
    pub revision: Option<String>,
    /// Hub tags of the repo at download time (empty for local imports).
    pub hub_tags: Vec<String>,
    /// Every file in order. A single-file model has one.
    pub files: Vec<ModelFile>,
    /// Support class of the architecture. `Unknown` models still load (llama-server decides).
    pub architecture: ArchInfo,
}

/// Aggregate disk usage of installed models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StorageUsage {
    pub models: usize,
    /// Bytes inside the library directory (removed on delete).
    pub managed_bytes: u64,
    /// Bytes of files the library only references.
    pub referenced_bytes: u64,
}

/// Classification of a GGUF found while scanning a folder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ScanStatus {
    New,
    AlreadyInLibrary {
        id: String,
    },
    /// Corrupt header, or an incomplete or inconsistent shard set (the reason says which).
    InvalidGguf {
        reason: String,
    },
}

/// One model discovered by [`Library::scan_folder`]. For a split model this is one item,
/// pointing at its first shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScanItem {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub status: ScanStatus,
    pub architecture: Option<String>,
    pub expert_count: Option<u64>,
}

/// Shard metadata kept beside an entry. Revision and tags live here too.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ShardRecord {
    file_name: String,
    location: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct EntryExtras {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    revision: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    hub_tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shards: Vec<ShardRecord>,
}

impl EntryExtras {
    fn is_empty(&self) -> bool {
        self.revision.is_none() && self.hub_tags.is_empty() && self.shards.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
struct State {
    entries: Vec<ModelEntry>,
    extras: BTreeMap<String, EntryExtras>,
}

#[derive(Serialize, Deserialize)]
struct IndexFile {
    version: u32,
    entries: Vec<ModelEntry>,
    #[serde(default)]
    extras: BTreeMap<String, EntryExtras>,
}

/// The model library. All methods take `&self`; the index is guarded by a mutex.
pub struct Library {
    models_dir: PathBuf,
    index_path: PathBuf,
    state: Mutex<State>,
    import_progress: Arc<HashProgress>,
}

/// One file read for import or registration: its header, size and digest.
struct Inspected {
    name: String,
    path: PathBuf,
    meta: GgufMeta,
    size: u64,
    sha: String,
}

impl Library {
    /// Opens the library under `dirs` (models in `dirs.models()`, index in `dirs.db()`).
    pub fn open(dirs: &DataDirs) -> Result<Self> {
        Self::open_at(dirs.models(), dirs.db().join(INDEX_FILE))
    }

    /// Opens a library with explicit paths. Missing index means an empty library;
    /// a corrupt or wrong-version index is an error and is never silently replaced.
    pub fn open_at(models_dir: impl Into<PathBuf>, index_path: impl Into<PathBuf>) -> Result<Self> {
        let models_dir = models_dir.into();
        let index_path = index_path.into();
        fs::create_dir_all(&models_dir)?;
        let state = match fs::read(&index_path) {
            Ok(bytes) => {
                let index: IndexFile = serde_json::from_slice(&bytes)
                    .map_err(|e| RaptorError::Integrity(format!("model index is corrupt: {e}")))?;
                if index.version != INDEX_VERSION {
                    return Err(RaptorError::Unsupported(format!(
                        "model index version {} is not supported",
                        index.version
                    )));
                }
                State {
                    entries: index.entries,
                    extras: index.extras,
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => State::default(),
            Err(e) => return Err(e.into()),
        };
        Ok(Self {
            models_dir,
            index_path,
            state: Mutex::new(state),
            import_progress: Arc::new(HashProgress::default()),
        })
    }

    /// The directory managed models are stored in.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Where a Hub file will be stored when managed. Validates the repo id and the repo path
    /// (subfolders allowed, traversal refused).
    pub fn managed_path(&self, repo: &str, file_path: &str) -> Result<PathBuf> {
        validate_repo_id(repo)?;
        safe_repo_path(file_path)?;
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| RaptorError::InvalidInput(format!("invalid repo id: {repo}")))?;
        let mut path = self.models_dir.join(owner).join(name);
        for segment in file_path.split('/') {
            path.push(segment);
        }
        Ok(path)
    }

    /// True when an entry with this id exists in the index (installed or not).
    pub fn contains_id(&self, id: &str) -> Result<bool> {
        Ok(self.lock()?.entries.iter().any(|e| e.id == id))
    }

    /// All installed models, in insertion order. Entries whose file (or any shard) is missing
    /// or has the wrong size are left out.
    pub fn list(&self) -> Result<Vec<ModelEntry>> {
        let state = self.lock()?;
        Ok(state
            .entries
            .iter()
            .filter(|e| self.installed(&state, e))
            .cloned()
            .collect())
    }

    /// One installed model by id.
    pub fn get(&self, id: &str) -> Result<ModelEntry> {
        let state = self.lock()?;
        state
            .entries
            .iter()
            .find(|e| e.id == id && self.installed(&state, e))
            .cloned()
            .ok_or_else(|| RaptorError::NotFound(format!("model not installed: {id}")))
    }

    /// An installed model with its files, revision and Hub tags.
    pub fn details(&self, id: &str) -> Result<ModelDetails> {
        let state = self.lock()?;
        let entry = state
            .entries
            .iter()
            .find(|e| e.id == id && self.installed(&state, e))
            .cloned()
            .ok_or_else(|| RaptorError::NotFound(format!("model not installed: {id}")))?;
        let extras = state.extras.get(id).cloned().unwrap_or_default();
        let files = if extras.shards.is_empty() {
            vec![ModelFile {
                file_name: entry.file_name.clone(),
                path: self.resolve(&entry),
                size_bytes: entry.size_bytes,
                sha256: entry.sha256.clone(),
            }]
        } else {
            extras
                .shards
                .iter()
                .map(|s| ModelFile {
                    file_name: s.file_name.clone(),
                    path: self.resolve_location(entry.managed, &s.location),
                    size_bytes: s.size_bytes,
                    sha256: s.sha256.clone(),
                })
                .collect()
        };
        let architecture = gguf::arch_info(entry.gguf.architecture.as_deref());
        Ok(ModelDetails {
            entry,
            revision: extras.revision,
            hub_tags: extras.hub_tags,
            files,
            architecture,
        })
    }

    /// Absolute path of an installed model's file. For a split model, the first shard.
    pub fn file_path(&self, id: &str) -> Result<PathBuf> {
        self.details(id)?
            .files
            .into_iter()
            .next()
            .map(|f| f.path)
            .ok_or_else(|| RaptorError::Internal(format!("model has no files: {id}")))
    }

    /// Copies a GGUF file (or every shard of a split model) into the library under `local/`.
    /// Headers are validated and content hashed before anything is written. For a split
    /// model, passing any shard imports the whole set, and a missing shard is an error.
    /// Duplicate content (same sha256) and name collisions are refused, never overwritten.
    pub fn import_copy(&self, src: &Path) -> Result<ModelEntry> {
        let _scope = ProgressScope::start(&self.import_progress);
        self.refuse_known_name(src)?;
        let files = inspect_import(src)?;
        let mut state = self.lock()?;
        let id = format!("{LOCAL_PUBLISHER}/{}", files[0].name);
        let identity = identity_of(&files);
        check_free(&state.entries, &id, &identity)?;
        let dir = self.models_dir.join(LOCAL_PUBLISHER);
        for f in &files {
            if dir.join(&f.name).exists() {
                return Err(RaptorError::Conflict(format!(
                    "file name already in use: {}",
                    f.name
                )));
            }
        }
        fs::create_dir_all(&dir)?;

        // Every copy lands under a name the index never lists, is verified, and only then
        // renamed into place. Any failure removes everything this call created.
        let mut tmps: Vec<PathBuf> = Vec::new();
        let mut placed: Vec<PathBuf> = Vec::new();
        let copied = (|| -> Result<()> {
            for f in &files {
                let tmp = dir.join(format!("{}.importing", f.name));
                tmps.push(tmp.clone());
                fs::copy(&f.path, &tmp)?;
                if sha256_file(&tmp)? != f.sha {
                    return Err(RaptorError::Integrity(
                        "copied file does not match the source hash".into(),
                    ));
                }
            }
            for (f, tmp) in files.iter().zip(&tmps) {
                let dest = dir.join(&f.name);
                fs::rename(tmp, &dest)?;
                placed.push(dest);
            }
            Ok(())
        })();
        if let Err(e) = copied {
            for tmp in &tmps {
                let _ = fs::remove_file(tmp);
            }
            for dest in &placed {
                let _ = fs::remove_file(dest);
            }
            return Err(e);
        }

        let locations: Vec<String> = files
            .iter()
            .map(|f| format!("{LOCAL_PUBLISHER}/{}", f.name))
            .collect();
        let shards = shard_records(&files, &locations);
        let entry = build_entry(
            id.clone(),
            files[0].name.clone(),
            locations[0].clone(),
            true,
            total_size(&files),
            identity,
            ModelSource::LocalCopy {
                original_path: files[0].path.display().to_string(),
            },
            files[0].meta.clone(),
        );
        self.commit_new(&mut state, entry, shards, EntryExtras::default(), &placed)
    }

    /// A single-file import whose name is already in the library is refused before the file is
    /// hashed, which takes minutes for a large model. Split models are checked after hashing.
    fn refuse_known_name(&self, src: &Path) -> Result<()> {
        let Some(name) = src.file_name().and_then(|n| n.to_str()) else {
            return Ok(());
        };
        if parse_shard_name(name).is_some_and(|s| s.total > 1) {
            return Ok(());
        }
        // A file already inside the library gets its own, clearer refusal later.
        if std::path::absolute(src)?.starts_with(&self.models_dir) {
            return Ok(());
        }
        let id = format!("{LOCAL_PUBLISHER}/{name}");
        if self.lock()?.entries.iter().any(|e| e.id == id) {
            return Err(RaptorError::Conflict(format!("already in library: {id}")));
        }
        Ok(())
    }

    /// `(bytes hashed, bytes to hash)` of the import running now or the last one. The total
    /// grows as each file of a split set is opened. `(0, 0)` before any import.
    pub fn import_progress(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.import_progress.done.load(Relaxed),
            self.import_progress.total.load(Relaxed),
        )
    }

    /// Indexes a GGUF file (or every shard of a split model) where it is, without copying.
    /// Deleting the entry leaves the files.
    pub fn import_reference(&self, src: &Path) -> Result<ModelEntry> {
        let _scope = ProgressScope::start(&self.import_progress);
        self.refuse_known_name(src)?;
        // `absolute` (unlike `canonicalize`) adds no Windows `\\?\` prefix, so stored paths stay readable.
        let absolute = std::path::absolute(src)?;
        if absolute.starts_with(&self.models_dir) {
            return Err(RaptorError::InvalidInput(
                "file is already inside the model library".into(),
            ));
        }
        let files = inspect_import(&absolute)?;
        if files.iter().any(|f| f.path.starts_with(&self.models_dir)) {
            return Err(RaptorError::InvalidInput(
                "file is already inside the model library".into(),
            ));
        }
        let mut state = self.lock()?;
        let id = format!("{LOCAL_PUBLISHER}/{}", files[0].name);
        let identity = identity_of(&files);
        check_free(&state.entries, &id, &identity)?;
        let locations: Vec<String> = files.iter().map(|f| f.path.display().to_string()).collect();
        let shards = shard_records(&files, &locations);
        let entry = build_entry(
            id,
            files[0].name.clone(),
            locations[0].clone(),
            false,
            total_size(&files),
            identity,
            ModelSource::LocalReference {
                original_path: locations[0].clone(),
            },
            files[0].meta.clone(),
        );
        self.commit_new(&mut state, entry, shards, EntryExtras::default(), &[])
    }

    /// Records a just-promoted download. `path` must be inside the models directory.
    /// `replace` allows the same id to be replaced (the caller already enforced overwrite).
    pub fn register_completed(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
    ) -> Result<ModelEntry> {
        self.register_completed_with(path, sha256, source, replace, &Provenance::default())
    }

    /// Like [`Library::register_completed`], also recording the revision and Hub tags.
    pub fn register_completed_with(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<ModelEntry> {
        let rel = path.strip_prefix(&self.models_dir).map_err(|_| {
            RaptorError::InvalidInput("download target is outside the model library".into())
        })?;
        let id = rel_id(rel);
        let file_name = dest_file_name(path);
        let meta = gguf::parse_file(path)?;
        let size = fs::metadata(path)?.len();
        let mut state = self.lock()?;
        if state.entries.iter().any(|e| e.id == id) && !replace {
            return Err(RaptorError::Conflict(format!("already in library: {id}")));
        }
        if let Some(dup) = state
            .entries
            .iter()
            .find(|e| e.sha256 == sha256 && e.id != id)
        {
            return Err(RaptorError::Conflict(format!(
                "duplicate content, already in library as {}",
                dup.id
            )));
        }
        let entry = build_entry(
            id.clone(),
            file_name,
            rel.to_string_lossy().replace('\\', "/"),
            true,
            size,
            sha256.to_string(),
            source.clone(),
            meta,
        );
        let extras = extras_for(provenance, Vec::new());
        // Build the next state first so a failed save leaves memory and disk in agreement.
        let mut next = State {
            entries: state
                .entries
                .iter()
                .filter(|e| e.id != id)
                .cloned()
                .collect(),
            extras: state.extras.clone(),
        };
        next.extras.remove(&id);
        if let Some(x) = extras {
            next.extras.insert(id.clone(), x);
        }
        next.entries.push(entry.clone());
        self.save(&next)?;
        *state = next;
        Ok(entry)
    }

    /// Records a complete shard set as one entry. Shards must be in index order and every
    /// header must agree (`split.count`, `split.no`, architecture). All or nothing.
    pub fn register_shards(
        &self,
        shards: &[StagedShard],
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<ModelEntry> {
        if shards.is_empty() {
            return Err(RaptorError::InvalidInput(
                "a shard set needs at least one shard".into(),
            ));
        }
        let mut files = Vec::with_capacity(shards.len());
        for s in shards {
            let rel = s.path.strip_prefix(&self.models_dir).map_err(|_| {
                RaptorError::InvalidInput("download target is outside the model library".into())
            })?;
            let (meta, split) = gguf::parse_file_with_split(&s.path)?;
            files.push((
                Inspected {
                    name: dest_file_name(&s.path),
                    path: s.path.clone(),
                    meta,
                    size: fs::metadata(&s.path)?.len(),
                    sha: s.sha256.clone(),
                },
                split,
                rel.to_string_lossy().replace('\\', "/"),
            ));
        }
        check_set_members(&files.iter().map(|(f, s, _)| (f, *s)).collect::<Vec<_>>())?;
        let id = rel_id(Path::new(&files[0].2));
        let inspected: Vec<Inspected> = files.iter().map(|(f, _, _)| clone_inspected(f)).collect();
        let identity = identity_of(&inspected);
        let locations: Vec<String> = files.iter().map(|(_, _, loc)| loc.clone()).collect();
        let records = shard_records(&inspected, &locations);

        let mut state = self.lock()?;
        if state.entries.iter().any(|e| e.id == id) && !replace {
            return Err(RaptorError::Conflict(format!("already in library: {id}")));
        }
        if let Some(dup) = state
            .entries
            .iter()
            .find(|e| e.sha256 == identity && e.id != id)
        {
            return Err(RaptorError::Conflict(format!(
                "duplicate content, already in library as {}",
                dup.id
            )));
        }
        let entry = build_entry(
            id.clone(),
            inspected[0].name.clone(),
            locations[0].clone(),
            true,
            total_size(&inspected),
            identity,
            source.clone(),
            inspected[0].meta.clone(),
        );
        let mut next = State {
            entries: state
                .entries
                .iter()
                .filter(|e| e.id != id)
                .cloned()
                .collect(),
            extras: state.extras.clone(),
        };
        next.extras.remove(&id);
        if let Some(x) = extras_for(provenance, records) {
            next.extras.insert(id.clone(), x);
        }
        next.entries.push(entry.clone());
        self.save(&next)?;
        *state = next;
        Ok(entry)
    }

    /// Removes an entry. Managed files (every shard of a split model) are deleted from disk;
    /// referenced files are kept.
    pub fn delete(&self, id: &str) -> Result<()> {
        let mut state = self.lock()?;
        let pos = state
            .entries
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| RaptorError::NotFound(format!("no such model: {id}")))?;
        let entry = state.entries[pos].clone();
        if entry.managed {
            for path in self.entry_paths(&state, &entry) {
                match fs::remove_file(path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        let mut next = state.clone();
        next.entries.remove(pos);
        next.extras.remove(id);
        self.save(&next)?;
        *state = next;
        Ok(())
    }

    /// Re-hashes every file of an installed model and compares with the index.
    pub fn verify(&self, id: &str) -> Result<()> {
        let details = self.details(id)?;
        let mut digests = Vec::with_capacity(details.files.len());
        for f in &details.files {
            let actual = sha256_file(&f.path)?;
            if actual != f.sha256 {
                return Err(RaptorError::Integrity(format!(
                    "{id} changed on disk: {} does not match its indexed sha256",
                    f.file_name
                )));
            }
            digests.push(actual);
        }
        if details.files.len() > 1 && fingerprint(&digests) != details.entry.sha256 {
            return Err(RaptorError::Integrity(format!(
                "{id}: shard set fingerprint does not match the index"
            )));
        }
        Ok(())
    }

    /// Disk usage of installed models, split by managed vs referenced.
    pub fn storage_usage(&self) -> Result<StorageUsage> {
        let state = self.lock()?;
        let mut usage = StorageUsage {
            models: 0,
            managed_bytes: 0,
            referenced_bytes: 0,
        };
        for e in state.entries.iter().filter(|e| self.installed(&state, e)) {
            usage.models += 1;
            if e.managed {
                usage.managed_bytes = usage.managed_bytes.saturating_add(e.size_bytes);
            } else {
                usage.referenced_bytes = usage.referenced_bytes.saturating_add(e.size_bytes);
            }
        }
        Ok(usage)
    }

    /// Finds `.gguf` files under `dir` (up to 4 levels deep, at most 10,000 files) and
    /// classifies each as new, already in the library, or invalid. Split shards are grouped:
    /// one item per set, pointing at its first shard. Hashes are computed only for candidates
    /// whose size matches an installed model.
    pub fn scan_folder(&self, dir: &Path) -> Result<Vec<ScanItem>> {
        let mut found = Vec::new();
        let mut partial = Vec::new();
        collect_ggufs(dir, 0, &mut found, &mut partial)?;
        let mut singles = Vec::new();
        let mut sets: BTreeMap<(PathBuf, String, u32), Vec<(u32, PathBuf)>> = BTreeMap::new();
        for path in found {
            let name = dest_file_name(&path);
            match parse_shard_name(&name) {
                Some(shard) if shard.total > 1 => {
                    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
                    sets.entry((parent, shard.prefix, shard.total))
                        .or_default()
                        .push((shard.index, path));
                }
                _ => singles.push(path),
            }
        }
        let state = self.lock()?;
        let mut items = Vec::with_capacity(singles.len() + sets.len());
        for path in singles {
            let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mut architecture = None;
            let mut expert_count = None;
            let status = match gguf::parse_file(&path) {
                Err(e) => ScanStatus::InvalidGguf {
                    reason: e.to_string(),
                },
                Ok(meta) => {
                    architecture = meta.architecture.clone();
                    expert_count = meta.expert_count;
                    // A file that is already referenced by path is known without reading it.
                    // Hashing is the slow part (minutes for a large model), so it comes last.
                    if let Some(id) = referenced_by_path(&state.entries, &path) {
                        ScanStatus::AlreadyInLibrary { id }
                    // Hash only when an installed model has the same size.
                    } else if self.size_matches_installed(&state, size_bytes) {
                        let sha = sha256_file(&path)?;
                        match self.find_by_identity(&state, size_bytes, &sha) {
                            Some(id) => ScanStatus::AlreadyInLibrary { id },
                            None => ScanStatus::New,
                        }
                    } else {
                        ScanStatus::New
                    }
                }
            };
            items.push(ScanItem {
                path,
                size_bytes,
                status,
                architecture,
                expert_count,
            });
        }
        for ((_, prefix, total), mut members) in sets {
            members.sort_by_key(|(i, _)| *i);
            let first = members[0].1.clone();
            let size_bytes: u64 = members
                .iter()
                .map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum();
            let present: Vec<u32> = members.iter().map(|(i, _)| *i).collect();
            let missing: Vec<String> = (1..=total)
                .filter(|i| !present.contains(i))
                .map(|i| format!("{i} of {total}"))
                .collect();
            let status = if !missing.is_empty() {
                ScanStatus::InvalidGguf {
                    reason: format!(
                        "incomplete shard set {prefix}: missing shard {}",
                        missing.join(", ")
                    ),
                }
            } else {
                let hash = self.size_matches_installed(&state, size_bytes);
                match scan_set(&members, hash) {
                    Err(e) => ScanStatus::InvalidGguf {
                        reason: e.to_string(),
                    },
                    Ok(None) => ScanStatus::New,
                    Ok(Some(identity)) => {
                        match self.find_by_identity(&state, size_bytes, &identity) {
                            Some(id) => ScanStatus::AlreadyInLibrary { id },
                            None => ScanStatus::New,
                        }
                    }
                }
            };
            let meta = gguf::parse_file(&first).ok();
            items.push(ScanItem {
                architecture: meta.as_ref().and_then(|meta| meta.architecture.clone()),
                expert_count: meta.and_then(|meta| meta.expert_count),
                path: first,
                size_bytes,
                status,
            });
        }
        for path in partial {
            let size_bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            items.push(ScanItem {
                path,
                size_bytes,
                status: ScanStatus::InvalidGguf {
                    reason: "incomplete download (.gguf.part): not installed and never listed as a model"
                        .into(),
                },
                architecture: None,
                expert_count: None,
            });
        }
        Ok(items)
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        crate::util::lock(&self.state)
    }

    fn size_matches_installed(&self, state: &State, size: u64) -> bool {
        state
            .entries
            .iter()
            .any(|e| e.size_bytes == size && self.installed(state, e))
    }

    fn find_by_identity(&self, state: &State, size: u64, identity: &str) -> Option<String> {
        state
            .entries
            .iter()
            .find(|e| e.size_bytes == size && e.sha256 == identity && self.installed(state, e))
            .map(|e| e.id.clone())
    }

    fn resolve(&self, entry: &ModelEntry) -> PathBuf {
        self.resolve_location(entry.managed, &entry.location)
    }

    fn resolve_location(&self, managed: bool, location: &str) -> PathBuf {
        if managed {
            self.models_dir.join(location)
        } else {
            PathBuf::from(location)
        }
    }

    /// Every file of an entry, from its shard list or its single location.
    fn entry_paths(&self, state: &State, entry: &ModelEntry) -> Vec<PathBuf> {
        match state.extras.get(&entry.id).filter(|x| !x.shards.is_empty()) {
            Some(x) => x
                .shards
                .iter()
                .map(|s| self.resolve_location(entry.managed, &s.location))
                .collect(),
            None => vec![self.resolve(entry)],
        }
    }

    /// True when every file of the entry exists with its recorded size.
    fn installed(&self, state: &State, entry: &ModelEntry) -> bool {
        let file_ok = |path: PathBuf, size: u64| matches!(fs::metadata(path), Ok(m) if m.is_file() && m.len() == size);
        match state.extras.get(&entry.id).filter(|x| !x.shards.is_empty()) {
            Some(x) => x.shards.iter().all(|s| {
                file_ok(
                    self.resolve_location(entry.managed, &s.location),
                    s.size_bytes,
                )
            }),
            None => file_ok(self.resolve(entry), entry.size_bytes),
        }
    }

    /// Adds `entry` (and its shard records) to the index and saves. On failure removes the
    /// files in `created`.
    fn commit_new(
        &self,
        state: &mut State,
        entry: ModelEntry,
        shards: Vec<ShardRecord>,
        mut extras: EntryExtras,
        created: &[PathBuf],
    ) -> Result<ModelEntry> {
        let mut next = state.clone();
        extras.shards = shards;
        if !extras.is_empty() {
            next.extras.insert(entry.id.clone(), extras);
        }
        next.entries.push(entry.clone());
        if let Err(e) = self.save(&next) {
            for path in created {
                let _ = fs::remove_file(path);
            }
            return Err(e);
        }
        *state = next;
        Ok(entry)
    }

    fn save(&self, state: &State) -> Result<()> {
        let index = IndexFile {
            version: INDEX_VERSION,
            entries: state.entries.clone(),
            extras: state.extras.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&index)?;
        crate::util::write_atomic(&self.index_path, &bytes)
    }
}

impl Registrar for Library {
    fn register(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
    ) -> Result<()> {
        self.register_completed(path, sha256, source, replace)
            .map(|_| ())
    }

    fn register_with(
        &self,
        path: &Path,
        sha256: &str,
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<()> {
        self.register_completed_with(path, sha256, source, replace, provenance)
            .map(|_| ())
    }

    fn register_set(
        &self,
        shards: &[StagedShard],
        source: &ModelSource,
        replace: bool,
        provenance: &Provenance,
    ) -> Result<()> {
        self.register_shards(shards, source, replace, provenance)
            .map(|_| ())
    }
}

/// Refuses an id already in use or a content identity already in the library.
/// Id of an entry that references exactly this file in place (same path, either separator,
/// any letter case), so a scan need not hash it to know it is already in the library.
fn referenced_by_path(entries: &[ModelEntry], path: &Path) -> Option<String> {
    let key = |p: &str| p.to_lowercase().replace('\\', "/");
    let want = key(&path.to_string_lossy());
    entries
        .iter()
        .find(|e| !e.managed && key(&e.location) == want)
        .map(|e| e.id.clone())
}

fn check_free(entries: &[ModelEntry], id: &str, identity: &str) -> Result<()> {
    if entries.iter().any(|e| e.id == id) {
        return Err(RaptorError::Conflict(format!("already in library: {id}")));
    }
    if let Some(dup) = entries.iter().find(|e| e.sha256 == identity) {
        return Err(RaptorError::Conflict(format!(
            "duplicate content, already in library as {}",
            dup.id
        )));
    }
    Ok(())
}

/// The identity of an import: the file's own digest, or the fingerprint of a shard set.
fn identity_of(files: &[Inspected]) -> String {
    if files.len() == 1 {
        return files[0].sha.clone();
    }
    let digests: Vec<String> = files.iter().map(|f| f.sha.clone()).collect();
    fingerprint(&digests)
}

fn total_size(files: &[Inspected]) -> u64 {
    files.iter().map(|f| f.size).sum()
}

fn shard_records(files: &[Inspected], locations: &[String]) -> Vec<ShardRecord> {
    if files.len() < 2 {
        return Vec::new();
    }
    files
        .iter()
        .zip(locations)
        .map(|(f, loc)| ShardRecord {
            file_name: f.name.clone(),
            location: loc.clone(),
            size_bytes: f.size,
            sha256: f.sha.clone(),
        })
        .collect()
}

fn extras_for(provenance: &Provenance, shards: Vec<ShardRecord>) -> Option<EntryExtras> {
    let extras = EntryExtras {
        revision: provenance.revision.clone(),
        hub_tags: provenance.hub_tags.clone(),
        shards,
    };
    (!extras.is_empty()).then_some(extras)
}

fn clone_inspected(f: &Inspected) -> Inspected {
    Inspected {
        name: f.name.clone(),
        path: f.path.clone(),
        meta: f.meta.clone(),
        size: f.size,
        sha: f.sha.clone(),
    }
}

/// Checks that shards form a consistent set, in the order given: each reports `split.count`
/// equal to the number of shards and `split.no` equal to its position (0-based), and every
/// reported architecture agrees.
fn check_set_members(members: &[(&Inspected, SplitInfo)]) -> Result<()> {
    let total = members.len() as u64;
    let first_arch = members
        .iter()
        .find_map(|(f, _)| f.meta.architecture.clone());
    for (i, (f, split)) in members.iter().enumerate() {
        if split.count != Some(total) {
            return Err(RaptorError::Integrity(format!(
                "{} reports split count {:?}, but the set has {total} shards",
                f.name, split.count
            )));
        }
        if split.no != Some(i as u64) {
            return Err(RaptorError::Integrity(format!(
                "{} is at position {:?} in the set, expected {i}",
                f.name, split.no
            )));
        }
        if let (Some(a), Some(b)) = (&first_arch, &f.meta.architecture) {
            if a != b {
                return Err(RaptorError::Integrity(format!(
                    "shard architectures disagree: {a} and {b}"
                )));
            }
        }
    }
    Ok(())
}

/// Validates the members of a scanned set in index order. Returns the set fingerprint when
/// `hash` is true, and `None` (headers checked, no hashing) otherwise.
fn scan_set(members: &[(u32, PathBuf)], hash: bool) -> Result<Option<String>> {
    let mut inspected = Vec::with_capacity(members.len());
    let mut splits = Vec::with_capacity(members.len());
    for (_, path) in members {
        let (meta, split) = gguf::parse_file_with_split(path)?;
        let size = fs::metadata(path)?.len();
        let sha = if hash {
            sha256_file(path)?
        } else {
            String::new()
        };
        inspected.push(Inspected {
            name: dest_file_name(path),
            path: path.clone(),
            meta,
            size,
            sha,
        });
        splits.push(split);
    }
    let pairs: Vec<(&Inspected, SplitInfo)> = inspected.iter().zip(splits).collect();
    check_set_members(&pairs)?;
    Ok(hash.then(|| identity_of(&inspected)))
}

/// Reads one import source (name, header with split keys, size, digest).
fn inspect_one(src: &Path) -> Result<(Inspected, SplitInfo)> {
    let name = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| RaptorError::InvalidInput("source has no usable file name".into()))?;
    safe_file_name(name)?;
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(RaptorError::InvalidInput(
            "only .gguf files can be imported".into(),
        ));
    }
    let (meta, split) = gguf::parse_file_with_split(src)?;
    let size = fs::metadata(src)?.len();
    let sha = sha256_file(src)?;
    Ok((
        Inspected {
            name: name.to_string(),
            path: src.to_path_buf(),
            meta,
            size,
            sha,
        },
        split,
    ))
}

/// Validates an import: one file, or every shard of the set `src` belongs to.
fn inspect_import(src: &Path) -> Result<Vec<Inspected>> {
    let name = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| RaptorError::InvalidInput("source has no usable file name".into()))?;
    let files = match parse_shard_name(name) {
        Some(shard) if shard.total > 1 => inspect_shard_set(src, &shard)?,
        _ => {
            let (one, split) = inspect_one(src)?;
            if split.count.unwrap_or(1) > 1 {
                return Err(RaptorError::Unsupported(
                    "this file is one shard of a split model; import it through the complete set's \
                     -00001-of-N file name"
                        .into(),
                ));
            }
            vec![one]
        }
    };
    // Only a known-unsupported architecture is refused. Unknown ones are the user's own files.
    if let Some(arch) = files[0].meta.architecture.as_deref() {
        gguf::refuse_known_unsupported(arch)?;
    }
    Ok(files)
}

fn inspect_shard_set(src: &Path, shard: &ShardName) -> Result<Vec<Inspected>> {
    let dir = src
        .parent()
        .ok_or_else(|| RaptorError::InvalidInput("shard has no parent directory".into()))?;
    let mut present: BTreeMap<u32, PathBuf> = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let name = dest_file_name(&path);
        if let Some(s) = parse_shard_name(&name) {
            if s.prefix == shard.prefix && s.total == shard.total {
                present.entry(s.index).or_insert(path);
            }
        }
    }
    let missing: Vec<String> = (1..=shard.total)
        .filter(|i| !present.contains_key(i))
        .map(|i| format!("{i} of {}", shard.total))
        .collect();
    if !missing.is_empty() {
        return Err(RaptorError::NotFound(format!(
            "incomplete shard set {}: missing shard {} in {}",
            shard.prefix,
            missing.join(", "),
            dir.display()
        )));
    }
    let mut members = Vec::with_capacity(shard.total as usize);
    for i in 1..=shard.total {
        let path = &present[&i];
        members.push(inspect_one(path)?);
    }
    let pairs: Vec<(&Inspected, SplitInfo)> = members.iter().map(|(f, s)| (f, *s)).collect();
    check_set_members(&pairs)?;
    Ok(members.into_iter().map(|(f, _)| f).collect())
}

fn rel_id(rel: &Path) -> String {
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

#[allow(clippy::too_many_arguments)]
fn build_entry(
    id: String,
    file_name: String,
    location: String,
    managed: bool,
    size_bytes: u64,
    sha256: String,
    source: ModelSource,
    gguf: GgufMeta,
) -> ModelEntry {
    ModelEntry {
        id,
        file_name,
        location,
        managed,
        size_bytes,
        sha256,
        added_at: now_unix(),
        source,
        gguf,
    }
}

fn dest_file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Collects `.gguf` files into `out` and unfinished downloads (`*.gguf.part`) into `partial`.
/// A partial file is never a model; the scan only reports it.
fn collect_ggufs(
    dir: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
    partial: &mut Vec<PathBuf>,
) -> Result<()> {
    if depth > MAX_SCAN_DEPTH {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        if out.len() + partial.len() >= MAX_SCAN_FILES {
            return Ok(());
        }
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_ggufs(&path, depth + 1, out, partial)?;
        } else if !file_type.is_file() {
            continue;
        } else if path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            out.push(path);
        } else if is_partial_download(&path) {
            partial.push(path);
        }
    }
    Ok(())
}

/// True for `<name>.gguf.part`, the name the downloader gives an unfinished file.
fn is_partial_download(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.to_ascii_lowercase().ends_with(".gguf.part"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{gguf_file, gguf_shard, sha256_hex, GgufWriter};
    use tempfile::TempDir;

    struct Fixture {
        _tmp: TempDir,
        root: PathBuf,
        lib: Library,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let lib = Library::open_at(root.join("models"), root.join("db").join(INDEX_FILE)).unwrap();
        Fixture {
            _tmp: tmp,
            root,
            lib,
        }
    }

    fn write_src(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, bytes).unwrap();
        p
    }

    /// Writes a split set of `count` shards into `dir` and returns their paths in order.
    fn write_set(dir: &Path, prefix: &str, count: u32) -> Vec<PathBuf> {
        (0..count)
            .map(|i| {
                let name = format!("{prefix}-{:05}-of-{:05}.gguf", i + 1, count);
                write_src(dir, &name, &gguf_shard("llama", i, count))
            })
            .collect()
    }

    #[test]
    fn import_copy_indexes_and_copies_valid_gguf() {
        let f = fixture();
        let bytes = gguf_file("llama", 15);
        let src = write_src(&f.root, "tiny.gguf", &bytes);
        let entry = f.lib.import_copy(&src).unwrap();
        assert_eq!(entry.id, "local/tiny.gguf");
        assert!(entry.managed);
        assert_eq!(entry.sha256, sha256_hex(&bytes));
        assert_eq!(entry.gguf.architecture.as_deref(), Some("llama"));
        assert!(f
            .root
            .join("models")
            .join("local")
            .join("tiny.gguf")
            .is_file());
        assert_eq!(f.lib.list().unwrap().len(), 1);
    }

    #[test]
    fn import_refuses_corrupt_gguf_and_writes_nothing() {
        let f = fixture();
        let src = write_src(&f.root, "bad.gguf", b"GGUF\x03\x00\x00\x00 truncated");
        assert!(f.lib.import_copy(&src).is_err());
        assert!(f.lib.list().unwrap().is_empty());
        assert!(!f
            .root
            .join("models")
            .join("local")
            .join("bad.gguf")
            .exists());
        assert!(!f
            .root
            .join("models")
            .join("local")
            .join("bad.gguf.importing")
            .exists());
    }

    #[test]
    fn import_refuses_non_gguf_extension() {
        let f = fixture();
        let src = write_src(&f.root, "model.bin", &gguf_file("llama", 15));
        assert!(matches!(
            f.lib.import_copy(&src),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn duplicate_content_under_another_name_is_refused() {
        let f = fixture();
        let bytes = gguf_file("llama", 15);
        let a = write_src(&f.root, "a.gguf", &bytes);
        f.lib.import_copy(&a).unwrap();
        let b = write_src(&f.root, "b.gguf", &bytes);
        match f.lib.import_copy(&b) {
            Err(RaptorError::Conflict(m)) => assert!(m.contains("local/a.gguf")),
            other => panic!("expected duplicate conflict, got {other:?}"),
        }
        assert!(!f.root.join("models").join("local").join("b.gguf").exists());
    }

    #[test]
    fn same_name_different_content_is_refused_not_overwritten() {
        let f = fixture();
        let a = write_src(&f.root, "m.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&a).unwrap();
        let other = write_src(&f.root, "other/m.gguf", &gguf_file("llama", 7));
        assert!(matches!(
            f.lib.import_copy(&other),
            Err(RaptorError::Conflict(_))
        ));
        let kept = fs::read(f.root.join("models").join("local").join("m.gguf")).unwrap();
        assert_eq!(kept, gguf_file("llama", 15));
    }

    #[test]
    fn index_survives_reopen() {
        let f = fixture();
        let src = write_src(&f.root, "keep.gguf", &gguf_file("qwen2", 15));
        f.lib.import_copy(&src).unwrap();
        let reopened =
            Library::open_at(f.root.join("models"), f.root.join("db").join(INDEX_FILE)).unwrap();
        let list = reopened.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].gguf.architecture.as_deref(), Some("qwen2"));
    }

    #[test]
    fn corrupt_index_is_an_error_not_a_silent_reset() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = tmp.path().join("idx.json");
        fs::write(&idx, b"{ not json").unwrap();
        assert!(matches!(
            Library::open_at(tmp.path().join("m"), &idx),
            Err(RaptorError::Integrity(_))
        ));
        // The bad index must still be there, untouched.
        assert_eq!(fs::read(&idx).unwrap(), b"{ not json");
    }

    #[test]
    fn wrong_index_version_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = tmp.path().join("idx.json");
        fs::write(&idx, br#"{"version": 99, "entries": []}"#).unwrap();
        assert!(matches!(
            Library::open_at(tmp.path().join("m"), &idx),
            Err(RaptorError::Unsupported(_))
        ));
    }

    #[test]
    fn index_written_before_extras_still_opens() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = tmp.path().join("idx.json");
        fs::write(&idx, br#"{"version": 1, "entries": []}"#).unwrap();
        let lib = Library::open_at(tmp.path().join("m"), &idx).unwrap();
        assert!(lib.list().unwrap().is_empty());
    }

    #[test]
    fn partial_or_missing_files_are_not_listed_as_installed() {
        let f = fixture();
        let src = write_src(&f.root, "gone.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&src).unwrap();
        fs::remove_file(f.root.join("models").join("local").join("gone.gguf")).unwrap();
        assert!(f.lib.list().unwrap().is_empty());
        assert!(matches!(
            f.lib.get("local/gone.gguf"),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn size_mismatch_is_not_installed() {
        let f = fixture();
        let src = write_src(&f.root, "grow.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&src).unwrap();
        let path = f.root.join("models").join("local").join("grow.gguf");
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(b"extra");
        fs::write(&path, bytes).unwrap();
        assert!(f.lib.list().unwrap().is_empty());
    }

    #[test]
    fn verify_detects_same_size_tampering() {
        let f = fixture();
        let src = write_src(&f.root, "t.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&src).unwrap();
        f.lib.verify("local/t.gguf").unwrap();
        let path = f.root.join("models").join("local").join("t.gguf");
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            f.lib.verify("local/t.gguf"),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn delete_managed_removes_file_and_entry() {
        let f = fixture();
        let src = write_src(&f.root, "d.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&src).unwrap();
        let path = f.root.join("models").join("local").join("d.gguf");
        f.lib.delete("local/d.gguf").unwrap();
        assert!(!path.exists());
        assert!(f.lib.list().unwrap().is_empty());
        assert!(matches!(
            f.lib.delete("local/d.gguf"),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn a_name_already_in_the_library_is_refused_before_the_file_is_hashed() {
        let f = fixture();
        let src = write_src(&f.root, "dup.gguf", &gguf_file("llama", 15));
        f.lib.import_reference(&src).unwrap();
        assert!(
            f.lib.import_progress().1 > 0,
            "the first import hashed the file"
        );
        for result in [f.lib.import_reference(&src), f.lib.import_copy(&src)] {
            assert!(matches!(result, Err(RaptorError::Conflict(_))));
            assert_eq!(
                f.lib.import_progress(),
                (0, 0),
                "nothing was hashed for the refused import"
            );
        }
    }

    #[test]
    fn a_scan_recognises_a_referenced_file_by_path_without_hashing_it() {
        let f = fixture();
        let src = write_src(&f.root, "known.gguf", &gguf_file("llama", 15));
        f.lib.import_reference(&src).unwrap();
        // Scanning the folder must not read the file again: the progress counter stays at the
        // import's total, and no hashing scope is open during the scan.
        let before = f.lib.import_progress();
        let items = f.lib.scan_folder(&f.root).unwrap();
        assert!(items.iter().any(|i| i.path == src
            && matches!(&i.status, ScanStatus::AlreadyInLibrary { id } if id == "local/known.gguf")));
        assert_eq!(f.lib.import_progress(), before);
    }

    #[test]
    fn referenced_file_survives_delete() {
        let f = fixture();
        let src = write_src(&f.root, "ext.gguf", &gguf_file("llama", 15));
        let entry = f.lib.import_reference(&src).unwrap();
        assert!(!entry.managed);
        f.lib.delete("local/ext.gguf").unwrap();
        assert!(
            src.exists(),
            "user's own file must never be deleted by the library"
        );
    }

    #[test]
    fn reference_inside_library_is_refused() {
        let f = fixture();
        let src = write_src(&f.root, "x.gguf", &gguf_file("llama", 15));
        f.lib.import_copy(&src).unwrap();
        let inside = f.root.join("models").join("local").join("x.gguf");
        assert!(matches!(
            f.lib.import_reference(&inside),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn storage_usage_splits_managed_and_referenced() {
        let f = fixture();
        let a_bytes = gguf_file("llama", 15);
        let a = write_src(&f.root, "a.gguf", &a_bytes);
        f.lib.import_copy(&a).unwrap();
        let b = write_src(&f.root, "b.gguf", &gguf_file("llama", 7));
        f.lib.import_reference(&b).unwrap();
        let usage = f.lib.storage_usage().unwrap();
        assert_eq!(usage.models, 2);
        assert_eq!(usage.managed_bytes, a_bytes.len() as u64);
        assert_eq!(usage.referenced_bytes, fs::metadata(&b).unwrap().len());
    }

    #[test]
    fn register_completed_for_download_path() {
        let f = fixture();
        let dest = f
            .lib
            .managed_path("owner/repo", "model-Q4_K_M.gguf")
            .unwrap();
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let bytes = gguf_file("llama", 15);
        fs::write(&dest, &bytes).unwrap();
        let source = ModelSource::HuggingFace {
            repo: "owner/repo".into(),
            file_name: "model-Q4_K_M.gguf".into(),
        };
        let entry = f
            .lib
            .register_completed(&dest, &sha256_hex(&bytes), &source, false)
            .unwrap();
        assert_eq!(entry.id, "owner/repo/model-Q4_K_M.gguf");
        assert_eq!(
            f.lib.get("owner/repo/model-Q4_K_M.gguf").unwrap().location,
            "owner/repo/model-Q4_K_M.gguf"
        );
    }

    #[test]
    fn register_refuses_outside_library_and_duplicate_id() {
        let f = fixture();
        let outside = write_src(&f.root, "elsewhere.gguf", &gguf_file("llama", 15));
        let source = ModelSource::HuggingFace {
            repo: "o/r".into(),
            file_name: "elsewhere.gguf".into(),
        };
        assert!(matches!(
            f.lib.register_completed(&outside, "00", &source, false),
            Err(RaptorError::InvalidInput(_))
        ));

        let dest = f.lib.managed_path("o/r", "m.gguf").unwrap();
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let bytes = gguf_file("llama", 15);
        fs::write(&dest, &bytes).unwrap();
        let sha = sha256_hex(&bytes);
        f.lib
            .register_completed(&dest, &sha, &source, false)
            .unwrap();
        assert!(matches!(
            f.lib.register_completed(&dest, &sha, &source, false),
            Err(RaptorError::Conflict(_))
        ));
        // With replace=true the same id may be replaced by the caller's explicit overwrite.
        f.lib
            .register_completed(&dest, &sha, &source, true)
            .unwrap();
        assert_eq!(f.lib.list().unwrap().len(), 1);
    }

    #[test]
    fn managed_path_rejects_hostile_inputs() {
        let f = fixture();
        assert!(f.lib.managed_path("../etc", "x.gguf").is_err());
        assert!(f.lib.managed_path("o/r", "../../x.gguf").is_err());
        assert!(f.lib.managed_path("o/r", "a\\b.gguf").is_err());
        assert!(f.lib.managed_path("o/r", "ok.gguf").is_ok());
        assert_eq!(
            f.lib.managed_path("o/r", "Q4/ok.gguf").unwrap(),
            f.root
                .join("models")
                .join("o")
                .join("r")
                .join("Q4")
                .join("ok.gguf")
        );
    }

    #[test]
    fn scan_classifies_new_duplicate_and_invalid() {
        let f = fixture();
        let known_bytes = gguf_file("llama", 15);
        let known = write_src(&f.root, "known.gguf", &known_bytes);
        f.lib.import_copy(&known).unwrap();

        let scan = f.root.join("lmstudio").join("publisher").join("repo");
        fs::create_dir_all(&scan).unwrap();
        write_src(&scan, "same.gguf", &known_bytes);
        write_src(&scan, "fresh.gguf", &gguf_file("qwen2", 12));
        write_src(&scan, "broken.gguf", b"GGUF garbage");
        write_src(&scan, "readme.txt", b"ignored");

        let items = f.lib.scan_folder(&f.root.join("lmstudio")).unwrap();
        assert_eq!(items.len(), 3);
        let by_name = |n: &str| items.iter().find(|i| i.path.ends_with(n)).unwrap();
        assert_eq!(
            by_name("same.gguf").status,
            ScanStatus::AlreadyInLibrary {
                id: "local/known.gguf".into()
            }
        );
        assert_eq!(by_name("fresh.gguf").status, ScanStatus::New);
        assert!(matches!(
            by_name("broken.gguf").status,
            ScanStatus::InvalidGguf { .. }
        ));
    }

    #[test]
    fn scan_depth_is_limited() {
        let f = fixture();
        let mut deep = f.root.join("deep");
        for i in 0..7 {
            deep = deep.join(format!("d{i}"));
        }
        fs::create_dir_all(&deep).unwrap();
        write_src(&deep, "too-deep.gguf", &gguf_file("llama", 15));
        let items = f.lib.scan_folder(&f.root.join("deep")).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn gguf_writer_round_trip_through_library_meta() {
        let f = fixture();
        let mut w = GgufWriter::new();
        w.kv_str("general.architecture", "llama");
        w.kv_u32("general.file_type", 17);
        let src = write_src(&f.root, "meta.gguf", &w.finish());
        let entry = f.lib.import_copy(&src).unwrap();
        assert_eq!(entry.gguf.quant_label.as_deref(), Some("Q5_K_M"));
    }

    #[test]
    fn complete_split_import_is_one_entry_with_summed_size() {
        let f = fixture();
        let shards = write_set(&f.root, "big-Q4_K_M", 3);
        let entry = f.lib.import_copy(&shards[1]).unwrap();
        assert_eq!(entry.id, "local/big-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(entry.file_name, "big-Q4_K_M-00001-of-00003.gguf");
        let expected: u64 = shards.iter().map(|p| fs::metadata(p).unwrap().len()).sum();
        assert_eq!(entry.size_bytes, expected);
        assert_eq!(f.lib.list().unwrap().len(), 1);
        let details = f.lib.details(&entry.id).unwrap();
        assert_eq!(details.files.len(), 3);
        assert_eq!(details.files[2].file_name, "big-Q4_K_M-00003-of-00003.gguf");
        assert_eq!(
            f.lib.file_path(&entry.id).unwrap(),
            f.root
                .join("models")
                .join("local")
                .join("big-Q4_K_M-00001-of-00003.gguf")
        );
        f.lib.verify(&entry.id).unwrap();
    }

    #[test]
    fn import_with_a_missing_shard_names_it_and_writes_nothing() {
        let f = fixture();
        let shards = write_set(&f.root, "gap", 3);
        fs::remove_file(&shards[1]).unwrap();
        match f.lib.import_copy(&shards[0]) {
            Err(RaptorError::NotFound(m)) => assert!(m.contains("missing shard 2 of 3"), "{m}"),
            other => panic!("expected missing shard error, got {other:?}"),
        }
        assert!(f.lib.list().unwrap().is_empty());
        assert!(!f
            .root
            .join("models")
            .join("local")
            .join("gap-00001-of-00003.gguf")
            .exists());
    }

    #[test]
    fn import_refuses_a_shard_set_whose_headers_disagree() {
        let f = fixture();
        let shards = write_set(&f.root, "bad", 2);
        // Shard 2 claims to be part of a 5-shard split.
        fs::write(&shards[1], gguf_shard("llama", 1, 5)).unwrap();
        assert!(matches!(
            f.lib.import_copy(&shards[0]),
            Err(RaptorError::Integrity(_))
        ));
        assert!(f.lib.list().unwrap().is_empty());
    }

    #[test]
    fn import_refuses_mixed_architectures_inside_one_set() {
        let f = fixture();
        let a = write_src(
            &f.root,
            "mix-00001-of-00002.gguf",
            &gguf_shard("llama", 0, 2),
        );
        write_src(
            &f.root,
            "mix-00002-of-00002.gguf",
            &gguf_shard("qwen2", 1, 2),
        );
        assert!(matches!(
            f.lib.import_copy(&a),
            Err(RaptorError::Integrity(m)) if m.contains("architectures")
        ));
    }

    #[test]
    fn a_lone_shard_with_a_plain_name_is_refused() {
        let f = fixture();
        let src = write_src(&f.root, "plain.gguf", &gguf_shard("llama", 0, 2));
        assert!(matches!(
            f.lib.import_copy(&src),
            Err(RaptorError::Unsupported(_))
        ));
    }

    #[test]
    fn deleting_a_split_entry_removes_every_shard() {
        let f = fixture();
        let shards = write_set(&f.root, "gone", 2);
        let entry = f.lib.import_copy(&shards[0]).unwrap();
        let folder = f.root.join("models").join("local");
        f.lib.delete(&entry.id).unwrap();
        assert!(!folder.join("gone-00001-of-00002.gguf").exists());
        assert!(!folder.join("gone-00002-of-00002.gguf").exists());
        assert!(f.lib.list().unwrap().is_empty());
    }

    #[test]
    fn one_missing_shard_makes_the_whole_set_not_installed() {
        let f = fixture();
        let shards = write_set(&f.root, "half", 2);
        let entry = f.lib.import_copy(&shards[0]).unwrap();
        fs::remove_file(
            f.root
                .join("models")
                .join("local")
                .join("half-00002-of-00002.gguf"),
        )
        .unwrap();
        assert!(f.lib.list().unwrap().is_empty());
        assert!(matches!(
            f.lib.get(&entry.id),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn hub_provenance_is_recorded_with_the_entry() {
        let f = fixture();
        let dest = f.lib.managed_path("owner/repo", "Q4/model.gguf").unwrap();
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let bytes = gguf_file("llama", 15);
        fs::write(&dest, &bytes).unwrap();
        let source = ModelSource::HuggingFace {
            repo: "owner/repo".into(),
            file_name: "Q4/model.gguf".into(),
        };
        let prov = Provenance {
            revision: Some("0123456789abcdef0123456789abcdef01234567".into()),
            hub_tags: vec!["gguf".into(), "license:mit".into()],
        };
        f.lib
            .register_completed_with(&dest, &sha256_hex(&bytes), &source, false, &prov)
            .unwrap();
        let details = f.lib.details("owner/repo/Q4/model.gguf").unwrap();
        assert_eq!(
            details.revision.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(details.hub_tags, vec!["gguf", "license:mit"]);
        assert_eq!(details.files.len(), 1);

        // The record survives a reopen of the index.
        let reopened =
            Library::open_at(f.root.join("models"), f.root.join("db").join(INDEX_FILE)).unwrap();
        assert_eq!(
            reopened
                .details("owner/repo/Q4/model.gguf")
                .unwrap()
                .revision,
            details.revision
        );
    }

    #[test]
    fn hub_shard_set_registers_as_one_entry_with_shard_records() {
        let f = fixture();
        let dir = f
            .root
            .join("models")
            .join("owner")
            .join("repo")
            .join("Q8_0");
        let paths = write_set(&dir, "m", 2);
        let staged: Vec<StagedShard> = paths
            .iter()
            .map(|p| StagedShard {
                path: p.clone(),
                sha256: sha256_hex(&fs::read(p).unwrap()),
            })
            .collect();
        let source = ModelSource::HuggingFace {
            repo: "owner/repo".into(),
            file_name: "Q8_0/m-00001-of-00002.gguf".into(),
        };
        let prov = Provenance {
            revision: Some("abcdefabcdefabcdefabcdefabcdefabcdefabcd".into()),
            hub_tags: vec![],
        };
        let entry = f
            .lib
            .register_shards(&staged, &source, false, &prov)
            .unwrap();
        assert_eq!(entry.id, "owner/repo/Q8_0/m-00001-of-00002.gguf");
        assert_eq!(
            entry.sha256,
            fingerprint(&staged.iter().map(|s| s.sha256.clone()).collect::<Vec<_>>())
        );
        assert_eq!(f.lib.list().unwrap().len(), 1);
        assert_eq!(
            f.lib.details(&entry.id).unwrap().revision.as_deref(),
            Some("abcdefabcdefabcdefabcdefabcdefabcdefabcd")
        );
        // Registering the same id again without replace is a conflict.
        assert!(matches!(
            f.lib.register_shards(&staged, &source, false, &prov),
            Err(RaptorError::Conflict(_))
        ));
    }

    #[test]
    fn scan_reports_each_split_set_once_and_flags_incomplete_ones() {
        let f = fixture();
        let scan = f.root.join("found");
        write_set(&scan, "ok", 2);
        let gap = write_set(&scan.join("sub"), "gap", 3);
        fs::remove_file(&gap[2]).unwrap();
        let items = f.lib.scan_folder(&scan).unwrap();
        assert_eq!(items.len(), 2, "one item per set, not per shard");
        let ok = items
            .iter()
            .find(|i| i.path.ends_with("ok-00001-of-00002.gguf"))
            .unwrap();
        assert_eq!(ok.status, ScanStatus::New);
        assert_eq!(
            ok.size_bytes,
            fs::metadata(scan.join("ok-00001-of-00002.gguf"))
                .unwrap()
                .len()
                + fs::metadata(scan.join("ok-00002-of-00002.gguf"))
                    .unwrap()
                    .len()
        );
        let gap_item = items
            .iter()
            .find(|i| i.path.ends_with("gap-00001-of-00003.gguf"))
            .unwrap();
        match &gap_item.status {
            ScanStatus::InvalidGguf { reason } => {
                assert!(reason.contains("missing shard 3"), "{reason}")
            }
            other => panic!("expected incomplete set, got {other:?}"),
        }
    }

    #[test]
    fn scanned_set_matching_an_installed_one_is_already_in_library() {
        let f = fixture();
        let shards = write_set(&f.root, "twin", 2);
        let entry = f.lib.import_copy(&shards[0]).unwrap();
        let other = f.root.join("other");
        write_set(&other, "twin", 2);
        let items = f.lib.scan_folder(&other).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0].status,
            ScanStatus::AlreadyInLibrary { id: entry.id }
        );
    }

    #[test]
    fn unknown_architecture_imports_and_is_flagged_in_details() {
        let f = fixture();
        let bytes = gguf_file("zzfake-arch", 15);
        let src = write_src(&f.root, "novel.gguf", &bytes);
        let entry = f.lib.import_copy(&src).unwrap();
        let details = f.lib.details(&entry.id).unwrap();
        assert_eq!(details.architecture.support, gguf::ArchSupport::Unknown);
        assert_eq!(
            details.architecture.note.as_deref(),
            Some("not in Raptor's tested list; llama-server will decide when you load it")
        );
    }

    #[test]
    fn known_unsupported_architecture_is_refused_on_import() {
        let f = fixture();
        let bytes = gguf_file("clip", 15);
        let src = write_src(&f.root, "mmproj.gguf", &bytes);
        match f.lib.import_copy(&src) {
            Err(RaptorError::Unsupported(m)) => assert!(m.contains("clip"), "{m}"),
            other => panic!("expected unsupported, got {other:?}"),
        }
        assert!(f.lib.list().unwrap().is_empty());
        assert!(!f.root.join("models/local/mmproj.gguf").exists());
    }

    #[test]
    fn incomplete_download_is_reported_and_never_listed() {
        let f = fixture();
        let scan = f.root.join("found");
        write_src(&scan, "Model-00001-of-00003.gguf.part", &[0u8; 64]);
        let items = f.lib.scan_folder(&scan).unwrap();
        assert_eq!(items.len(), 1);
        match &items[0].status {
            ScanStatus::InvalidGguf { reason } => {
                assert!(reason.contains("incomplete"), "{reason}")
            }
            other => panic!("expected incomplete report, got {other:?}"),
        }
        assert!(f.lib.list().unwrap().is_empty());
    }

    /// The user's own models, read by header only (nothing is hashed or copied). Skipped when
    /// the folder is absent. `RAPTOR_GGUF_SAMPLES` overrides the folder.
    #[test]
    fn local_models_pass_the_import_gate_and_the_partial_file_is_incomplete() {
        let root = match std::env::var_os("RAPTOR_GGUF_SAMPLES") {
            Some(v) => PathBuf::from(v),
            None => PathBuf::from(r"D:\AI_Storage\lmstudio\lmstudio-community"),
        };
        if !root.is_dir() {
            eprintln!("skipping: {} not found", root.display());
            return;
        }
        let f = fixture();
        let items = f.lib.scan_folder(&root).unwrap();
        let mut importable = Vec::new();
        let mut incomplete = Vec::new();
        for item in &items {
            match &item.status {
                ScanStatus::New => importable.push(item.path.clone()),
                ScanStatus::InvalidGguf { reason } if reason.contains("incomplete") => {
                    incomplete.push(item.path.clone())
                }
                other => panic!("{}: unexpected {other:?}", item.path.display()),
            }
        }
        // Every model the user owns must pass the import gate (header read only).
        for wanted in [
            "Ornith 1.5",
            "Qwen 3.8 27b",
            "granite-4.2-30b",
            "flint-GGUF",
        ] {
            let path = importable
                .iter()
                .find(|p| p.to_string_lossy().contains(wanted))
                .unwrap_or_else(|| panic!("{wanted} was not found as importable"));
            let meta = gguf::parse_file(path).unwrap();
            let arch = meta.architecture.as_deref().unwrap();
            gguf::refuse_known_unsupported(arch).unwrap();
            assert_ne!(
                gguf::classify_architecture(arch),
                gguf::ArchSupport::KnownUnsupported
            );
        }
        assert_eq!(importable.len(), 4, "{importable:?}");
        // The unfinished Flash-Next download is reported, never a model.
        assert!(
            incomplete
                .iter()
                .any(|p| p.to_string_lossy().ends_with(".gguf.part")),
            "{incomplete:?}"
        );
        assert!(f.lib.list().unwrap().is_empty());
    }
}
