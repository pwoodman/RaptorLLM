//! Repository walk: `.gitignore` / `.raptorignore` filtering, exclusion reasons,
//! content hashing. Files are streamed one at a time through a visitor, so the
//! whole repository is never held in memory as strings.

use crate::parse::Language;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

/// Directory names pruned from the walk (dependencies, build output, VCS metadata).
const VENDOR_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".gradle",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
];

const LOCKFILES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "go.sum",
    "composer.lock",
    "bun.lockb",
    "uv.lock",
];

const BINARY_EXTENSIONS: &[&str] = &[
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "ico",
    "pdf",
    "zip",
    "gz",
    "tgz",
    "tar",
    "7z",
    "rar",
    "exe",
    "dll",
    "so",
    "dylib",
    "a",
    "o",
    "obj",
    "lib",
    "class",
    "jar",
    "wasm",
    "bin",
    "gguf",
    "onnx",
    "safetensors",
    "mp3",
    "mp4",
    "mov",
    "wav",
    "woff",
    "woff2",
    "ttf",
    "otf",
    "sqlite",
    "db",
    "pyc",
];

/// Why a path was not indexed. Every exclusion is reported, never silently skipped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum ExcludeReason {
    /// Credentials, keys or `.env` files. Never read into the index.
    Secret,
    /// Dependency lock file: pins, not source.
    Lockfile,
    /// Inside a dependency, build-output or VCS directory.
    VendorDir,
    /// Carries a generated-code marker in its header.
    Generated,
    /// Binary content or a binary extension.
    Binary,
    /// Larger than the configured limit.
    TooLarge { bytes: u64 },
    /// The file or directory could not be read.
    Unreadable { error: String },
}

impl ExcludeReason {
    /// Human-readable explanation for manifests and logs.
    pub fn describe(&self) -> String {
        match self {
            ExcludeReason::Secret => "secret or credential file (never indexed)".into(),
            ExcludeReason::Lockfile => "lock file (dependency pins, not source)".into(),
            ExcludeReason::VendorDir => "dependency, build-output or VCS directory".into(),
            ExcludeReason::Generated => "generated file (marker found in header)".into(),
            ExcludeReason::Binary => "binary content".into(),
            ExcludeReason::TooLarge { bytes } => {
                format!("{bytes} bytes, above the indexing size limit")
            }
            ExcludeReason::Unreadable { error } => format!("could not be read: {error}"),
        }
    }
}

/// A path that was considered and not indexed, with the reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exclusion {
    pub path: String,
    pub reason: ExcludeReason,
}

/// Metadata for one indexed file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Workspace-relative path with `/` separators.
    pub path: String,
    /// Lowercase hex SHA-256 of the file content.
    pub hash: String,
    pub size: u64,
    /// Modification time in unix seconds, 0 when unavailable. Used only for relative ranking.
    pub mtime: u64,
    pub language: Language,
    /// Unix seconds taken just before the content was read (0 when unknown). The stat fast path
    /// trusts `size` and `mtime` only when this is at least two seconds after `mtime`.
    #[serde(default)]
    pub hashed_at: u64,
}

/// Scan limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanConfig {
    /// Files above this size are excluded with [`ExcludeReason::TooLarge`].
    pub max_file_bytes: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            max_file_bytes: 512 * 1024,
        }
    }
}

/// A scanned file: its record plus the text needed for parsing. Dropped after the visit.
pub struct ScannedFile {
    pub record: FileRecord,
    pub text: String,
}

/// Pure path classification: secrets, lock files, vendor dirs and binary extensions.
pub fn classify_path(rel: &str) -> Option<ExcludeReason> {
    let segments: Vec<&str> = rel.split('/').collect();
    if segments.iter().any(|s| VENDOR_DIRS.contains(s)) {
        return Some(ExcludeReason::VendorDir);
    }
    let name = *segments.last().unwrap_or(&rel);
    let lower = name.to_ascii_lowercase();
    if LOCKFILES.contains(&name) {
        return Some(ExcludeReason::Lockfile);
    }
    if is_secret_name(&lower) {
        return Some(ExcludeReason::Secret);
    }
    let ext = lower.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if BINARY_EXTENSIONS.contains(&ext) {
        return Some(ExcludeReason::Binary);
    }
    None
}

fn is_secret_name(lower: &str) -> bool {
    if lower == ".env" || (lower.starts_with(".env.") && !is_template_name(lower)) {
        return true;
    }
    if lower.ends_with(".pem")
        || lower.ends_with(".key")
        || lower.ends_with(".p12")
        || lower.ends_with(".pfx")
        || lower.ends_with(".keystore")
        || lower.ends_with(".jks")
        || lower.ends_with(".ppk")
    {
        return true;
    }
    if matches!(
        lower,
        "id_rsa" | "id_dsa" | "id_ecdsa" | "id_ed25519" | ".npmrc" | ".netrc" | ".pypirc"
    ) {
        return true;
    }
    lower.contains("secret")
        && (lower.ends_with(".json")
            || lower.ends_with(".yaml")
            || lower.ends_with(".yml")
            || lower.ends_with(".toml"))
}

fn is_template_name(lower: &str) -> bool {
    lower.ends_with(".example") || lower.ends_with(".sample") || lower.ends_with(".template")
}

/// True when the bytes look binary (a NUL in the first 8000 bytes).
pub fn is_binary_content(bytes: &[u8]) -> bool {
    bytes.iter().take(8000).any(|b| *b == 0)
}

/// True when the first kilobyte carries a generated-code marker.
pub fn is_generated_header(text: &str) -> bool {
    let head: String = text
        .chars()
        .take(1024)
        .collect::<String>()
        .to_ascii_lowercase();
    [
        "@generated",
        "do not edit",
        "code generated",
        "auto-generated",
        "autogenerated",
    ]
    .iter()
    .any(|m| head.contains(m))
}

/// Lowercase hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn to_rel(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    Some(parts.join("/"))
}

/// One visit of [`scan_each_with`].
pub enum ScanEvent {
    /// The file was read and hashed.
    Read(ScannedFile),
    /// Size and mtime match what the caller already knows, so the file was not read.
    Unchanged { path: String },
}

/// Seconds since the unix epoch, 0 before it.
pub fn unix_secs(t: std::time::SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Walks `root`, calling `visit` for each indexable file, and returns every exclusion.
///
/// Honours `.gitignore` (even outside a git checkout) and `.raptorignore`. Gitignored
/// paths are pruned by the walker and therefore not listed individually; vendor
/// directories pruned by Raptor LLM are listed once each.
pub fn scan_each<F>(root: &Path, cfg: &ScanConfig, mut visit: F) -> Result<Vec<Exclusion>>
where
    F: FnMut(ScannedFile) -> Result<()>,
{
    scan_each_with(
        root,
        cfg,
        |_| None,
        |event| match event {
            ScanEvent::Read(file) => visit(file),
            ScanEvent::Unchanged { .. } => Ok(()),
        },
    )
}

/// Like [`scan_each`], with a stat fast path. `known` maps a workspace-relative path to the
/// `(size, mtime)` the caller last recorded. When the file's size and mtime still match, the
/// file is reported as [`ScanEvent::Unchanged`] without being read or hashed. Exclusion checks
/// that need content (binary, generated) are skipped for such files: the caller only passes
/// paths it already indexed, which passed those checks when they were read.
pub fn scan_each_with<K, F>(
    root: &Path,
    cfg: &ScanConfig,
    known: K,
    mut visit: F,
) -> Result<Vec<Exclusion>>
where
    K: Fn(&str) -> Option<(u64, u64)>,
    F: FnMut(ScanEvent) -> Result<()>,
{
    let pruned: Arc<Mutex<Vec<Exclusion>>> = Arc::new(Mutex::new(Vec::new()));
    let pruned_in_walk = Arc::clone(&pruned);
    let root_owned = root.to_path_buf();

    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .hidden(false)
        .require_git(false)
        .add_custom_ignore_filename(".raptorignore")
        .sort_by_file_name(|a, b| a.cmp(b))
        .filter_entry(move |entry| {
            if entry.depth() == 0 || !entry.file_type().is_some_and(|t| t.is_dir()) {
                return true;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == ".git" {
                return false;
            }
            if VENDOR_DIRS.contains(&name.as_str()) {
                if let (Some(rel), Ok(mut list)) =
                    (to_rel(&root_owned, entry.path()), pruned_in_walk.lock())
                {
                    list.push(Exclusion {
                        path: rel,
                        reason: ExcludeReason::VendorDir,
                    });
                }
                return false;
            }
            true
        });

    let mut excluded = Vec::new();
    for result in builder.build() {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                excluded.push(Exclusion {
                    path: "(walk)".into(),
                    reason: ExcludeReason::Unreadable {
                        error: e.to_string(),
                    },
                });
                continue;
            }
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Some(rel) = to_rel(root, entry.path()) else {
            continue;
        };
        if let Some(reason) = classify_path(&rel) {
            excluded.push(Exclusion { path: rel, reason });
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                excluded.push(Exclusion {
                    path: rel,
                    reason: ExcludeReason::Unreadable {
                        error: e.to_string(),
                    },
                });
                continue;
            }
        };
        if meta.len() > cfg.max_file_bytes {
            excluded.push(Exclusion {
                path: rel,
                reason: ExcludeReason::TooLarge { bytes: meta.len() },
            });
            continue;
        }
        let mtime = meta.modified().map(unix_secs).unwrap_or(0);
        if let Some((size, known_mtime)) = known(&rel) {
            if size == meta.len() && known_mtime == mtime {
                visit(ScanEvent::Unchanged { path: rel })?;
                continue;
            }
        }
        // Taken before the read: a write that lands during the read has a later mtime,
        // so the stat fast path does not trust this hash (see `FileRecord::hashed_at`).
        let hashed_at = unix_secs(std::time::SystemTime::now());
        let bytes = match std::fs::read(entry.path()) {
            Ok(b) => b,
            Err(e) => {
                excluded.push(Exclusion {
                    path: rel,
                    reason: ExcludeReason::Unreadable {
                        error: e.to_string(),
                    },
                });
                continue;
            }
        };
        if is_binary_content(&bytes) {
            excluded.push(Exclusion {
                path: rel,
                reason: ExcludeReason::Binary,
            });
            continue;
        }
        let hash = sha256_hex(&bytes);
        let Ok(text) = String::from_utf8(bytes) else {
            excluded.push(Exclusion {
                path: rel,
                reason: ExcludeReason::Binary,
            });
            continue;
        };
        if is_generated_header(&text) {
            excluded.push(Exclusion {
                path: rel,
                reason: ExcludeReason::Generated,
            });
            continue;
        }
        let record = FileRecord {
            language: Language::from_path(&rel),
            path: rel,
            hash,
            size: meta.len(),
            mtime,
            hashed_at,
        };
        visit(ScanEvent::Read(ScannedFile { record, text }))?;
    }

    let mut out = match pruned.lock() {
        Ok(list) => list.clone(),
        Err(_) => return Err(RaptorError::Internal("scan exclusion list poisoned".into())),
    };
    out.append(&mut excluded);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_secrets_and_keeps_templates() {
        assert_eq!(classify_path(".env"), Some(ExcludeReason::Secret));
        assert_eq!(
            classify_path("config/.env.production"),
            Some(ExcludeReason::Secret)
        );
        assert_eq!(classify_path(".env.example"), None);
        assert_eq!(
            classify_path("keys/server.pem"),
            Some(ExcludeReason::Secret)
        );
        assert_eq!(classify_path("id_ed25519"), Some(ExcludeReason::Secret));
        assert_eq!(
            classify_path("config/secrets.yaml"),
            Some(ExcludeReason::Secret)
        );
    }

    #[test]
    fn classifies_lockfiles_vendor_and_binary() {
        assert_eq!(classify_path("Cargo.lock"), Some(ExcludeReason::Lockfile));
        assert_eq!(
            classify_path("web/package-lock.json"),
            Some(ExcludeReason::Lockfile)
        );
        assert_eq!(
            classify_path("web/node_modules/x/index.js"),
            Some(ExcludeReason::VendorDir)
        );
        assert_eq!(
            classify_path("target/debug/app.exe"),
            Some(ExcludeReason::VendorDir)
        );
        assert_eq!(
            classify_path("assets/logo.PNG"),
            Some(ExcludeReason::Binary)
        );
        assert_eq!(classify_path("src/main.rs"), None);
    }

    #[test]
    fn detects_binary_and_generated_content() {
        assert!(is_binary_content(b"abc\0def"));
        assert!(!is_binary_content("plain text".as_bytes()));
        assert!(is_generated_header(
            "// Code generated by protoc. DO NOT EDIT.\nfn x() {}"
        ));
        assert!(!is_generated_header("fn ordinary() {}"));
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn scan_reports_reasons_and_respects_ignore_files() {
        let t = tempfile::tempdir().expect("tempdir");
        let root = t.path();
        std::fs::create_dir_all(root.join("src")).expect("mkdir");
        std::fs::create_dir_all(root.join("node_modules/pkg")).expect("mkdir");
        std::fs::write(root.join("src/lib.rs"), "pub fn a() {}\n").expect("write");
        std::fs::write(root.join(".env"), "TOKEN=1\n").expect("write");
        std::fs::write(root.join("node_modules/pkg/i.js"), "x\n").expect("write");
        std::fs::write(root.join(".gitignore"), "ignored.txt\n").expect("write");
        std::fs::write(root.join("ignored.txt"), "hidden\n").expect("write");
        std::fs::write(root.join(".raptorignore"), "notes/\n").expect("write");
        std::fs::create_dir_all(root.join("notes")).expect("mkdir");
        std::fs::write(root.join("notes/n.md"), "n\n").expect("write");

        let mut seen = Vec::new();
        let excl = scan_each(root, &ScanConfig::default(), |f| {
            seen.push(f.record.path);
            Ok(())
        })
        .expect("scan");
        assert!(seen.contains(&"src/lib.rs".to_string()));
        assert!(!seen.contains(&"ignored.txt".to_string()));
        assert!(!seen.iter().any(|p| p.starts_with("notes/")));
        assert!(excl
            .iter()
            .any(|e| e.path == ".env" && e.reason == ExcludeReason::Secret));
        assert!(excl
            .iter()
            .any(|e| e.path == "node_modules" && e.reason == ExcludeReason::VendorDir));
    }

    #[test]
    fn scan_enforces_size_limit() {
        let t = tempfile::tempdir().expect("tempdir");
        std::fs::write(t.path().join("big.rs"), "x".repeat(100)).expect("write");
        let cfg = ScanConfig { max_file_bytes: 10 };
        let excl = scan_each(t.path(), &cfg, |_| Ok(())).expect("scan");
        assert_eq!(excl.len(), 1);
        assert_eq!(excl[0].reason, ExcludeReason::TooLarge { bytes: 100 });
    }

    #[test]
    fn visitor_error_aborts_scan() {
        let t = tempfile::tempdir().expect("tempdir");
        std::fs::write(t.path().join("a.rs"), "fn a() {}\n").expect("write");
        let res = scan_each(t.path(), &ScanConfig::default(), |_| {
            Err(RaptorError::Internal("stop".into()))
        });
        assert!(res.is_err());
    }

    #[test]
    fn record_has_language_and_hash() {
        let t = tempfile::tempdir().expect("tempdir");
        std::fs::write(t.path().join("a.py"), "def f():\n    pass\n").expect("write");
        let mut recs = Vec::new();
        scan_each(t.path(), &ScanConfig::default(), |f| {
            recs.push(f.record);
            Ok(())
        })
        .expect("scan");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].language, Language::Python);
        assert_eq!(recs[0].hash.len(), 64);
    }
}
