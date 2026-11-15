//! Small shared helpers: poison-aware locking, atomic writes, hashing, idempotent removal.

use raptor_core::{RaptorError, Result};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Bytes hashed so far and expected in total by the import running on this library. Imports hash
/// whole multi-GB files, so the shell polls this to show progress instead of a frozen button.
#[derive(Debug, Default)]
pub struct HashProgress {
    pub done: AtomicU64,
    pub total: AtomicU64,
}

thread_local! {
    static SINK: RefCell<Option<Arc<HashProgress>>> = const { RefCell::new(None) };
}

/// While alive, every `sha256_file` on this thread reports into the given progress.
pub(crate) struct ProgressScope;

impl ProgressScope {
    pub(crate) fn start(progress: &Arc<HashProgress>) -> Self {
        progress.done.store(0, Ordering::Relaxed);
        progress.total.store(0, Ordering::Relaxed);
        SINK.with(|s| *s.borrow_mut() = Some(progress.clone()));
        ProgressScope
    }
}

impl Drop for ProgressScope {
    fn drop(&mut self) {
        SINK.with(|s| *s.borrow_mut() = None);
    }
}

/// Locks a mutex, mapping poisoning to an internal error instead of panicking.
pub(crate) fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| RaptorError::Internal("model state lock poisoned".into()))
}

/// Writes `bytes` to `path` through a sibling temp file and an atomic rename,
/// so readers never observe a half-written index or record.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Removes a file, treating "already gone" as success (idempotent cleanup).
pub(crate) fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Streaming SHA-256 of a file, lowercase hex. Blocking: call from a blocking context.
pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let sink = SINK.with(|s| s.borrow().clone());
    if let Some(p) = &sink {
        p.total.fetch_add(file.metadata()?.len(), Ordering::Relaxed);
    }
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        if let Some(p) = &sink {
            p.done.fetch_add(n as u64, Ordering::Relaxed);
        }
    }
    Ok(hex::encode(hasher.finalize()))
}

/// True for a 64-character hex SHA-256 digest (either case).
pub(crate) fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Identity of an ordered list of files (a shard set): SHA-256 over the newline-joined
/// lowercase digests. A single file's identity is simply its own digest.
pub(crate) fn fingerprint(digests: &[String]) -> String {
    let mut hasher = Sha256::new();
    for d in digests {
        hasher.update(d.to_ascii_lowercase().as_bytes());
        hasher.update(b"\n");
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_reports_progress_only_inside_a_scope() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.bin");
        fs::write(&p, vec![7u8; 3_000_000]).unwrap();
        let progress = Arc::new(HashProgress::default());
        sha256_file(&p).unwrap();
        assert_eq!(progress.total.load(Ordering::Relaxed), 0);
        {
            let _scope = ProgressScope::start(&progress);
            sha256_file(&p).unwrap();
        }
        assert_eq!(progress.total.load(Ordering::Relaxed), 3_000_000);
        assert_eq!(progress.done.load(Ordering::Relaxed), 3_000_000);
        sha256_file(&p).unwrap();
        assert_eq!(progress.done.load(Ordering::Relaxed), 3_000_000);
    }

    #[test]
    fn sha256_known_vector() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("abc.bin");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn write_atomic_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("x.json");
        write_atomic(&p, b"{}").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"{}");
        assert!(!dir.path().join("sub").join("x.json.tmp").exists());
    }

    #[test]
    fn remove_if_exists_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("gone");
        remove_if_exists(&p).unwrap();
        fs::write(&p, b"1").unwrap();
        remove_if_exists(&p).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn hex_check() {
        assert!(is_sha256_hex(&"A".repeat(64)));
        assert!(!is_sha256_hex(&"g".repeat(64)));
        assert!(!is_sha256_hex("abc"));
    }
}
