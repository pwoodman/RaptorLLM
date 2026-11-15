//! Small file helpers shared by the shell's JSON stores.

use raptor_core::{RaptorError, Result};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Writes `bytes` to `path` atomically: write a sibling temp file, fsync it, then
/// rename over the destination. A crash leaves either the old or the new file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| RaptorError::InvalidInput(format!("no parent: {}", path.display())))?;
    std::fs::create_dir_all(parent)?;
    let tmp = temp_sibling(path);
    let result = (|| -> Result<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Seconds since the Unix epoch. Falls back to 0 if the clock is before 1970.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Lowercase hex of `n` random bytes.
pub fn random_hex(n: usize) -> Result<String> {
    let mut buf = vec![0u8; n];
    getrandom::getrandom(&mut buf)
        .map_err(|e| RaptorError::Internal(format!("random source unavailable: {e}")))?;
    Ok(hex::encode(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_replaces_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.json");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two");
        assert!(!temp_sibling(&p).exists());
    }

    #[test]
    fn random_hex_has_expected_length() {
        let a = random_hex(16).unwrap();
        assert_eq!(a.len(), 32);
        assert_ne!(a, random_hex(16).unwrap());
    }
}
