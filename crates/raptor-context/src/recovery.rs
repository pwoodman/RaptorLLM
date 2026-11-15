//! Recovery handles: stable references to a chunk that an agent can redeem later
//! to fetch exactly that code again, without rebuilding the whole context.
//!
//! A handle is `chunk:<path>:<start>-<end>:<hash16>`. It stays valid while the file's
//! content hash keeps that prefix. Once the file changes, redeeming it fails with a
//! `Conflict` error whose message starts with `stale handle`.

use raptor_core::{RaptorError, Result};

/// Prefix of the message returned when a handle no longer matches the file.
pub const STALE_PREFIX: &str = "stale handle";

const HASH_PREFIX_LEN: usize = 16;

/// A decoded handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedHandle {
    pub path: String,
    pub start: u32,
    pub end: u32,
    pub hash_prefix: String,
}

/// Builds the handle for a chunk of `path` at `hash`.
pub fn make_handle(path: &str, start: u32, end: u32, hash: &str) -> String {
    let prefix = &hash[..hash.len().min(HASH_PREFIX_LEN)];
    format!("chunk:{path}:{start}-{end}:{prefix}")
}

/// True for a workspace-relative path that cannot escape the workspace.
pub fn is_safe_relative(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return false;
    }
    let first = path.split('/').next().unwrap_or("");
    if first.contains(':') {
        return false;
    }
    !path
        .split('/')
        .any(|seg| seg.is_empty() || seg == "." || seg == "..")
}

/// Decodes a handle. Rejects anything that is not a well-formed, safe chunk handle.
pub fn parse_handle(handle: &str) -> Result<ParsedHandle> {
    let bad = || RaptorError::InvalidInput(format!("malformed recovery handle: {handle}"));
    let rest = handle.strip_prefix("chunk:").ok_or_else(bad)?;
    let mut parts = rest.rsplitn(3, ':');
    let hash_prefix = parts.next().ok_or_else(bad)?;
    let range = parts.next().ok_or_else(bad)?;
    let path = parts.next().ok_or_else(bad)?;
    let (s, e) = range.split_once('-').ok_or_else(bad)?;
    let start: u32 = s.parse().map_err(|_| bad())?;
    let end: u32 = e.parse().map_err(|_| bad())?;
    if start == 0 || end < start || hash_prefix.is_empty() || !is_safe_relative(path) {
        return Err(bad());
    }
    Ok(ParsedHandle {
        path: path.to_string(),
        start,
        end,
        hash_prefix: hash_prefix.to_string(),
    })
}

/// Stale-handle error for `path`.
pub fn stale_error(path: &str) -> RaptorError {
    RaptorError::Conflict(format!(
        "{STALE_PREFIX}: {path} changed since the handle was issued; search again"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trips() {
        let h = make_handle("src/ledger/refund.rs", 10, 40, "abcdef0123456789ffff");
        assert_eq!(h, "chunk:src/ledger/refund.rs:10-40:abcdef0123456789");
        let p = parse_handle(&h).expect("parse");
        assert_eq!(p.path, "src/ledger/refund.rs");
        assert_eq!((p.start, p.end), (10, 40));
        assert_eq!(p.hash_prefix, "abcdef0123456789");
    }

    #[test]
    fn rejects_malformed_handles() {
        for bad in [
            "",
            "chunk:",
            "file:a.rs:1-2:ab",
            "chunk:a.rs:x-2:ab",
            "chunk:a.rs:5-2:ab",
            "chunk:a.rs:0-2:ab",
            "chunk:a.rs:1-2:",
        ] {
            assert!(parse_handle(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_traversal_in_handles() {
        assert!(parse_handle("chunk:../secret.rs:1-2:ab").is_err());
        assert!(parse_handle("chunk:/etc/passwd:1-2:ab").is_err());
        assert!(parse_handle("chunk:C:/x.rs:1-2:ab").is_err());
    }

    #[test]
    fn safe_relative_paths() {
        assert!(is_safe_relative("src/a.rs"));
        assert!(!is_safe_relative("a/../b"));
        assert!(!is_safe_relative("a\\b"));
        assert!(!is_safe_relative("/abs"));
        assert!(!is_safe_relative("a//b"));
    }

    #[test]
    fn stale_error_is_a_conflict_with_marker() {
        match stale_error("a.rs") {
            RaptorError::Conflict(msg) => assert!(msg.starts_with(STALE_PREFIX)),
            other => panic!("unexpected {other:?}"),
        }
    }
}
