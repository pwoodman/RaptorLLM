//! SHA-256 helpers for download verification and model fingerprints.

use raptor_core::Result;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Lowercase hex SHA-256 of an in-memory buffer.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Streams a file through SHA-256 without loading it into memory.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// True when `s` is exactly 64 hexadecimal characters.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Case-insensitive digest comparison.
pub fn digests_equal(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn file_hash_matches_buffer_hash() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.bin");
        let data = vec![7u8; 3 * (1 << 20) + 17];
        std::fs::write(&p, &data).unwrap();
        assert_eq!(sha256_file(&p).unwrap(), sha256_hex(&data));
    }

    #[test]
    fn validates_digest_shape() {
        assert!(is_sha256_hex(&"a".repeat(64)));
        assert!(!is_sha256_hex(&"a".repeat(63)));
        assert!(!is_sha256_hex(&format!("{}g", "a".repeat(63))));
        assert!(digests_equal("ABCD", "abcd"));
    }
}
