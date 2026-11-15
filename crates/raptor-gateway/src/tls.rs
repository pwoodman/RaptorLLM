//! TLS for the listener.
//!
//! Two sources:
//! * [`TlsConfig::Files`]: a PEM certificate chain and PEM private key the user
//!   provides. Nothing is written.
//! * [`TlsConfig::SelfSigned`]: a certificate generated with `rcgen` and kept in
//!   a caller-supplied directory, so it survives restarts and clients can pin
//!   its SHA-256 fingerprint. The private key file is created owner-only on
//!   Unix. On Windows it inherits the directory's ACL, so keep the directory
//!   under the user's profile.
//!
//! The crypto provider is `ring`. It is pure Rust with no CMake or NASM step,
//! and it is the provider the rest of the workspace already links.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum_server::tls_rustls::RustlsConfig;
use raptor_core::{RaptorError, Result};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;
use sha2::{Digest, Sha256};

/// File name of the certificate inside a self-signed directory.
pub const CERT_FILE: &str = "gateway-cert.pem";
/// File name of the private key inside a self-signed directory.
pub const KEY_FILE: &str = "gateway-key.pem";

/// Where the listener's TLS certificate comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlsConfig {
    /// A user-provided PEM certificate chain (leaf first) and PEM private key.
    Files { cert_pem: PathBuf, key_pem: PathBuf },
    /// A self-signed certificate created under `dir` when none is there yet.
    /// `subject_alt_names` are added to `localhost`, `127.0.0.1` and `::1`;
    /// each entry is a DNS name or an IP address.
    SelfSigned {
        dir: PathBuf,
        subject_alt_names: Vec<String>,
    },
}

/// The certificate a listener presents, with what clients need to pin it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsIdentity {
    /// The leaf certificate, DER encoded.
    pub cert_der: Vec<u8>,
    /// Lowercase hex SHA-256 of `cert_der`. This is the value clients pin.
    pub fingerprint_sha256: String,
}

/// A certificate ready to serve, plus its identity.
pub(crate) struct Loaded {
    pub(crate) server_config: Arc<ServerConfig>,
    pub(crate) identity: TlsIdentity,
}

/// SHA-256 of a DER certificate as lowercase hex.
pub fn fingerprint_of_der(der: &[u8]) -> String {
    hex::encode(Sha256::digest(der))
}

/// Formats a hex fingerprint as `AB:CD:...` for display next to a pinning hint.
pub fn display_fingerprint(hex_fp: &str) -> String {
    hex_fp
        .as_bytes()
        .chunks(2)
        .map(|pair| String::from_utf8_lossy(pair).to_ascii_uppercase())
        .collect::<Vec<_>>()
        .join(":")
}

/// Loads the self-signed certificate from `dir`, or creates one if none is
/// there or the stored files do not parse. Existing files are reused as they are.
pub fn ensure_self_signed(dir: &Path, subject_alt_names: &[String]) -> Result<TlsIdentity> {
    Ok(load_or_generate(dir, subject_alt_names, false)?.identity)
}

/// Replaces the self-signed certificate and key in `dir` with a new pair.
/// Clients that pinned the old fingerprint must re-pin.
pub fn regenerate_self_signed(dir: &Path, subject_alt_names: &[String]) -> Result<TlsIdentity> {
    Ok(load_or_generate(dir, subject_alt_names, true)?.identity)
}

/// Loads (or generates) the material for a configuration.
pub(crate) fn load(config: &TlsConfig) -> Result<Loaded> {
    match config {
        TlsConfig::Files { cert_pem, key_pem } => load_files(cert_pem, key_pem),
        TlsConfig::SelfSigned {
            dir,
            subject_alt_names,
        } => load_or_generate(dir, subject_alt_names, false),
    }
}

/// Rotation: regenerates a self-signed pair, or re-reads user files.
pub(crate) fn rotate(config: &TlsConfig) -> Result<Loaded> {
    match config {
        TlsConfig::Files { cert_pem, key_pem } => load_files(cert_pem, key_pem),
        TlsConfig::SelfSigned {
            dir,
            subject_alt_names,
        } => load_or_generate(dir, subject_alt_names, true),
    }
}

/// Wraps a server config for `axum-server`, which can swap it without a restart.
pub(crate) fn rustls_config(loaded: &Loaded) -> RustlsConfig {
    RustlsConfig::from_config(Arc::clone(&loaded.server_config))
}

fn load_files(cert_pem: &Path, key_pem: &Path) -> Result<Loaded> {
    let chain = CertificateDer::pem_file_iter(cert_pem)
        .map_err(|e| invalid_pem("certificate", cert_pem, e))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| invalid_pem("certificate", cert_pem, e))?;
    let key = PrivateKeyDer::from_pem_file(key_pem)
        .map_err(|e| invalid_pem("private key", key_pem, e))?;
    warn_if_key_readable_by_others(key_pem);
    build(chain, key)
}

fn invalid_pem(what: &str, path: &Path, e: impl std::fmt::Display) -> RaptorError {
    RaptorError::InvalidInput(format!("cannot read TLS {what} {}: {e}", path.display()))
}

fn load_or_generate(dir: &Path, sans: &[String], force_new: bool) -> Result<Loaded> {
    let cert_path = dir.join(CERT_FILE);
    let key_path = dir.join(KEY_FILE);
    if !force_new && cert_path.exists() && key_path.exists() {
        match stored_pair(&cert_path, &key_path) {
            Ok(loaded) => return Ok(loaded),
            Err(e) => tracing::warn!("stored TLS certificate is unusable, regenerating: {e}"),
        }
    }
    std::fs::create_dir_all(dir)
        .map_err(|e| RaptorError::Io(format!("create TLS directory {}: {e}", dir.display())))?;
    let (cert_der, cert_pem, key_pem) = generate(sans)?;
    write_file(&cert_path, cert_pem.as_bytes(), false)?;
    write_file(&key_path, key_pem.as_bytes(), true)?;
    build(
        vec![CertificateDer::from(cert_der)],
        PrivateKeyDer::from_pem_slice(key_pem.as_bytes())
            .map_err(|e| RaptorError::Internal(format!("generated key unreadable: {e}")))?,
    )
}

fn stored_pair(cert_path: &Path, key_path: &Path) -> Result<Loaded> {
    let chain = CertificateDer::pem_file_iter(cert_path)
        .map_err(|e| invalid_pem("certificate", cert_path, e))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| invalid_pem("certificate", cert_path, e))?;
    let key = PrivateKeyDer::from_pem_file(key_path)
        .map_err(|e| invalid_pem("private key", key_path, e))?;
    tighten_key_permissions(key_path)?;
    build(chain, key)
}

/// Generates a P-256 self-signed certificate. Returns the DER certificate and
/// both PEM blobs.
fn generate(sans: &[String]) -> Result<(Vec<u8>, String, String)> {
    let mut names: Vec<String> = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    for n in sans {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    let gen_err = |e: rcgen::Error| RaptorError::Internal(format!("certificate generation: {e}"));
    let key = rcgen::KeyPair::generate().map_err(gen_err)?;
    let mut params = rcgen::CertificateParams::new(names).map_err(gen_err)?;
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "Raptor LLM gateway");
    let cert = params.self_signed(&key).map_err(gen_err)?;
    Ok((cert.der().to_vec(), cert.pem(), key.serialize_pem()))
}

fn build(chain: Vec<CertificateDer<'static>>, key: PrivateKeyDer<'static>) -> Result<Loaded> {
    let leaf = chain.first().ok_or_else(|| {
        RaptorError::InvalidInput("TLS certificate file holds no certificate".into())
    })?;
    let identity = TlsIdentity {
        cert_der: leaf.as_ref().to_vec(),
        fingerprint_sha256: fingerprint_of_der(leaf.as_ref()),
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])
        .map_err(|e| RaptorError::InvalidInput(format!("TLS protocol setup: {e}")))?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|e| {
            RaptorError::InvalidInput(format!("TLS certificate and key do not match: {e}"))
        })?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(Loaded {
        server_config: Arc::new(config),
        identity,
    })
}

/// Writes a file through a temporary sibling and a rename, so a crash never
/// leaves a half-written key. Private files are created owner-only on Unix.
fn write_file(path: &Path, data: &[u8], private: bool) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let _ = std::fs::remove_file(&tmp);
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(if private { 0o600 } else { 0o644 });
    }
    let _ = private;
    let io_err = |e: std::io::Error| RaptorError::Io(format!("write {}: {e}", path.display()));
    let mut file = opts.open(&tmp).map_err(io_err)?;
    file.write_all(data).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(io_err)
}

/// Owner-only on Unix: any group or other bit on our own key file is removed,
/// with a warning.
#[cfg(unix)]
fn tighten_key_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path)
        .map_err(|e| RaptorError::Io(format!("stat {}: {e}", path.display())))?;
    if meta.permissions().mode() & 0o077 != 0 {
        tracing::warn!(
            "TLS private key {} was readable by others; restricting it to its owner",
            path.display()
        );
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| RaptorError::Io(format!("chmod {}: {e}", path.display())))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn tighten_key_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn warn_if_key_readable_by_others(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.permissions().mode() & 0o077 != 0 {
            tracing::warn!(
                "user-provided TLS private key {} is readable by other users",
                path.display()
            );
        }
    }
}

#[cfg(not(unix))]
fn warn_if_key_readable_by_others(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_sha256_hex_and_displays_in_pairs() {
        let fp = fingerprint_of_der(b"abc");
        assert_eq!(
            fp,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(display_fingerprint("0a1b"), "0A:1B");
    }

    #[test]
    fn generated_certificate_is_reused_until_regenerated() {
        let dir = tempfile::tempdir().unwrap();
        let first = ensure_self_signed(dir.path(), &[]).unwrap();
        let again = ensure_self_signed(dir.path(), &[]).unwrap();
        assert_eq!(first, again, "restart must keep the same identity");
        assert_eq!(first.fingerprint_sha256.len(), 64);
        assert_eq!(
            first.fingerprint_sha256,
            fingerprint_of_der(&first.cert_der)
        );

        let rotated = regenerate_self_signed(dir.path(), &[]).unwrap();
        assert_ne!(first.fingerprint_sha256, rotated.fingerprint_sha256);
        let after = ensure_self_signed(dir.path(), &[]).unwrap();
        assert_eq!(rotated, after, "rotation persists to the directory");
    }

    #[test]
    fn corrupt_stored_key_is_replaced_not_served() {
        let dir = tempfile::tempdir().unwrap();
        let first = ensure_self_signed(dir.path(), &[]).unwrap();
        std::fs::write(dir.path().join(KEY_FILE), "not a key").unwrap();
        let healed = ensure_self_signed(dir.path(), &[]).unwrap();
        assert_ne!(first.fingerprint_sha256, healed.fingerprint_sha256);
    }

    #[test]
    fn extra_names_are_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let names = vec!["box.lan".to_string(), "192.168.1.20".to_string()];
        let id = ensure_self_signed(dir.path(), &names).unwrap();
        assert!(!id.cert_der.is_empty());
        let stored = std::fs::read_to_string(dir.path().join(CERT_FILE)).unwrap();
        assert!(stored.contains("BEGIN CERTIFICATE"));
    }

    #[test]
    fn user_files_are_loaded_and_bad_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = generate(&[]).unwrap();
        let cert = dir.path().join("c.pem");
        let key = dir.path().join("k.pem");
        std::fs::write(&cert, &loaded.1).unwrap();
        std::fs::write(&key, &loaded.2).unwrap();
        let ok = load(&TlsConfig::Files {
            cert_pem: cert.clone(),
            key_pem: key.clone(),
        })
        .unwrap();
        assert_eq!(ok.identity.cert_der, loaded.0);

        let missing = load(&TlsConfig::Files {
            cert_pem: dir.path().join("nope.pem"),
            key_pem: key.clone(),
        });
        assert!(matches!(missing, Err(RaptorError::InvalidInput(_))));

        // A key that belongs to another certificate is refused.
        let other = generate(&[]).unwrap();
        std::fs::write(&key, &other.2).unwrap();
        let mismatched = load(&TlsConfig::Files {
            cert_pem: cert,
            key_pem: key,
        });
        assert!(matches!(mismatched, Err(RaptorError::InvalidInput(_))));
    }

    #[cfg(unix)]
    #[test]
    fn private_key_is_owner_only_and_loose_permissions_are_tightened() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        ensure_self_signed(dir.path(), &[]).unwrap();
        let key = dir.path().join(KEY_FILE);
        let mode = std::fs::metadata(&key).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);

        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
        ensure_self_signed(dir.path(), &[]).unwrap();
        let mode = std::fs::metadata(&key).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode & 0o077, 0, "group/other bits must be cleared");
    }
}
