//! Runtime installation: download -> verify SHA-256 -> safe extract into a
//! staging directory -> atomic rename into `runtimes/<id>/<version>`.
//! Previous versions are kept so `rollback` can restore them.

use crate::hash::{digests_equal, sha256_file};
use crate::manifest::{Backend, Manifest};
use async_trait::async_trait;
use futures_util::StreamExt;
use raptor_core::{DataDirs, RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

const MARKER_FILE: &str = ".raptor-runtime.json";
const STATE_FILE: &str = "state.json";
const MAX_LOCATE_DEPTH: usize = 3;

/// Streams a URL to a local file. Behind a trait so tests never touch the network.
#[async_trait]
pub trait Fetcher: Send + Sync {
    /// Writes the body of `url` to `dest` (created or truncated) and returns the
    /// byte count. Must return promptly with an error once `cancel` fires.
    async fn fetch(&self, url: &str, dest: &Path, cancel: &CancellationToken) -> Result<u64>;
}

/// reqwest-backed fetcher with streaming writes and cancellation.
pub struct ReqwestFetcher {
    client: reqwest::Client,
    /// Bytes written so far by the current fetch, for a progress display. Reset to 0 at the
    /// start of every fetch.
    progress: Option<Arc<AtomicU64>>,
}

impl ReqwestFetcher {
    /// Builds the HTTP client. Connect and read timeouts are set; there is no
    /// total timeout because runtime archives are large.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(5))
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| RaptorError::Network(format!("download client init failed: {e}")))?;
        Ok(Self {
            client,
            progress: None,
        })
    }

    /// Reports the bytes written so far into `progress` while a fetch runs.
    pub fn with_progress(mut self, progress: Arc<AtomicU64>) -> Self {
        self.progress = Some(progress);
        self
    }
}

fn cancelled() -> RaptorError {
    RaptorError::Runtime("operation cancelled".into())
}

#[async_trait]
impl Fetcher for ReqwestFetcher {
    async fn fetch(&self, url: &str, dest: &Path, cancel: &CancellationToken) -> Result<u64> {
        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(cancelled()),
            r = self.client.get(url).send() => r.map_err(|e| RaptorError::Network(format!("GET {url}: {e}")))?,
        };
        if !resp.status().is_success() {
            return Err(RaptorError::Network(format!(
                "GET {url} returned HTTP {}",
                resp.status().as_u16()
            )));
        }
        let mut file = tokio::fs::File::create(dest).await?;
        let mut stream = resp.bytes_stream();
        let mut total = 0u64;
        if let Some(p) = &self.progress {
            p.store(0, Ordering::Relaxed);
        }
        loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => return Err(cancelled()),
                n = stream.next() => n,
            };
            match next {
                None => break,
                Some(Ok(bytes)) => {
                    file.write_all(&bytes).await?;
                    total += bytes.len() as u64;
                    if let Some(p) = &self.progress {
                        p.store(total, Ordering::Relaxed);
                    }
                }
                Some(Err(e)) => {
                    return Err(RaptorError::Network(format!("download interrupted: {e}")))
                }
            }
        }
        file.flush().await?;
        file.sync_all().await?;
        Ok(total)
    }
}

/// Archive formats the installer can extract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    Zip,
    TarGz,
}

impl ArchiveKind {
    /// Infers the format from the URL path (query and fragment are ignored).
    pub fn from_url(url: &str) -> Result<Self> {
        let path = url
            .split(['?', '#'])
            .next()
            .unwrap_or(url)
            .to_ascii_lowercase();
        if path.ends_with(".zip") {
            Ok(ArchiveKind::Zip)
        } else if path.ends_with(".tar.gz") || path.ends_with(".tgz") {
            Ok(ArchiveKind::TarGz)
        } else {
            Err(RaptorError::Unsupported(format!(
                "unsupported archive type for {url}"
            )))
        }
    }
}

/// Converts an archive entry name into a relative path, rejecting anything that
/// could escape the destination (absolute paths, `..`, drive prefixes, backslashes).
pub fn safe_entry_path(name: &str) -> Result<PathBuf> {
    let bad = || RaptorError::InvalidInput(format!("unsafe archive entry: {name:?}"));
    if name.is_empty() || name.contains('\\') || name.contains(':') || name.starts_with('/') {
        return Err(bad());
    }
    let mut out = PathBuf::new();
    for c in Path::new(name).components() {
        match c {
            Component::Normal(p) => out.push(p),
            Component::CurDir => {}
            _ => return Err(bad()),
        }
    }
    Ok(out)
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<()> {
    let file = File::open(archive)?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|e| RaptorError::Integrity(format!("bad zip archive: {e}")))?;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| RaptorError::Integrity(format!("bad zip entry: {e}")))?;
        if entry.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000) {
            return Err(RaptorError::Unsupported(format!(
                "symlink entry {} is not allowed",
                entry.name()
            )));
        }
        let rel = safe_entry_path(entry.name())?;
        if rel.as_os_str().is_empty() {
            continue;
        }
        let out = dest.join(&rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = File::create(&out)?;
        std::io::copy(&mut entry, &mut f)?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode & 0o777))?;
        }
    }
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest: &Path) -> Result<()> {
    let file = File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(BufReader::new(file)));
    for entry in tar.entries()? {
        let mut entry = entry?;
        let kind = entry.header().entry_type();
        if kind.is_symlink() || kind.is_hard_link() {
            return Err(RaptorError::Unsupported(
                "link entries in tar archives are not allowed".into(),
            ));
        }
        let name = entry.path()?.to_string_lossy().into_owned();
        safe_entry_path(&name)?;
        if !entry.unpack_in(dest)? {
            return Err(RaptorError::InvalidInput(format!(
                "unsafe tar entry: {name:?}"
            )));
        }
    }
    Ok(())
}

/// Extracts `archive` into the (empty) directory `dest`.
pub fn extract_archive(kind: ArchiveKind, archive: &Path, dest: &Path) -> Result<()> {
    match kind {
        ArchiveKind::Zip => extract_zip(archive, dest),
        ArchiveKind::TarGz => extract_tar_gz(archive, dest),
    }
}

/// Finds the llama-server executable within `root` (shallowest match, up to a
/// few directory levels deep, since archives often wrap a folder).
pub fn locate_binary(root: &Path) -> Result<PathBuf> {
    let names = ["llama-server.exe", "llama-server"];
    let mut level = vec![root.to_path_buf()];
    for _ in 0..=MAX_LOCATE_DEPTH {
        let mut next = Vec::new();
        for dir in &level {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            let mut entries: Vec<_> = entries.flatten().collect();
            entries.sort_by_key(|e| e.file_name());
            for e in entries {
                let p = e.path();
                if p.is_file() && names.iter().any(|n| e.file_name() == *n) {
                    return Ok(p);
                }
                if p.is_dir() {
                    next.push(p);
                }
            }
        }
        level = next;
    }
    Err(RaptorError::NotFound(format!(
        "llama-server not found under {}",
        root.display()
    )))
}

/// Metadata written into every installed version directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallMarker {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub sha256: String,
}

/// A runtime version present on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstalledRuntime {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub sha256: String,
    pub dir: PathBuf,
    pub binary: PathBuf,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RuntimeState {
    current: Option<String>,
    previous: Option<String>,
}

/// Installs, lists, and rolls back runtime versions under `DataDirs::runtimes()`.
pub struct RuntimeInstaller {
    runtimes: PathBuf,
    downloads: PathBuf,
    fetcher: Arc<dyn Fetcher>,
}

impl RuntimeInstaller {
    /// Creates an installer rooted at `dirs`. Directories are created on demand.
    pub fn new(dirs: &DataDirs, fetcher: Arc<dyn Fetcher>) -> Self {
        Self {
            runtimes: dirs.runtimes(),
            downloads: dirs.downloads(),
            fetcher,
        }
    }

    /// Installs `m` (idempotent). If the version is already installed with the
    /// same SHA-256, nothing is downloaded. Otherwise the archive is fetched,
    /// verified, extracted into a staging directory and renamed into place. On
    /// any failure the partial download and staging directory are removed.
    pub async fn install(
        &self,
        m: &Manifest,
        cancel: &CancellationToken,
    ) -> Result<InstalledRuntime> {
        m.validate()?;
        let kind = ArchiveKind::from_url(&m.url)?;
        let id_dir = self.runtimes.join(&m.id);
        let ver_dir = id_dir.join(&m.version);

        if let Some(existing) = self.installed_version(&m.id, &m.version)? {
            if digests_equal(&existing.sha256, &m.sha256) {
                self.note_installed(&m.id, &m.version)?;
                return Ok(existing);
            }
            tracing::warn!(id = %m.id, version = %m.version, "installed sha256 differs from manifest; reinstalling");
        }

        std::fs::create_dir_all(&id_dir)?;
        std::fs::create_dir_all(&self.downloads)?;
        let part = self.downloads.join(format!("{}-{}.part", m.id, m.version));
        let stage = id_dir.join(format!(".stage-{}-{}", m.version, unique_suffix()));

        let result = self
            .install_inner(m, kind, &part, &stage, &ver_dir, cancel)
            .await;
        remove_file_if_exists(&part);
        remove_dir_if_exists(&stage);
        result
    }

    async fn install_inner(
        &self,
        m: &Manifest,
        kind: ArchiveKind,
        part: &Path,
        stage: &Path,
        ver_dir: &Path,
        cancel: &CancellationToken,
    ) -> Result<InstalledRuntime> {
        self.fetcher.fetch(&m.url, part, cancel).await?;
        if cancel.is_cancelled() {
            return Err(cancelled());
        }

        let part_owned = part.to_path_buf();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&part_owned))
            .await
            .map_err(|e| RaptorError::Internal(format!("hash task failed: {e}")))??;
        if !digests_equal(&actual, &m.sha256) {
            return Err(RaptorError::Integrity(format!(
                "sha256 mismatch for {} {}: expected {}, got {}",
                m.id, m.version, m.sha256, actual
            )));
        }

        let (part_owned, stage_owned, ver_owned) = (
            part.to_path_buf(),
            stage.to_path_buf(),
            ver_dir.to_path_buf(),
        );
        let marker = InstallMarker {
            id: m.id.clone(),
            version: m.version.clone(),
            backend: m.backend,
            sha256: m.sha256.clone(),
        };
        tokio::task::spawn_blocking(move || -> Result<()> {
            std::fs::create_dir_all(&stage_owned)?;
            extract_archive(kind, &part_owned, &stage_owned)?;
            locate_binary(&stage_owned)?;
            std::fs::write(
                stage_owned.join(MARKER_FILE),
                serde_json::to_vec_pretty(&marker)?,
            )?;
            if ver_owned.exists() {
                std::fs::remove_dir_all(&ver_owned)?;
            }
            std::fs::rename(&stage_owned, &ver_owned)?;
            Ok(())
        })
        .await
        .map_err(|e| RaptorError::Internal(format!("extract task failed: {e}")))??;

        self.note_installed(&m.id, &m.version)?;
        self.installed_version(&m.id, &m.version)?
            .ok_or_else(|| RaptorError::Internal("installed runtime vanished after rename".into()))
    }

    /// Reads one installed version. `Ok(None)` when it is absent or has no marker.
    pub fn installed_version(&self, id: &str, version: &str) -> Result<Option<InstalledRuntime>> {
        crate::manifest::validate_ids(id, version)?;
        let dir = self.runtimes.join(id).join(version);
        let marker_path = dir.join(MARKER_FILE);
        if !marker_path.is_file() {
            return Ok(None);
        }
        let marker: InstallMarker =
            serde_json::from_slice(&std::fs::read(&marker_path)?).map_err(|e| {
                RaptorError::Integrity(format!("corrupt install marker for {id} {version}: {e}"))
            })?;
        let binary = locate_binary(&dir)?;
        Ok(Some(InstalledRuntime {
            id: marker.id,
            version: marker.version,
            backend: marker.backend,
            sha256: marker.sha256,
            dir,
            binary,
        }))
    }

    /// The version marked current for `id`, if any.
    pub fn current(&self, id: &str) -> Result<Option<InstalledRuntime>> {
        let state = self.load_state(id)?;
        match state.current {
            Some(v) => self.installed_version(id, &v),
            None => Ok(None),
        }
    }

    /// Makes the previous version current again and returns it. The version
    /// being replaced becomes the new "previous", so a rollback can be undone.
    pub fn rollback(&self, id: &str) -> Result<InstalledRuntime> {
        let mut state = self.load_state(id)?;
        let prev = state.previous.clone().ok_or_else(|| {
            RaptorError::NotFound(format!("no previous runtime version for {id}"))
        })?;
        let restored = self.installed_version(id, &prev)?.ok_or_else(|| {
            RaptorError::NotFound(format!("previous version {prev} of {id} is not on disk"))
        })?;
        state.previous = state.current.take();
        state.current = Some(prev);
        self.save_state(id, &state)?;
        Ok(restored)
    }

    /// Deletes every installed version of runtime `id`, plus its rollback state.
    /// Only `runtimes/<id>` is removed. Models and downloads are never touched.
    /// Callers must not remove a runtime whose server is still running.
    pub fn uninstall(&self, id: &str) -> Result<()> {
        crate::manifest::validate_ids(id, "x")?;
        let dir = self.runtimes.join(id);
        if !dir.is_dir() {
            return Err(RaptorError::NotFound(format!(
                "runtime {id} is not installed"
            )));
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Lists all installed runtime versions on disk.
    pub fn list_installed(&self) -> Result<Vec<InstalledRuntime>> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.runtimes) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e.into()),
        };
        for id_entry in entries.flatten() {
            if !id_entry.path().is_dir() {
                continue;
            }
            let id = id_entry.file_name().to_string_lossy().into_owned();
            let ver_entries = match std::fs::read_dir(id_entry.path()) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for ver_entry in ver_entries.flatten() {
                if !ver_entry.path().is_dir() {
                    continue;
                }
                let ver = ver_entry.file_name().to_string_lossy().into_owned();
                if let Ok(Some(inst)) = self.installed_version(&id, &ver) {
                    out.push(inst);
                }
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.version.cmp(&b.version)));
        Ok(out)
    }

    fn note_installed(&self, id: &str, version: &str) -> Result<()> {
        let mut state = self.load_state(id)?;
        if state.current.as_deref() != Some(version) {
            state.previous = state.current.take();
            state.current = Some(version.to_string());
            self.save_state(id, &state)?;
        }
        Ok(())
    }

    fn load_state(&self, id: &str) -> Result<RuntimeState> {
        crate::manifest::validate_ids(id, "x")?;
        let path = self.runtimes.join(id).join(STATE_FILE);
        if !path.is_file() {
            return Ok(RuntimeState::default());
        }
        serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| RaptorError::Integrity(format!("corrupt runtime state for {id}: {e}")))
    }

    fn save_state(&self, id: &str, state: &RuntimeState) -> Result<()> {
        let dir = self.runtimes.join(id);
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!("{STATE_FILE}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec_pretty(state)?)?;
        std::fs::rename(&tmp, dir.join(STATE_FILE))?;
        Ok(())
    }
}

fn unique_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}-{}", nanos, std::process::id())
}

fn remove_file_if_exists(p: &Path) {
    if let Err(e) = std::fs::remove_file(p) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %p.display(), error = %e, "could not remove temporary file");
        }
    }
}

fn remove_dir_if_exists(p: &Path) {
    if let Err(e) = std::fs::remove_dir_all(p) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %p.display(), error = %e, "could not remove staging directory");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hex;
    use crate::manifest::test_support::manifest;
    use crate::testing::{make_tar_gz, make_zip, FakeFetcher};

    const LLAMA: &[u8] = b"fake llama-server";

    fn setup() -> (tempfile::TempDir, DataDirs) {
        let t = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(t.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        (t, dirs)
    }

    fn mani(archive: &[u8], url: &str, version: &str) -> Manifest {
        let mut m = manifest("llama-hip-win", version, Backend::Hip, &["gfx1200"]);
        m.url = url.into();
        m.sha256 = sha256_hex(archive);
        m
    }

    fn installer(
        dirs: &DataDirs,
        archive_url: &str,
        archive: Vec<u8>,
    ) -> (RuntimeInstaller, Arc<FakeFetcher>) {
        let fetcher = Arc::new(FakeFetcher::default().serve(archive_url, archive));
        (RuntimeInstaller::new(dirs, fetcher.clone()), fetcher)
    }

    fn stage_dirs(dirs: &DataDirs) -> Vec<String> {
        let id_dir = dirs.runtimes().join("llama-hip-win");
        std::fs::read_dir(id_dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.starts_with(".stage-"))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn install_extracts_verifies_and_marks_current() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/hip-b7100.zip";
        let zip = make_zip(&[
            ("llama-b7100/llama-server.exe", LLAMA),
            ("llama-b7100/readme.txt", b"hi"),
        ]);
        let (inst, _) = installer(&dirs, url, zip.clone());
        let m = mani(&zip, url, "b7100");
        let got = inst.install(&m, &CancellationToken::new()).await.unwrap();
        assert_eq!(got.version, "b7100");
        assert_eq!(std::fs::read(&got.binary).unwrap(), LLAMA);
        assert!(got.binary.ends_with("llama-server.exe"));
        assert_eq!(
            inst.current("llama-hip-win").unwrap().unwrap().version,
            "b7100"
        );
        assert!(
            stage_dirs(&dirs).is_empty(),
            "staging directory must be renamed away"
        );
        assert!(!dirs.downloads().join("llama-hip-win-b7100.part").exists());
    }

    #[tokio::test]
    async fn reinstall_of_same_version_does_not_download_again() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/a.zip";
        let zip = make_zip(&[("llama-server.exe", LLAMA)]);
        let (inst, fetcher) = installer(&dirs, url, zip.clone());
        let m = mani(&zip, url, "b7100");
        inst.install(&m, &CancellationToken::new()).await.unwrap();
        inst.install(&m, &CancellationToken::new()).await.unwrap();
        assert_eq!(fetcher.calls(), 1);
    }

    #[tokio::test]
    async fn sha_mismatch_is_integrity_error_and_leaves_nothing_behind() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/a.zip";
        let zip = make_zip(&[("llama-server.exe", LLAMA)]);
        let (inst, _) = installer(&dirs, url, zip.clone());
        let mut m = mani(&zip, url, "b7100");
        m.sha256 = "0".repeat(64);
        let err = inst
            .install(&m, &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(
            matches!(err, RaptorError::Integrity(ref msg) if msg.contains("sha256 mismatch")),
            "{err:?}"
        );
        assert!(!dirs.runtimes().join("llama-hip-win").join("b7100").exists());
        assert!(stage_dirs(&dirs).is_empty());
        assert!(!dirs.downloads().join("llama-hip-win-b7100.part").exists());
        assert!(inst.current("llama-hip-win").unwrap().is_none());
    }

    #[tokio::test]
    async fn zip_slip_entry_is_rejected_and_writes_nothing_outside() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/evil.zip";
        let zip = make_zip(&[
            ("../escape/evil.txt", b"pwned"),
            ("llama-server.exe", LLAMA),
        ]);
        let (inst, _) = installer(&dirs, url, zip.clone());
        let err = inst
            .install(&mani(&zip, url, "b7100"), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::InvalidInput(_)), "{err:?}");
        assert!(!dirs.root.join("escape").exists());
        assert!(!dirs.runtimes().join("escape").exists());
    }

    #[tokio::test]
    async fn tar_gz_archive_installs() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/linux-b7100.tar.gz";
        let tgz = make_tar_gz(&[("build/bin/llama-server", LLAMA)]);
        let (inst, _) = installer(&dirs, url, tgz.clone());
        let got = inst
            .install(&mani(&tgz, url, "b7100"), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(std::fs::read(&got.binary).unwrap(), LLAMA);
    }

    #[tokio::test]
    async fn tar_symlink_entry_is_rejected() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/sym.tar.gz";
        let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut b = tar::Builder::new(gz);
        let mut h = tar::Header::new_gnu();
        h.set_entry_type(tar::EntryType::Symlink);
        h.set_size(0);
        b.append_link(&mut h, "llama-server", "/etc/passwd")
            .unwrap();
        let tgz = b.into_inner().unwrap().finish().unwrap();
        let (inst, _) = installer(&dirs, url, tgz.clone());
        let err = inst
            .install(&mani(&tgz, url, "b7100"), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Unsupported(_)), "{err:?}");
    }

    #[tokio::test]
    async fn archive_without_llama_server_fails_and_cleans_stage() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/empty.zip";
        let zip = make_zip(&[("README.txt", b"no binary here")]);
        let (inst, _) = installer(&dirs, url, zip.clone());
        let err = inst
            .install(&mani(&zip, url, "b7100"), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::NotFound(_)), "{err:?}");
        assert!(stage_dirs(&dirs).is_empty());
        assert!(!dirs.runtimes().join("llama-hip-win").join("b7100").exists());
    }

    #[tokio::test]
    async fn unsupported_archive_type_is_rejected_before_download() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/runtime.7z";
        let (inst, fetcher) = installer(&dirs, url, vec![1, 2, 3]);
        let err = inst
            .install(&mani(&[1, 2, 3], url, "b7100"), &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, RaptorError::Unsupported(_)));
        assert_eq!(fetcher.calls(), 0);
    }

    #[tokio::test]
    async fn cancelled_install_fails_and_installs_nothing() {
        let (_t, dirs) = setup();
        let url = "https://example.invalid/a.zip";
        let zip = make_zip(&[("llama-server.exe", LLAMA)]);
        let (inst, _) = installer(&dirs, url, zip.clone());
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(inst
            .install(&mani(&zip, url, "b7100"), &cancel)
            .await
            .is_err());
        assert!(inst.current("llama-hip-win").unwrap().is_none());
    }

    #[tokio::test]
    async fn rollback_restores_previous_version_and_can_be_undone() {
        let (_t, dirs) = setup();
        let url_old = "https://example.invalid/b7000.zip";
        let url_new = "https://example.invalid/b7100.zip";
        let zip_old = make_zip(&[("llama-server.exe", b"old")]);
        let zip_new = make_zip(&[("llama-server.exe", b"new")]);
        let fetcher = Arc::new(
            FakeFetcher::default()
                .serve(url_old, zip_old.clone())
                .serve(url_new, zip_new.clone()),
        );
        let inst = RuntimeInstaller::new(&dirs, fetcher);
        let cancel = CancellationToken::new();
        inst.install(&mani(&zip_old, url_old, "b7000"), &cancel)
            .await
            .unwrap();
        inst.install(&mani(&zip_new, url_new, "b7100"), &cancel)
            .await
            .unwrap();
        assert_eq!(
            inst.current("llama-hip-win").unwrap().unwrap().version,
            "b7100"
        );

        let back = inst.rollback("llama-hip-win").unwrap();
        assert_eq!(back.version, "b7000");
        assert_eq!(std::fs::read(&back.binary).unwrap(), b"old");
        assert_eq!(
            inst.current("llama-hip-win").unwrap().unwrap().version,
            "b7000"
        );

        let forward = inst.rollback("llama-hip-win").unwrap();
        assert_eq!(forward.version, "b7100", "rollback is reversible");
    }

    #[tokio::test]
    async fn rollback_without_previous_is_not_found() {
        let (_t, dirs) = setup();
        let inst = RuntimeInstaller::new(&dirs, Arc::new(FakeFetcher::default()));
        assert!(matches!(
            inst.rollback("llama-hip-win"),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn corrupt_install_marker_is_integrity_error() {
        let (_t, dirs) = setup();
        let ver = dirs.runtimes().join("llama-hip-win").join("b7100");
        std::fs::create_dir_all(&ver).unwrap();
        std::fs::write(ver.join(MARKER_FILE), b"{oops").unwrap();
        std::fs::write(ver.join("llama-server.exe"), LLAMA).unwrap();
        let inst = RuntimeInstaller::new(&dirs, Arc::new(FakeFetcher::default()));
        assert!(matches!(
            inst.installed_version("llama-hip-win", "b7100"),
            Err(RaptorError::Integrity(_))
        ));
    }

    #[test]
    fn safe_entry_path_accepts_relative_and_rejects_escapes() {
        assert_eq!(
            safe_entry_path("a/b.txt").unwrap(),
            PathBuf::from("a").join("b.txt")
        );
        assert_eq!(safe_entry_path("./a").unwrap(), PathBuf::from("a"));
        for bad in [
            "",
            "../x",
            "a/../../x",
            "/abs",
            "C:x",
            "a\\b",
            "..",
            "a/..\\b",
        ] {
            assert!(safe_entry_path(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn archive_kind_is_inferred_from_url() {
        assert_eq!(
            ArchiveKind::from_url("https://h/x.zip").unwrap(),
            ArchiveKind::Zip
        );
        assert_eq!(
            ArchiveKind::from_url("https://h/x.tar.gz?sig=1").unwrap(),
            ArchiveKind::TarGz
        );
        assert_eq!(
            ArchiveKind::from_url("https://h/x.TGZ").unwrap(),
            ArchiveKind::TarGz
        );
        assert!(ArchiveKind::from_url("https://h/x.7z").is_err());
    }

    #[test]
    fn locate_binary_searches_shallow_levels_only() {
        let t = tempfile::tempdir().unwrap();
        let shallow = t.path().join("wrap").join("bin");
        std::fs::create_dir_all(&shallow).unwrap();
        std::fs::write(shallow.join("llama-server"), b"x").unwrap();
        assert_eq!(
            locate_binary(t.path()).unwrap(),
            shallow.join("llama-server")
        );

        let deep_root = tempfile::tempdir().unwrap();
        let deep = deep_root.path().join("a").join("b").join("c").join("d");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("llama-server.exe"), b"x").unwrap();
        assert!(matches!(
            locate_binary(deep_root.path()),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn zip_symlink_entry_is_rejected() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("sym.zip");
        let mut w = zip::ZipWriter::new(std::fs::File::create(&p).unwrap());
        let opts = zip::write::SimpleFileOptions::default().unix_permissions(0o777);
        w.add_symlink("llama-server", "/etc/passwd", opts).unwrap();
        w.finish().unwrap();
        let out = t.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        assert!(matches!(
            extract_archive(ArchiveKind::Zip, &p, &out),
            Err(RaptorError::Unsupported(_))
        ));
    }

    #[tokio::test]
    async fn list_installed_finds_all_versions() {
        let (_t, dirs) = setup();
        let url1 = "https://example.invalid/hip-b7000.zip";
        let url2 = "https://example.invalid/hip-b7100.zip";
        let zip1 = make_zip(&[("llama-server.exe", b"fake-b7000" as &[u8])]);
        let zip2 = make_zip(&[("llama-server.exe", b"fake-b7100" as &[u8])]);
        let m1 = mani(&zip1, url1, "b7000");
        let m2 = mani(&zip2, url2, "b7100");
        let fetcher = Arc::new(FakeFetcher::default().serve(url1, zip1).serve(url2, zip2));
        let inst = RuntimeInstaller::new(&dirs, fetcher);
        inst.install(&m1, &CancellationToken::new()).await.unwrap();
        inst.install(&m2, &CancellationToken::new()).await.unwrap();
        let list = inst.list_installed().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].version, "b7000");
        assert_eq!(list[1].version, "b7100");
    }
}
