use crate::{RaptorError, Result};
use std::path::{Path, PathBuf};

/// All on-disk locations Raptor LLM uses. Models are never stored under `config`
/// or `runtimes`, so uninstall/reset never touches user models unless asked.
#[derive(Debug, Clone)]
pub struct DataDirs {
    pub root: PathBuf,
}

impl DataDirs {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    /// The per-user platform location, `<platform data dir>/RaptorLLM`.
    pub fn platform_location() -> Result<Self> {
        dirs::data_dir()
            .map(|d| Self::new(d.join("RaptorLLM")))
            .ok_or_else(|| RaptorError::Io("no platform data directory".into()))
    }

    /// `<folder of the app>/data`, when that folder can be written to.
    pub fn next_to_app() -> Option<Self> {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?.join("data");
        let writable = std::fs::create_dir_all(&dir).is_ok() && {
            let probe = dir.join(".write-test");
            let ok = std::fs::write(&probe, b"x").is_ok();
            let _ = std::fs::remove_file(&probe);
            ok
        };
        writable.then(|| Self::new(dir))
    }

    /// Where data lives when the user has not chosen a folder: next to the app. An install that
    /// already keeps its data in the platform location keeps using it, so an update never hides
    /// existing models. When the app folder cannot be written to (a system-wide install), the
    /// platform location is used.
    pub fn default_location() -> Result<Self> {
        let platform = Self::platform_location()?;
        Ok(Self::choose_default(
            platform.root.join("config").is_dir(),
            platform,
            Self::next_to_app(),
        ))
    }

    fn choose_default(legacy_in_use: bool, platform: Self, app: Option<Self>) -> Self {
        match (legacy_in_use, app) {
            (false, Some(app)) => app,
            _ => platform,
        }
    }

    pub fn config(&self) -> PathBuf {
        self.root.join("config")
    }
    pub fn runtimes(&self) -> PathBuf {
        self.root.join("runtimes")
    }
    pub fn models(&self) -> PathBuf {
        self.root.join("models")
    }
    pub fn downloads(&self) -> PathBuf {
        self.root.join("downloads")
    }
    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
    pub fn db(&self) -> PathBuf {
        self.root.join("db")
    }
    pub fn context_index(&self) -> PathBuf {
        self.root.join("context")
    }

    pub fn ensure(&self) -> Result<()> {
        for d in [
            self.config(),
            self.runtimes(),
            self.models(),
            self.downloads(),
            self.logs(),
            self.db(),
            self.context_index(),
        ] {
            std::fs::create_dir_all(&d)?;
        }
        Ok(())
    }
}

/// Reject path components that could escape a base directory (`..`, absolute,
/// separators). Use for every filename that originates from the network or UI.
pub fn safe_file_name(name: &str) -> Result<&str> {
    let p = Path::new(name);
    let ok = !name.is_empty()
        && p.components().count() == 1
        && !name.contains(['/', '\\', ':'])
        && name != "."
        && name != "..";
    if ok {
        Ok(name)
    } else {
        Err(RaptorError::InvalidInput(format!(
            "unsafe file name: {name}"
        )))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_default_is_next_to_the_app_unless_existing_data_or_a_read_only_folder() {
        let p = || DataDirs::new("/platform");
        let a = || Some(DataDirs::new("/app/data"));
        assert_eq!(
            DataDirs::choose_default(false, p(), a()).root,
            std::path::PathBuf::from("/app/data")
        );
        assert_eq!(
            DataDirs::choose_default(true, p(), a()).root,
            std::path::PathBuf::from("/platform")
        );
        assert_eq!(
            DataDirs::choose_default(false, p(), None).root,
            std::path::PathBuf::from("/platform")
        );
    }

    use super::*;
    #[test]
    fn rejects_traversal() {
        for bad in ["", "..", "../x", "a/b", "a\\b", "C:evil", "/abs"] {
            assert!(safe_file_name(bad).is_err(), "{bad}");
        }
        assert!(safe_file_name("model-Q4_K_M.gguf").is_ok());
    }
    #[test]
    fn ensure_creates_dirs() {
        let t = tempfile::tempdir().unwrap();
        let d = DataDirs::new(t.path().join("n"));
        d.ensure().unwrap();
        assert!(d.models().is_dir() && d.runtimes().is_dir());
    }
}
