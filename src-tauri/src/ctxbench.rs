//! Saved results of the context-length benchmark matrix, in `config/ctxbench.json`, keyed like the
//! tuning store (machine, model content, runtime id and version) so a new runtime build, GPU or
//! file is measured again. A corrupt file is treated as empty.

use raptor_core::Result;
use raptor_runtime::ctxmatrix::MatrixRow;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One full run of the matrix for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixReport {
    pub model_id: String,
    pub measured_at: u64,
    /// Layers on the GPU and micro-batch used (the tuned values when the model was tuned).
    pub gpu_layers: u32,
    pub ubatch: u32,
    /// The largest depth that was asked for.
    pub max_depth: u32,
    pub rows: Vec<MatrixRow>,
    /// Seconds to ingest a prompt of each measured depth from empty, from the measured speeds.
    pub prompt_seconds: Vec<(u32, f64)>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct File {
    entries: BTreeMap<String, MatrixReport>,
}

/// The persisted reports.
#[derive(Debug)]
pub struct MatrixStore {
    path: PathBuf,
    file: File,
}

impl MatrixStore {
    /// Opens the store at `path`. A missing, unreadable or corrupt file gives an empty store.
    pub fn open(path: &Path) -> Self {
        let file = std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Self {
            path: path.to_path_buf(),
            file,
        }
    }

    pub fn get(&self, key: &str) -> Option<MatrixReport> {
        self.file.entries.get(key).cloned()
    }

    /// Saves `report` for `key`, replacing an older one, atomically.
    pub fn put(&mut self, key: &str, report: MatrixReport) -> Result<()> {
        let mut next = File {
            entries: self.file.entries.clone(),
        };
        next.entries.insert(key.to_string(), report);
        let bytes = serde_json::to_vec_pretty(&next)?;
        crate::fsutil::write_atomic(&self.path, &bytes)?;
        self.file = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raptor_runtime::config::KvCacheType;

    fn report(depths: &[u32]) -> MatrixReport {
        MatrixReport {
            model_id: "local/m.gguf".into(),
            measured_at: 1,
            gpu_layers: 66,
            ubatch: 256,
            max_depth: 32_768,
            rows: depths
                .iter()
                .map(|d| MatrixRow {
                    depth: *d,
                    kv_cache: KvCacheType::F16,
                    prompt_tps: Some(500.0),
                    gen_tps: Some(20.0),
                    turn_ms: Some(1024.0),
                    peak_dedicated_mib: Some(13_000),
                    peak_shared_growth_mib: Some(10),
                    spilled: false,
                    note: None,
                })
                .collect(),
            prompt_seconds: vec![(0, 0.0)],
        }
    }

    #[test]
    fn a_report_survives_a_restart_and_a_newer_run_replaces_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config").join("ctxbench.json");
        let mut store = MatrixStore::open(&path);
        assert!(store.get("k").is_none());
        store.put("k", report(&[0, 4096])).unwrap();
        assert_eq!(MatrixStore::open(&path).get("k"), Some(report(&[0, 4096])));
        store.put("k", report(&[0, 4096, 16_384])).unwrap();
        assert_eq!(MatrixStore::open(&path).get("k").unwrap().rows.len(), 3);
    }

    #[test]
    fn a_corrupt_file_is_an_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ctxbench.json");
        std::fs::write(&path, b"not json").unwrap();
        assert!(MatrixStore::open(&path).get("k").is_none());
    }
}
