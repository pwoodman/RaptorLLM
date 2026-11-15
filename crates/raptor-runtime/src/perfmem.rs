//! Performance memory: benchmark records keyed by (machine, model, runtime) and
//! a versioned JSON store with atomic writes.
//!
//! Rules: repeated runs of the same config are merged (not duplicated); an
//! unstable run never replaces a stable one; `best_known` only ever returns a
//! stable record.

use crate::config::{KvCacheType, LaunchConfig};
use crate::hardware::HardwareInfo;
use crate::hash::sha256_hex;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Current on-disk schema version.
pub const SCHEMA_VERSION: u32 = 2;

/// Identifies what a benchmark applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecordKey {
    /// Machine fingerprint from [`machine_fingerprint`].
    pub machine: String,
    /// SHA-256 of the model file.
    pub model_hash: String,
    pub runtime_id: String,
}

/// The launch parameters that affect speed or memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub kv_cache: KvCacheType,
}

impl From<&LaunchConfig> for ConfigSnapshot {
    fn from(c: &LaunchConfig) -> Self {
        Self {
            gpu_layers: c.gpu_layers,
            ctx_size: c.ctx_size,
            batch_size: c.batch_size,
            ubatch_size: c.ubatch_size,
            threads: c.threads,
            flash_attn: c.flash_attn,
            kv_cache: c.kv_cache,
        }
    }
}

/// One benchmark result (or a merged set of results for the same config).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRecord {
    pub machine: String,
    pub model_hash: String,
    pub runtime_id: String,
    pub runtime_version: String,
    pub config: ConfigSnapshot,
    pub ttft_ms: f64,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    /// True when the run finished without crashes, timeouts or GPU fallback.
    pub stable: bool,
    /// Number of runs merged into this record.
    pub samples: u32,
    /// Unix seconds of the most recent run.
    pub measured_at: u64,
}

impl BenchRecord {
    /// The key this record is stored under.
    pub fn key(&self) -> RecordKey {
        RecordKey {
            machine: self.machine.clone(),
            model_hash: self.model_hash.clone(),
            runtime_id: self.runtime_id.clone(),
        }
    }

    fn validate(&self) -> Result<()> {
        let finite_nonneg = |v: f64| v.is_finite() && v >= 0.0;
        if self.machine.is_empty() || self.model_hash.is_empty() || self.runtime_id.is_empty() {
            return Err(RaptorError::InvalidInput(
                "benchmark key fields must be non-empty".into(),
            ));
        }
        if !(finite_nonneg(self.ttft_ms)
            && finite_nonneg(self.prompt_tps)
            && finite_nonneg(self.gen_tps))
        {
            return Err(RaptorError::InvalidInput(
                "benchmark numbers must be finite and non-negative".into(),
            ));
        }
        if self.samples == 0 {
            return Err(RaptorError::InvalidInput(
                "samples must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Result of recording a benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// A new entry was created.
    Inserted,
    /// Merged into an existing entry for the same key and config.
    Merged { samples: u32 },
    /// Ignored: the existing entry is stable and this run is not.
    KeptStable,
}

/// Relative change between two runs, in percent (positive = candidate is higher).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Comparison {
    pub gen_tps_delta_pct: f64,
    pub prompt_tps_delta_pct: f64,
    /// Negative means the candidate responds faster.
    pub ttft_delta_pct: f64,
    pub vram_delta_bytes: i64,
}

fn pct(base: f64, cand: f64) -> f64 {
    (cand - base) / base * 100.0
}

/// Compares `candidate` against `base`. Fails when the base has zero speed (no meaningful ratio).
pub fn compare(base: &BenchRecord, candidate: &BenchRecord) -> Result<Comparison> {
    if base.gen_tps <= 0.0 || base.prompt_tps <= 0.0 || base.ttft_ms <= 0.0 {
        return Err(RaptorError::InvalidInput(
            "base record has zero timing; cannot compare".into(),
        ));
    }
    Ok(Comparison {
        gen_tps_delta_pct: pct(base.gen_tps, candidate.gen_tps),
        prompt_tps_delta_pct: pct(base.prompt_tps, candidate.prompt_tps),
        ttft_delta_pct: pct(base.ttft_ms, candidate.ttft_ms),
        vram_delta_bytes: candidate.vram_bytes as i64 - base.vram_bytes as i64,
    })
}

/// Stable fingerprint of the hardware that matters for performance. Excludes
/// volatile values such as available RAM.
pub fn machine_fingerprint(hw: &HardwareInfo) -> String {
    let gpus: Vec<String> = hw
        .gpus
        .iter()
        .map(|g| format!("{}|{}|{:?}", g.name, g.vram_total, g.arch))
        .collect();
    let canonical = format!(
        "os={:?};cpu={};cores={};ram={};gpus={}",
        hw.os,
        hw.cpu.brand,
        hw.cpu.cores,
        hw.ram_total,
        gpus.join(";")
    );
    sha256_hex(canonical.as_bytes())
}

/// Upgrades a parsed store file to [`SCHEMA_VERSION`]. Version 1 lacked
/// `runtime_version`, `samples` and `measured_at`; defaults are filled in.
pub fn migrate(mut file: serde_json::Value) -> Result<Vec<BenchRecord>> {
    let version = file
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RaptorError::Integrity("perf store has no schema_version".into()))?;
    match version {
        1 => {
            if let Some(records) = file.get_mut("records").and_then(|r| r.as_array_mut()) {
                for r in records.iter_mut() {
                    if let Some(obj) = r.as_object_mut() {
                        obj.entry("runtime_version")
                            .or_insert(serde_json::json!("unknown"));
                        obj.entry("samples").or_insert(serde_json::json!(1));
                        obj.entry("measured_at").or_insert(serde_json::json!(0));
                    }
                }
            }
        }
        v if v == u64::from(SCHEMA_VERSION) => {}
        other => {
            return Err(RaptorError::Unsupported(format!(
                "perf store schema {other} is newer than this build supports ({SCHEMA_VERSION})"
            )))
        }
    }
    let records = file
        .get("records")
        .cloned()
        .ok_or_else(|| RaptorError::Integrity("perf store has no records array".into()))?;
    serde_json::from_value(records)
        .map_err(|e| RaptorError::Integrity(format!("perf store records are invalid: {e}")))
}

/// JSON-backed store of benchmark records.
#[derive(Debug, Clone)]
pub struct PerfStore {
    path: PathBuf,
    records: Vec<BenchRecord>,
}

fn weighted(a: f64, na: u32, b: f64, nb: u32) -> f64 {
    (a * f64::from(na) + b * f64::from(nb)) / f64::from(na + nb)
}

impl PerfStore {
    /// Opens the store at `path`. A missing file is an empty store. A corrupt
    /// file is an error and is never overwritten.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            return Ok(Self {
                path,
                records: Vec::new(),
            });
        }
        let bytes = std::fs::read(&path)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            RaptorError::Integrity(format!("perf store {} is corrupt: {e}", path.display()))
        })?;
        let records = migrate(value)?;
        Ok(Self { path, records })
    }

    /// All records.
    pub fn records(&self) -> &[BenchRecord] {
        &self.records
    }

    /// Path this store saves to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Records a benchmark. Changes are in memory until [`PerfStore::save`].
    ///
    /// - Same key, runtime version and config: merged (sample-weighted averages).
    /// - Existing stable and new run unstable: ignored (`KeptStable`).
    pub fn record(&mut self, run: BenchRecord) -> Result<RecordOutcome> {
        run.validate()?;
        let existing = self.records.iter_mut().find(|r| {
            r.key() == run.key()
                && r.config == run.config
                && r.runtime_version == run.runtime_version
        });
        match existing {
            None => {
                self.records.push(run);
                Ok(RecordOutcome::Inserted)
            }
            Some(e) => {
                if e.stable && !run.stable {
                    return Ok(RecordOutcome::KeptStable);
                }
                let (na, nb) = (e.samples, run.samples);
                e.ttft_ms = weighted(e.ttft_ms, na, run.ttft_ms, nb);
                e.prompt_tps = weighted(e.prompt_tps, na, run.prompt_tps, nb);
                e.gen_tps = weighted(e.gen_tps, na, run.gen_tps, nb);
                e.vram_bytes = e.vram_bytes.max(run.vram_bytes);
                e.ram_bytes = e.ram_bytes.max(run.ram_bytes);
                e.stable = e.stable || run.stable;
                e.samples = na + nb;
                e.measured_at = e.measured_at.max(run.measured_at);
                Ok(RecordOutcome::Merged { samples: e.samples })
            }
        }
    }

    /// The fastest stable record for `key` (by generation speed), if any.
    pub fn best_known(&self, key: &RecordKey) -> Option<&BenchRecord> {
        self.records
            .iter()
            .filter(|r| r.stable && r.key() == *key)
            .max_by(|a, b| {
                a.gen_tps
                    .partial_cmp(&b.gen_tps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// The record for an exact config under `key`, stable or not.
    pub fn lookup(&self, key: &RecordKey, config: &ConfigSnapshot) -> Option<&BenchRecord> {
        self.records
            .iter()
            .find(|r| r.key() == *key && r.config == *config)
    }

    /// Writes the store atomically: a temporary file is synced and then renamed over the target.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let doc = serde_json::json!({ "schema_version": SCHEMA_VERSION, "records": self.records });
        let bytes = serde_json::to_vec_pretty(&doc)?;
        let name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "perf.json".into());
        let tmp = self.path.with_file_name(format!("{name}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, GpuInfo, GpuVendor, Os};

    fn cfg(ctx: u32) -> ConfigSnapshot {
        ConfigSnapshot {
            gpu_layers: 40,
            ctx_size: ctx,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 8,
            flash_attn: true,
            kv_cache: KvCacheType::F16,
        }
    }

    fn rec(ctx: u32, gen: f64, stable: bool) -> BenchRecord {
        BenchRecord {
            machine: "m1".into(),
            model_hash: "h1".into(),
            runtime_id: "llama-hip-win".into(),
            runtime_version: "b7100".into(),
            config: cfg(ctx),
            ttft_ms: 300.0,
            prompt_tps: 900.0,
            gen_tps: gen,
            vram_bytes: 10 << 30,
            ram_bytes: 2 << 30,
            stable,
            samples: 1,
            measured_at: 1_700_000_000,
        }
    }

    fn key() -> RecordKey {
        RecordKey {
            machine: "m1".into(),
            model_hash: "h1".into(),
            runtime_id: "llama-hip-win".into(),
        }
    }

    fn temp_store(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("perf.json")
    }

    #[test]
    fn save_and_reopen_preserves_records() {
        let dir = tempfile::tempdir().unwrap();
        let p = temp_store(&dir);
        let mut s = PerfStore::open(&p).unwrap();
        s.record(rec(8192, 62.0, true)).unwrap();
        s.save().unwrap();
        let reopened = PerfStore::open(&p).unwrap();
        assert_eq!(reopened.records(), s.records());
        assert!(
            !dir.path().join("perf.json.tmp").exists(),
            "temporary file must be renamed away"
        );
    }

    #[test]
    fn same_config_is_merged_with_weighted_average() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        assert_eq!(
            s.record(rec(8192, 60.0, true)).unwrap(),
            RecordOutcome::Inserted
        );
        assert_eq!(
            s.record(rec(8192, 70.0, true)).unwrap(),
            RecordOutcome::Merged { samples: 2 }
        );
        assert_eq!(s.records().len(), 1);
        assert!((s.records()[0].gen_tps - 65.0).abs() < 1e-9);
    }

    #[test]
    fn unstable_run_never_replaces_stable_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        s.record(rec(8192, 60.0, true)).unwrap();
        assert_eq!(
            s.record(rec(8192, 999.0, false)).unwrap(),
            RecordOutcome::KeptStable
        );
        let best = s.best_known(&key()).unwrap();
        assert_eq!(best.gen_tps, 60.0);
        assert_eq!(best.samples, 1);
    }

    #[test]
    fn unstable_record_is_never_best_known() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        s.record(rec(4096, 500.0, false)).unwrap();
        assert!(s.best_known(&key()).is_none());
        assert!(
            s.lookup(&key(), &cfg(4096)).is_some(),
            "lookup still finds unstable records"
        );
    }

    #[test]
    fn best_known_picks_fastest_stable_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        s.record(rec(4096, 55.0, true)).unwrap();
        s.record(rec(8192, 62.0, true)).unwrap();
        s.record(rec(16384, 40.0, true)).unwrap();
        assert_eq!(s.best_known(&key()).unwrap().config.ctx_size, 8192);
    }

    #[test]
    fn different_runtime_version_is_a_separate_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        s.record(rec(8192, 60.0, true)).unwrap();
        let mut newer = rec(8192, 70.0, true);
        newer.runtime_version = "b7200".into();
        assert_eq!(s.record(newer).unwrap(), RecordOutcome::Inserted);
        assert_eq!(s.records().len(), 2);
    }

    #[test]
    fn compare_reports_percentages() {
        let base = rec(8192, 50.0, true);
        let mut cand = rec(8192, 60.0, true);
        cand.ttft_ms = 240.0;
        cand.vram_bytes = 9 << 30;
        let c = compare(&base, &cand).unwrap();
        assert!((c.gen_tps_delta_pct - 20.0).abs() < 1e-9);
        assert!((c.ttft_delta_pct + 20.0).abs() < 1e-9);
        assert_eq!(c.vram_delta_bytes, -(1 << 30));
    }

    #[test]
    fn compare_rejects_zero_baseline() {
        let mut base = rec(8192, 0.0, true);
        base.gen_tps = 0.0;
        assert!(compare(&base, &rec(8192, 10.0, true)).is_err());
    }

    #[test]
    fn validation_rejects_nan_and_zero_samples() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = PerfStore::open(temp_store(&dir)).unwrap();
        assert!(s.record(rec(8192, f64::NAN, true)).is_err());
        let mut zero = rec(8192, 1.0, true);
        zero.samples = 0;
        assert!(s.record(zero).is_err());
        assert!(s.records().is_empty());
    }

    #[test]
    fn v1_store_migrates_to_current_schema() {
        let v1 = serde_json::json!({
            "schema_version": 1,
            "records": [{
                "machine": "m1", "model_hash": "h1", "runtime_id": "llama-hip-win",
                "config": {"gpu_layers": 40, "ctx_size": 8192, "batch_size": 2048, "ubatch_size": 512,
                           "threads": 8, "flash_attn": true, "kv_cache": "f16"},
                "ttft_ms": 300.0, "prompt_tps": 900.0, "gen_tps": 61.0,
                "vram_bytes": 1, "ram_bytes": 1, "stable": true
            }]
        });
        let records = migrate(v1).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].runtime_version, "unknown");
        assert_eq!(records[0].samples, 1);
        assert_eq!(records[0].measured_at, 0);
    }

    #[test]
    fn corrupt_store_is_an_error_and_is_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let p = temp_store(&dir);
        std::fs::write(&p, b"{not json").unwrap();
        assert!(matches!(
            PerfStore::open(&p),
            Err(RaptorError::Integrity(_))
        ));
        assert_eq!(std::fs::read(&p).unwrap(), b"{not json");
    }

    #[test]
    fn future_schema_is_unsupported() {
        let future = serde_json::json!({"schema_version": 99, "records": []});
        assert!(matches!(migrate(future), Err(RaptorError::Unsupported(_))));
    }

    #[test]
    fn fingerprint_is_stable_and_ignores_available_ram() {
        let hw = HardwareInfo {
            os: Os::Windows,
            cpu: CpuInfo {
                brand: "Ryzen".into(),
                cores: 8,
                threads: 16,
                features: vec![],
            },
            ram_total: 32 << 30,
            ram_avail: 10 << 30,
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Amd,
                name: "AMD Radeon RX 9060 XT".into(),
                vram_total: 16 << 30,
                vram_free: None,
                arch: Some("gfx1200".into()),
                driver: None,
            }],
        };
        let mut busier = hw.clone();
        busier.ram_avail = 2 << 30;
        assert_eq!(machine_fingerprint(&hw), machine_fingerprint(&busier));
        let mut other_gpu = hw.clone();
        other_gpu.gpus[0].name = "AMD Radeon RX 7900 XTX".into();
        assert_ne!(machine_fingerprint(&hw), machine_fingerprint(&other_gpu));
    }

    #[test]
    fn config_snapshot_from_launch_config() {
        let lc = LaunchConfig::builder("C:/m.gguf", crate::manifest::Backend::Hip)
            .ctx_size(8192)
            .kv_cache(KvCacheType::Q8_0)
            .build()
            .unwrap();
        let snap = ConfigSnapshot::from(&lc);
        assert_eq!(snap.ctx_size, 8192);
        assert_eq!(snap.kv_cache, KvCacheType::Q8_0);
        assert_eq!(snap.gpu_layers, 999);
    }
}
