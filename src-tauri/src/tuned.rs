//! What tuning found for each model on this machine, kept in `config/tuned.json` so the choice
//! survives a restart. An entry is keyed by machine, model content and runtime build: a new
//! runtime build, a different GPU or a different file is measured again, never assumed.
//! A corrupt or unreadable file is treated as empty and never stops the app.

use raptor_core::Result;
use raptor_runtime::config::KvCacheType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The settings tuning chose, and the evidence for them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunedEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualification: Option<Qualification>,
    pub gpu_layers: u32,
    pub ubatch_size: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    /// Time of the representative request with these settings, seconds.
    pub seconds: f64,
    /// The same for the settings used without tuning, seconds.
    pub baseline_seconds: f64,
    pub measured_at: u64,
    /// Load settings that affect the benchmark's transferability. Older entries without this
    /// evidence are ignored by the loader and can be regenerated with Optimize speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<TuneProfile>,
    /// Matched streamed-request measurements. Optional for older tuning files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_live: Option<TuneMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_live: Option<TuneMetrics>,
    /// The setting measured with a long context, from an opt-in deep tune. Absent in files written
    /// before deep tuning existed, and when only the quick tune was run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long: Option<LongTuned>,
}

/// Independent full-window confirmations in (baseline seconds, candidate seconds) order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qualification {
    pub context_size: u32,
    pub signature: String,
    pub pairs: Vec<(f64, f64)>,
}

impl Qualification {
    pub fn passed(&self) -> bool {
        self.context_size >= crate::settings::MIN_CONTEXT_TARGET
            && self.pairs.len() == 3
            && self.pairs.iter().all(|&(base, candidate)| {
                base.is_finite()
                    && candidate.is_finite()
                    && candidate > 0.0
                    && base >= candidate * 1.5
            })
    }
}

/// Median metrics from repeated, warmed-up streamed requests on one launch configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneMetrics {
    pub gpu_layers: u32,
    pub ubatch_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_slots: Option<u32>,
    pub kv_cache: KvCacheType,
    pub flash_attn: bool,
    pub speculative: bool,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub ttft_ms: f64,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub samples: u32,
    pub prompt_tokens_exact: bool,
}

/// Configuration held constant by the quick and deep tuning passes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneProfile {
    pub kv_cache: KvCacheType,
    pub flash_attn: bool,
}

/// The fastest setting with [`raptor_runtime::tune::DEEP_DEPTH`] tokens already in the context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongTuned {
    pub gpu_layers: u32,
    pub ubatch_size: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    /// Time of the representative request at that depth with these settings, seconds.
    pub seconds: f64,
    /// The same for the quick-tune setting at that depth (equal to `seconds` when it won).
    pub baseline_seconds: f64,
    pub depth: u32,
    /// Cache and attention mode held constant during the deep measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<TuneProfile>,
}

/// The setting a load uses for a given context window.
#[derive(Debug, Clone, PartialEq)]
pub struct Setting {
    pub gpu_layers: u32,
    pub ubatch_size: u32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub seconds: f64,
    pub baseline_seconds: f64,
    /// True for the long-prompt setting.
    pub long: bool,
}

impl TunedEntry {
    /// Prefer compatible deep measurements, falling back to the compatible quick profile.
    pub fn compatible_setting(
        &self,
        ctx_size: u32,
        kv_cache: KvCacheType,
        flash_attn: bool,
    ) -> Option<Setting> {
        if self.long.as_ref().is_some_and(|l| ctx_size >= l.depth)
            && self.matches_long(kv_cache, flash_attn)
        {
            return Some(self.setting_for(ctx_size));
        }
        self.matches(kv_cache, flash_attn)
            .then(|| self.setting_for(0))
    }

    /// Whether this profile was measured with the settings that the current load will use.
    pub fn matches(&self, kv_cache: KvCacheType, flash_attn: bool) -> bool {
        self.profile
            .as_ref()
            .is_some_and(|p| p.kv_cache == kv_cache && p.flash_attn == flash_attn)
    }

    /// Whether a deep profile matches the cache and attention settings for this context range.
    pub fn matches_long(&self, kv_cache: KvCacheType, flash_attn: bool) -> bool {
        self.long.as_ref().is_some_and(|l| {
            l.profile
                .as_ref()
                .is_some_and(|p| p.kv_cache == kv_cache && p.flash_attn == flash_attn)
        })
    }

    /// The long-prompt setting for a window of at least its depth when a deep tune measured one,
    /// otherwise the quick-tune setting.
    pub fn setting_for(&self, ctx_size: u32) -> Setting {
        match &self.long {
            Some(l) if ctx_size >= l.depth => Setting {
                gpu_layers: l.gpu_layers,
                ubatch_size: l.ubatch_size,
                prompt_tps: l.prompt_tps,
                gen_tps: l.gen_tps,
                seconds: l.seconds,
                baseline_seconds: l.baseline_seconds,
                long: true,
            },
            _ => Setting {
                gpu_layers: self.gpu_layers,
                ubatch_size: self.ubatch_size,
                prompt_tps: self.prompt_tps,
                gen_tps: self.gen_tps,
                seconds: self.seconds,
                baseline_seconds: self.baseline_seconds,
                long: false,
            },
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct File {
    entries: BTreeMap<String, TunedEntry>,
}

/// The persisted tuning results.
#[derive(Debug)]
pub struct TunedStore {
    path: PathBuf,
    file: File,
}

/// Key of one machine, model file and runtime build.
pub fn key(machine: &str, model_sha256: &str, runtime_id: &str, runtime_version: &str) -> String {
    format!("{machine}|{model_sha256}|{runtime_id}|{runtime_version}")
}

impl TunedStore {
    pub fn for_model(&self, machine: &str, sha: &str) -> Vec<SavedTuning> {
        let prefix = format!("{machine}|{sha}|");
        self.file
            .entries
            .iter()
            .filter_map(|(key, entry)| {
                let (runtime_id, runtime_version) = key.strip_prefix(&prefix)?.split_once('|')?;
                Some(SavedTuning {
                    runtime_id: runtime_id.into(),
                    runtime_version: runtime_version.into(),
                    tuning: entry.clone(),
                })
            })
            .collect()
    }

    /// Reset all builds/backends for this model in one atomic write.
    pub fn remove_model(&mut self, machine: &str, sha: &str) -> Result<bool> {
        let prefix = format!("{machine}|{sha}|");
        let mut next = File {
            entries: self.file.entries.clone(),
        };
        next.entries.retain(|key, _| !key.starts_with(&prefix));
        if next.entries.len() == self.file.entries.len() {
            return Ok(false);
        }
        crate::fsutil::write_atomic(&self.path, &serde_json::to_vec_pretty(&next)?)?;
        self.file = next;
        Ok(true)
    }

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

    pub fn get(&self, key: &str) -> Option<TunedEntry> {
        self.file.entries.get(key).cloned()
    }

    /// Saves `entry` for `key`, replacing an older one. The file is written atomically.
    pub fn put(&mut self, key: &str, entry: TunedEntry) -> Result<()> {
        let mut next = File {
            entries: self.file.entries.clone(),
        };
        next.entries.insert(key.to_string(), entry);
        let bytes = serde_json::to_vec_pretty(&next)?;
        crate::fsutil::write_atomic(&self.path, &bytes)?;
        self.file = next;
        Ok(())
    }

    /// Forgets the tuning of `key`. True when there was one.
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if !self.file.entries.contains_key(key) {
            return Ok(false);
        }
        let mut next = File {
            entries: self.file.entries.clone(),
        };
        next.entries.remove(key);
        let bytes = serde_json::to_vec_pretty(&next)?;
        crate::fsutil::write_atomic(&self.path, &bytes)?;
        self.file = next;
        Ok(true)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedTuning {
    pub runtime_id: String,
    pub runtime_version: String,
    pub tuning: TunedEntry,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(ub: u32) -> TunedEntry {
        TunedEntry {
            qualification: None,
            gpu_layers: 41,
            ubatch_size: ub,
            prompt_tps: 2234.7,
            gen_tps: 68.4,
            seconds: 5.6,
            baseline_seconds: 8.5,
            measured_at: 1,
            profile: Some(TuneProfile {
                kv_cache: KvCacheType::F16,
                flash_attn: true,
            }),
            baseline_live: None,
            best_live: None,
            long: None,
        }
    }

    #[test]
    fn qualification_requires_every_pair_and_the_hard_floor() {
        let mut proof = Qualification {
            context_size: 65536,
            signature: "config".into(),
            pairs: vec![(15.0, 10.0); 3],
        };
        assert!(proof.passed());
        proof.context_size -= 1;
        assert!(!proof.passed());
        proof.context_size += 1;
        proof.pairs[2].0 = 14.999;
        assert!(!proof.passed());
        proof.pairs.pop();
        assert!(!proof.passed());
    }

    fn long(ub: u32) -> LongTuned {
        LongTuned {
            gpu_layers: 39,
            ubatch_size: ub,
            prompt_tps: 600.0,
            gen_tps: 30.0,
            seconds: 15.4,
            baseline_seconds: 18.0,
            depth: 16_384,
            profile: Some(TuneProfile {
                kv_cache: KvCacheType::F16,
                flash_attn: true,
            }),
        }
    }

    #[test]
    fn a_file_written_before_deep_tuning_still_loads_and_has_no_long_setting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuned.json");
        std::fs::write(
            &path,
            br#"{"entries":{"k":{"gpu_layers":41,"ubatch_size":256,"prompt_tps":2234.7,"gen_tps":68.4,"seconds":5.6,"baseline_seconds":8.5,"measured_at":1}}}"#,
        )
        .unwrap();
        let got = TunedStore::open(&path).get("k").unwrap();
        let mut expected = entry(256);
        expected.profile = None;
        assert_eq!(got, expected);
        assert!(got.long.is_none());
        assert!(!got.matches(KvCacheType::F16, true));
        let saved = serde_json::to_string(&got).unwrap();
        assert!(!saved.contains("long"), "{saved}");
    }

    #[test]
    fn the_long_setting_is_saved_and_used_only_for_windows_at_least_its_depth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuned.json");
        let mut e = entry(256);
        e.long = Some(long(512));
        let mut store = TunedStore::open(&path);
        store.put("k", e).unwrap();
        let got = TunedStore::open(&path).get("k").unwrap();

        let short = got.setting_for(8_192);
        assert!(!short.long);
        assert_eq!((short.gpu_layers, short.ubatch_size), (41, 256));
        let at = got.setting_for(16_384);
        assert!(at.long);
        assert_eq!((at.gpu_layers, at.ubatch_size), (39, 512));
        assert!((at.baseline_seconds - 18.0).abs() < 1e-9);
        assert!(got.setting_for(131_072).long);
    }

    #[test]
    fn without_a_long_setting_every_window_uses_the_quick_tune() {
        let s = entry(256).setting_for(131_072);
        assert!(!s.long);
        assert_eq!(s.ubatch_size, 256);
    }

    #[test]
    fn profiles_match_the_cache_and_attention_mode_of_their_context_range() {
        let mut e = entry(256);
        e.long = Some(long(512));
        assert!(e.matches(KvCacheType::F16, true));
        assert!(!e.matches(KvCacheType::Q8_0, true));
        assert!(e.matches_long(KvCacheType::F16, true));
        assert!(!e.matches_long(KvCacheType::Q8_0, true));
        if let Some(deep) = e.long.as_mut() {
            deep.profile = Some(TuneProfile {
                kv_cache: KvCacheType::Q8_0,
                flash_attn: true,
            });
        }
        assert!(e.matches_long(KvCacheType::Q8_0, true));
        assert!(!e.matches_long(KvCacheType::F16, true));
    }

    #[test]
    fn incompatible_deep_profile_falls_back_to_compatible_quick_profile() {
        let mut e = entry(256);
        e.long = Some(long(512));
        e.long.as_mut().unwrap().profile.as_mut().unwrap().kv_cache = KvCacheType::Q8_0;
        let quick = e
            .compatible_setting(65_536, KvCacheType::F16, true)
            .unwrap();
        assert!(!quick.long);
        assert_eq!(quick.ubatch_size, 256);
        assert!(
            e.compatible_setting(65_536, KvCacheType::Q8_0, true)
                .unwrap()
                .long
        );
        assert!(e
            .compatible_setting(65_536, KvCacheType::Q4_0, true)
            .is_none());
    }

    #[test]
    fn reset_removes_all_runtime_builds_for_only_this_model_and_machine() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuned.json");
        let mut store = TunedStore::open(&path);
        for (machine, sha, version) in [
            ("pc", "sha", "v1"),
            ("pc", "sha", "v2"),
            ("pc", "other", "v1"),
            ("other", "sha", "v1"),
        ] {
            store
                .put(&key(machine, sha, "hip", version), entry(256))
                .unwrap();
        }
        assert_eq!(store.for_model("pc", "sha").len(), 2);
        assert!(store.remove_model("pc", "sha").unwrap());
        let reopened = TunedStore::open(&path);
        assert!(reopened.for_model("pc", "sha").is_empty());
        assert_eq!(reopened.for_model("pc", "other").len(), 1);
        assert_eq!(reopened.for_model("other", "sha").len(), 1);
    }

    #[test]
    fn results_survive_a_restart_and_are_keyed_by_runtime_build() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config").join("tuned.json");
        let k1 = key("m1", "sha", "llama-hip-win", "b12360");
        let k2 = key("m1", "sha", "llama-hip-win", "b12400");
        let mut store = TunedStore::open(&path);
        assert!(store.get(&k1).is_none());
        store.put(&k1, entry(256)).unwrap();

        let reopened = TunedStore::open(&path);
        assert_eq!(reopened.get(&k1), Some(entry(256)));
        assert!(
            reopened.get(&k2).is_none(),
            "a new runtime build is measured again"
        );
    }

    #[test]
    fn a_corrupt_file_is_an_empty_store_and_is_replaced_on_the_next_save() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuned.json");
        std::fs::write(&path, b"{ not json").unwrap();
        let mut store = TunedStore::open(&path);
        assert!(store.get("k").is_none());
        store.put("k", entry(512)).unwrap();
        assert_eq!(TunedStore::open(&path).get("k"), Some(entry(512)));
    }

    #[test]
    fn remove_forgets_one_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tuned.json");
        let mut store = TunedStore::open(&path);
        store.put("a", entry(256)).unwrap();
        store.put("b", entry(512)).unwrap();
        assert!(store.remove("a").unwrap());
        assert!(!store.remove("a").unwrap());
        let reopened = TunedStore::open(&path);
        assert!(reopened.get("a").is_none() && reopened.get("b").is_some());
    }
}
