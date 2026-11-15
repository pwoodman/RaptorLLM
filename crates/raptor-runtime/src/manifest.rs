//! Runtime manifests and runtime selection.
//!
//! A manifest describes one downloadable llama.cpp build. `select_runtime` picks
//! the best build for the detected hardware and never falls back to CPU silently:
//! a CPU selection always carries `degraded = true` and an explanation.

use crate::hardware::{GpuVendor, HardwareInfo, Os};
use crate::hash::is_sha256_hex;
use raptor_core::paths::safe_file_name;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};

/// Compute backend a llama.cpp build was compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// AMD ROCm/HIP.
    Hip,
    /// NVIDIA CUDA.
    Cuda,
    /// Vulkan (any vendor).
    Vulkan,
    /// CPU only.
    Cpu,
}

impl Backend {
    /// True for every backend that offloads to a GPU.
    pub fn is_gpu(self) -> bool {
        self != Backend::Cpu
    }

    /// Flash attention is only enabled for GPU builds here. Quantized KV
    /// cache requires it, so this also gates `kv_cache` choices.
    pub fn supports_flash_attn(self) -> bool {
        self.is_gpu()
    }

    /// Short lowercase name used in logs and UI.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Hip => "hip",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }
}

/// One downloadable runtime build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Stable runtime id, used as a directory name (e.g. "llama-hip-win").
    pub id: String,
    /// Version directory name (e.g. "b7100").
    pub version: String,
    pub llama_cpp_build: String,
    pub backend: Backend,
    pub os: Os,
    /// AMD gfx targets this build supports. Required for HIP builds.
    #[serde(default)]
    pub gfx_archs: Vec<String>,
    /// Archive URL. Must be https; the archive is hash-checked before extraction.
    pub url: String,
    /// Expected SHA-256 of the archive file.
    pub sha256: String,
    /// Minimum GPU driver version, dotted numeric (e.g. "31.0.0").
    #[serde(default)]
    pub min_driver: Option<String>,
    /// Reserved for detached signatures. Not verified yet (see README Status).
    #[serde(default)]
    pub signature: Option<String>,
}

/// Checks an id and version before they are used as path components.
pub fn validate_ids(id: &str, version: &str) -> Result<()> {
    safe_file_name(id)?;
    safe_file_name(version)?;
    Ok(())
}

impl Manifest {
    /// Validates every field that reaches the filesystem or a network request.
    pub fn validate(&self) -> Result<()> {
        validate_ids(&self.id, &self.version)?;
        if !self.url.starts_with("https://") {
            return Err(RaptorError::InvalidInput(format!(
                "runtime {} url must be https",
                self.id
            )));
        }
        if !is_sha256_hex(&self.sha256) {
            return Err(RaptorError::InvalidInput(format!(
                "runtime {} sha256 is not 64 hex chars",
                self.id
            )));
        }
        for gfx in &self.gfx_archs {
            let ok = gfx.starts_with("gfx")
                && gfx.len() > 3
                && gfx.bytes().all(|b| b.is_ascii_alphanumeric());
            if !ok {
                return Err(RaptorError::InvalidInput(format!(
                    "bad gfx target {gfx:?} in {}",
                    self.id
                )));
            }
        }
        if self.backend == Backend::Hip && self.gfx_archs.is_empty() {
            return Err(RaptorError::InvalidInput(format!(
                "HIP runtime {} lists no gfx archs",
                self.id
            )));
        }
        if let Some(min) = &self.min_driver {
            if version_key(min).is_empty() {
                return Err(RaptorError::InvalidInput(format!(
                    "bad min_driver {min:?} in {}",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct ManifestIndex {
    schema_version: u32,
    runtimes: Vec<Manifest>,
}

/// Parses and validates a runtime index file (`{"schema_version":1,"runtimes":[...]}`).
pub fn parse_manifest_index(json: &str) -> Result<Vec<Manifest>> {
    let index: ManifestIndex = serde_json::from_str(json)
        .map_err(|e| RaptorError::InvalidInput(format!("bad runtime index: {e}")))?;
    if index.schema_version != 1 {
        return Err(RaptorError::Unsupported(format!(
            "runtime index schema {} is not supported",
            index.schema_version
        )));
    }
    for m in &index.runtimes {
        m.validate()?;
    }
    for (i, a) in index.runtimes.iter().enumerate() {
        if index.runtimes[..i]
            .iter()
            .any(|b| b.id == a.id && b.version == a.version)
        {
            return Err(RaptorError::InvalidInput(format!(
                "duplicate runtime {} {}",
                a.id, a.version
            )));
        }
    }
    Ok(index.runtimes)
}

/// Splits a version string into numeric components so "b6900" < "b7100"
/// and "31.0.10" > "31.0.9".
pub fn version_key(v: &str) -> Vec<u64> {
    v.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>().unwrap_or(u64::MAX))
        .collect()
}

/// Result of runtime selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Selection {
    pub manifest: Manifest,
    pub backend: Backend,
    /// The gfx target that matched, for HIP selections.
    pub gpu_arch: Option<String>,
    /// Human-readable explanation of the choice.
    pub reason: String,
    /// True when no GPU runtime is compatible and a CPU runtime was chosen.
    pub degraded: bool,
    /// Candidates that were rejected, each with its reason.
    pub rejected: Vec<String>,
}

fn backend_rank(b: Backend) -> u8 {
    match b {
        Backend::Hip | Backend::Cuda => 3,
        Backend::Vulkan => 1,
        Backend::Cpu => 0,
    }
}

/// Returns Ok((matched_arch, note)) when `m` can run on `hw`, or Err(reason).
fn compatibility(
    hw: &HardwareInfo,
    m: &Manifest,
) -> std::result::Result<(Option<String>, String), String> {
    let driver_note = |gpu_driver: Option<&String>| -> std::result::Result<String, String> {
        match (&m.min_driver, gpu_driver) {
            (None, _) => Ok(String::new()),
            (Some(min), Some(have)) => {
                if version_key(have) < version_key(min) {
                    Err(format!("driver {have} is older than required {min}"))
                } else {
                    Ok(String::new())
                }
            }
            (Some(min), None) => Ok(format!(
                "; driver version unknown, minimum {min} not verified"
            )),
        }
    };
    match m.backend {
        Backend::Hip => {
            let amd: Vec<_> = hw
                .gpus
                .iter()
                .filter(|g| g.vendor == GpuVendor::Amd)
                .collect();
            if amd.is_empty() {
                return Err("no AMD GPU detected".into());
            }
            for g in &amd {
                if let Some(arch) = &g.arch {
                    if m.gfx_archs.iter().any(|a| a == arch) {
                        let note = driver_note(g.driver.as_ref())?;
                        return Ok((Some(arch.clone()), format!("{} is {arch}{note}", g.name)));
                    }
                }
            }
            let found: Vec<String> = amd
                .iter()
                .map(|g| {
                    format!(
                        "{} ({})",
                        g.name,
                        g.arch.as_deref().unwrap_or("arch unknown")
                    )
                })
                .collect();
            Err(format!(
                "needs {:?}, found {}",
                m.gfx_archs,
                found.join(", ")
            ))
        }
        Backend::Cuda => {
            let nv = hw
                .gpus
                .iter()
                .find(|g| g.vendor == GpuVendor::Nvidia)
                .ok_or("no NVIDIA GPU detected")?;
            let note = driver_note(nv.driver.as_ref())?;
            Ok((None, format!("{}{note}", nv.name)))
        }
        Backend::Vulkan => {
            let g = hw
                .gpus
                .iter()
                .filter(|g| g.vram_total > 0)
                .max_by_key(|g| g.vram_total)
                .ok_or("no GPU with reported VRAM detected")?;
            let note = driver_note(g.driver.as_ref())?;
            Ok((None, format!("{}{note}", g.name)))
        }
        Backend::Cpu => Ok((None, "CPU".into())),
    }
}

/// Chooses the best runtime for `hw` among `manifests` (matching the host OS).
///
/// Order: backend rank (HIP/CUDA > Vulkan), then highest version. If no GPU
/// runtime is compatible, the newest CPU runtime is returned with
/// `degraded = true`. Errors only if the host has no compatible build at all.
pub fn select_runtime(hw: &HardwareInfo, manifests: &[Manifest]) -> Result<Selection> {
    /// (backend rank, version key, manifest, matched arch, note)
    type Candidate<'a> = (u8, Vec<u64>, &'a Manifest, Option<String>, String);
    let mut rejected = Vec::new();
    let mut best: Option<Candidate<'_>> = None;
    let mut cpu_best: Option<(Vec<u64>, &Manifest)> = None;

    for m in manifests.iter().filter(|m| m.os == hw.os) {
        let label = format!("{}@{}", m.id, m.version);
        if m.backend == Backend::Cpu {
            let key = version_key(&m.version);
            if cpu_best.as_ref().is_none_or(|(k, _)| key > *k) {
                cpu_best = Some((key, m));
            }
            continue;
        }
        match compatibility(hw, m) {
            Ok((arch, note)) => {
                let cand = (
                    backend_rank(m.backend),
                    version_key(&m.version),
                    m,
                    arch,
                    note,
                );
                let better = match &best {
                    None => true,
                    Some((r, v, ..)) => (cand.0, &cand.1) > (*r, v),
                };
                if better {
                    best = Some(cand);
                }
            }
            Err(why) => rejected.push(format!("{label}: {why}")),
        }
    }

    if let Some((_, _, m, arch, note)) = best {
        let reason = format!(
            "{} {} ({}) selected for {}",
            m.id,
            m.version,
            m.backend.label(),
            note
        );
        return Ok(Selection {
            manifest: m.clone(),
            backend: m.backend,
            gpu_arch: arch,
            reason,
            degraded: false,
            rejected,
        });
    }

    let (_, cpu) = cpu_best.ok_or_else(|| {
        RaptorError::NotFound(format!(
            "no runtime for {:?} and no CPU fallback build",
            hw.os
        ))
    })?;
    let why = if rejected.is_empty() {
        "no GPU runtime available for this OS".to_string()
    } else {
        format!("no compatible GPU runtime ({})", rejected.join("; "))
    };
    Ok(Selection {
        manifest: cpu.clone(),
        backend: Backend::Cpu,
        gpu_arch: None,
        reason: format!(
            "DEGRADED: {why}. Using CPU runtime {}; inference will NOT be GPU accelerated.",
            cpu.version
        ),
        degraded: true,
        rejected,
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::hardware::{CpuInfo, GpuInfo};

    pub fn manifest(id: &str, version: &str, backend: Backend, gfx: &[&str]) -> Manifest {
        Manifest {
            id: id.into(),
            version: version.into(),
            llama_cpp_build: version.into(),
            backend,
            os: Os::Windows,
            gfx_archs: gfx.iter().map(|s| s.to_string()).collect(),
            url: format!("https://example.invalid/{id}-{version}.zip"),
            sha256: "a".repeat(64),
            min_driver: None,
            signature: None,
        }
    }

    pub fn gpu(
        vendor: GpuVendor,
        name: &str,
        arch: Option<&str>,
        vram_gib: u64,
        driver: Option<&str>,
    ) -> GpuInfo {
        GpuInfo {
            vendor,
            name: name.into(),
            vram_total: vram_gib * 1024 * 1024 * 1024,
            vram_free: None,
            arch: arch.map(str::to_string),
            driver: driver.map(str::to_string),
        }
    }

    pub fn hw(gpus: Vec<GpuInfo>) -> HardwareInfo {
        HardwareInfo {
            os: Os::Windows,
            cpu: CpuInfo {
                brand: "test cpu".into(),
                cores: 8,
                threads: 16,
                features: vec![],
            },
            ram_total: 32 << 30,
            ram_avail: 20 << 30,
            gpus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{gpu, hw, manifest};
    use super::*;
    use crate::hardware::GpuVendor;

    fn rx9060() -> HardwareInfo {
        hw(vec![gpu(
            GpuVendor::Amd,
            "AMD Radeon RX 9060 XT",
            Some("gfx1200"),
            16,
            Some("32.0.12019.1024"),
        )])
    }

    const INDEX: &str = r#"{
      "schema_version": 1,
      "runtimes": [
        {"id":"llama-hip-win","version":"b7100","llama_cpp_build":"b7100","backend":"hip","os":"windows",
         "gfx_archs":["gfx1200","gfx1201"],"url":"https://example.invalid/hip.zip","sha256":"ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12"},
        {"id":"llama-cpu-win","version":"b7100","llama_cpp_build":"b7100","backend":"cpu","os":"windows",
         "url":"https://example.invalid/cpu.zip","sha256":"cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34cd34"}
      ]
    }"#;

    #[test]
    fn index_parses_and_validates() {
        let ms = parse_manifest_index(INDEX).unwrap();
        assert_eq!(ms.len(), 2);
        assert_eq!(ms[0].backend, Backend::Hip);
        assert_eq!(ms[0].gfx_archs, vec!["gfx1200", "gfx1201"]);
    }

    #[test]
    fn index_rejects_unknown_schema_version() {
        let bad = INDEX.replace("\"schema_version\": 1", "\"schema_version\": 2");
        assert!(matches!(
            parse_manifest_index(&bad),
            Err(RaptorError::Unsupported(_))
        ));
    }

    #[test]
    fn index_rejects_http_urls() {
        let bad = INDEX.replace(
            "https://example.invalid/hip.zip",
            "http://example.invalid/hip.zip",
        );
        assert!(matches!(
            parse_manifest_index(&bad),
            Err(RaptorError::InvalidInput(_))
        ));
    }

    #[test]
    fn index_rejects_bad_sha256() {
        let bad = INDEX.replace(
            "ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12",
            "xyz",
        );
        assert!(parse_manifest_index(&bad).is_err());
    }

    #[test]
    fn index_rejects_path_traversal_ids() {
        let bad = INDEX.replace("\"id\":\"llama-cpu-win\"", "\"id\":\"../evil\"");
        assert!(parse_manifest_index(&bad).is_err());
    }

    #[test]
    fn index_rejects_duplicate_id_version() {
        let mut ms = parse_manifest_index(INDEX).unwrap();
        ms.push(ms[0].clone());
        let json = serde_json::json!({"schema_version": 1, "runtimes": ms}).to_string();
        assert!(parse_manifest_index(&json).is_err());
    }

    #[test]
    fn hip_without_gfx_archs_is_invalid() {
        let mut m = manifest("h", "b1", Backend::Hip, &[]);
        m.os = Os::Windows;
        assert!(m.validate().is_err());
    }

    #[test]
    fn selects_hip_for_rx_9060_xt() {
        let ms = vec![
            manifest(
                "llama-hip-win",
                "b7100",
                Backend::Hip,
                &["gfx1200", "gfx1201"],
            ),
            manifest("llama-cpu-win", "b7100", Backend::Cpu, &[]),
        ];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert_eq!(sel.backend, Backend::Hip);
        assert_eq!(sel.gpu_arch.as_deref(), Some("gfx1200"));
        assert!(!sel.degraded);
        assert!(sel.reason.contains("gfx1200"));
    }

    #[test]
    fn hip_build_for_other_arch_is_rejected_with_reason() {
        let ms = vec![
            manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1201"]),
            manifest("llama-cpu-win", "b7100", Backend::Cpu, &[]),
        ];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert!(sel.degraded);
        assert_eq!(sel.backend, Backend::Cpu);
        assert!(sel
            .rejected
            .iter()
            .any(|r| r.contains("needs") && r.contains("gfx1200")));
        assert!(sel.reason.contains("NOT be GPU accelerated"));
    }

    #[test]
    fn vulkan_is_chosen_when_hip_arch_does_not_match() {
        let ms = vec![
            manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1201"]),
            manifest("llama-vk-win", "b7100", Backend::Vulkan, &[]),
            manifest("llama-cpu-win", "b7100", Backend::Cpu, &[]),
        ];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert_eq!(sel.backend, Backend::Vulkan);
        assert!(!sel.degraded);
    }

    #[test]
    fn cuda_is_chosen_for_nvidia() {
        let hw = hw(vec![gpu(
            GpuVendor::Nvidia,
            "GeForce RTX 4090",
            None,
            24,
            Some("560.0.0"),
        )]);
        let ms = vec![
            manifest("llama-cuda-win", "b7100", Backend::Cuda, &[]),
            manifest("llama-cpu-win", "b7100", Backend::Cpu, &[]),
        ];
        let sel = select_runtime(&hw, &ms).unwrap();
        assert_eq!(sel.backend, Backend::Cuda);
    }

    #[test]
    fn driver_below_minimum_is_rejected() {
        let mut m = manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1200"]);
        m.min_driver = Some("33.0.0".into());
        let ms = vec![m, manifest("llama-cpu-win", "b7100", Backend::Cpu, &[])];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert!(sel.degraded);
        assert!(sel
            .rejected
            .iter()
            .any(|r| r.contains("older than required 33.0.0")));
    }

    #[test]
    fn unknown_driver_is_selected_with_visible_note() {
        let mut hw = rx9060();
        hw.gpus[0].driver = None;
        let mut m = manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1200"]);
        m.min_driver = Some("33.0.0".into());
        let sel = select_runtime(
            &hw,
            &[m, manifest("llama-cpu-win", "b7100", Backend::Cpu, &[])],
        )
        .unwrap();
        assert!(!sel.degraded);
        assert!(sel.reason.contains("not verified"));
    }

    #[test]
    fn newest_version_wins_within_backend() {
        let ms = vec![
            manifest("llama-hip-win", "b6900", Backend::Hip, &["gfx1200"]),
            manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1200"]),
            manifest("llama-hip-win", "b7099", Backend::Hip, &["gfx1200"]),
        ];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert_eq!(sel.manifest.version, "b7100");
    }

    #[test]
    fn no_gpu_selects_cpu_degraded_with_reason() {
        let cpu_only = hw(vec![]);
        let ms = vec![
            manifest("llama-hip-win", "b7100", Backend::Hip, &["gfx1200"]),
            manifest("llama-cpu-win", "b7100", Backend::Cpu, &[]),
        ];
        let sel = select_runtime(&cpu_only, &ms).unwrap();
        assert_eq!(sel.backend, Backend::Cpu);
        assert!(sel.degraded);
        assert!(sel.reason.contains("no AMD GPU"));
    }

    #[test]
    fn no_cpu_fallback_is_an_error() {
        let ms = vec![manifest(
            "llama-hip-win",
            "b7100",
            Backend::Hip,
            &["gfx1200"],
        )];
        assert!(matches!(
            select_runtime(&hw(vec![]), &ms),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[test]
    fn other_os_builds_are_ignored() {
        let mut linux = manifest("llama-hip-linux", "b7100", Backend::Hip, &["gfx1200"]);
        linux.os = Os::Linux;
        let ms = vec![linux, manifest("llama-cpu-win", "b7100", Backend::Cpu, &[])];
        let sel = select_runtime(&rx9060(), &ms).unwrap();
        assert_eq!(sel.manifest.id, "llama-cpu-win");
        assert!(sel.rejected.is_empty());
    }

    #[test]
    fn version_key_orders_numerically() {
        assert!(version_key("b6900") < version_key("b7100"));
        assert!(version_key("31.0.10") > version_key("31.0.9"));
        assert!(version_key("").is_empty());
    }
}
