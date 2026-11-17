//! Hardware model and detection.
//!
//! The parsers are pure functions over captured tool output (PowerShell JSON,
//! /proc text, `lspci -Dnn`, `rocminfo`, sysfs). `SystemProbe` only gathers the
//! text and hands it to them, so the logic is tested without real hardware.

use crate::sys::{powershell_encoded_args, CommandRunner};
use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Host operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Os {
    Windows,
    Linux,
    MacOs,
    Other,
}

impl Os {
    /// The OS this binary was compiled for.
    pub fn current() -> Self {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_os = "macos") {
            Os::MacOs
        } else {
            Os::Other
        }
    }
}

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Amd,
    Nvidia,
    Intel,
    Other,
}

impl GpuVendor {
    /// Maps a PCI vendor id (e.g. `0x1002` from sysfs) to a vendor.
    pub fn from_pci_id(id: u16) -> Self {
        match id {
            0x1002 => GpuVendor::Amd,
            0x10de => GpuVendor::Nvidia,
            0x8086 => GpuVendor::Intel,
            _ => GpuVendor::Other,
        }
    }

    /// Maps a human-readable provider or device name. Deliberately does not
    /// match "ati": it appears inside "Corporation" and would misclassify vendors.
    pub fn from_name(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        if n.contains("amd") || n.contains("advanced micro") || n.contains("radeon") {
            GpuVendor::Amd
        } else if n.contains("nvidia") || n.contains("geforce") {
            GpuVendor::Nvidia
        } else if n.contains("intel") {
            GpuVendor::Intel
        } else {
            GpuVendor::Other
        }
    }
}

/// CPU summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub cores: u32,
    pub threads: u32,
    /// Notable SIMD features detected at runtime (e.g. "avx2", "avx512f").
    pub features: Vec<String>,
}

/// One GPU. `vram_free` is `None` when the platform cannot report it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: Option<u64>,
    /// AMD gfx target (e.g. "gfx1200"). Name-derived on Windows; from rocminfo on Linux.
    pub arch: Option<String>,
    pub driver: Option<String>,
}

/// Full hardware snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: Os,
    pub cpu: CpuInfo,
    pub ram_total: u64,
    pub ram_avail: u64,
    pub gpus: Vec<GpuInfo>,
}

impl HardwareInfo {
    /// The GPU with the most VRAM, optionally restricted to one vendor.
    pub fn gpu_of(&self, vendor: Option<GpuVendor>) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .filter(|g| vendor.is_none_or(|v| g.vendor == v))
            .max_by_key(|g| g.vram_total)
    }
}

/// Best-effort map from an AMD marketing name to its gfx target. Only used when
/// the platform does not report the arch itself.
pub fn amd_arch_from_name(name: &str) -> Option<&'static str> {
    const TABLE: &[(&str, &str)] = &[
        ("RX 9070", "gfx1201"), // 9070 / 9070 XT (Navi 48)
        ("RX 9060", "gfx1200"), // 9060 / 9060 XT (Navi 44)
        ("RX 7900", "gfx1100"), // 7900 XTX / XT / GRE (Navi 31)
        ("RX 7800", "gfx1101"), // Navi 32
        ("RX 7700", "gfx1101"), // Navi 32
        ("RX 7600", "gfx1102"), // Navi 33
        ("RX 6950", "gfx1030"), // Navi 21 refresh
        ("RX 6900", "gfx1030"), // Navi 21
        ("RX 6800", "gfx1030"), // Navi 21
        ("RX 6750", "gfx1031"), // Navi 22 refresh
        ("RX 6700", "gfx1031"), // Navi 22
        ("RX 6650", "gfx1032"), // Navi 23 refresh
        ("RX 6600", "gfx1032"), // Navi 23
        ("RX 6500", "gfx1034"), // Navi 24
        ("RX 6400", "gfx1034"), // Navi 24
    ];
    let upper = name.to_ascii_uppercase();
    TABLE
        .iter()
        .find(|(needle, _)| upper.contains(needle))
        .map(|(_, arch)| *arch)
}

/// Pulls the hardware snapshot. Behind a trait so tests use [`FakeProbe`]-style fakes.
#[async_trait]
pub trait HardwareProbe: Send + Sync {
    async fn probe(&self) -> Result<HardwareInfo>;
}

/// PowerShell script for Windows.
///
/// Two sources are read and joined in Rust:
/// - `live_gpus`: `Win32_VideoController`, the adapters that are actually present.
/// - `registry_gpus`: display-class registry keys, used only for VRAM because
///   `AdapterRAM` is a 32-bit field and reports 4 GiB for larger cards.
///
/// Registry keys can outlive a removed card, so the registry is never reported on its own.
pub const WINDOWS_PROBE_SCRIPT: &str = r#"
$ErrorActionPreference = 'SilentlyContinue'
$cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$live = @()
foreach ($v in (Get-CimInstance -ClassName Win32_VideoController)) {
  $live += [pscustomobject]@{ name = [string]$v.Name; pnp = [string]$v.PNPDeviceID; driver = [string]$v.DriverVersion; adapter_ram = [uint64]$v.AdapterRAM }
}
$reg = @()
$classKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
foreach ($k in (Get-ChildItem -Path $classKey)) {
  $p = Get-ItemProperty -Path $k.PSPath
  if (-not $p.DriverDesc) { continue }
  $mem = $p.'HardwareInformation.qwMemorySize'
  if ($mem -is [byte[]]) { $mem = [BitConverter]::ToUInt64($mem, 0) }
  $reg += [pscustomobject]@{ name = [string]$p.DriverDesc; provider = [string]$p.ProviderName; driver = [string]$p.DriverVersion; vram_bytes = [uint64]$mem; matching = [string]$p.MatchingDeviceId }
}
[pscustomobject]@{
  cpu = [pscustomobject]@{ name = [string]$cpu.Name; cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors }
  ram_total_kb = $os.TotalVisibleMemorySize
  ram_free_kb = $os.FreePhysicalMemory
  live_gpus = @($live)
  registry_gpus = @($reg)
} | ConvertTo-Json -Depth 4 -Compress
"#;

#[derive(Deserialize)]
struct WinProbe {
    cpu: Option<WinCpu>,
    ram_total_kb: Option<u64>,
    ram_free_kb: Option<u64>,
    #[serde(default)]
    live_gpus: Vec<WinLiveGpu>,
    #[serde(default)]
    registry_gpus: Vec<WinRegGpu>,
}

#[derive(Deserialize)]
struct WinCpu {
    name: Option<String>,
    cores: Option<u32>,
    threads: Option<u32>,
}

/// An adapter currently present (Win32_VideoController).
#[derive(Deserialize)]
struct WinLiveGpu {
    name: String,
    pnp: Option<String>,
    driver: Option<String>,
    adapter_ram: Option<u64>,
}

/// A display-class registry key. May be stale (card removed).
#[derive(Deserialize)]
struct WinRegGpu {
    name: String,
    provider: Option<String>,
    driver: Option<String>,
    vram_bytes: Option<u64>,
    matching: Option<String>,
}

/// Joins live adapters with registry entries.
///
/// Only live adapters are reported. A registry entry supplies VRAM only when
/// its name matches and, where possible, its MatchingDeviceId appears in the
/// live PNP id. Each registry entry is used at most once. Stale registry keys
/// (cards no longer present) are dropped. A live adapter without a registry
/// match falls back to `AdapterRAM`.
fn join_windows_gpus(live: &[WinLiveGpu], registry: &[WinRegGpu]) -> Vec<GpuInfo> {
    let mut used = vec![false; registry.len()];
    let mut seen_pnp = std::collections::HashSet::new();
    let mut out = Vec::new();
    for l in live {
        let name = l.name.trim();
        if name.is_empty() || name.contains("Basic Display") {
            continue;
        }
        let pnp = l.pnp.clone().unwrap_or_default().to_ascii_uppercase();
        if !pnp.is_empty() && !seen_pnp.insert(pnp.clone()) {
            continue;
        }
        let same_name: Vec<usize> = registry
            .iter()
            .enumerate()
            .filter(|(i, r)| !used[*i] && r.name.trim().eq_ignore_ascii_case(name))
            .map(|(i, _)| i)
            .collect();
        let chosen = same_name
            .iter()
            .copied()
            .find(|&i| {
                let m = registry[i]
                    .matching
                    .clone()
                    .unwrap_or_default()
                    .to_ascii_uppercase();
                !m.is_empty() && pnp.contains(&m)
            })
            .or_else(|| same_name.first().copied());
        let reg = chosen.map(|i| {
            used[i] = true;
            &registry[i]
        });
        let vram_total = reg
            .and_then(|r| r.vram_bytes)
            .filter(|v| *v > 0)
            .or(l.adapter_ram)
            .unwrap_or(0);
        let vendor = GpuVendor::from_name(&format!(
            "{} {}",
            reg.and_then(|r| r.provider.as_deref()).unwrap_or(""),
            name
        ));
        let driver = l
            .driver
            .clone()
            .or_else(|| reg.and_then(|r| r.driver.clone()))
            .filter(|d| !d.trim().is_empty());
        out.push(GpuInfo {
            vendor,
            arch: if vendor == GpuVendor::Amd {
                amd_arch_from_name(name).map(str::to_string)
            } else {
                None
            },
            name: name.to_string(),
            vram_total,
            vram_free: None,
            driver,
        });
    }
    out
}

/// Parses the JSON emitted by [`WINDOWS_PROBE_SCRIPT`]. CPU features are not
/// included here; the real probe fills them in.
pub fn parse_windows_probe(json: &str) -> Result<HardwareInfo> {
    let raw: WinProbe = serde_json::from_str(json.trim())
        .map_err(|e| RaptorError::Runtime(format!("unreadable Windows probe output: {e}")))?;
    let cpu = raw.cpu.unwrap_or(WinCpu {
        name: None,
        cores: None,
        threads: None,
    });
    let threads = cpu.threads.unwrap_or(1).max(1);
    let gpus = join_windows_gpus(&raw.live_gpus, &raw.registry_gpus);
    Ok(HardwareInfo {
        os: Os::Windows,
        cpu: CpuInfo {
            brand: cpu.name.unwrap_or_default().trim().to_string(),
            cores: cpu.cores.unwrap_or(threads).max(1),
            threads,
            features: Vec::new(),
        },
        ram_total: raw.ram_total_kb.unwrap_or(0).saturating_mul(1024),
        ram_avail: raw.ram_free_kb.unwrap_or(0).saturating_mul(1024),
        gpus,
    })
}

/// Parses `/proc/cpuinfo`: brand, logical threads, and physical cores
/// (unique physical-id/core-id pairs).
pub fn parse_cpuinfo(text: &str) -> (String, u32, u32) {
    let mut brand = String::new();
    let mut threads = 0u32;
    let mut cores = std::collections::HashSet::new();
    let (mut phys, mut core): (Option<String>, Option<String>) = (None, None);
    let flush = |phys: &mut Option<String>,
                 core: &mut Option<String>,
                 cores: &mut std::collections::HashSet<(String, String)>| {
        if let (Some(p), Some(c)) = (phys.take(), core.take()) {
            cores.insert((p, c));
        }
    };
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            flush(&mut phys, &mut core, &mut cores);
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "processor" => threads += 1,
            "model name" if brand.is_empty() => brand = value.to_string(),
            "physical id" => phys = Some(value.to_string()),
            "core id" => core = Some(value.to_string()),
            _ => {}
        }
    }
    flush(&mut phys, &mut core, &mut cores);
    let physical = if cores.is_empty() {
        threads
    } else {
        cores.len() as u32
    };
    (brand, physical.max(1), threads.max(1))
}

/// Parses `/proc/meminfo`. Returns (total, available) in bytes.
pub fn parse_meminfo(text: &str) -> Result<(u64, u64)> {
    let field = |name: &str| -> Option<u64> {
        text.lines().find_map(|l| {
            let rest = l.strip_prefix(name)?.strip_prefix(':')?;
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            Some(kb * 1024)
        })
    };
    match (field("MemTotal"), field("MemAvailable")) {
        (Some(total), Some(avail)) => Ok((total, avail)),
        _ => Err(RaptorError::Runtime(
            "meminfo lacks MemTotal/MemAvailable".into(),
        )),
    }
}

/// One GPU as listed by `lspci -Dnn`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspciGpu {
    /// Full PCI address, e.g. "0000:03:00.0".
    pub slot: String,
    pub vendor_id: u16,
    pub device_id: u16,
    /// Marketing name, e.g. "Radeon RX 9060 XT" (last bracketed group).
    pub name: String,
}

/// Parses `lspci -Dnn` output, keeping only display-class devices.
///
/// Example line:
/// `0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 44 [Radeon RX 9060 XT] [1002:7590] (rev c0)`
pub fn parse_lspci(text: &str) -> Vec<LspciGpu> {
    text.lines().filter_map(parse_lspci_line).collect()
}

fn parse_lspci_line(line: &str) -> Option<LspciGpu> {
    let (slot, rest) = line.split_once(' ')?;
    let is_display = [
        "VGA compatible controller",
        "3D controller",
        "Display controller",
    ]
    .iter()
    .any(|c| rest.contains(c));
    if !is_display {
        return None;
    }
    let (_, description) = rest.split_once("]: ")?;
    let groups = bracket_groups(description);
    let (vendor_id, device_id) = groups
        .iter()
        .find_map(|(_, content)| parse_pci_pair(content))?;
    let name = groups
        .iter()
        .filter(|(_, content)| parse_pci_pair(content).is_none())
        .map(|(_, content)| content.clone())
        .next_back()
        .unwrap_or_else(|| {
            description
                .split(" [")
                .next()
                .unwrap_or(description)
                .trim()
                .to_string()
        });
    Some(LspciGpu {
        slot: slot.to_string(),
        vendor_id,
        device_id,
        name,
    })
}

/// Returns each `[...]` group with its content, in order.
fn bracket_groups(s: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in s.char_indices() {
        match c {
            '[' => start = Some(i),
            ']' => {
                if let Some(st) = start.take() {
                    out.push((st, s[st + 1..i].to_string()));
                }
            }
            _ => {}
        }
    }
    out
}

/// Parses "1002:7590" into (vendor, device).
fn parse_pci_pair(s: &str) -> Option<(u16, u16)> {
    let (v, d) = s.split_once(':')?;
    if v.len() != 4 || d.len() != 4 {
        return None;
    }
    Some((
        u16::from_str_radix(v, 16).ok()?,
        u16::from_str_radix(d, 16).ok()?,
    ))
}

/// An AMD GPU agent reported by `rocminfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmAgent {
    /// gfx target, e.g. "gfx1200".
    pub gfx: String,
    /// e.g. "AMD Radeon RX 9060 XT".
    pub marketing_name: String,
}

/// Parses `rocminfo` output and returns only GPU agents with a gfx target.
pub fn parse_rocminfo(text: &str) -> Vec<RocmAgent> {
    let mut agents = Vec::new();
    let mut cur: Option<(Option<String>, Option<String>, bool)> = None;
    let flush = |cur: &mut Option<(Option<String>, Option<String>, bool)>,
                 agents: &mut Vec<RocmAgent>| {
        if let Some((Some(gfx), marketing, true)) = cur.take() {
            agents.push(RocmAgent {
                gfx,
                marketing_name: marketing.unwrap_or_default(),
            });
        }
    };
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with("Agent ") && line[6..].trim().chars().all(|c| c.is_ascii_digit()) {
            flush(&mut cur, &mut agents);
            cur = Some((None, None, false));
            continue;
        }
        let Some(state) = cur.as_mut() else { continue };
        if let Some(v) = line.strip_prefix("Name:") {
            let v = v.trim();
            if v.starts_with("gfx") && state.0.is_none() {
                state.0 = Some(v.to_string());
            }
        } else if let Some(v) = line.strip_prefix("Marketing Name:") {
            state.1 = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Device Type:") {
            state.2 = v.trim() == "GPU";
        }
    }
    flush(&mut cur, &mut agents);
    agents
}

/// One DRM card from sysfs (`/sys/class/drm/cardN/device`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrmCard {
    /// PCI address from the device `uevent` (PCI_SLOT_NAME).
    pub slot: Option<String>,
    pub vendor_id: u16,
    pub vram_total: Option<u64>,
    pub vram_used: Option<u64>,
}

/// Extracts `PCI_SLOT_NAME=...` from a device `uevent` file.
pub fn parse_uevent_slot(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("PCI_SLOT_NAME="))
        .map(|s| s.trim().to_string())
}

/// Parses a sysfs hex id such as "0x1002".
pub fn parse_sysfs_hex(text: &str) -> Option<u16> {
    u16::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
}

/// Combines `lspci`, `rocminfo` and sysfs data into GPU records.
///
/// `lspci` is the device list. sysfs contributes VRAM (joined by PCI slot).
/// rocminfo contributes the gfx target for AMD GPUs, matched by marketing name,
/// or the only AMD agent if there is exactly one. Otherwise the name table is used.
pub fn build_linux_gpus(lspci: &[LspciGpu], drm: &[DrmCard], rocm: &[RocmAgent]) -> Vec<GpuInfo> {
    lspci
        .iter()
        .map(|g| {
            let vendor = GpuVendor::from_pci_id(g.vendor_id);
            let card = drm
                .iter()
                .find(|c| c.slot.as_deref() == Some(g.slot.as_str()));
            let vram_total = card.and_then(|c| c.vram_total).unwrap_or(0);
            let vram_free = card.and_then(|c| Some(c.vram_total?.saturating_sub(c.vram_used?)));
            let display_name = match vendor {
                GpuVendor::Amd => format!("AMD {}", g.name),
                GpuVendor::Nvidia => format!("NVIDIA {}", g.name),
                GpuVendor::Intel => format!("Intel {}", g.name),
                GpuVendor::Other => g.name.clone(),
            };
            let arch = if vendor == GpuVendor::Amd {
                let lower = display_name.to_ascii_lowercase();
                rocm.iter()
                    .find(|a| {
                        !a.marketing_name.is_empty()
                            && lower.contains(&a.marketing_name.to_ascii_lowercase())
                    })
                    .or_else(|| if rocm.len() == 1 { rocm.first() } else { None })
                    .map(|a| a.gfx.clone())
                    .or_else(|| amd_arch_from_name(&display_name).map(str::to_string))
            } else {
                None
            };
            GpuInfo {
                vendor,
                name: display_name,
                vram_total,
                vram_free,
                arch,
                driver: None,
            }
        })
        .collect()
}

/// Runtime-detected CPU SIMD features relevant to llama.cpp kernels.
pub fn detect_cpu_features() -> Vec<String> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut f = Vec::new();
        if std::is_x86_feature_detected!("avx") {
            f.push("avx".to_string());
        }
        if std::is_x86_feature_detected!("avx2") {
            f.push("avx2".to_string());
        }
        if std::is_x86_feature_detected!("avx512f") {
            f.push("avx512f".to_string());
        }
        if std::is_x86_feature_detected!("fma") {
            f.push("fma".to_string());
        }
        if std::is_x86_feature_detected!("f16c") {
            f.push("f16c".to_string());
        }
        f
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

/// Real hardware probe. Windows uses one PowerShell call; Linux reads /proc and
/// /sys directly and uses `lspci` and `rocminfo` when present.
pub struct SystemProbe {
    runner: Arc<dyn CommandRunner>,
}

impl SystemProbe {
    /// Creates a probe that runs helper tools through `runner`.
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self { runner }
    }

    async fn probe_windows(&self) -> Result<HardwareInfo> {
        let args = powershell_encoded_args(WINDOWS_PROBE_SCRIPT);
        let out = self
            .runner
            .run("powershell.exe", &args, Duration::from_secs(30))
            .await?;
        parse_windows_probe(&out)
    }

    async fn probe_linux(&self) -> Result<HardwareInfo> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")?;
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let (brand, cores, threads) = parse_cpuinfo(&cpuinfo);
        let (ram_total, ram_avail) = parse_meminfo(&meminfo)?;

        let lspci = match self
            .runner
            .run("lspci", &["-Dnn".to_string()], Duration::from_secs(10))
            .await
        {
            Ok(out) => parse_lspci(&out),
            Err(_) => Vec::new(), // Missing lspci means no GPU list; the CPU-only path still reports honestly.
        };
        let rocm = match self
            .runner
            .run("rocminfo", &[], Duration::from_secs(20))
            .await
        {
            Ok(out) => parse_rocminfo(&out),
            Err(_) => Vec::new(),
        };
        let drm = read_drm_cards(Path::new("/sys/class/drm"));
        Ok(HardwareInfo {
            os: Os::Linux,
            cpu: CpuInfo {
                brand,
                cores,
                threads,
                features: detect_cpu_features(),
            },
            ram_total,
            ram_avail,
            gpus: build_linux_gpus(&lspci, &drm, &rocm),
        })
    }
}

fn read_drm_cards(root: &Path) -> Vec<DrmCard> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut cards = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Connector nodes (card0-DP-1) are not devices; only "cardN" is.
        let is_card = name
            .strip_prefix("card")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
        if !is_card {
            continue;
        }
        let dev = entry.path().join("device");
        let read = |f: &str| std::fs::read_to_string(dev.join(f)).ok();
        let Some(vendor_id) = read("vendor").as_deref().and_then(parse_sysfs_hex) else {
            continue;
        };
        let num = |f: &str| read(f).and_then(|s| s.trim().parse::<u64>().ok());
        cards.push(DrmCard {
            slot: read("uevent").as_deref().and_then(parse_uevent_slot),
            vendor_id,
            vram_total: num("mem_info_vram_total"),
            vram_used: num("mem_info_vram_used"),
        });
    }
    cards
}

#[async_trait]
impl HardwareProbe for SystemProbe {
    async fn probe(&self) -> Result<HardwareInfo> {
        match Os::current() {
            Os::Windows => {
                let mut hw = self.probe_windows().await?;
                hw.cpu.features = detect_cpu_features();
                Ok(hw)
            }
            Os::Linux => self.probe_linux().await,
            other => Err(RaptorError::Unsupported(format!(
                "hardware probe for {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN_RX9060: &str = r#"{"cpu":{"name":"AMD Ryzen 7 7800X3D 8-Core Processor","cores":8,"threads":16},"ram_total_kb":33554432,"ram_free_kb":20971520,"live_gpus":[{"name":"AMD Radeon RX 9060 XT","pnp":"PCI\\VEN_1002&DEV_7590&SUBSYS_00000000&REV_C0\\4&1A2B3C&0&0008","driver":"32.0.12019.1024","adapter_ram":4293918720}],"registry_gpus":[{"name":"AMD Radeon RX 9060 XT","provider":"Advanced Micro Devices, Inc.","driver":"32.0.12019.1024","vram_bytes":17163091968,"matching":"PCI\\VEN_1002&DEV_7590&SUBSYS_00000000&REV_C0"}]}"#;
    const WIN_MULTI: &str = r#"{"cpu":{"name":"Intel Core i7-13700H","cores":14,"threads":20},"ram_total_kb":16777216,"ram_free_kb":8388608,"live_gpus":[{"name":"Intel(R) Iris(R) Xe Graphics","pnp":"PCI\\VEN_8086&DEV_A7A0","driver":"31.0.101.5186","adapter_ram":134217728},{"name":"NVIDIA GeForce RTX 4070 Laptop GPU","pnp":"PCI\\VEN_10DE&DEV_2820","driver":"31.0.15.4601","adapter_ram":4293918720},{"name":"Microsoft Basic Display Adapter","pnp":"ROOT\\BASICDISPLAY\\0000","driver":"10.0.26100.1","adapter_ram":0}],"registry_gpus":[{"name":"Intel(R) Iris(R) Xe Graphics","provider":"Intel Corporation","driver":"31.0.101.5186","vram_bytes":134217728,"matching":"PCI\\VEN_8086&DEV_A7A0"},{"name":"NVIDIA GeForce RTX 4070 Laptop GPU","provider":"NVIDIA","driver":"31.0.15.4601","vram_bytes":8589934592,"matching":"PCI\\VEN_10DE&DEV_2820"}]}"#;
    const WIN_CPU_ONLY: &str = r#"{"cpu":{"name":"Intel Xeon","cores":4,"threads":8},"ram_total_kb":8388608,"ram_free_kb":4194304,"live_gpus":[],"registry_gpus":[]}"#;
    const WIN_BYTE_VRAM: &str = r#"{"cpu":null,"ram_total_kb":1048576,"ram_free_kb":524288,"live_gpus":[{"name":"AMD Radeon RX 7900 XTX","pnp":"PCI\\VEN_1002&DEV_744C","driver":null,"adapter_ram":4293918720}],"registry_gpus":[{"name":"AMD Radeon RX 7900 XTX","provider":"Advanced Micro Devices, Inc.","driver":null,"vram_bytes":25769803776,"matching":null}]}"#;
    /// Two stale registry keys for a removed RX 580 plus one live RX 9060 XT.
    /// Only the live card may be reported.
    const WIN_STALE_REGISTRY: &str = r#"{"cpu":{"name":"AMD Ryzen 7 3800X 8-Core Processor","cores":8,"threads":16},"ram_total_kb":67108864,"ram_free_kb":56623104,"live_gpus":[{"name":"AMD Radeon RX 9060 XT","pnp":"PCI\\VEN_1002&DEV_7590&SUBSYS_00000000&REV_C0\\4&1A2B3C&0&0008","driver":"32.0.31044.16","adapter_ram":4293918720}],"registry_gpus":[{"name":"AMD Radeon RX 580 2048SP","provider":"Advanced Micro Devices, Inc.","driver":"31.0.21925.1001","vram_bytes":8589934592,"matching":"PCI\\VEN_1002&DEV_67DF&SUBSYS_E387148C"},{"name":"AMD Radeon RX 580 2048SP","provider":"Advanced Micro Devices, Inc.","driver":"31.0.21925.1001","vram_bytes":8589934592,"matching":"PCI\\VEN_1002&DEV_67DF&SUBSYS_E387148C"},{"name":"AMD Radeon RX 9060 XT","provider":"Advanced Micro Devices, Inc.","driver":"32.0.31044.16","vram_bytes":17095983104,"matching":"PCI\\VEN_1002&DEV_7590&SUBSYS_00000000&REV_C0"}]}"#;

    #[test]
    fn stale_registry_keys_are_not_reported_as_gpus() {
        let hw = parse_windows_probe(WIN_STALE_REGISTRY).unwrap();
        assert_eq!(hw.gpus.len(), 1, "{:?}", hw.gpus);
        let g = &hw.gpus[0];
        assert_eq!(g.name, "AMD Radeon RX 9060 XT");
        assert_eq!(g.arch.as_deref(), Some("gfx1200"));
        assert_eq!(
            g.vram_total, 17_095_983_104,
            "VRAM comes from the registry, not the 4 GiB AdapterRAM"
        );
    }

    #[test]
    fn registry_only_without_live_controller_reports_nothing() {
        let json = r#"{"cpu":null,"ram_total_kb":1,"ram_free_kb":1,"live_gpus":[],"registry_gpus":[{"name":"AMD Radeon RX 580 2048SP","provider":"AMD","driver":"x","vram_bytes":8589934592,"matching":"PCI\\VEN_1002&DEV_67DF"}]}"#;
        assert!(parse_windows_probe(json).unwrap().gpus.is_empty());
    }

    #[test]
    fn duplicate_live_name_prefers_matching_registry_entry_and_dedupes() {
        let json = r#"{"cpu":null,"ram_total_kb":1,"ram_free_kb":1,"live_gpus":[{"name":"AMD Radeon RX 9060 XT","pnp":"PCI\\VEN_1002&DEV_7590&X","driver":"1","adapter_ram":0},{"name":"AMD Radeon RX 9060 XT","pnp":"PCI\\VEN_1002&DEV_7590&X","driver":"1","adapter_ram":0}],"registry_gpus":[{"name":"AMD Radeon RX 9060 XT","provider":"AMD","driver":"1","vram_bytes":100,"matching":"PCI\\VEN_1002&DEV_1111"},{"name":"AMD Radeon RX 9060 XT","provider":"AMD","driver":"1","vram_bytes":200,"matching":"PCI\\VEN_1002&DEV_7590"}]}"#;
        let hw = parse_windows_probe(json).unwrap();
        assert_eq!(hw.gpus.len(), 1, "same PNP id reported twice must dedupe");
        assert_eq!(hw.gpus[0].vram_total, 200);
    }

    #[test]
    fn windows_probe_identifies_rx_9060_xt_gfx1200() {
        let hw = parse_windows_probe(WIN_RX9060).unwrap();
        assert_eq!(hw.os, Os::Windows);
        assert_eq!(hw.gpus.len(), 1);
        let g = &hw.gpus[0];
        assert_eq!(g.vendor, GpuVendor::Amd);
        assert_eq!(g.arch.as_deref(), Some("gfx1200"));
        assert_eq!(g.vram_total, 17_163_091_968);
        assert_eq!(g.vram_free, None);
        assert_eq!(g.driver.as_deref(), Some("32.0.12019.1024"));
    }

    #[test]
    fn windows_probe_converts_ram_and_cpu() {
        let hw = parse_windows_probe(WIN_RX9060).unwrap();
        assert_eq!(hw.ram_total, 32 * 1024 * 1024 * 1024);
        assert_eq!(hw.ram_avail, 20_971_520 * 1024);
        assert_eq!(hw.cpu.cores, 8);
        assert_eq!(hw.cpu.threads, 16);
        assert!(hw.cpu.brand.starts_with("AMD Ryzen 7"));
    }

    #[test]
    fn windows_probe_multi_vendor_drops_basic_adapter() {
        let hw = parse_windows_probe(WIN_MULTI).unwrap();
        let vendors: Vec<_> = hw.gpus.iter().map(|g| g.vendor).collect();
        assert_eq!(vendors, vec![GpuVendor::Intel, GpuVendor::Nvidia]);
        assert_eq!(
            hw.gpu_of(Some(GpuVendor::Nvidia)).unwrap().vram_total,
            8_589_934_592
        );
        assert_eq!(hw.gpu_of(None).unwrap().vendor, GpuVendor::Nvidia);
        assert!(hw.gpu_of(Some(GpuVendor::Amd)).is_none());
    }

    #[test]
    fn windows_probe_cpu_only_has_no_gpus() {
        let hw = parse_windows_probe(WIN_CPU_ONLY).unwrap();
        assert!(hw.gpus.is_empty());
        assert_eq!(hw.cpu.cores, 4);
    }

    #[test]
    fn windows_probe_tolerates_null_cpu_and_null_driver() {
        let hw = parse_windows_probe(WIN_BYTE_VRAM).unwrap();
        assert_eq!(hw.cpu.threads, 1);
        assert_eq!(hw.gpus[0].arch.as_deref(), Some("gfx1100"));
        assert_eq!(hw.gpus[0].driver, None);
    }

    #[test]
    fn windows_probe_rejects_garbage() {
        assert!(parse_windows_probe("not json").is_err());
    }

    #[test]
    fn vendor_from_name_avoids_ati_false_positive() {
        assert_eq!(
            GpuVendor::from_name("Advanced Micro Devices, Inc."),
            GpuVendor::Amd
        );
        assert_eq!(GpuVendor::from_name("Intel Corporation"), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_name("Acme Corporation"), GpuVendor::Other);
    }

    #[test]
    fn amd_arch_table_covers_rdna4_and_rdna3() {
        assert_eq!(amd_arch_from_name("AMD Radeon RX 9060 XT"), Some("gfx1200"));
        assert_eq!(amd_arch_from_name("AMD Radeon RX 9070 XT"), Some("gfx1201"));
        assert_eq!(
            amd_arch_from_name("AMD Radeon RX 7900 XTX"),
            Some("gfx1100")
        );
        assert_eq!(amd_arch_from_name("AMD Radeon Graphics"), None);
    }

    const CPUINFO: &str = "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 7 7800X3D 8-Core Processor\nphysical id\t: 0\ncore id\t\t: 0\n\nprocessor\t: 1\nmodel name\t: AMD Ryzen 7 7800X3D 8-Core Processor\nphysical id\t: 0\ncore id\t\t: 0\n\nprocessor\t: 2\nmodel name\t: AMD Ryzen 7 7800X3D 8-Core Processor\nphysical id\t: 0\ncore id\t\t: 1\n\nprocessor\t: 3\nmodel name\t: AMD Ryzen 7 7800X3D 8-Core Processor\nphysical id\t: 0\ncore id\t\t: 1\n";

    #[test]
    fn cpuinfo_counts_physical_and_logical_cores() {
        let (brand, physical, logical) = parse_cpuinfo(CPUINFO);
        assert!(brand.contains("7800X3D"));
        assert_eq!(logical, 4);
        assert_eq!(physical, 2);
    }

    #[test]
    fn cpuinfo_without_topology_falls_back_to_threads() {
        let (_, physical, logical) = parse_cpuinfo("processor : 0\nprocessor : 1\n");
        assert_eq!((physical, logical), (2, 2));
    }

    #[test]
    fn meminfo_parses_total_and_available() {
        let text =
            "MemTotal:       32768000 kB\nMemFree:        1000 kB\nMemAvailable:   20000000 kB\n";
        let (total, avail) = parse_meminfo(text).unwrap();
        assert_eq!(total, 32_768_000 * 1024);
        assert_eq!(avail, 20_000_000 * 1024);
    }

    #[test]
    fn meminfo_missing_fields_is_error() {
        assert!(parse_meminfo("MemTotal: 10 kB\n").is_err());
    }

    const LSPCI: &str = "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation Raptor Lake-P [Iris Xe Graphics] [8086:a7a0] (rev 04)
0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 44 [Radeon RX 9060 XT] [1002:7590] (rev c0)
0000:01:00.0 3D controller [0302]: NVIDIA Corporation GA107M [GeForce RTX 3050 Mobile] [10de:25a0] (rev a1)
0000:00:14.0 USB controller [0c03]: Intel Corporation Alder Lake USB 3.2 [8086:51ed]
";

    #[test]
    fn lspci_keeps_only_display_devices_with_ids() {
        let gpus = parse_lspci(LSPCI);
        assert_eq!(gpus.len(), 3);
        assert_eq!(gpus[0].vendor_id, 0x8086);
        assert_eq!(gpus[0].name, "Iris Xe Graphics");
        assert_eq!(gpus[1].slot, "0000:03:00.0");
        assert_eq!(gpus[1].vendor_id, 0x1002);
        assert_eq!(gpus[1].device_id, 0x7590);
        assert_eq!(gpus[1].name, "Radeon RX 9060 XT");
        assert_eq!(gpus[2].vendor_id, 0x10de);
        assert_eq!(gpus[2].name, "GeForce RTX 3050 Mobile");
    }

    const ROCMINFO: &str = "ROCk module version 6.10.5 is loaded
=====================
HSA System Attributes
=====================
*******
Agent 1
*******
  Name:                    AMD Ryzen 7 7800X3D 8-Core Processor
  Marketing Name:          AMD Ryzen 7 7800X3D 8-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1200
  Uuid:                    GPU-XX11223344556677
  Marketing Name:          AMD Radeon RX 9060 XT
  Vendor Name:             AMD
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1200
";

    #[test]
    fn rocminfo_returns_gpu_agents_only() {
        let agents = parse_rocminfo(ROCMINFO);
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].gfx, "gfx1200");
        assert_eq!(agents[0].marketing_name, "AMD Radeon RX 9060 XT");
    }

    #[test]
    fn rocminfo_without_gpus_is_empty() {
        assert!(parse_rocminfo("=====\nHSA System Attributes\n").is_empty());
    }

    #[test]
    fn uevent_and_sysfs_hex_parse() {
        assert_eq!(
            parse_uevent_slot("DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\n").as_deref(),
            Some("0000:03:00.0")
        );
        assert_eq!(parse_uevent_slot("DRIVER=x\n"), None);
        assert_eq!(parse_sysfs_hex("0x1002\n"), Some(0x1002));
        assert_eq!(parse_sysfs_hex("garbage"), None);
    }

    #[test]
    fn linux_merge_identifies_rx_9060_xt_gfx1200_with_vram() {
        let lspci = parse_lspci(LSPCI);
        let rocm = parse_rocminfo(ROCMINFO);
        let drm = vec![DrmCard {
            slot: Some("0000:03:00.0".into()),
            vendor_id: 0x1002,
            vram_total: Some(16 * 1024 * 1024 * 1024),
            vram_used: Some(1024 * 1024 * 1024),
        }];
        let gpus = build_linux_gpus(&lspci, &drm, &rocm);
        let amd = gpus.iter().find(|g| g.vendor == GpuVendor::Amd).unwrap();
        assert_eq!(amd.name, "AMD Radeon RX 9060 XT");
        assert_eq!(amd.arch.as_deref(), Some("gfx1200"));
        assert_eq!(amd.vram_total, 16 * 1024 * 1024 * 1024);
        assert_eq!(amd.vram_free, Some(15 * 1024 * 1024 * 1024));
        let nv = gpus.iter().find(|g| g.vendor == GpuVendor::Nvidia).unwrap();
        assert_eq!(nv.arch, None);
        assert_eq!(nv.vram_total, 0, "no sysfs card means unknown VRAM stays 0");
    }

    #[test]
    fn linux_merge_falls_back_to_name_table_without_rocminfo() {
        let lspci = parse_lspci(LSPCI);
        let gpus = build_linux_gpus(&lspci, &[], &[]);
        let amd = gpus.iter().find(|g| g.vendor == GpuVendor::Amd).unwrap();
        assert_eq!(amd.arch.as_deref(), Some("gfx1200"));
        assert_eq!(amd.vram_free, None);
    }

    #[test]
    fn linux_merge_single_amd_agent_is_used_when_names_differ() {
        let lspci = vec![LspciGpu {
            slot: "0000:03:00.0".into(),
            vendor_id: 0x1002,
            device_id: 1,
            name: "Navi 99 [Mystery]".into(),
        }];
        let rocm = vec![RocmAgent {
            gfx: "gfx1201".into(),
            marketing_name: "Unrelated".into(),
        }];
        let gpus = build_linux_gpus(&lspci, &[], &rocm);
        assert_eq!(gpus[0].arch.as_deref(), Some("gfx1201"));
    }

    #[test]
    fn cpu_features_detection_does_not_panic() {
        let f = detect_cpu_features();
        assert!(f.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn os_current_is_known_on_ci_platforms() {
        assert_ne!(Os::current(), Os::Other);
    }

    /// Manual check against the real machine (runs PowerShell or /proc reads):
    /// `cargo test -p raptor-runtime -- --ignored real_probe --nocapture`.
    #[tokio::test]
    #[ignore = "touches real hardware and runs PowerShell"]
    async fn real_probe_reports_this_machine() {
        let probe = SystemProbe::new(Arc::new(crate::sys::SystemCommandRunner));
        let hw = probe.probe().await.unwrap();
        println!("{hw:#?}");
        assert!(hw.ram_total > 0);
        assert!(hw.cpu.threads >= 1);
    }
}
