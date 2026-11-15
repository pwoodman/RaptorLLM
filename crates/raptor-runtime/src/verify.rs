//! GPU verification. A GPU is only reported as used when llama-server's own log
//! shows it: an initialized device of the requested backend and offloaded
//! layers or device-resident model buffers. Telemetry can add evidence or
//! downgrade a claim, but it never upgrades a verdict on its own.

use crate::manifest::Backend;
use crate::sys::CommandRunner;
use async_trait::async_trait;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

/// Facts extracted from llama-server's startup log.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LogFacts {
    /// Backend the log says initialized a device.
    pub detected_backend: Option<Backend>,
    /// Device descriptions, e.g. "AMD Radeon RX 9060 XT (gfx1200)".
    pub devices: Vec<String>,
    pub layers_offloaded: Option<u32>,
    pub layers_total: Option<u32>,
    /// Sum of non-CPU model buffers, in MiB.
    pub gpu_buffer_mib: Option<f64>,
    /// Sum of CPU model buffers, in MiB.
    pub cpu_buffer_mib: Option<f64>,
    /// Sum of non-CPU KV cache buffers, in MiB.
    pub kv_gpu_mib: Option<f64>,
    pub flash_attn: Option<bool>,
    pub server_listening: bool,
    /// Lines that look like errors, trimmed and capped.
    pub errors: Vec<String>,
}

const MAX_ERROR_LINES: usize = 30;

/// Parses llama-server output. Pure: no I/O.
pub fn parse_server_log(text: &str) -> LogFacts {
    let mut f = LogFacts::default();
    for raw in text.lines() {
        let line = raw.trim_end();
        let trimmed = line.trim();

        if line.contains("ROCm devices") {
            set_backend(&mut f, Backend::Hip);
        } else if line.contains("CUDA devices") {
            set_backend(&mut f, Backend::Cuda);
        } else if line.contains("Vulkan devices") {
            set_backend(&mut f, Backend::Vulkan);
        }

        if let Some(dev) = parse_hip_or_cuda_device(trimmed) {
            if dev.1 == Backend::Hip {
                set_backend(&mut f, Backend::Hip);
            } else {
                set_backend(&mut f, Backend::Cuda);
            }
            f.devices.push(dev.0);
        } else if let Some(name) = parse_vulkan_device(trimmed) {
            set_backend(&mut f, Backend::Vulkan);
            f.devices.push(name);
        }

        // The first offload line is the main model's. A draft model (`-md`) loads afterwards and
        // logs its own (DFlash2 draft: "offloaded 4/4"), which must not replace the target's.
        if let (Some((done, total)), None) = (parse_offload(line), f.layers_offloaded) {
            f.layers_offloaded = Some(done);
            f.layers_total = Some(total);
        }

        if let Some((name, mib)) = parse_buffer(line, "model buffer size") {
            if is_cpu_buffer(&name) {
                add(&mut f.cpu_buffer_mib, mib);
            } else {
                add(&mut f.gpu_buffer_mib, mib);
                if let Some(b) = backend_from_buffer_name(&name) {
                    set_backend(&mut f, b);
                }
            }
        }

        if let Some((name, mib)) = parse_buffer(line, "KV buffer size") {
            if !is_cpu_buffer(&name) {
                add(&mut f.kv_gpu_mib, mib);
            }
        }

        if let Some(v) = parse_flash_attn(line) {
            f.flash_attn = Some(v);
        }

        if line.contains("server is listening") {
            f.server_listening = true;
        }

        let lower = line.to_ascii_lowercase();
        if (lower.contains("error") || lower.contains("failed to"))
            && f.errors.len() < MAX_ERROR_LINES
        {
            f.errors.push(trimmed.to_string());
        }
    }
    f
}

fn set_backend(f: &mut LogFacts, b: Backend) {
    match f.detected_backend {
        None => f.detected_backend = Some(b),
        Some(existing) if existing != b => {
            let msg = format!(
                "conflicting backend evidence: {} and {}",
                existing.label(),
                b.label()
            );
            if f.errors.len() < MAX_ERROR_LINES {
                f.errors.push(msg);
            }
        }
        Some(_) => {}
    }
}

fn add(slot: &mut Option<f64>, v: f64) {
    *slot = Some(slot.unwrap_or(0.0) + v);
}

/// "Device 0: AMD Radeon RX 9060 XT, gfx1200 (0x1200), VMM: no, Wave Size: 32"
/// or "Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes".
fn parse_hip_or_cuda_device(trimmed: &str) -> Option<(String, Backend)> {
    let rest = trimmed.strip_prefix("Device ")?;
    let (idx, body) = rest.split_once(": ")?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if let Some((name, after)) = body.split_once(", gfx") {
        let arch: String = format!(
            "gfx{}",
            after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
        );
        return Some((format!("{} ({arch})", name.trim()), Backend::Hip));
    }
    if let Some((name, _)) = body.split_once(", compute capability") {
        return Some((name.trim().to_string(), Backend::Cuda));
    }
    None
}

/// "ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) (radv) | uma: 0 | ..."
fn parse_vulkan_device(trimmed: &str) -> Option<String> {
    let rest = trimmed.strip_prefix("ggml_vulkan: ")?;
    let (idx, body) = rest.split_once(" = ")?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let name = body.split(" | ").next()?.trim();
    Some(strip_trailing_parens(name))
}

/// Removes trailing "(...)" groups: "AMD Radeon RX 9060 XT (RADV GFX1200) (radv)" -> "AMD Radeon RX 9060 XT".
fn strip_trailing_parens(s: &str) -> String {
    let mut out = s.trim().to_string();
    while out.ends_with(')') {
        match out.rfind('(') {
            Some(i) if i > 0 => out.truncate(i),
            _ => break,
        }
        out = out.trim_end().to_string();
    }
    out
}

/// "load_tensors: offloaded 33/33 layers to GPU" -> (33, 33)
fn parse_offload(line: &str) -> Option<(u32, u32)> {
    let idx = line.find("offloaded ")?;
    let rest = &line[idx + "offloaded ".len()..];
    let token = rest.split_whitespace().next()?;
    if !rest[token.len()..].trim_start().starts_with("layers") {
        return None;
    }
    let (done, total) = token.split_once('/')?;
    Some((done.parse().ok()?, total.parse().ok()?))
}

/// "load_tensors:        ROCm0 model buffer size =  4168.09 MiB" -> ("ROCm0", 4168.09)
fn parse_buffer(line: &str, marker: &str) -> Option<(String, f64)> {
    let (left, right) = line.split_once(marker)?;
    let name = left.split_whitespace().last()?.to_string();
    let value = right.trim_start().strip_prefix('=')?.trim_start();
    let mut parts = value.split_whitespace();
    let number: f64 = parts.next()?.parse().ok()?;
    if parts.next()? != "MiB" {
        return None;
    }
    Some((name, number))
}

fn is_cpu_buffer(name: &str) -> bool {
    name.starts_with("CPU") || name.contains("Host") || name.starts_with("Host")
}

fn backend_from_buffer_name(name: &str) -> Option<Backend> {
    if name.starts_with("ROCm") || name.starts_with("HIP") {
        Some(Backend::Hip)
    } else if name.starts_with("CUDA") {
        Some(Backend::Cuda)
    } else if name.starts_with("Vulkan") {
        Some(Backend::Vulkan)
    } else {
        None
    }
}

/// "llama_context: flash_attn = 1" / "flash_attn    = enabled" -> Some(true)
fn parse_flash_attn(line: &str) -> Option<bool> {
    let idx = line
        .find("flash_attn")
        .or_else(|| line.find("flash attention"))?;
    let after = &line[idx..];
    let (_, value) = after.split_once('=')?;
    match value.split_whitespace().next()? {
        "1" | "enabled" | "on" | "true" => Some(true),
        "0" | "disabled" | "off" | "false" => Some(false),
        _ => None,
    }
}

/// Outcome of GPU verification.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum GpuVerdict {
    /// The log shows the requested GPU backend running the model.
    Verified {
        evidence: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offloaded_layers: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_layers: Option<u32>,
    },
    /// The model is running on the CPU. `reason` is a loud warning when a GPU was requested.
    CpuOnly { reason: String },
    /// Not enough evidence either way. Never shown as "GPU active".
    Unverified { reason: String },
}

impl GpuVerdict {
    /// True only for `Verified`.
    pub fn is_verified(&self) -> bool {
        matches!(self, GpuVerdict::Verified { .. })
    }
}

/// Decides whether `requested` actually ran on the GPU, given the parsed log.
pub fn verify_gpu(requested: Backend, facts: &LogFacts) -> GpuVerdict {
    if !requested.is_gpu() {
        return GpuVerdict::CpuOnly {
            reason: "CPU runtime selected; no GPU was requested".into(),
        };
    }
    let label = requested.label();
    let total = facts
        .layers_total
        .map(|t| t.to_string())
        .unwrap_or_else(|| "?".into());

    if facts.layers_offloaded == Some(0) {
        return GpuVerdict::CpuOnly {
            reason: format!(
                "WARNING: {label} backend requested but 0/{total} layers were offloaded to the GPU; inference is running on the CPU"
            ),
        };
    }
    if let Some(d) = facts.detected_backend {
        if d != requested {
            return GpuVerdict::Unverified {
                reason: format!(
                    "log reports the {} backend but {label} was requested",
                    d.label()
                ),
            };
        }
    } else if facts.gpu_buffer_mib.is_none() {
        if facts.cpu_buffer_mib.is_some() {
            return GpuVerdict::CpuOnly {
                reason: format!(
                    "WARNING: {label} backend requested but no GPU device was initialized; model buffers are on the CPU"
                ),
            };
        }
        return GpuVerdict::Unverified {
            reason: "log contains no backend or buffer evidence".into(),
        };
    }

    let offload = facts.layers_offloaded.unwrap_or(0);
    let gpu_mib = facts.gpu_buffer_mib.unwrap_or(0.0);
    if offload == 0 && gpu_mib <= 0.0 {
        return GpuVerdict::Unverified {
            reason: format!(
                "{label} device present but the log shows no offloaded layers or GPU buffers"
            ),
        };
    }

    let mut evidence = vec![format!("backend: {label}")];
    evidence.extend(facts.devices.iter().map(|d| format!("device: {d}")));
    if let (Some(done), Some(tot)) = (facts.layers_offloaded, facts.layers_total) {
        if done < tot {
            evidence.push(format!("partial offload: {done}/{tot} layers on GPU"));
        } else {
            evidence.push(format!("offloaded {done}/{tot} layers to GPU"));
        }
    }
    if gpu_mib > 0.0 {
        evidence.push(format!("GPU model buffers: {gpu_mib:.0} MiB"));
    }
    if let Some(kv) = facts.kv_gpu_mib {
        evidence.push(format!("GPU KV cache: {kv:.0} MiB"));
    }
    if facts.flash_attn == Some(true) {
        evidence.push("flash attention: on".into());
    }
    GpuVerdict::Verified {
        evidence,
        offloaded_layers: facts.layers_offloaded,
        total_layers: facts.layers_total,
    }
}

/// One GPU's live numbers from rocm-smi or a similar tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuSample {
    pub index: u32,
    pub vram_used: u64,
    pub vram_total: u64,
    pub util_pct: Option<u8>,
}

/// Optional source of live GPU numbers. Implementations may fail; callers
/// must treat a failure as "no telemetry", not as evidence.
#[async_trait]
pub trait GpuTelemetry: Send + Sync {
    async fn sample(&self) -> Result<Vec<GpuSample>>;
}

fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    match v.get(key)? {
        serde_json::Value::String(s) => s.trim().parse().ok(),
        serde_json::Value::Number(n) => n.as_u64(),
        _ => None,
    }
}

/// Parses `rocm-smi --showmeminfo vram --showuse --json`. Entries not named
/// `cardN` (such as "system") are ignored.
pub fn parse_rocm_smi_json(text: &str) -> Result<Vec<GpuSample>> {
    let root: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| RaptorError::Runtime(format!("unreadable rocm-smi output: {e}")))?;
    let obj = root
        .as_object()
        .ok_or_else(|| RaptorError::Runtime("rocm-smi output is not a JSON object".into()))?;
    let mut out = Vec::new();
    for (key, val) in obj {
        let Some(index) = key.strip_prefix("card").and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        let Some(total) = json_u64(val, "VRAM Total Memory (B)") else {
            continue;
        };
        out.push(GpuSample {
            index,
            vram_used: json_u64(val, "VRAM Total Used Memory (B)").unwrap_or(0),
            vram_total: total,
            util_pct: json_u64(val, "GPU use (%)").and_then(|u| u8::try_from(u).ok()),
        });
    }
    out.sort_by_key(|s| s.index);
    Ok(out)
}

/// rocm-smi backed telemetry (AMD only).
pub struct RocmSmiTelemetry {
    runner: Arc<dyn CommandRunner>,
}

impl RocmSmiTelemetry {
    /// Creates telemetry that runs `rocm-smi` through `runner`.
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl GpuTelemetry for RocmSmiTelemetry {
    async fn sample(&self) -> Result<Vec<GpuSample>> {
        let args = ["--showmeminfo", "vram", "--showuse", "--json"].map(String::from);
        let out = self
            .runner
            .run("rocm-smi", &args, Duration::from_secs(10))
            .await?;
        parse_rocm_smi_json(&out)
    }
}

/// Combines a log verdict with telemetry.
///
/// - `Verified` gains one evidence line per card with VRAM in use. If telemetry
///   is present and shows no VRAM in use at all, the claim is downgraded to
///   `Unverified`, because a loaded GPU model always occupies VRAM.
/// - Other verdicts are returned unchanged.
pub fn apply_telemetry(verdict: GpuVerdict, samples: &[GpuSample]) -> GpuVerdict {
    match verdict {
        GpuVerdict::Verified {
            mut evidence,
            offloaded_layers,
            total_layers,
        } if !samples.is_empty() => {
            let used_cards: Vec<&GpuSample> = samples.iter().filter(|s| s.vram_used > 0).collect();
            if used_cards.is_empty() {
                return GpuVerdict::Unverified {
                    reason: "log reports GPU offload but telemetry shows no VRAM in use".into(),
                };
            }
            for s in used_cards {
                let util = s
                    .util_pct
                    .map(|u| format!(", utilization {u}%"))
                    .unwrap_or_default();
                evidence.push(format!(
                    "telemetry: card{} VRAM {} of {} MiB in use{util}",
                    s.index,
                    s.vram_used / (1024 * 1024),
                    s.vram_total / (1024 * 1024)
                ));
            }
            GpuVerdict::Verified {
                evidence,
                offloaded_layers,
                total_layers,
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HIP_LOG: &str = "\
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon RX 9060 XT, gfx1200 (0x1200), VMM: no, Wave Size: 32
load_backend: loaded ROCm backend from C:\\raptor\\runtimes\\llama-hip-win\\b7100\\ggml-hip.dll
load_backend: loaded CPU backend from C:\\raptor\\runtimes\\llama-hip-win\\b7100\\ggml-cpu-zen4.dll
build: 7100 (abcdef1) with clang version 19 for x86_64-pc-windows-msvc
llama_model_loader: loaded meta data with 35 key-value pairs and 291 tensors
print_info: arch             = llama
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 33/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
load_tensors:        ROCm0 model buffer size =  4168.09 MiB
llama_context: n_ctx         = 8192
llama_context: flash_attn    = 1
llama_kv_cache:      ROCm0 KV buffer size =  1024.00 MiB
llama_kv_cache: size = 1024.00 MiB (  8192 cells,  32 layers,  1/1 seqs), K (f16):  512.00 MiB, V (f16):  512.00 MiB
main: model loaded
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
";

    const CUDA_LOG: &str = "\
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
load_backend: loaded CUDA backend from /opt/raptor/runtimes/llama-cuda-linux/b7100/libggml-cuda.so
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        CUDA0 model buffer size =  4300.00 MiB
llama_context: flash_attn    = 1
main: server is listening on http://127.0.0.1:8080
";

    const VULKAN_LOG: &str = "\
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
load_backend: loaded Vulkan backend from C:\\raptor\\runtimes\\llama-vk-win\\b7100\\ggml-vulkan.dll
load_tensors: offloaded 33/33 layers to GPU
load_tensors:      Vulkan0 model buffer size =  4168.09 MiB
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
llama_context: flash_attn    = enabled
main: server is listening on http://127.0.0.1:8080
";

    const CPU_LOG: &str = "\
load_backend: loaded CPU backend from C:\\raptor\\runtimes\\llama-cpu-win\\b7100\\ggml-cpu-zen4.dll
load_tensors: offloaded 0/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  4483.39 MiB
llama_context: flash_attn    = 0
main: server is listening on http://127.0.0.1:8080
";

    const PARTIAL_LOG: &str = "\
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
load_tensors: offloaded 20/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  2600.00 MiB
load_tensors:        CUDA0 model buffer size =  1700.00 MiB
main: server is listening on http://127.0.0.1:8080
";

    const OOM_LOG: &str = "\
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4300.00 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
llama_model_load_from_file_impl: failed to load model
";

    #[test]
    fn parses_hip_log_with_gfx1200_device() {
        let f = parse_server_log(HIP_LOG);
        assert_eq!(f.detected_backend, Some(Backend::Hip));
        assert_eq!(
            f.devices,
            vec!["AMD Radeon RX 9060 XT (gfx1200)".to_string()]
        );
        assert_eq!(f.layers_offloaded, Some(33));
        assert_eq!(f.layers_total, Some(33));
        assert_eq!(f.gpu_buffer_mib, Some(4168.09));
        assert_eq!(f.cpu_buffer_mib, Some(315.30));
        assert_eq!(f.kv_gpu_mib, Some(1024.0));
        assert_eq!(f.flash_attn, Some(true));
        assert!(f.server_listening);
    }

    #[test]
    fn hip_log_is_verified_for_hip() {
        let v = verify_gpu(Backend::Hip, &parse_server_log(HIP_LOG));
        match v {
            GpuVerdict::Verified {
                evidence,
                offloaded_layers,
                total_layers,
            } => {
                assert_eq!(offloaded_layers, Some(33));
                assert_eq!(total_layers, Some(33));
                assert!(evidence.iter().any(|e| e.contains("gfx1200")));
                assert!(evidence.iter().any(|e| e.contains("33/33")));
                assert!(evidence.iter().any(|e| e.contains("4168")));
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn parses_cuda_log() {
        let f = parse_server_log(CUDA_LOG);
        assert_eq!(f.detected_backend, Some(Backend::Cuda));
        assert_eq!(
            f.devices,
            vec!["NVIDIA GeForce RTX 4070 Laptop GPU".to_string()]
        );
        assert_eq!(f.layers_offloaded, Some(41));
        assert!(verify_gpu(Backend::Cuda, &f).is_verified());
    }

    #[test]
    fn parses_vulkan_log_and_strips_driver_suffix() {
        let f = parse_server_log(VULKAN_LOG);
        assert_eq!(f.detected_backend, Some(Backend::Vulkan));
        assert_eq!(f.devices, vec!["AMD Radeon RX 9060 XT".to_string()]);
        assert_eq!(f.flash_attn, Some(true));
        assert!(verify_gpu(Backend::Vulkan, &f).is_verified());
    }

    #[test]
    fn cpu_log_for_cpu_request_is_cpu_only_without_warning() {
        let f = parse_server_log(CPU_LOG);
        assert_eq!(
            f.detected_backend, None,
            "CPU backend load line alone is not a GPU"
        );
        match verify_gpu(Backend::Cpu, &f) {
            GpuVerdict::CpuOnly { reason } => assert!(!reason.starts_with("WARNING")),
            other => panic!("expected CpuOnly, got {other:?}"),
        }
    }

    #[test]
    fn zero_offloaded_layers_for_gpu_request_is_loud_cpu_only() {
        let f = parse_server_log(CPU_LOG);
        match verify_gpu(Backend::Hip, &f) {
            GpuVerdict::CpuOnly { reason } => {
                assert!(reason.starts_with("WARNING"));
                assert!(reason.contains("0/33"));
            }
            other => panic!("expected CpuOnly warning, got {other:?}"),
        }
    }

    #[test]
    fn explicit_zero_offload_with_device_is_cpu_only_warning() {
        let log = HIP_LOG.replace("offloaded 33/33 layers", "offloaded 0/33 layers");
        match verify_gpu(Backend::Hip, &parse_server_log(&log)) {
            GpuVerdict::CpuOnly { reason } => assert!(reason.contains("0/33")),
            other => panic!("expected CpuOnly, got {other:?}"),
        }
    }

    #[test]
    fn a_draft_models_offload_line_does_not_replace_the_main_models() {
        // Order measured with a DFlash2 draft (b12374): the target's line, then the draft's.
        let log = "load_tensors: offloaded 45/53 layers to GPU\n\
                   common_speculative_init_result: loading draft model 'D:/d.gguf'\n\
                   load_tensors: offloaded 4/4 layers to GPU\n";
        let f = parse_server_log(log);
        assert_eq!((f.layers_offloaded, f.layers_total), (Some(45), Some(53)));
    }

    #[test]
    fn partial_offload_is_verified_and_says_so() {
        let f = parse_server_log(PARTIAL_LOG);
        match verify_gpu(Backend::Cuda, &f) {
            GpuVerdict::Verified {
                evidence,
                offloaded_layers,
                total_layers,
            } => {
                assert_eq!(offloaded_layers, Some(20));
                assert_eq!(total_layers, Some(41));
                assert!(evidence
                    .iter()
                    .any(|e| e.contains("partial offload: 20/41")));
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn backend_mismatch_is_unverified() {
        match verify_gpu(Backend::Vulkan, &parse_server_log(HIP_LOG)) {
            GpuVerdict::Unverified { reason } => assert!(reason.contains("hip")),
            other => panic!("expected Unverified, got {other:?}"),
        }
    }

    #[test]
    fn oom_log_is_not_verified() {
        let f = parse_server_log(OOM_LOG);
        assert!(!f.errors.is_empty());
        assert!(!verify_gpu(Backend::Cuda, &f).is_verified());
    }

    #[test]
    fn empty_log_is_unverified_not_verified() {
        let f = parse_server_log("");
        assert!(matches!(
            verify_gpu(Backend::Hip, &f),
            GpuVerdict::Unverified { .. }
        ));
    }

    #[test]
    fn gpu_request_with_device_but_no_offload_lines_is_unverified() {
        let log = "ggml_cuda_init: found 1 ROCm devices:\n  Device 0: AMD Radeon RX 9060 XT, gfx1200 (0x1200), VMM: no\n";
        assert!(matches!(
            verify_gpu(Backend::Hip, &parse_server_log(log)),
            GpuVerdict::Unverified { .. }
        ));
    }

    #[test]
    fn cpu_requested_is_always_cpu_only() {
        let f = parse_server_log(HIP_LOG);
        assert!(matches!(
            verify_gpu(Backend::Cpu, &f),
            GpuVerdict::CpuOnly { .. }
        ));
    }

    #[test]
    fn buffer_names_identify_backend_when_device_line_missing() {
        let log = "load_tensors:        CUDA0 model buffer size =  100.00 MiB\nload_tensors: offloaded 5/5 layers to GPU\n";
        let f = parse_server_log(log);
        assert_eq!(f.detected_backend, Some(Backend::Cuda));
        assert!(verify_gpu(Backend::Cuda, &f).is_verified());
    }

    #[test]
    fn conflicting_backends_are_reported_as_errors() {
        let log = format!("{HIP_LOG}\nggml_vulkan: Found 1 Vulkan devices:\n");
        let f = parse_server_log(&log);
        assert!(f.errors.iter().any(|e| e.contains("conflicting backend")));
    }

    #[test]
    fn flash_attention_variants_parse() {
        assert_eq!(
            parse_flash_attn("llama_context: flash_attn    = 1"),
            Some(true)
        );
        assert_eq!(
            parse_flash_attn("llama_context: flash_attn = enabled"),
            Some(true)
        );
        assert_eq!(
            parse_flash_attn("llama_context: flash_attn = disabled"),
            Some(false)
        );
        assert_eq!(parse_flash_attn("llama_context: flash_attn = auto"), None);
        assert_eq!(parse_flash_attn("nothing here"), None);
    }

    #[test]
    fn offload_and_buffer_parsers_reject_malformed_lines() {
        assert_eq!(parse_offload("offloaded 3/x layers"), None);
        assert_eq!(parse_offload("offloaded 3/4 tokens"), None);
        assert_eq!(
            parse_buffer("ROCm0 model buffer size = 12 GiB", "model buffer size"),
            None
        );
    }

    #[test]
    fn strip_trailing_parens_keeps_name_with_inner_text() {
        assert_eq!(
            strip_trailing_parens("AMD Radeon RX 9060 XT (RADV GFX1200) (radv)"),
            "AMD Radeon RX 9060 XT"
        );
        assert_eq!(strip_trailing_parens("Plain Name"), "Plain Name");
    }

    const ROCM_SMI: &str = r#"{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "4500000000", "GPU use (%)": "12"}, "system": {"Driver version": "6.10.5"}}"#;

    #[test]
    fn rocm_smi_json_parses_cards_only() {
        let s = parse_rocm_smi_json(ROCM_SMI).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].index, 0);
        assert_eq!(s[0].vram_total, 17_163_091_968);
        assert_eq!(s[0].vram_used, 4_500_000_000);
        assert_eq!(s[0].util_pct, Some(12));
    }

    #[test]
    fn rocm_smi_rejects_non_json() {
        assert!(parse_rocm_smi_json("GPU[0] : VRAM").is_err());
    }

    #[test]
    fn telemetry_adds_vram_evidence_to_verified() {
        let v = verify_gpu(Backend::Hip, &parse_server_log(HIP_LOG));
        let samples = parse_rocm_smi_json(ROCM_SMI).unwrap();
        match apply_telemetry(v, &samples) {
            GpuVerdict::Verified { evidence, .. } => {
                assert!(evidence
                    .iter()
                    .any(|e| e.starts_with("telemetry: card0 VRAM")));
            }
            other => panic!("expected Verified, got {other:?}"),
        }
    }

    #[test]
    fn telemetry_with_no_vram_use_downgrades_verified() {
        let v = verify_gpu(Backend::Hip, &parse_server_log(HIP_LOG));
        let idle = vec![GpuSample {
            index: 0,
            vram_used: 0,
            vram_total: 1,
            util_pct: Some(0),
        }];
        assert!(matches!(
            apply_telemetry(v, &idle),
            GpuVerdict::Unverified { .. }
        ));
    }

    #[tokio::test]
    async fn rocm_smi_telemetry_runs_tool_and_parses_output() {
        let runner = Arc::new(crate::testing::FakeRunner::default().with_ok("rocm-smi", ROCM_SMI));
        let samples = RocmSmiTelemetry::new(runner).sample().await.unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].util_pct, Some(12));
    }

    #[tokio::test]
    async fn missing_rocm_smi_is_an_error_not_empty_evidence() {
        let runner =
            Arc::new(crate::testing::FakeRunner::default().with_err("rocm-smi", "not found"));
        assert!(RocmSmiTelemetry::new(runner).sample().await.is_err());
    }

    #[test]
    fn telemetry_never_upgrades_unverified() {
        let v = verify_gpu(Backend::Hip, &parse_server_log(""));
        let busy = vec![GpuSample {
            index: 0,
            vram_used: 9,
            vram_total: 10,
            util_pct: None,
        }];
        assert!(matches!(
            apply_telemetry(v, &busy),
            GpuVerdict::Unverified { .. }
        ));
    }
}
