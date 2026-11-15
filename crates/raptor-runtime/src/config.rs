//! llama-server launch configuration and memory-aware recommendation.
//!
//! `LaunchConfig` is validated on build, so an invalid combination (for example
//! a quantized KV cache without flash attention) never reaches the process spawner.
//! `recommend` estimates VRAM/RAM use and explains every decision it makes.

use crate::fit::FitProjection;
use crate::hardware::{GpuVendor, HardwareInfo};
use crate::manifest::Backend;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::net::IpAddr;
use std::path::PathBuf;

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

/// Estimated scratch/compute buffers per GPU model load. Not measured.
const COMPUTE_OVERHEAD: u64 = 384 * MIB;
/// Context lengths are multiples of this.
const CTX_STEP: u32 = 256;
/// Smallest context the recommender will settle for before reducing layers instead.
const CTX_FLOOR: u32 = 2048;
/// Headroom is max(10% of total, this).
const MIN_HEADROOM: u64 = 512 * MIB;
const BATCH_MAX: u32 = 2048;
const UBATCH_MAX: u32 = 512;

/// Secret wrapper that never prints its contents.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// KV cache element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    /// Value for llama-server `--cache-type-k/-v`.
    pub fn llama_arg(self) -> &'static str {
        match self {
            KvCacheType::F16 => "f16",
            KvCacheType::Q8_0 => "q8_0",
            KvCacheType::Q4_0 => "q4_0",
        }
    }

    /// Average bytes per cached element, including block scales for quantized types.
    pub fn bytes_per_element(self) -> f64 {
        match self {
            KvCacheType::F16 => 2.0,
            KvCacheType::Q8_0 => 34.0 / 32.0,
            KvCacheType::Q4_0 => 18.0 / 32.0,
        }
    }
}

/// Speculative decoding without a second model. Measured on a real RX 9060 XT (llama.cpp b12360,
/// temperature 0, a code-editing prompt): Qwen3.8-27B 21.5 -> 47.4 tok/s with `Mtp` (95% of the
/// drafted tokens accepted); granite 20.4 -> 25.9 and Ornith 66.8 -> 77.8 tok/s with `Ngram` on a
/// first reply, rising as the n-gram table learns repeated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecMode {
    /// `--spec-type ngram-mod`: drafts from text already in the prompt and reply. Works with any
    /// model, costs nothing when it cannot predict.
    Ngram,
    /// `--spec-type draft-mtp,ngram-simple`: the model's own multi-token-prediction head, stacked
    /// with prompt n-grams. Only for a model whose GGUF has `nextn_predict_layers` > 0, and the
    /// server refuses to start when the tensors are missing (measured on Ornith), so callers must
    /// be ready to retry without it. Measured on Qwen3.8-27B (b12374, six code-rewrite prompts):
    /// `draft-mtp` alone 28-46 tok/s, stacked 186-213, and 25-29 -> 41-42 on new code.
    Mtp,
    /// `--spec-type draft-dflash,ngram-simple` with a separate DFlash block-diffusion draft model
    /// (`-md`, [`LaunchConfig::draft_model`]), stacked with prompt n-grams. Measured on
    /// Nanbeige4.3-33B IQ4_XS + its DFlash2 draft (Q4_K_M, 0.54 GiB), b12374 ROCm, p_min 0.3:
    /// whole-file code edits 31.7 / 38.8 tok/s (n-grams alone 25.6), new code 16.7 vs 14.2 plain,
    /// prose about even or slightly slower (the draft's VRAM can push the target into shared memory).
    Dflash,
}

/// `--spec-draft-p-min` used with [`SpecMode::Dflash`]: a draft stops at the first position the
/// draft is less sure of than this. Measured: p_min 0.3 gave code 16.7 / story 13.1 tok/s vs
/// 16.6 / 12.4 without it and 15.2 / 11.9 at 0.5.
pub const DFLASH_P_MIN: &str = "0.3";

impl SpecMode {
    /// The value passed to `--spec-type`.
    pub fn arg(self) -> &'static str {
        match self {
            SpecMode::Ngram => "ngram-mod",
            SpecMode::Mtp => "draft-mtp,ngram-simple",
            SpecMode::Dflash => "draft-dflash,ngram-simple",
        }
    }
}

/// Validated llama-server launch parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchConfig {
    pub model_path: PathBuf,
    pub backend: Backend,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
    pub kv_cache: KvCacheType,
    pub host: String,
    pub port: u16,
    /// When a GPU backend has no device, the engine refuses to load unless this is
    /// set. Setting it accepts a CPU run that is reported with a warning verdict.
    #[serde(default)]
    pub allow_cpu_fallback: bool,
    /// Also serve `/v1/embeddings` (`--embeddings`). Off by default. Measured on a real GPU
    /// (b12360, a 27B chat model): chat keeps working at full speed with it on.
    #[serde(default)]
    pub embeddings: bool,
    /// Pooling for embeddings (`--pooling`): `mean`, `cls`, `last` or `rank`. A chat model has
    /// no pooling of its own, and without one the server refuses `/v1/embeddings` ("Pooling
    /// type 'none' is not OAI compatible"). None leaves the model's own pooling (embedding models).
    #[serde(default)]
    pub pooling: Option<String>,
    /// Speculative decoding (`--spec-type`). Off by default. See [`SpecMode`].
    #[serde(default)]
    pub speculative: Option<SpecMode>,
    /// The draft model for [`SpecMode::Dflash`] (`-md`, every layer on the GPU). Required by that
    /// mode and ignored by the others.
    #[serde(default)]
    pub draft_model: Option<PathBuf>,
    /// Parallel request slots (`-np`). None leaves llama-server's default (4 on the builds
    /// measured). Each slot keeps its own recurrent state and prompt checkpoints, and a hybrid
    /// model's recurrent memory grows with the slot count.
    #[serde(default)]
    pub parallel_slots: Option<u32>,
    /// Threads for prompt processing (`-tb`). None leaves it equal to `threads`.
    #[serde(default)]
    pub threads_batch: Option<u32>,
    /// Passed through the environment (`LLAMA_API_KEY`), never on the command line.
    #[serde(skip)]
    api_key: Option<Secret>,
}

/// llama-server log verbosity passed as `-lv`. Verbosity 3 (the llama-server default)
/// omits the library lines that GPU verification needs: `llama_prepare_model_devices:
/// using device ...` and `load_tensors: offloaded N/M layers to GPU`. Measured on
/// llama.cpp b12340 (Windows, ROCm): levels 0 to 3 emit no such lines, level 4 emits
/// both. Level 4 adds only a few lines per request (see the gated hardware test).
pub const LLAMA_LOG_VERBOSITY: u32 = 4;

/// Fluent builder. `build` validates the result.
#[derive(Debug, Clone)]
pub struct LaunchConfigBuilder {
    cfg: LaunchConfig,
}

impl LaunchConfig {
    /// Starts a builder with defaults appropriate for `backend`.
    pub fn builder(model_path: impl Into<PathBuf>, backend: Backend) -> LaunchConfigBuilder {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);
        LaunchConfigBuilder {
            cfg: LaunchConfig {
                model_path: model_path.into(),
                backend,
                gpu_layers: if backend.is_gpu() { 999 } else { 0 },
                ctx_size: 4096,
                batch_size: 2048,
                ubatch_size: 512,
                threads: threads.max(1),
                flash_attn: backend.supports_flash_attn(),
                kv_cache: KvCacheType::F16,
                host: "127.0.0.1".into(),
                port: 8080,
                allow_cpu_fallback: false,
                embeddings: false,
                pooling: None,
                speculative: None,
                draft_model: None,
                parallel_slots: None,
                threads_batch: None,
                api_key: None,
            },
        }
    }

    /// Checks every cross-field rule. Called by the builder and before spawning.
    pub fn validate(&self) -> Result<()> {
        let inv = |m: &str| Err(RaptorError::InvalidInput(m.to_string()));
        if self.model_path.as_os_str().is_empty() {
            return inv("model path is empty");
        }
        if self.model_path.to_str().is_none() {
            return inv(
                "model path is not valid UTF-8; it cannot be passed to llama-server losslessly",
            );
        }
        if let Some(p) = &self.pooling {
            if !["mean", "cls", "last", "rank"].contains(&p.as_str()) {
                return inv("pooling must be mean, cls, last or rank");
            }
        }
        if self.ctx_size < 256 {
            return inv("ctx_size must be at least 256");
        }
        if self.speculative == Some(SpecMode::Dflash) {
            match &self.draft_model {
                None => return inv("DFlash speculation needs a draft model"),
                Some(p) if p.as_os_str().is_empty() || p.to_str().is_none() => {
                    return inv("draft model path must be non-empty valid UTF-8");
                }
                Some(_) => {}
            }
        }
        if self.parallel_slots.is_some_and(|n| !(1..=16).contains(&n)) {
            return inv("parallel_slots must be 1-16");
        }
        if self.threads_batch.is_some_and(|n| !(1..=256).contains(&n)) {
            return inv("threads_batch must be 1-256");
        }
        if self.batch_size == 0 || self.ubatch_size == 0 || self.threads == 0 {
            return inv("batch, ubatch and threads must be positive");
        }
        if self.ubatch_size > self.batch_size {
            return inv("ubatch_size cannot exceed batch_size");
        }
        if self.port == 0 {
            return inv("port must be non-zero");
        }
        let ip: IpAddr = match self.host.parse() {
            Ok(ip) => ip,
            Err(_) => return inv("host must be an IP address (e.g. 127.0.0.1)"),
        };
        if !self.backend.is_gpu() && self.gpu_layers > 0 {
            return inv("gpu_layers > 0 requires a GPU backend; CPU runtimes cannot offload");
        }
        if self.flash_attn && !self.backend.supports_flash_attn() {
            return inv("flash attention is not enabled for the CPU backend");
        }
        if self.kv_cache != KvCacheType::F16 && !self.flash_attn {
            return inv("a quantized KV cache requires flash attention");
        }
        if !ip.is_loopback() && self.api_key.is_none() {
            return inv("binding to a non-loopback host requires an api_key");
        }
        if let Some(Secret(key)) = &self.api_key {
            if key.is_empty() || key.chars().any(|c| c.is_control() || c.is_whitespace()) {
                return inv(
                    "api_key must be non-empty and contain no whitespace or control characters",
                );
            }
        }
        Ok(())
    }

    /// Command-line arguments for llama-server (no secrets).
    pub fn args(&self) -> Vec<String> {
        let model = self.model_path.to_string_lossy().into_owned();
        let mut args: Vec<String> = vec![
            "-m".into(),
            model,
            "-ngl".into(),
            self.gpu_layers.to_string(),
            "-c".into(),
            self.ctx_size.to_string(),
            "-b".into(),
            self.batch_size.to_string(),
            "-ub".into(),
            self.ubatch_size.to_string(),
            "-t".into(),
            self.threads.to_string(),
            "-fa".into(),
            if self.flash_attn { "on" } else { "off" }.into(),
            "--cache-type-k".into(),
            self.kv_cache.llama_arg().into(),
            "--cache-type-v".into(),
            self.kv_cache.llama_arg().into(),
            "--host".into(),
            self.host.clone(),
            "--port".into(),
            self.port.to_string(),
            "-lv".into(),
            LLAMA_LOG_VERBOSITY.to_string(),
        ];
        if let Some(n) = self.parallel_slots {
            let at = args.len() - 2;
            args.splice(at..at, ["-np".to_string(), n.to_string()]);
        }
        if let Some(n) = self.threads_batch {
            let at = args.len() - 2;
            args.splice(at..at, ["-tb".to_string(), n.to_string()]);
        }
        // Speculative decoding is not combined with the embeddings mode.
        if let (Some(mode), false) = (self.speculative, self.embeddings) {
            let at = args.len() - 2;
            args.splice(at..at, ["--spec-type".to_string(), mode.arg().to_string()]);
            if let (SpecMode::Dflash, Some(draft)) = (mode, &self.draft_model) {
                let at = args.len() - 2;
                args.splice(
                    at..at,
                    [
                        "-md".to_string(),
                        draft.to_string_lossy().into_owned(),
                        "-ngld".to_string(),
                        "999".to_string(),
                        "--spec-draft-p-min".to_string(),
                        DFLASH_P_MIN.to_string(),
                    ],
                );
            }
        }
        if self.embeddings {
            args.insert(args.len() - 2, "--embeddings".into());
            if let Some(pooling) = &self.pooling {
                let at = args.len() - 2;
                args.splice(at..at, ["--pooling".to_string(), pooling.clone()]);
            }
        }
        args
    }

    /// Extra environment for the child process. Carries the API key, if any.
    pub fn env(&self) -> Vec<(String, String)> {
        match &self.api_key {
            Some(Secret(key)) => vec![("LLAMA_API_KEY".into(), key.clone())],
            None => Vec::new(),
        }
    }

    /// True when an API key is configured.
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// Base URL for HTTP calls to this server (IPv6 hosts are bracketed).
    pub fn endpoint(&self) -> String {
        if self.host.contains(':') {
            format!("http://[{}]:{}", self.host, self.port)
        } else {
            format!("http://{}:{}", self.host, self.port)
        }
    }
}

impl LaunchConfigBuilder {
    pub fn gpu_layers(mut self, n: u32) -> Self {
        self.cfg.gpu_layers = n;
        self
    }
    pub fn ctx_size(mut self, n: u32) -> Self {
        self.cfg.ctx_size = n;
        self
    }
    pub fn batch_size(mut self, n: u32) -> Self {
        self.cfg.batch_size = n;
        self
    }
    pub fn ubatch_size(mut self, n: u32) -> Self {
        self.cfg.ubatch_size = n;
        self
    }
    pub fn threads(mut self, n: u32) -> Self {
        self.cfg.threads = n;
        self
    }
    pub fn flash_attn(mut self, on: bool) -> Self {
        self.cfg.flash_attn = on;
        self
    }
    pub fn kv_cache(mut self, kv: KvCacheType) -> Self {
        self.cfg.kv_cache = kv;
        self
    }
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.cfg.host = host.into();
        self
    }
    pub fn port(mut self, port: u16) -> Self {
        self.cfg.port = port;
        self
    }
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.cfg.api_key = Some(Secret(key.into()));
        self
    }
    /// Allows a GPU backend to run on the CPU when it has no device. Off by default.
    /// Also serves embeddings (`--embeddings`). Off by default.
    pub fn embeddings(mut self, on: bool) -> Self {
        self.cfg.embeddings = on;
        self
    }
    /// Pooling used for embeddings. Only applied with [`LaunchConfigBuilder::embeddings`].
    pub fn pooling(mut self, pooling: &str) -> Self {
        self.cfg.pooling = Some(pooling.to_string());
        self
    }
    /// Parallel request slots (`-np`).
    pub fn parallel_slots(mut self, n: u32) -> Self {
        self.cfg.parallel_slots = Some(n);
        self
    }
    /// Threads for prompt processing (`-tb`).
    pub fn threads_batch(mut self, n: u32) -> Self {
        self.cfg.threads_batch = Some(n);
        self
    }
    /// Speculative decoding mode. Off by default.
    pub fn speculative(mut self, mode: SpecMode) -> Self {
        self.cfg.speculative = Some(mode);
        self
    }
    /// Draft model for [`SpecMode::Dflash`].
    pub fn draft_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.cfg.draft_model = Some(path.into());
        self
    }
    pub fn allow_cpu_fallback(mut self, allow: bool) -> Self {
        self.cfg.allow_cpu_fallback = allow;
        self
    }
    /// Validates and returns the configuration.
    pub fn build(self) -> Result<LaunchConfig> {
        self.cfg.validate()?;
        Ok(self.cfg)
    }
}

/// Facts about a GGUF model needed for memory estimates. Produced by the caller
/// (for example from GGUF metadata); this crate does not parse GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub size_bytes: u64,
    pub n_layers: u32,
    pub n_ctx_train: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

/// Memory estimate for the recommended configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub kv_bytes: u64,
    pub vram_budget: u64,
    pub ram_budget: u64,
}

/// A recommended config with the reasoning behind it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recommendation {
    pub config: LaunchConfig,
    pub rationale: Vec<String>,
    pub estimate: MemoryEstimate,
}

/// Architecture facts that `ModelMeta` cannot express. `ModelArch::dense` describes a
/// model where every layer holds KV and no state is resident besides the weights.
/// Hybrid models mix attention layers (KV cache) with recurrent or state-space layers
/// (fixed-size state, independent of context). For Qwen3.5-family GGUFs the KV layer
/// count is the repeating layers divided by `<arch>.full_attention_interval`.
///
/// The caller supplies these facts; this crate does not parse GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArch {
    /// Layers that hold a KV cache.
    pub kv_layers: u32,
    /// Recurrent-state bytes for llama-server's default slot count. Independent of context.
    pub recurrent_bytes: u64,
    /// Layers llama-server offloads with `-ngl`: repeating layers plus the output layer.
    pub offload_layers: u32,
    /// Device bytes the load needs besides this model, resident whenever any layer is on the
    /// GPU: a draft model loaded with it (`-md`, all of its layers on the GPU). 0 for none.
    #[serde(default)]
    pub extra_device_bytes: u64,
}

impl ModelArch {
    /// Every layer holds KV, nothing else is resident, and `-ngl` equals `n_layers`.
    pub fn dense(model: &ModelMeta) -> Self {
        Self {
            kv_layers: model.n_layers,
            recurrent_bytes: 0,
            offload_layers: model.n_layers,
            extra_device_bytes: 0,
        }
    }
}

/// Reserve kept free for the OS, drivers and other apps: max(10%, 512 MiB).
pub fn headroom(total: u64) -> u64 {
    (total / 10).max(MIN_HEADROOM)
}

fn kv_bytes_per_token_per_layer(m: &ModelMeta, kv: KvCacheType) -> u64 {
    let elems = 2 * u64::from(m.n_kv_heads) * u64::from(m.head_dim); // K and V
    (elems as f64 * kv.bytes_per_element()).ceil() as u64
}

fn round_down_step(ctx: u64) -> u64 {
    ctx / u64::from(CTX_STEP) * u64::from(CTX_STEP)
}

fn ctx_floor(cap: u32) -> u64 {
    round_down_step(u64::from(CTX_FLOOR.min(cap)))
}

/// Device and host costs in bytes for one model, architecture and KV cache type.
/// Weights and recurrent state are spread evenly over the offloadable layers. KV
/// follows the layers that carry it, so its per-layer share is `kv_per_token / layers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Costs {
    /// Offloadable layers (`ModelArch::offload_layers`), at least 1.
    layers: u64,
    /// Weights of all layers.
    weights: u64,
    /// Recurrent state of all layers.
    recurrent: u64,
    /// KV bytes per token across all KV layers at the chosen cache type.
    kv_per_token: u64,
    /// Compute buffers. Counted once any layer is on the device.
    compute: u64,
}

impl Costs {
    /// Device bytes for weights and recurrent state of `k` offloaded layers.
    fn fixed_on(&self, k: u64) -> u64 {
        (k * (self.weights + self.recurrent)).div_ceil(self.layers)
    }

    /// Device bytes for the KV cache of `k` offloaded layers at `ctx` tokens.
    fn kv_on(&self, k: u64, ctx: u64) -> u64 {
        (k * self.kv_per_token * ctx).div_ceil(self.layers)
    }

    /// Total device bytes with `k` layers offloaded at `ctx` tokens.
    fn vram(&self, k: u64, ctx: u64) -> u64 {
        if k == 0 {
            return 0;
        }
        self.compute + self.fixed_on(k) + self.kv_on(k, ctx)
    }

    /// Host bytes for the layers that stay on the CPU, with their share of KV.
    fn ram(&self, k: u64, ctx: u64) -> u64 {
        let off = self.layers - k;
        (off * (self.weights + self.recurrent)).div_ceil(self.layers)
            + (off * self.kv_per_token * ctx).div_ceil(self.layers)
    }
}

/// Largest context (a multiple of `CTX_STEP`, at most `cap`) for `k` layers on the
/// device within `vram_budget`. `None` when it is below the context floor.
fn max_ctx_for(c: &Costs, k: u64, vram_budget: u64, cap: u32) -> Option<u32> {
    if k == 0 || k > c.layers {
        return None;
    }
    let left = vram_budget.checked_sub(c.compute + c.fixed_on(k))?;
    let per_token_num = k * c.kv_per_token;
    let ctx = (left.saturating_mul(c.layers) / per_token_num.max(1)).min(u64::from(cap));
    let ctx = round_down_step(ctx);
    if ctx >= ctx_floor(cap) && ctx > 0 {
        Some(ctx as u32)
    } else {
        None
    }
}

/// Finds (layers, ctx) for GPU offload. It first tries every layer with the context
/// reduced as needed. Failing that, it offloads the most layers that still allow the
/// context floor, keeps the rest in RAM, and requires that RAM to hold them. Returns
/// `None` when neither fits.
fn plan_gpu(c: &Costs, vram_budget: u64, ram_budget: u64, cap: u32) -> Option<(u32, u32)> {
    if let Some(ctx) = max_ctx_for(c, c.layers, vram_budget, cap) {
        return Some((c.layers as u32, ctx));
    }
    (1..c.layers).rev().find_map(|k| {
        let ctx = max_ctx_for(c, k, vram_budget, cap)?;
        (c.ram(k, u64::from(ctx)) <= ram_budget).then_some((k as u32, ctx))
    })
}

fn validate_model(m: &ModelMeta, ctx_target: u32) -> Result<()> {
    if m.size_bytes == 0
        || m.n_layers == 0
        || m.n_ctx_train == 0
        || m.n_kv_heads == 0
        || m.head_dim == 0
    {
        return Err(RaptorError::InvalidInput(
            "model metadata has zero-valued fields".into(),
        ));
    }
    if ctx_target < 256 {
        return Err(RaptorError::InvalidInput(
            "ctx_target must be at least 256".into(),
        ));
    }
    Ok(())
}

fn validate_arch(a: &ModelArch) -> Result<()> {
    if a.offload_layers == 0 || a.kv_layers == 0 || a.kv_layers > a.offload_layers {
        return Err(RaptorError::InvalidInput(
            "model architecture needs 1 or more KV layers, not more than the offloadable layers"
                .into(),
        ));
    }
    Ok(())
}

fn gpu_vendor_for(backend: Backend) -> Option<GpuVendor> {
    match backend {
        Backend::Hip => Some(GpuVendor::Amd),
        Backend::Cuda => Some(GpuVendor::Nvidia),
        Backend::Vulkan => None,
        Backend::Cpu => None,
    }
}

/// Replaces the GGUF-based weight, recurrent and compute figures with llama.cpp's
/// projection for a full-offload launch. Returns the reason when the projection
/// does not agree with the model, so the caller can say why it was not used.
fn calibrate(
    model: &ModelMeta,
    arch: &ModelArch,
    fit: &FitProjection,
) -> std::result::Result<(u64, u64, u64), String> {
    let weights = fit.model_mib * MIB;
    if weights < model.size_bytes / 2 || weights > model.size_bytes + model.size_bytes / 20 {
        return Err(format!(
            "projected weights {} MiB do not match the {} MiB file",
            fit.model_mib,
            model.size_bytes / MIB
        ));
    }
    let kv_f16 = u64::from(arch.kv_layers) * kv_bytes_per_token_per_layer(model, KvCacheType::F16);
    let context = fit.context_mib * MIB;
    let kv_at_ctx = u64::from(fit.ctx) * kv_f16;
    let recurrent = context.checked_sub(kv_at_ctx).ok_or_else(|| {
        format!(
            "projected context {} MiB is below the {} MiB KV cache for ctx {}",
            fit.context_mib,
            kv_at_ctx / MIB,
            fit.ctx
        )
    })?;
    Ok((weights, recurrent, fit.compute_mib * MIB))
}

/// Recommends a launch configuration for a dense `model` (see [`recommend_with_arch`]).
pub fn recommend(
    hw: &HardwareInfo,
    model: &ModelMeta,
    ctx_target: u32,
    backend: Backend,
) -> Result<Recommendation> {
    recommend_with_arch(
        hw,
        model,
        &ModelArch::dense(model),
        ctx_target,
        backend,
        None,
    )
}

/// Recommends a launch configuration for `model` on `hw` with runtime `backend`.
///
/// Rules: keep `headroom` free, computed on the estimate; prefer full GPU offload
/// with the requested context (reducing the context if needed); otherwise offload
/// fewer layers and say so. Quantized KV is tried only after F16 fails. If no GPU
/// layers fit, this returns an error instead of a CPU configuration, so GPU use is
/// never dropped silently.
///
/// Without `fit`, weights are the file size and compute is a fixed 384 MiB, which
/// overestimates. With a full-offload projection from [`crate::fit::probe`], those
/// figures come from llama.cpp itself. A projection that disagrees with the model is
/// ignored, and the rationale says so.
pub fn recommend_with_arch(
    hw: &HardwareInfo,
    model: &ModelMeta,
    arch: &ModelArch,
    ctx_target: u32,
    backend: Backend,
    fit: Option<&FitProjection>,
) -> Result<Recommendation> {
    validate_model(model, ctx_target)?;
    validate_arch(arch)?;
    let mut rationale = Vec::new();
    let cap = if ctx_target > model.n_ctx_train {
        rationale.push(format!(
            "requested context {ctx_target} exceeds the model's trained context {}; capped",
            model.n_ctx_train
        ));
        model.n_ctx_train
    } else {
        ctx_target
    };
    let threads = hw.cpu.cores.max(1);
    let ram_reserve = headroom(hw.ram_total);
    let ram_budget = hw.ram_avail.saturating_sub(ram_reserve);
    rationale.push(format!(
        "RAM available {} MiB, reserve {} MiB, budget {} MiB",
        hw.ram_avail / MIB,
        ram_reserve / MIB,
        ram_budget / MIB
    ));

    if backend == Backend::Cpu {
        let kv = KvCacheType::F16;
        let kvpl = kv_bytes_per_token_per_layer(model, kv);
        let all_kv_per_token = kvpl * u64::from(arch.kv_layers);
        let left = ram_budget
            .checked_sub(model.size_bytes + arch.recurrent_bytes)
            .ok_or_else(|| {
                RaptorError::Runtime(format!(
                    "model needs {} MiB of RAM but the budget is {} MiB",
                    model.size_bytes / MIB,
                    ram_budget / MIB
                ))
            })?;
        let ctx = round_down_step((left / all_kv_per_token.max(1)).min(u64::from(cap)));
        if ctx < ctx_floor(cap) {
            return Err(RaptorError::Runtime(
                "not enough RAM for the model plus a minimal KV cache".into(),
            ));
        }
        if ctx < u64::from(cap) {
            rationale.push(format!(
                "context reduced from {cap} to {ctx} to fit the RAM budget"
            ));
        }
        rationale.push("CPU backend: all weights and KV cache in RAM, no GPU layers".into());
        let ctx = ctx as u32;
        let kv_bytes = u64::from(ctx) * all_kv_per_token;
        let ram_bytes = model.size_bytes + arch.recurrent_bytes + kv_bytes;
        let config = build_config(backend, 0, ctx, threads, false, kv)?;
        rationale.push(format!("threads set to {threads} physical cores"));
        return Ok(Recommendation {
            config,
            rationale,
            estimate: MemoryEstimate {
                vram_bytes: 0,
                ram_bytes,
                kv_bytes,
                vram_budget: 0,
                ram_budget,
            },
        });
    }

    let vendor = gpu_vendor_for(backend);
    let gpu = hw
        .gpu_of(vendor)
        .filter(|g| g.vram_total > 0)
        .ok_or_else(|| {
            RaptorError::Unsupported(format!(
                "no GPU with VRAM for the {} backend",
                backend.label()
            ))
        })?;
    let free = match gpu.vram_free {
        Some(f) => f,
        None => {
            rationale.push(
                "free VRAM not reported; using total VRAM as the upper bound (optimistic)".into(),
            );
            gpu.vram_total
        }
    };
    let vram_reserve = headroom(gpu.vram_total);
    let vram_budget = free.saturating_sub(vram_reserve);
    rationale.push(format!(
        "{}: VRAM free {} MiB, reserve {} MiB, budget {} MiB",
        gpu.name,
        free / MIB,
        vram_reserve / MIB,
        vram_budget / MIB
    ));

    // Weights, recurrent state and compute: llama.cpp's projection when it agrees
    // with the model, otherwise the GGUF-based estimate.
    let n = u64::from(arch.offload_layers);
    let (weights, recurrent, compute) = match fit {
        Some(f) => match calibrate(model, arch, f) {
            Ok(v) => {
                rationale.push(format!(
                    "memory model: llama.cpp projection at ctx {} with all {n} layers on the GPU \
                     (weights {} MiB, recurrent state {} MiB, compute {} MiB)",
                    f.ctx,
                    v.0 / MIB,
                    v.1 / MIB,
                    v.2 / MIB
                ));
                v
            }
            Err(why) => {
                rationale.push(format!(
                    "llama.cpp projection not used ({why}); memory model is the GGUF estimate"
                ));
                (model.size_bytes, arch.recurrent_bytes, COMPUTE_OVERHEAD)
            }
        },
        None => {
            rationale.push(format!(
                "memory model: GGUF estimate (file size for weights, {} MiB recurrent state, \
                 {} MiB compute); no llama.cpp projection supplied",
                arch.recurrent_bytes / MIB,
                COMPUTE_OVERHEAD / MIB
            ));
            (model.size_bytes, arch.recurrent_bytes, COMPUTE_OVERHEAD)
        }
    };
    if arch.kv_layers < arch.offload_layers {
        rationale.push(format!(
            "KV cache on {} of {n} layers (hybrid model)",
            arch.kv_layers
        ));
    }
    let compute = compute + arch.extra_device_bytes;
    if arch.extra_device_bytes > 0 {
        rationale.push(format!(
            "{} MiB of VRAM kept for the draft model loaded with it",
            arch.extra_device_bytes / MIB
        ));
    }

    // Every cache type is planned, and the one that keeps the most layers on the GPU wins; on a
    // tie the earlier (larger) type wins. A smaller cache costs little speed (measured within 1%
    // at up to 16K tokens, about 6% at 32K, perplexity within noise), while each layer left on
    // the CPU costs a lot (a dense model with 25 of 65 layers on the CPU: 1-4 tok/s vs 20.5).
    let mut kv_candidates = vec![KvCacheType::F16];
    if backend.supports_flash_attn() {
        kv_candidates.push(KvCacheType::Q8_0);
        kv_candidates.push(KvCacheType::Q4_0);
    }
    let costs_for = |kv: KvCacheType| Costs {
        layers: n,
        weights,
        recurrent,
        kv_per_token: u64::from(arch.kv_layers) * kv_bytes_per_token_per_layer(model, kv),
        compute,
    };
    let plans: Vec<(KvCacheType, Option<(u32, u32)>)> = kv_candidates
        .iter()
        .map(|&kv| (kv, plan_gpu(&costs_for(kv), vram_budget, ram_budget, cap)))
        .collect();
    let most_layers = plans
        .iter()
        .filter_map(|(_, p)| p.map(|(layers, _)| layers))
        .max();
    let chosen = plans
        .iter()
        .find(|(_, p)| p.map(|(layers, _)| layers) == most_layers && p.is_some())
        .copied();
    if let Some((kv, Some((layers, ctx)))) = chosen {
        let costs = costs_for(kv);
        if kv != KvCacheType::F16 {
            let f16_layers = plans[0].1.map(|(l, _)| l);
            rationale.push(match f16_layers {
                Some(l) => format!(
                    "{} KV cache (requires flash attention): it keeps {layers} layers on the GPU, \
                     the f16 cache only {l}",
                    kv.llama_arg()
                ),
                None => format!(
                    "f16 KV cache does not fit; using {} KV cache (requires flash attention)",
                    kv.llama_arg()
                ),
            });
        }
        let (layers_u, ctx_u) = (u64::from(layers), u64::from(ctx));
        if layers_u == n {
            rationale.push(format!("all {n} layers fit on the GPU"));
        } else {
            rationale.push(format!(
                "offloading {layers} of {n} layers to the GPU; the other {} stay in RAM (slower)",
                n - layers_u
            ));
        }
        if ctx < cap {
            rationale.push(format!(
                "context reduced from {cap} to {ctx} to fit the VRAM budget"
            ));
        } else {
            rationale.push(format!("context {ctx} kept as requested"));
        }
        rationale.push("flash attention on: supported by the GPU backend".into());
        rationale.push(format!("threads set to {threads} physical cores"));

        let vram_bytes = costs.vram(layers_u, ctx_u);
        let ram_bytes = costs.ram(layers_u, ctx_u);
        let kv_bytes = costs.kv_per_token * ctx_u;
        let config = build_config(backend, layers, ctx, threads, true, kv)?;
        return Ok(Recommendation {
            config,
            rationale,
            estimate: MemoryEstimate {
                vram_bytes,
                ram_bytes,
                kv_bytes,
                vram_budget,
                ram_budget,
            },
        });
    }

    Err(RaptorError::Runtime(format!(
        "no GPU layers fit in the {} MiB VRAM budget at a context of at least {} tokens. \
         Choose a smaller quantization, or select the CPU runtime explicitly. \
         Raptor LLM will not fall back to CPU silently.",
        vram_budget / MIB,
        ctx_floor(cap)
    )))
}

/// Assembles a recommended config. The model path stays empty until
/// `with_model` sets it, so everything else is validated here with a placeholder.
fn build_config(
    backend: Backend,
    gpu_layers: u32,
    ctx: u32,
    threads: u32,
    flash: bool,
    kv: KvCacheType,
) -> Result<LaunchConfig> {
    let batch = BATCH_MAX.min(ctx);
    let cfg = LaunchConfig {
        model_path: PathBuf::new(),
        backend,
        gpu_layers,
        ctx_size: ctx,
        batch_size: batch,
        ubatch_size: UBATCH_MAX.min(batch),
        threads,
        flash_attn: flash,
        kv_cache: kv,
        host: "127.0.0.1".into(),
        port: 8080,
        allow_cpu_fallback: false,
        embeddings: false,
        pooling: None,
        speculative: None,
        draft_model: None,
        parallel_slots: None,
        threads_batch: None,
        api_key: None,
    };
    let mut probe = cfg.clone();
    probe.model_path = PathBuf::from("model.gguf");
    probe.validate()?;
    Ok(cfg)
}

impl LaunchConfig {
    /// Sets the model path on a recommended config and re-validates it.
    pub fn with_model(mut self, path: impl Into<PathBuf>) -> Result<Self> {
        self.model_path = path.into();
        self.validate()?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, GpuInfo, Os};

    fn cfg_builder(backend: Backend) -> LaunchConfigBuilder {
        LaunchConfig::builder("C:/models/m.gguf", backend)
    }

    fn model_8b() -> ModelMeta {
        ModelMeta {
            size_bytes: 8 * GIB,
            n_layers: 40,
            n_ctx_train: 32768,
            n_kv_heads: 8,
            head_dim: 128,
        }
    }

    fn hw_rx9060(vram_free_mib: Option<u64>) -> HardwareInfo {
        HardwareInfo {
            os: Os::Windows,
            cpu: CpuInfo {
                brand: "cpu".into(),
                cores: 8,
                threads: 16,
                features: vec![],
            },
            ram_total: 32 * GIB,
            ram_avail: 20 * GIB,
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Amd,
                name: "AMD Radeon RX 9060 XT".into(),
                vram_total: 16 * GIB,
                vram_free: vram_free_mib.map(|m| m * MIB),
                arch: Some("gfx1200".into()),
                driver: None,
            }],
        }
    }

    fn hw_with_vram(total: u64, free: u64) -> HardwareInfo {
        let mut hw = hw_rx9060(None);
        hw.gpus[0].vram_total = total;
        hw.gpus[0].vram_free = Some(free);
        hw
    }

    #[test]
    fn builder_defaults_differ_for_gpu_and_cpu() {
        let gpu = cfg_builder(Backend::Hip).build().unwrap();
        assert_eq!(gpu.gpu_layers, 999);
        assert!(gpu.flash_attn);
        let cpu = cfg_builder(Backend::Cpu).build().unwrap();
        assert_eq!(cpu.gpu_layers, 0);
        assert!(!cpu.flash_attn);
    }

    #[test]
    fn args_contain_expected_flags_and_no_secret() {
        let cfg = cfg_builder(Backend::Hip)
            .ctx_size(8192)
            .api_key("sekrit")
            .kv_cache(KvCacheType::Q8_0)
            .build()
            .unwrap();
        let args = cfg.args();
        let pairs: Vec<(&str, &str)> = args
            .chunks(2)
            .map(|c| (c[0].as_str(), c[1].as_str()))
            .collect();
        assert!(pairs.contains(&("-ngl", "999")));
        assert!(pairs.contains(&("-c", "8192")));
        assert!(pairs.contains(&("-fa", "on")));
        assert!(pairs.contains(&("--cache-type-k", "q8_0")));
        assert!(pairs.contains(&("--cache-type-v", "q8_0")));
        assert!(pairs.contains(&("--host", "127.0.0.1")));
        assert!(pairs.contains(&("--port", "8080")));
        assert!(
            !args.iter().any(|a| a.contains("sekrit")),
            "api key must not appear in argv"
        );
    }

    #[test]
    fn args_request_device_level_log_verbosity() {
        let args = cfg_builder(Backend::Hip).build().unwrap().args();
        let n = args.len();
        assert_eq!(&args[n - 2..], ["-lv", "4"]);
        assert_eq!(LLAMA_LOG_VERBOSITY, 4);
    }

    #[test]
    fn cpu_fallback_is_off_by_default_and_opt_in() {
        let cfg = cfg_builder(Backend::Hip).build().unwrap();
        assert!(!cfg.allow_cpu_fallback);
        let cfg = cfg_builder(Backend::Hip)
            .allow_cpu_fallback(true)
            .build()
            .unwrap();
        assert!(cfg.allow_cpu_fallback);
    }

    #[test]
    fn api_key_travels_in_env_and_is_redacted_in_debug() {
        let cfg = cfg_builder(Backend::Cpu)
            .api_key("sekrit")
            .host("0.0.0.0")
            .build()
            .unwrap();
        assert_eq!(
            cfg.env(),
            vec![("LLAMA_API_KEY".to_string(), "sekrit".to_string())]
        );
        assert!(!format!("{cfg:?}").contains("sekrit"));
        assert!(cfg.has_api_key());
    }

    #[test]
    fn rejects_gpu_layers_on_cpu_backend() {
        assert!(cfg_builder(Backend::Cpu).gpu_layers(10).build().is_err());
    }

    #[test]
    fn rejects_quantized_kv_without_flash_attention() {
        let r = cfg_builder(Backend::Hip)
            .flash_attn(false)
            .kv_cache(KvCacheType::Q4_0)
            .build();
        assert!(matches!(r, Err(RaptorError::InvalidInput(m)) if m.contains("flash attention")));
    }

    #[test]
    fn rejects_flash_attention_on_cpu() {
        assert!(cfg_builder(Backend::Cpu).flash_attn(true).build().is_err());
    }

    #[test]
    fn rejects_non_loopback_host_without_key() {
        assert!(cfg_builder(Backend::Cpu).host("0.0.0.0").build().is_err());
        assert!(cfg_builder(Backend::Cpu)
            .host("0.0.0.0")
            .api_key("k")
            .build()
            .is_ok());
    }

    #[test]
    fn rejects_hostnames_and_injection_like_hosts() {
        assert!(cfg_builder(Backend::Cpu)
            .host("127.0.0.1 --evil")
            .build()
            .is_err());
        assert!(cfg_builder(Backend::Cpu).host("localhost").build().is_err());
    }

    #[test]
    fn rejects_bad_numbers_and_empty_path() {
        assert!(cfg_builder(Backend::Cpu).ubatch_size(4096).build().is_err());
        assert!(cfg_builder(Backend::Cpu).ctx_size(10).build().is_err());
        assert!(cfg_builder(Backend::Cpu).port(0).build().is_err());
        assert!(LaunchConfig::builder("", Backend::Cpu).build().is_err());
    }

    #[test]
    fn rejects_whitespace_in_api_key() {
        assert!(cfg_builder(Backend::Cpu)
            .host("0.0.0.0")
            .api_key("a b")
            .build()
            .is_err());
    }

    #[test]
    fn endpoint_brackets_ipv6() {
        let v6 = cfg_builder(Backend::Cpu).host("::1").build().unwrap();
        assert_eq!(v6.endpoint(), "http://[::1]:8080");
        let v4 = cfg_builder(Backend::Cpu).build().unwrap();
        assert_eq!(v4.endpoint(), "http://127.0.0.1:8080");
    }

    #[test]
    fn headroom_is_max_of_ten_percent_and_512mib() {
        assert_eq!(headroom(16 * GIB), 16 * GIB / 10);
        assert_eq!(headroom(2 * GIB), MIN_HEADROOM);
    }

    #[test]
    fn kv_bytes_per_token_per_layer_matches_hand_calc() {
        assert_eq!(
            kv_bytes_per_token_per_layer(&model_8b(), KvCacheType::F16),
            4096
        );
        assert_eq!(
            kv_bytes_per_token_per_layer(&model_8b(), KvCacheType::Q8_0),
            2176
        );
    }

    #[test]
    fn recommend_full_offload_keeps_requested_context() {
        let rec = recommend(&hw_rx9060(None), &model_8b(), 8192, Backend::Hip).unwrap();
        assert_eq!(rec.config.gpu_layers, 40);
        assert_eq!(rec.config.ctx_size, 8192);
        assert_eq!(rec.config.kv_cache, KvCacheType::F16);
        assert!(rec.config.flash_attn);
        assert!(rec
            .rationale
            .iter()
            .any(|r| r.contains("all 40 layers fit")));
        assert!(rec.estimate.vram_bytes <= rec.estimate.vram_budget + COMPUTE_OVERHEAD);
    }

    #[test]
    fn recommend_reduces_context_before_layers() {
        let model = ModelMeta {
            size_bytes: 13 * GIB,
            ..model_8b()
        };
        let rec = recommend(&hw_rx9060(None), &model, 32768, Backend::Hip).unwrap();
        assert_eq!(
            rec.config.gpu_layers, 40,
            "layers must stay on GPU when context can shrink"
        );
        assert!(rec.config.ctx_size < 32768);
        assert!(rec.config.ctx_size.is_multiple_of(CTX_STEP));
        assert!(rec.rationale.iter().any(|r| r.contains("context reduced")));
    }

    #[test]
    fn recommend_offloads_fewer_layers_when_context_floor_does_not_fit() {
        let model = ModelMeta {
            size_bytes: 20 * GIB,
            ..model_8b()
        };
        let rec = recommend(&hw_rx9060(None), &model, 8192, Backend::Hip).unwrap();
        assert!(
            rec.config.gpu_layers > 0 && rec.config.gpu_layers < 40,
            "got {}",
            rec.config.gpu_layers
        );
        assert!(rec.rationale.iter().any(|r| r.contains("stay in RAM")));
        assert!(rec.estimate.ram_bytes > 0);
    }

    #[test]
    fn recommend_uses_q8_kv_only_when_f16_cannot_fit_any_layer() {
        // 40 layers of 100 MiB, with a very KV-heavy shape (64 KV heads).
        let model = ModelMeta {
            size_bytes: 40 * 100 * MIB,
            n_layers: 40,
            n_ctx_train: 32768,
            n_kv_heads: 64,
            head_dim: 128,
        };
        // Budget = free - reserve(512 MiB) = 534 MiB: F16 fits zero layers, Q8 fits one.
        let hw = hw_with_vram(4 * GIB, (534 + 512) * MIB);
        let rec = recommend(&hw, &model, 8192, Backend::Hip).unwrap();
        assert_eq!(rec.config.kv_cache, KvCacheType::Q8_0);
        assert!(rec.config.flash_attn);
        assert!(rec.rationale.iter().any(|r| r.contains("q8_0 KV cache")));
    }

    #[test]
    fn a_smaller_cache_wins_when_it_keeps_more_layers_on_the_gpu() {
        // f16 at the 2,048-token floor needs 320 MiB of KV, q8_0 170 and q4_0 90: with these
        // weights f16 cannot keep all 40 layers, the quantized caches can, and of those the larger
        // (q8_0) wins.
        let model = ModelMeta {
            size_bytes: 14150 * MIB,
            ..model_8b()
        };
        let rec = recommend(&hw_rx9060(None), &model, 8192, Backend::Hip).unwrap();
        assert_eq!(rec.config.gpu_layers, 40, "{:?}", rec.rationale);
        assert_eq!(rec.config.kv_cache, KvCacheType::Q8_0);
        assert!(rec
            .rationale
            .iter()
            .any(|r| r.contains("keeps 40 layers on the GPU, the f16 cache only")));
        // When f16 keeps every layer it is still the choice.
        let small = recommend(&hw_rx9060(None), &model_8b(), 8192, Backend::Hip).unwrap();
        assert_eq!(small.config.kv_cache, KvCacheType::F16);
    }

    #[test]
    fn a_draft_models_memory_is_kept_free() {
        let model = ModelMeta {
            size_bytes: 13 * GIB,
            ..model_8b()
        };
        let plain = ModelArch::dense(&model);
        let with_draft = ModelArch {
            extra_device_bytes: 560 * MIB,
            ..plain
        };
        let hw = hw_rx9060(None);
        let a = recommend_with_arch(&hw, &model, &plain, 32768, Backend::Hip, None).unwrap();
        let b = recommend_with_arch(&hw, &model, &with_draft, 32768, Backend::Hip, None).unwrap();
        assert!(
            b.config.ctx_size < a.config.ctx_size,
            "{} vs {}",
            b.config.ctx_size,
            a.config.ctx_size
        );
        assert!(b.estimate.vram_bytes <= b.estimate.vram_budget);
        assert!(b
            .rationale
            .iter()
            .any(|r| r.contains("560 MiB of VRAM kept for the draft")));
    }

    #[test]
    fn recommend_errors_instead_of_silent_cpu_fallback() {
        let model = ModelMeta {
            size_bytes: 20 * GIB,
            ..model_8b()
        };
        let hw = hw_with_vram(4 * GIB, 3 * GIB);
        let err = recommend(&hw, &model, 8192, Backend::Hip).unwrap_err();
        assert!(
            matches!(err, RaptorError::Runtime(m) if m.contains("Raptor LLM will not fall back to CPU"))
        );
    }

    #[test]
    fn recommend_rejects_missing_gpu_for_backend() {
        let err = recommend(&hw_rx9060(None), &model_8b(), 8192, Backend::Cuda).unwrap_err();
        assert!(matches!(err, RaptorError::Unsupported(_)));
    }

    #[test]
    fn recommend_cpu_backend_uses_ram_and_zero_layers() {
        let model = ModelMeta {
            size_bytes: 4 * GIB,
            ..model_8b()
        };
        let rec = recommend(&hw_rx9060(None), &model, 8192, Backend::Cpu).unwrap();
        assert_eq!(rec.config.gpu_layers, 0);
        assert!(!rec.config.flash_attn);
        assert_eq!(rec.config.ctx_size, 8192);
        assert_eq!(rec.estimate.vram_bytes, 0);
        assert!(rec.estimate.ram_bytes >= 4 * GIB);
    }

    #[test]
    fn recommend_cpu_reduces_context_to_ram_budget() {
        let model = ModelMeta {
            size_bytes: 16 * GIB,
            n_ctx_train: 131072,
            ..model_8b()
        };
        let hw = hw_rx9060(None);
        let rec = recommend(&hw, &model, 131072, Backend::Cpu).unwrap();
        assert!(rec.config.ctx_size < 131072);
        assert!(rec.estimate.ram_bytes <= rec.estimate.ram_budget);
    }

    #[test]
    fn recommend_cpu_fails_when_model_exceeds_ram() {
        let model = ModelMeta {
            size_bytes: 40 * GIB,
            ..model_8b()
        };
        assert!(recommend(&hw_rx9060(None), &model, 4096, Backend::Cpu).is_err());
    }

    #[test]
    fn recommend_caps_context_at_trained_length_with_rationale() {
        let model = ModelMeta {
            n_ctx_train: 4096,
            ..model_8b()
        };
        let rec = recommend(&hw_rx9060(None), &model, 16384, Backend::Hip).unwrap();
        assert_eq!(rec.config.ctx_size, 4096);
        assert!(rec.rationale.iter().any(|r| r.contains("capped")));
    }

    #[test]
    fn recommend_rejects_zero_metadata() {
        let model = ModelMeta {
            n_layers: 0,
            ..model_8b()
        };
        assert!(recommend(&hw_rx9060(None), &model, 4096, Backend::Hip).is_err());
    }

    #[test]
    fn recommend_marks_optimistic_when_free_vram_unknown() {
        let rec = recommend(&hw_rx9060(None), &model_8b(), 4096, Backend::Hip).unwrap();
        assert!(rec.rationale.iter().any(|r| r.contains("optimistic")));
    }

    #[test]
    fn with_model_validates_path() {
        let rec = recommend(&hw_rx9060(None), &model_8b(), 4096, Backend::Hip).unwrap();
        let cfg = rec.config.with_model("D:/models/a.gguf").unwrap();
        assert_eq!(cfg.model_path, PathBuf::from("D:/models/a.gguf"));
        assert!(cfg.clone().with_model("").is_err());
    }
}

/// Regression tests for hybrid models and llama.cpp projections. The fixtures are the
/// measured values for Qwen3.8-27B-UD-IQ3_S on an RX 9060 XT (16304 MiB total, 16158
/// MiB free, llama.cpp b12340): see tests/fixtures/real_b12340_hip_gpu_lv4.log.
#[cfg(test)]
mod hybrid_tests {
    use super::*;
    use crate::fit::FitProjection;
    use crate::hardware::{CpuInfo, GpuInfo, Os};

    /// File 11483 MiB; `block_count` 65 (GGUF key); 16 KV layers (64 repeating layers
    /// with full_attention_interval 4); 66 layers offloadable (repeating + output);
    /// 4 KV heads of 256 dims; recurrent state 598.5 MiB at 4 slots (rounded up).
    fn qwen35_meta() -> ModelMeta {
        ModelMeta {
            size_bytes: 11483 * MIB,
            n_layers: 65,
            n_ctx_train: 262144,
            n_kv_heads: 4,
            head_dim: 256,
        }
    }

    fn qwen35_arch() -> ModelArch {
        ModelArch {
            kv_layers: 16,
            recurrent_bytes: 599 * MIB,
            offload_layers: 66,
            extra_device_bytes: 0,
        }
    }

    /// Server log at ctx 4096, 4 slots: ROCm0 model 10616, KV+recurrent 854, compute 146.
    fn projection_4096() -> FitProjection {
        FitProjection {
            ctx: 4096,
            model_mib: 10616,
            context_mib: 854,
            compute_mib: 146,
        }
    }

    /// `llama-fit-params --fit-print on` at ctx 8192, 4 slots (measured on this machine).
    fn projection_8192() -> FitProjection {
        FitProjection {
            ctx: 8192,
            model_mib: 10616,
            context_mib: 1110,
            compute_mib: 527,
        }
    }

    fn rx9060_16158_free() -> HardwareInfo {
        HardwareInfo {
            os: Os::Windows,
            cpu: CpuInfo {
                brand: "AMD Ryzen 7 3800X".into(),
                cores: 8,
                threads: 16,
                features: vec![],
            },
            ram_total: 65459 * MIB,
            ram_avail: 52433 * MIB,
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Amd,
                name: "AMD Radeon RX 9060 XT".into(),
                vram_total: 16304 * MIB,
                vram_free: Some(16158 * MIB),
                arch: Some("gfx1200".into()),
                driver: None,
            }],
        }
    }

    /// Budget in these fixtures: 16158 - max(10%, 512) = 14528 MiB.
    const FIXTURE_VRAM_BUDGET: u64 = 14528 * MIB;

    fn gpu_layers_of(rec: &Recommendation) -> u32 {
        rec.config.gpu_layers
    }

    #[test]
    fn qwen35_full_offload_at_8192_from_gguf_estimate() {
        let rec = recommend_with_arch(
            &rx9060_16158_free(),
            &qwen35_meta(),
            &qwen35_arch(),
            8192,
            Backend::Hip,
            None,
        )
        .unwrap();
        assert_eq!(gpu_layers_of(&rec), 66, "all 66 offloadable layers");
        assert_eq!(rec.config.ctx_size, 8192);
        assert!(rec
            .rationale
            .iter()
            .any(|r| r.contains("all 66 layers fit")));
        assert!(rec.rationale.iter().any(|r| r.contains("GGUF estimate")));
        // 384 compute + 11483 + 599 weights/state + 512 KV = 12978 MiB, within 14528.
        assert_eq!(rec.estimate.vram_bytes, (384 + 11483 + 599 + 512) * MIB);
        assert!(rec.estimate.vram_bytes <= FIXTURE_VRAM_BUDGET);
        assert!(rec.config.args().windows(2).any(|w| w == ["-ngl", "66"]));
    }

    #[test]
    fn qwen35_full_offload_at_4096_and_8192_with_llama_projection() {
        for (ctx, proj, expected_mib) in [
            (4096, projection_4096(), 11616),
            (8192, projection_8192(), 12253),
        ] {
            let rec = recommend_with_arch(
                &rx9060_16158_free(),
                &qwen35_meta(),
                &qwen35_arch(),
                ctx,
                Backend::Hip,
                Some(&proj),
            )
            .unwrap();
            assert_eq!(gpu_layers_of(&rec), 66, "ctx {ctx}: full offload");
            assert_eq!(
                rec.config.ctx_size, ctx,
                "ctx {ctx}: requested context kept"
            );
            assert!(rec
                .rationale
                .iter()
                .any(|r| r.contains("llama.cpp projection at ctx")));
            // Calibrated total equals llama.cpp's own figure (fixture: 11616 at 4096; the
            // server reports 11617 with rounding; fit-print at 8192: 12254, here 12253).
            assert_eq!(rec.estimate.vram_bytes, expected_mib * MIB, "ctx {ctx}");
            assert!(rec.estimate.vram_bytes <= FIXTURE_VRAM_BUDGET);
        }
    }

    #[test]
    fn qwen35_requested_huge_context_reduces_context_not_layers() {
        let rec = recommend_with_arch(
            &rx9060_16158_free(),
            &qwen35_meta(),
            &qwen35_arch(),
            262144,
            Backend::Hip,
            None,
        )
        .unwrap();
        assert_eq!(gpu_layers_of(&rec), 66);
        assert!(rec.config.ctx_size < 262144);
        assert!(rec.config.ctx_size.is_multiple_of(CTX_STEP));
        assert!(rec.estimate.vram_bytes <= rec.estimate.vram_budget);
    }

    #[test]
    fn too_big_model_reduces_layers_and_stays_within_budget() {
        let model = ModelMeta {
            size_bytes: 20 * GIB,
            ..qwen35_meta()
        };
        let arch = qwen35_arch();
        let rec = recommend_with_arch(
            &rx9060_16158_free(),
            &model,
            &arch,
            8192,
            Backend::Hip,
            None,
        )
        .unwrap();
        let layers = gpu_layers_of(&rec);
        assert!(layers > 0 && layers < 66, "got {layers}");
        assert!(rec.config.ctx_size >= 2048);
        assert!(rec.estimate.vram_bytes <= rec.estimate.vram_budget);
        assert!(rec.estimate.ram_bytes <= rec.estimate.ram_budget);
        assert!(rec.rationale.iter().any(|r| r.contains("stay in RAM")));
    }

    #[test]
    fn model_too_big_for_vram_and_ram_errors_instead_of_overplanning() {
        let model = ModelMeta {
            size_bytes: 80 * GIB,
            ..qwen35_meta()
        };
        let err = recommend_with_arch(
            &rx9060_16158_free(),
            &model,
            &qwen35_arch(),
            8192,
            Backend::Hip,
            None,
        );
        assert!(
            err.is_err(),
            "no feasible plan must be an error, not an overcommit"
        );
    }

    #[test]
    fn projection_that_disagrees_with_the_model_is_not_used() {
        let mut bad = projection_8192();
        bad.model_mib = 3000;
        let rec = recommend_with_arch(
            &rx9060_16158_free(),
            &qwen35_meta(),
            &qwen35_arch(),
            8192,
            Backend::Hip,
            Some(&bad),
        )
        .unwrap();
        assert!(rec
            .rationale
            .iter()
            .any(|r| r.contains("llama.cpp projection not used")));
        assert!(rec.rationale.iter().any(|r| r.contains("GGUF estimate")));
        assert_eq!(gpu_layers_of(&rec), 66);
    }

    #[test]
    fn projection_context_below_kv_cache_is_not_used() {
        let mut bad = projection_8192();
        bad.context_mib = 100;
        let rec = recommend_with_arch(
            &rx9060_16158_free(),
            &qwen35_meta(),
            &qwen35_arch(),
            8192,
            Backend::Hip,
            Some(&bad),
        )
        .unwrap();
        assert!(rec
            .rationale
            .iter()
            .any(|r| r.contains("llama.cpp projection not used")));
    }

    #[test]
    fn invalid_architecture_is_rejected() {
        let no_kv = ModelArch {
            kv_layers: 0,
            ..qwen35_arch()
        };
        let err = recommend_with_arch(
            &rx9060_16158_free(),
            &qwen35_meta(),
            &no_kv,
            8192,
            Backend::Hip,
            None,
        );
        assert!(matches!(err, Err(RaptorError::InvalidInput(_))));
    }

    #[test]
    fn dense_model_recommendation_is_unchanged_by_the_arch_api() {
        let model = ModelMeta {
            size_bytes: 8 * GIB,
            n_layers: 40,
            n_ctx_train: 32768,
            n_kv_heads: 8,
            head_dim: 128,
        };
        let hw = rx9060_16158_free();
        let old = recommend(&hw, &model, 8192, Backend::Hip).unwrap();
        let new = recommend_with_arch(
            &hw,
            &model,
            &ModelArch::dense(&model),
            8192,
            Backend::Hip,
            None,
        )
        .unwrap();
        assert_eq!(old, new);
        assert_eq!(old.config.gpu_layers, 40);
    }

    #[test]
    fn slots_and_batch_threads_are_off_by_default_validated_and_the_only_difference() {
        let base = || LaunchConfig::builder("C:/m.gguf", Backend::Cpu);
        let plain = base().build().unwrap().args();
        assert!(!plain.iter().any(|a| a == "-np" || a == "-tb"));
        let tuned = base()
            .parallel_slots(1)
            .threads_batch(12)
            .build()
            .unwrap()
            .args();
        assert!(tuned.windows(2).any(|w| w == ["-np", "1"]));
        assert!(tuned.windows(2).any(|w| w == ["-tb", "12"]));
        assert_eq!(&tuned[tuned.len() - 2..], ["-lv", "4"]);
        let mut without = tuned.clone();
        for flag in ["-np", "-tb"] {
            let at = without.iter().position(|a| a == flag).unwrap();
            without.drain(at..at + 2);
        }
        assert_eq!(without, plain);
        assert!(base().parallel_slots(0).build().is_err());
        assert!(base().parallel_slots(17).build().is_err());
        assert!(base().threads_batch(0).build().is_err());
    }

    #[test]
    fn speculative_decoding_is_off_by_default_and_never_combined_with_embeddings() {
        let base = || LaunchConfig::builder("C:/m.gguf", Backend::Cpu);
        let plain = base().build().unwrap().args();
        assert!(!plain.iter().any(|a| a == "--spec-type"));
        let mtp = base().speculative(SpecMode::Mtp).build().unwrap().args();
        let at = mtp.iter().position(|a| a == "--spec-type").unwrap();
        assert_eq!(mtp[at + 1], "draft-mtp,ngram-simple");
        assert_eq!(&mtp[mtp.len() - 2..], ["-lv", "4"]);
        let ngram = base().speculative(SpecMode::Ngram).build().unwrap().args();
        assert!(ngram.windows(2).any(|w| w == ["--spec-type", "ngram-mod"]));
        let emb = base()
            .speculative(SpecMode::Mtp)
            .embeddings(true)
            .build()
            .unwrap()
            .args();
        assert!(!emb.iter().any(|a| a == "--spec-type"));
        let mut without = mtp.clone();
        without.drain(at..at + 2);
        assert_eq!(without, plain, "the flag is the only difference");
    }

    #[test]
    fn dflash_needs_a_draft_model_and_passes_it_with_every_layer_on_the_gpu() {
        let base = || LaunchConfig::builder("C:/m.gguf", Backend::Hip);
        assert!(base().speculative(SpecMode::Dflash).build().is_err());
        let args = base()
            .speculative(SpecMode::Dflash)
            .draft_model("C:/draft.gguf")
            .build()
            .unwrap()
            .args();
        assert!(args
            .windows(2)
            .any(|w| w == ["--spec-type", "draft-dflash,ngram-simple"]));
        let at = args.iter().position(|a| a == "-md").unwrap();
        assert_eq!(
            &args[at..at + 6],
            [
                "-md",
                "C:/draft.gguf",
                "-ngld",
                "999",
                "--spec-draft-p-min",
                "0.3"
            ]
        );
        assert_eq!(&args[args.len() - 2..], ["-lv", "4"]);
        // A draft model alone, without the mode, changes nothing.
        let plain = base().build().unwrap().args();
        let unused = base().draft_model("C:/draft.gguf").build().unwrap().args();
        assert_eq!(unused, plain);
    }

    #[test]
    fn pooling_is_passed_only_with_embeddings_and_only_known_values() {
        let args = |b: LaunchConfigBuilder| b.build().unwrap().args();
        let base = || LaunchConfig::builder("C:/m.gguf", Backend::Cpu);
        let with = args(base().embeddings(true).pooling("mean"));
        let at = with.iter().position(|a| a == "--pooling").unwrap();
        assert_eq!(with[at + 1], "mean");
        assert_eq!(&with[with.len() - 2..], ["-lv", "4"]);
        assert!(!args(base().pooling("mean"))
            .iter()
            .any(|a| a == "--pooling"));
        assert!(base().embeddings(true).pooling("mean; rm").build().is_err());
    }

    #[test]
    fn embeddings_flag_is_off_by_default_and_additive() {
        let chat = LaunchConfig::builder("C:/m.gguf", Backend::Cpu)
            .build()
            .unwrap();
        assert!(!chat.embeddings);
        assert!(!chat.args().iter().any(|a| a == "--embeddings"));
        let emb = LaunchConfig::builder("C:/m.gguf", Backend::Cpu)
            .embeddings(true)
            .build()
            .unwrap();
        let args = emb.args();
        assert!(args.iter().any(|a| a == "--embeddings"));
        // -lv stays last, and the chat args are otherwise identical.
        assert_eq!(&args[args.len() - 2..], ["-lv", "4"]);
        let mut without: Vec<String> = args.clone();
        without.retain(|a| a != "--embeddings");
        assert_eq!(without, chat.args());
    }
}
