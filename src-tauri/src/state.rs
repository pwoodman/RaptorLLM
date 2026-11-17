//! `AppState`: the composition root. It builds every subsystem from the data directories
//! and owns the gateway listener, the download outcomes, the running chat streams, the
//! loaded-model record and the performance store.

use crate::adapters::{
    loopback_connect, Connect, ContextAdapter, ContextHub, InstalledModels, RuntimeUpstream,
    ServerProbe,
};
use crate::conversations::ConversationStore;
use crate::job::JobSpawner;
use crate::perf::{self, BenchmarkOutcome, KnownGood, PerfComparison, PerfEntry};
use crate::semantic::{ActiveModel, ActiveServer, LlamaHttp, LoopbackHttp, SemanticSetup};
use crate::settings::{PinnedRuntime, Settings, SettingsStore};
use crate::setup::{self, AvailableRuntime, SetupInput, SetupStatus};
use crate::tokens::ServerAwareCounter;
use raptor_core::{DataDirs, RaptorError, Result};
use raptor_gateway::{
    BindMode, ContextService, Gateway, GatewayConfig, GatewayDeps, Handle, Metrics, TlsConfig,
    Upstream,
};
use raptor_models::Preset;
use raptor_models::{DownloadProgress, JobState, ModelEntry, ModelManager};
use raptor_runtime::config::{LaunchConfig, MemoryEstimate, ModelArch, ModelMeta, Recommendation};
use raptor_runtime::devices::BackendLoadFailure;
use raptor_runtime::devices::HipSdkSearch;
use raptor_runtime::fit::SERVER_DEFAULT_SLOTS;
use raptor_runtime::hardware::{HardwareInfo, SystemProbe};
use raptor_runtime::http::ReqwestClient;
use raptor_runtime::install::ReqwestFetcher;
use raptor_runtime::manifest::{parse_manifest_index, select_runtime, Backend, Manifest};
use raptor_runtime::perfmem::{BenchRecord, ConfigSnapshot, PerfStore};
use raptor_runtime::process::{FileLogSink, RestartPolicy, ServerState, Timing};
use raptor_runtime::sys::{CommandRunner, SystemCommandRunner, TcpPortCheck};
use raptor_runtime::tokenizer::{LoopbackTransport, TokenizeTransport};
use raptor_runtime::{EngineDeps, EngineStatus, RuntimeEngine, RuntimeSummary};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

pub const RUNTIME_MANIFEST_FILE: &str = "runtime-manifest.json";

/// Gateway listener and its last error.
#[derive(Default)]
pub struct GatewayControl {
    handle: Option<Handle>,
    last_error: Option<String>,
}

/// Gateway state as the UI sees it.
#[derive(Debug, Clone, Serialize)]
pub struct GatewayStatus {
    pub running: bool,
    pub addr: Option<String>,
    pub metrics: Option<Metrics>,
    pub last_error: Option<String>,
    /// True when the listener binds all interfaces.
    pub lan: bool,
    pub tls: bool,
    /// Lowercase hex SHA-256 of the serving certificate. Clients pin this value.
    pub tls_fingerprint_sha256: Option<String>,
    /// True only with `allow_insecure_lan`, which the shell never sets.
    pub insecure: bool,
}

/// How a download ended. `status` is `completed`, `cancelled` or `failed`.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadFinished {
    pub key: String,
    pub status: String,
    pub path: Option<String>,
    pub sha256: Option<String>,
    pub error: Option<String>,
}

/// One row of the download queue.
#[derive(Debug, Clone, Serialize)]
pub struct QueueEntry {
    pub key: String,
    pub repo: String,
    pub file_name: String,
    pub progress: DownloadProgress,
    pub finished: Option<DownloadFinished>,
}

/// Where the launch configuration of a load came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    /// The fastest stable configuration this machine has measured for this model and runtime.
    KnownGood,
    /// Computed from the hardware and the model header. Used when nothing is known yet, or
    /// when the known-good context does not match the requested one.
    Recommended,
}

/// Result of a successful model load.
#[derive(Debug, Clone, Serialize)]
pub struct LoadModelResult {
    /// Includes the GPU verdict. A GPU build that is not verified is reported as such.
    pub status: EngineStatus,
    /// Why this configuration was chosen, in plain words.
    pub rationale: Vec<String>,
    /// Memory estimate of the recommended configuration.
    pub estimate: MemoryEstimate,
    pub model_id: String,
    pub config_source: ConfigSource,
    /// The benchmark run right after the model became Ready. Its stability is always reported.
    pub benchmark: BenchmarkOutcome,
    /// Name of the preset the load applied (its context length and model), if any.
    pub preset: Option<String>,
    /// True only when a GPU backend had no device and the load went ahead on CPU because
    /// `allow_cpu_fallback` is set. The GPU verdict is then CPU-only; the UI must say so.
    pub cpu_fallback: bool,
    /// What the load did to GPU memory, when it could be read (Windows).
    pub gpu_memory: Option<GpuMemoryReport>,
    /// Whether the loaded model has a thinking-effort control, and what each level maps to.
    pub thinking: crate::thinking::ThinkingView,
}

/// What a load did to GPU memory. `spilled` without `expected` is a problem the user must hear
/// about: the model works but generation is much slower (Qwen3.8-27B at a 167k window: 47 -> ~10 tok/s).
#[derive(Debug, Clone, Serialize)]
pub struct GpuMemoryReport {
    pub dedicated_mib: u64,
    pub shared_growth_mib: u64,
    pub spilled: bool,
    /// True when the spill is the plan: a sparse Mixture-of-experts model larger than VRAM.
    pub expected: bool,
}

/// The backend choice of one model as the Library shows it.
#[derive(Debug, Clone, Serialize)]
pub struct BackendView {
    pub model_id: String,
    pub preference: crate::backend_choice::BackendPref,
    /// What a load would use now: `rocm`, `vulkan`, `default` (the machine's normal runtime) or
    /// `unavailable` (the chosen backend is not installed; `note` says so).
    pub effective: String,
    pub rocm_installed: bool,
    pub cuda_installed: bool,
    pub sycl_installed: bool,
    pub vulkan_installed: bool,
    /// A sparse Mixture-of-experts model larger than VRAM.
    pub spills: bool,
    pub note: Option<String>,
    /// The measured comparison, quoted next to the selector.
    pub measured: &'static str,
}

fn auto_backend_for(
    hw: &HardwareInfo,
    rocm_installed: bool,
    cuda_installed: bool,
    sycl_installed: bool,
    vulkan_installed: bool,
) -> Option<Backend> {
    let gpu = hw.gpu_of(None)?;
    let native = match gpu.vendor {
        raptor_runtime::hardware::GpuVendor::Amd => (rocm_installed, Backend::Hip),
        raptor_runtime::hardware::GpuVendor::Nvidia => (cuda_installed, Backend::Cuda),
        raptor_runtime::hardware::GpuVendor::Intel => (sycl_installed, Backend::Sycl),
        raptor_runtime::hardware::GpuVendor::Other => (false, Backend::Vulkan),
    };
    if native.0 {
        Some(native.1)
    } else if vulkan_installed {
        Some(Backend::Vulkan)
    } else {
        None
    }
}

/// Layers `-ngl` can place: `block_count` plus the output layer. `block_count` already includes
/// any NextN/MTP layers, so a model with an MTP head counts it (Qwen3.8-27B: 65 blocks, 66
/// offloadable). Counting only the main stack left that layer on the CPU and cut generation from
/// 47 to 5 tok/s. None when the header has no layer count.
fn offload_layers_of(entry: &ModelEntry) -> Option<u32> {
    Some(raptor_models::runtime_hints(&entry.gguf).offload_layers).filter(|n| *n > 0)
}

/// Overrides for one load that come from a [`ContextOption`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadExtras {
    pub kv_cache: Option<raptor_runtime::config::KvCacheType>,
    pub slots: Option<u32>,
    /// `Some(false)` turns speculative decoding off for this load (the maximum window).
    pub speculation: Option<bool>,
}

/// A context window the GPU can hold for a model, as shown in the Library.
#[derive(Debug, Clone, Serialize)]
pub struct ContextOption {
    /// `default`, `large` or `maximum`.
    pub id: &'static str,
    pub label: String,
    pub ctx: u32,
    pub kv_cache: raptor_runtime::config::KvCacheType,
    pub slots: u32,
    /// Whether speculative decoding runs with this window. False for the maximum window of a model
    /// with an MTP head: the draft context needs memory, and without it generation is about half as fast.
    pub speculation: bool,
    /// Projected device memory at `ctx` including measured overhead, MiB, when it was projected.
    pub projected_mib: Option<u64>,
    pub budget_mib: u64,
    /// True when `ctx` is the model's trained maximum.
    pub capped_by_model: bool,
    pub note: String,
}

/// The "Long (4-bit cache)" window offered for a sparse Mixture-of-experts model that spills into
/// shared memory (`tuning::oversubscription_applies`): `min(32768, trained maximum)` tokens with a
/// q4_0 cache and one slot. Such a model is never offered the 8-bit "Large" window: at 32K its
/// generation is the slowest with q8_0 (Ornith 10.8 tok/s vs 24.0 f16 and 46.9 q4_0; flint 12.3 vs
/// 15.5 and 16.9), because the smaller q4_0 cache spills least. `None` when the default window is
/// already about that long.
fn long_option(default_ctx: u32, n_ctx_train: u32, budget_mib: u64) -> Option<ContextOption> {
    let ctx = crate::tuning::LONG_WINDOW_CTX.min(n_ctx_train);
    if u64::from(ctx) * 4 <= u64::from(default_ctx) * 5 {
        return None;
    }
    Some(ContextOption {
        id: "long",
        label: "Long (4-bit cache)".into(),
        ctx,
        kv_cache: raptor_runtime::config::KvCacheType::Q4_0,
        slots: 1,
        speculation: true,
        projected_mib: None,
        budget_mib,
        capped_by_model: ctx >= n_ctx_train,
        note: "This model is larger than VRAM, so part of it lives in shared system memory (expected). At a 32K context the 4-bit cache was fastest, measured with llama-bench: Ornith 46.9 tok/s vs 24.0 (16-bit) and 10.8 (8-bit); flint 16.9 vs 15.5 and 12.3. The 8-bit cache is not offered because it was slowest. Quality at long context with a 4-bit cache is not measured beyond a 2K-context perplexity test on another model (within noise).".into(),
    })
}

/// What tuning found. `improvement_percent` is how much faster the best setup is than the
/// untuned one for a long prompt plus a reply (0 when the untuned setup was already best).
#[derive(Debug, Clone, Serialize)]
pub struct TuneReport {
    pub model_id: String,
    pub runtime_id: String,
    pub runtime_version: String,
    pub backend: Backend,
    pub context_size: u32,
    pub gpu_name: Option<String>,
    pub gpu_driver: Option<String>,
    pub results: Vec<raptor_runtime::tune::Measured>,
    pub best: raptor_runtime::tune::Measured,
    pub baseline: Option<raptor_runtime::tune::Measured>,
    /// Matched streamed-request metrics for the untuned settings.
    pub baseline_live: Option<crate::tuned::TuneMetrics>,
    /// Matched streamed-request metrics for the selected optimized settings.
    pub best_live: Option<crate::tuned::TuneMetrics>,
    pub improvement_percent: f64,
    /// Layer counts that could not be measured, with the reason.
    pub errors: Vec<String>,
    /// Deep tune only: every setup measured with `long_depth` tokens in the context.
    pub long_results: Vec<raptor_runtime::tune::Measured>,
    pub long_best: Option<raptor_runtime::tune::Measured>,
    /// How much less time the long-prompt setting takes than the quick-tune one at that depth.
    pub long_improvement_percent: Option<f64>,
    /// The depth `long_results` were measured at (0 when no deep tune ran).
    pub long_depth: u32,
}

/// Semantic retrieval uses the loaded model, so the server must offer embeddings when this is the
/// configured embedding model. A chat model has no pooling of its own and would refuse them, so it
/// gets mean pooling; BERT-style embedding models keep their own. An embedding input must fit in
/// one micro-batch (the default 512 rejected 524-token chunks with HTTP 500 on a real GPU), so the
/// micro-batch is raised to the batch size.
fn apply_embedding_mode(
    config: &mut LaunchConfig,
    settings: &crate::settings::Settings,
    model_id: &str,
    entry: &ModelEntry,
) {
    if settings.embedding_model.as_deref() != Some(model_id) {
        return;
    }
    config.embeddings = true;
    config.ubatch_size = config.batch_size;
    let is_bert = entry
        .gguf
        .architecture
        .as_deref()
        .is_some_and(|a| a.contains("bert"));
    config.pooling = (!is_bert).then(|| "mean".to_string());
}

/// Median of finite benchmark values. Median is less sensitive to OS scheduling spikes than the
/// mean, while keeping every retained sample available in the profile's sample count.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

/// A load in progress: when it started, and whether it has reached the benchmark.
#[derive(Debug, Clone, Copy)]
struct LoadPhase {
    started: std::time::Instant,
    benchmarking: bool,
}

/// A load report kept for a window that was reloaded: the result plus the launch settings.
#[derive(Debug, Clone, Serialize)]
pub struct StoredLoadReport {
    pub result: LoadModelResult,
    pub ctx_size: u32,
    pub gpu_layers: u32,
}

/// A failed load. `backend_failure` is set when the GPU backend had no device, so the UI can
/// show the cause and its suggested actions. Serializes as the usual `{kind, message}` plus
/// `backend_failure` (null when there is none).
#[derive(Debug, Clone)]
pub struct LoadFailure {
    pub error: RaptorError,
    pub backend_failure: Option<BackendLoadFailure>,
}

impl std::fmt::Display for LoadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl From<RaptorError> for LoadFailure {
    fn from(error: RaptorError) -> Self {
        Self {
            error,
            backend_failure: None,
        }
    }
}

impl Serialize for LoadFailure {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.error).map_err(serde::ser::Error::custom)?;
        value["backend_failure"] =
            serde_json::to_value(&self.backend_failure).map_err(serde::ser::Error::custom)?;
        value.serialize(serializer)
    }
}

/// Reads the active server from the runtime and the loaded-model record. Each successful load
/// bumps `seq`, so anything cached per server can tell a restart apart.
struct RuntimeActive {
    runtime: Arc<RuntimeEngine>,
    loaded: Arc<Mutex<Option<LoadedModel>>>,
    seq: Arc<AtomicU64>,
}

impl ActiveModel for RuntimeActive {
    fn active(&self) -> Option<ActiveServer> {
        let status = self.runtime.status();
        let serving = matches!(
            status.server.state,
            ServerState::Ready | ServerState::Degraded { .. }
        );
        if !serving {
            return None;
        }
        let endpoint = status.server.endpoint?;
        let model_id = lock(&self.loaded).as_ref()?.model_id.clone();
        Some(ActiveServer {
            endpoint,
            model_id,
            generation: self.seq.load(Ordering::SeqCst),
        })
    }
}

/// Hardware as the model library and comparisons need it: RAM, and the VRAM of all GPUs.
pub fn hardware_summary(hw: &HardwareInfo) -> raptor_models::HardwareSummary {
    raptor_models::HardwareSummary {
        ram_bytes: hw.ram_total,
        vram_bytes: hw.gpus.iter().map(|g| g.vram_total).sum(),
    }
}

/// A runtime index after parsing. `refused` lists entries that were not installable because
/// their sha256 is missing or null. They are never passed to the installer.
#[derive(Debug, Clone, Default)]
pub struct RuntimeIndex {
    pub manifests: Vec<Manifest>,
    pub refused: Vec<String>,
}

/// The model the shell last reported as loaded, with the exact configuration it was started
/// with. Cleared on unload and before every load, so it never outlives the server.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub model_id: String,
    pub model_sha256: String,
    pub backend: Backend,
    pub runtime_id: String,
    pub runtime_version: String,
    pub config: LaunchConfig,
}

/// Dependencies the shell wires in. Production builds them from the system; tests pass fakes.
pub struct ShellDeps {
    pub engine: EngineDeps,
    /// Builds the upstream client for a llama-server endpoint.
    pub connect: Connect,
    /// Counts tokens through the loaded model's `/tokenize`.
    pub tokenizer: Arc<dyn TokenizeTransport>,
    /// Embeddings and summaries from the loaded llama-server. Loopback only in production.
    pub llama_http: Arc<dyn LlamaHttp>,
    /// Bytes of the runtime download written so far (production only; tests leave it at 0).
    pub install_progress: Arc<AtomicU64>,
    /// Reads GPU memory use, to report a load that spilled into shared system memory.
    pub gpu_memory: Arc<dyn raptor_runtime::gpumem::GpuMemoryProbe>,
}

impl ShellDeps {
    /// The real system: probes, downloads, child processes under the kill-on-close job, and
    /// loopback HTTP only.
    pub fn production(dirs: &DataDirs) -> Result<Self> {
        let install_progress = Arc::new(AtomicU64::new(0));
        Ok(Self {
            engine: system_engine_deps(dirs, install_progress.clone())?,
            connect: loopback_connect(),
            tokenizer: Arc::new(LoopbackTransport::default()),
            llama_http: Arc::new(LoopbackHttp),
            install_progress,
            gpu_memory: Arc::new(raptor_runtime::gpumem::WindowsGpuMemory::new(Arc::new(
                SystemCommandRunner,
            ))),
        })
    }
}

pub struct AppState {
    /// Where the active data (models, runtimes, conversations, context index) lives.
    pub dirs: DataDirs,
    pub settings: Arc<SettingsStore>,
    pub runtime: Arc<RuntimeEngine>,
    pub models: Arc<ModelManager>,
    pub conversations: ConversationStore,
    pub contexts: Arc<ContextHub>,
    pub upstream: Arc<RuntimeUpstream>,
    pub context_service: Arc<dyn ContextService>,
    /// Token counter used for benchmark prompts. Context builds use their own instance.
    pub token_counter: Arc<ServerAwareCounter>,
    pub gateway: AsyncMutex<GatewayControl>,
    /// Final outcome details of downloads, by job key. Live job state comes from the
    /// download manager; this map only keeps what the manager does not record (the path,
    /// the hash and the error text).
    pub download_outcomes: Mutex<HashMap<String, DownloadFinished>>,
    pub chats: Arc<Mutex<HashMap<String, CancellationToken>>>,
    /// Serializes runtime changes (install, rollback, load, unload, benchmark).
    pub runtime_lock: AsyncMutex<()>,
    /// The model the shell loaded, if any. See [`LoadedModel`]. Shared with the semantic
    /// backends, which read which model the server is serving.
    pub loaded: Arc<Mutex<Option<LoadedModel>>>,
    /// Incremented by every successful load. See [`RuntimeActive`].
    load_seq: Arc<AtomicU64>,
    /// Bytes of the runtime download written so far. See [`ShellDeps::install_progress`].
    pub install_progress: Arc<AtomicU64>,
    /// See [`ShellDeps::gpu_memory`]. Replaceable so tests can script the readings.
    gpu_memory: Mutex<Arc<dyn raptor_runtime::gpumem::GpuMemoryProbe>>,
    /// Newest llama.cpp release. GitHub in production; tests replace it.
    pub releases: Mutex<Arc<dyn crate::updates::ReleaseSource>>,
    /// Set while `load_model` runs, for the progress display.
    load_phase: Mutex<Option<LoadPhase>>,
    /// What tuning found for each model on this machine. See [`crate::tuned`].
    tuned: Mutex<crate::tuned::TunedStore>,
    /// Saved context-length benchmark matrices. See [`crate::ctxbench`].
    matrix: Mutex<crate::ctxbench::MatrixStore>,
    /// Steps done and total of the tuning run in progress, for the progress display.
    tune_steps: (std::sync::atomic::AtomicU32, std::sync::atomic::AtomicU32),
    /// What the last successful load reported, so a reloaded window can show it again.
    /// Cleared whenever `loaded` is cleared.
    load_report: Mutex<Option<StoredLoadReport>>,
    /// Hardware detected once at startup (or on the next explicit probe).
    pub hardware: Mutex<Option<HardwareInfo>>,
    /// Serializes hardware probes and avoids duplicate uncached probes during startup.
    hardware_probe_lock: AsyncMutex<()>,
    /// Performance memory. An open error is kept as text and never overwritten.
    pub perf: Mutex<std::result::Result<PerfStore, String>>,
    /// Cancelled at exit. Stops the status watcher and any other background loop.
    pub background: CancellationToken,
    /// Notes produced at startup (settings recovery, storage fallback).
    pub startup_notes: Vec<String>,
}

impl AppState {
    /// Builds every subsystem from the platform data directory. A failure is returned to
    /// the caller, and the app does not start.
    pub fn initialize() -> Result<Self> {
        let platform = DataDirs::default_location()?;
        platform.ensure()?;
        let settings = Arc::new(SettingsStore::open(platform.config().join("settings.json")));
        let mut startup_notes = Vec::new();
        if let Some(note) = settings.recovery() {
            startup_notes.push(note);
        }
        let snapshot = settings.snapshot();
        let dirs = match snapshot.storage_dir.clone() {
            Some(dir) => {
                let custom = DataDirs::new(dir.clone());
                match custom.ensure() {
                    Ok(()) => custom,
                    Err(e) => {
                        startup_notes.push(format!(
                            "storage directory {} is unavailable ({e}); using the default location for this session",
                            dir.display()
                        ));
                        platform
                    }
                }
            }
            None => platform,
        };
        let deps = ShellDeps::production(&dirs)?;
        Self::from_parts(dirs, settings, startup_notes, deps)
    }

    /// Assembles the shell from explicit parts. `initialize` uses the platform directories;
    /// the end-to-end gate test uses a temporary data directory.
    pub fn from_parts(
        dirs: DataDirs,
        settings: Arc<SettingsStore>,
        startup_notes: Vec<String>,
        deps: ShellDeps,
    ) -> Result<Self> {
        let runtime = Arc::new(RuntimeEngine::new(&dirs, deps.engine));
        let models = Arc::new(ModelManager::with_hugging_face(&dirs, None)?);
        let server: Arc<dyn ServerProbe> = runtime.clone();
        let library: Arc<dyn InstalledModels> = models.clone();
        let token_counter = Arc::new(ServerAwareCounter::new(
            server.clone(),
            deps.tokenizer.clone(),
        ));
        let loaded = Arc::new(Mutex::new(None));
        let load_seq = Arc::new(AtomicU64::new(0));
        let tuned_store = crate::tuned::TunedStore::open(&dirs.config().join("tuned.json"));
        let matrix_path = dirs.config().join("ctxbench.json");
        let active: Arc<dyn ActiveModel> = Arc::new(RuntimeActive {
            runtime: runtime.clone(),
            loaded: loaded.clone(),
            seq: load_seq.clone(),
        });
        let semantic = Arc::new(SemanticSetup {
            active,
            http: deps.llama_http,
        });
        let contexts = Arc::new(ContextHub::connected(
            dirs.clone(),
            settings.clone(),
            server.clone(),
            deps.tokenizer,
            semantic,
        ));
        let upstream = Arc::new(RuntimeUpstream::new(server, library, deps.connect));
        let context_service: Arc<dyn ContextService> =
            Arc::new(ContextAdapter::new(contexts.clone()));
        let perf = perf::open_store(&dirs);
        Ok(Self {
            conversations: ConversationStore::new(dirs.db().join("conversations")),
            dirs,
            settings,
            runtime,
            models,
            contexts,
            upstream,
            context_service,
            token_counter,
            gateway: AsyncMutex::new(GatewayControl::default()),
            download_outcomes: Mutex::new(HashMap::new()),
            chats: Arc::new(Mutex::new(HashMap::new())),
            runtime_lock: AsyncMutex::new(()),
            loaded,
            load_seq,
            install_progress: deps.install_progress,
            gpu_memory: Mutex::new(deps.gpu_memory),
            releases: Mutex::new(match crate::updates::GitHubReleases::new() {
                Ok(g) => Arc::new(g) as Arc<dyn crate::updates::ReleaseSource>,
                Err(e) => Arc::new(crate::updates::Unavailable(e.to_string())),
            }),
            load_phase: Mutex::new(None),
            tuned: Mutex::new(tuned_store),
            matrix: Mutex::new(crate::ctxbench::MatrixStore::open(&matrix_path)),
            tune_steps: Default::default(),
            load_report: Mutex::new(None),
            hardware: Mutex::new(None),
            hardware_probe_lock: AsyncMutex::new(()),
            perf: Mutex::new(perf),
            background: CancellationToken::new(),
            startup_notes,
        })
    }

    /// Path of the runtime index the shell reads manifests from.
    pub fn runtime_manifest_path(&self) -> PathBuf {
        self.dirs.config().join(RUNTIME_MANIFEST_FILE)
    }

    /// Reads and parses the runtime index. Without one, installing a runtime is refused
    /// with a clear message rather than guessing a download location.
    /// Selects the runtime that is already installed for this machine, so an app restart does
    /// not forget it. Never downloads: when the best match is not on disk it returns false and
    /// the user installs it from the UI. True when a runtime is selected afterwards.
    pub async fn restore_installed_runtime(&self) -> Result<bool> {
        if self.runtime.status().runtime.is_some() {
            return Ok(true);
        }
        let index = self.load_runtime_index()?;
        let pin = self.settings.snapshot().pinned_runtime;
        let candidates = candidates_for_pin(index.manifests, pin.as_ref())?;
        let hw = self.hardware_for_use().await?;
        let selection = raptor_runtime::manifest::select_runtime(&hw, &candidates)?;
        let on_disk = self
            .runtime
            .installed_runtimes()?
            .iter()
            .any(|r| r.id == selection.manifest.id && r.version == selection.manifest.version);
        if !on_disk {
            return Ok(false);
        }
        let _guard = self.runtime_lock.lock().await;
        self.runtime
            .ensure_runtime(&candidates, &tokio_util::sync::CancellationToken::new())
            .await?;
        Ok(true)
    }

    /// Which backends have a runtime on disk: (ROCm, CUDA, SYCL, Vulkan).
    fn installed_backends(&self) -> Result<(bool, bool, bool, bool)> {
        let installed = self.runtime.installed_runtimes()?;
        let has = |b: Backend| installed.iter().any(|r| r.backend == b);
        Ok((
            has(Backend::Hip),
            has(Backend::Cuda),
            has(Backend::Sycl),
            has(Backend::Vulkan),
        ))
    }

    /// The backend the user chose for a model (`Auto` when nothing was chosen).
    pub fn backend_pref(&self, model_id: &str) -> crate::backend_choice::BackendPref {
        self.settings
            .snapshot()
            .backend_prefs
            .get(model_id)
            .copied()
            .unwrap_or_default()
    }

    /// A sparse Mixture-of-experts model larger than the GPU's memory: the kind Vulkan generated
    /// faster than ROCm on (see `backend_choice`).
    fn model_spills(entry: &ModelEntry, hw: &HardwareInfo) -> bool {
        let vram_total = hw.gpus.iter().map(|g| g.vram_total).max().unwrap_or(0);
        crate::tuning::is_sparse_moe(entry.gguf.expert_count.zip(entry.gguf.expert_used_count))
            && entry.size_bytes > vram_total
    }

    /// What the Library shows for the backend selector of one model: the preference, what a load
    /// would use now, which runtimes are installed, and why a choice cannot be honoured.
    /// Whether the loaded model has a thinking-effort control, read from the server's `/props`
    /// (kept for ten seconds). Unsupported when no model is loaded or the server does not say.
    pub async fn thinking_view(&self) -> crate::thinking::ThinkingView {
        let caps = self.upstream.thinking_caps().await;
        crate::thinking::ThinkingView::from_caps(&caps, self.settings.snapshot().thinking_effort)
    }

    /// Like [`Self::thinking_view`], but asks the server again: called after a load.
    pub async fn refresh_thinking(&self) -> crate::thinking::ThinkingView {
        self.upstream.forget_server_facts();
        self.thinking_view().await
    }

    pub async fn backend_view(&self, model_id: &str) -> Result<BackendView> {
        use crate::backend_choice::{resolve, Inputs};
        let entry = self.models.library().get(model_id)?;
        let hw = self.hardware_for_use().await?;
        let (rocm_installed, cuda_installed, sycl_installed, vulkan_installed) =
            self.installed_backends()?;
        let spills = Self::model_spills(&entry, &hw);
        let preference = self.backend_pref(model_id);
        let auto_backend = auto_backend_for(
            &hw,
            rocm_installed,
            cuda_installed,
            sycl_installed,
            vulkan_installed,
        );
        let resolved = resolve(&Inputs {
            pref: preference,
            amd_gpu: hw
                .gpu_of(None)
                .is_some_and(|g| g.vendor == raptor_runtime::hardware::GpuVendor::Amd),
            spills,
            rocm_installed,
            cuda_installed,
            sycl_installed,
            vulkan_installed,
        });
        let active_backend = self.runtime.status().runtime.map(|r| r.backend);
        let (effective, note) = match resolved {
            Ok(c) => (
                match c.backend.or(auto_backend).or(active_backend) {
                    Some(Backend::Vulkan) => "vulkan",
                    Some(Backend::Hip) => "rocm",
                    Some(Backend::Cuda) => "cuda",
                    Some(Backend::Sycl) => "sycl",
                    _ => "default",
                }
                .to_string(),
                c.notes.first().cloned().or_else(|| {
                    (preference == crate::backend_choice::BackendPref::Auto)
                        .then(|| {
                            auto_backend.map(|b| {
                                format!("Auto selected the {} runtime for this GPU.", b.label())
                            })
                        })
                        .flatten()
                }),
            ),
            Err(message) => ("unavailable".to_string(), Some(message)),
        };
        Ok(BackendView {
            model_id: model_id.to_string(),
            preference,
            effective,
            rocm_installed,
            cuda_installed,
            sycl_installed,
            vulkan_installed,
            spills,
            note,
            measured: crate::backend_choice::MEASURED,
        })
    }

    /// Stores the backend the user chose for a model. Takes effect at the next load.
    pub fn set_backend_pref(
        &self,
        model_id: &str,
        pref: crate::backend_choice::BackendPref,
    ) -> Result<()> {
        self.models.library().get(model_id)?;
        self.settings.update(|s| {
            if pref == crate::backend_choice::BackendPref::Auto {
                s.backend_prefs.remove(model_id);
            } else {
                s.backend_prefs.insert(model_id.to_string(), pref);
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Picks the backend for one model from its preference and puts that runtime in use. Nothing
    /// is downloaded; a chosen backend that is not installed is an error. Needs `runtime_lock`
    /// held and the server stopped. Returns notes for the load report.
    async fn apply_backend_choice(
        &self,
        model_id: &str,
        entry: &ModelEntry,
        hw: &HardwareInfo,
    ) -> Result<Vec<String>> {
        use crate::backend_choice::{resolve, Inputs};
        let (rocm_installed, cuda_installed, sycl_installed, vulkan_installed) =
            self.installed_backends()?;
        let preference = self.backend_pref(model_id);
        let choice = resolve(&Inputs {
            pref: preference,
            amd_gpu: hw
                .gpu_of(None)
                .is_some_and(|g| g.vendor == raptor_runtime::hardware::GpuVendor::Amd),
            spills: Self::model_spills(entry, hw),
            rocm_installed,
            cuda_installed,
            sycl_installed,
            vulkan_installed,
        })
        .map_err(RaptorError::Conflict)?;
        let auto_backend = auto_backend_for(
            &hw,
            rocm_installed,
            cuda_installed,
            sycl_installed,
            vulkan_installed,
        );
        let active = self.runtime.status().runtime.map(|r| r.backend);
        match choice.backend.or(auto_backend) {
            Some(b) if active != Some(b) => {
                self.runtime.activate_installed(b)?;
            }
            Some(_) => {}
            None if active == Some(Backend::Vulkan) => {
                // An earlier model chose Vulkan; this one wants the normal runtime again (ROCm, or
                // CUDA on NVIDIA). With neither on disk the Vulkan one stays: it is the only one.
                for normal in [Backend::Hip, Backend::Cuda, Backend::Sycl] {
                    match self.runtime.activate_installed(normal) {
                        Ok(_) => break,
                        Err(RaptorError::NotFound(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            None => {}
        }
        let mut notes = choice.notes;
        // Auto may try Vulkan when the selected native runtime cannot initialize a device.
        // Explicit backend preferences remain strict: their failure must be visible to the user.
        if preference == crate::backend_choice::BackendPref::Auto {
            if let Some(selected) = self.runtime.status().runtime.map(|r| r.backend) {
                if selected != Backend::Vulkan && selected.is_gpu() && vulkan_installed {
                    let visibility = match self.runtime.preflight_devices().await {
                        Ok(preflight) if preflight.has_usable_device() => None,
                        Ok(preflight) => {
                            Some(preflight.failure.map(|f| f.message).unwrap_or_else(|| {
                                format!("{} reported no usable device", selected.label())
                            }))
                        }
                        Err(error) => {
                            Some(format!("{} preflight failed: {error}", selected.label()))
                        }
                    };
                    if let Some(reason) = visibility {
                        let fallback = self.runtime.activate_installed(Backend::Vulkan);
                        match fallback {
                            Ok(_) => match self.runtime.preflight_devices().await {
                                Ok(preflight) if preflight.has_usable_device() => notes.push(format!(
                                    "Auto selected Vulkan because {} could not initialize a visible device ({reason}); Vulkan device visibility was verified.",
                                    selected.label()
                                )),
                                Ok(preflight) => {
                                    let fallback_reason = preflight.failure.map(|f| f.message).unwrap_or_else(|| "no usable device was listed".into());
                                    self.runtime.activate_installed(selected)?;
                                    notes.push(format!(
                                        "Auto kept {}: its device check failed ({reason}), and Vulkan also found no usable device ({fallback_reason}).",
                                        selected.label()
                                    ));
                                }
                                Err(error) => {
                                    self.runtime.activate_installed(selected)?;
                                    notes.push(format!(
                                        "Auto kept {}: its device check failed ({reason}), and Vulkan preflight failed ({error}).",
                                        selected.label()
                                    ));
                                }
                            },
                            Err(error) => notes.push(format!(
                                "Auto kept {} after its device check failed ({reason}); Vulkan fallback could not be selected ({error}).",
                                selected.label()
                            )),
                        }
                    }
                }
            }
        }
        Ok(notes)
    }

    pub fn load_runtime_index(&self) -> Result<RuntimeIndex> {
        let path = self.runtime_manifest_path();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(RaptorError::NotFound(format!(
                    "no runtime index at {}; add a Raptor LLM runtime manifest there before installing llama.cpp",
                    path.display()
                )));
            }
            Err(e) => return Err(e.into()),
        };
        parse_runtime_index(&text)
    }

    fn gateway_deps(&self) -> GatewayDeps {
        let upstream: Arc<dyn Upstream> = self.upstream.clone();
        GatewayDeps {
            upstream,
            context: Some(self.context_service.clone()),
        }
    }

    /// Makes the running gateway match the settings: starts, reconfigures or stops it.
    /// The error is also kept for `gateway_status`.
    pub async fn apply_gateway(&self) -> Result<GatewayStatus> {
        let snapshot = self.settings.snapshot();
        let mut g = self.gateway.lock().await;
        let result = self.apply_gateway_locked(&mut g, &snapshot).await;
        g.last_error = result.as_ref().err().map(|e| e.to_string());
        result?;
        Ok(gateway_status(&g))
    }

    async fn apply_gateway_locked(&self, g: &mut GatewayControl, s: &Settings) -> Result<()> {
        if !s.gateway_enabled {
            if let Some(h) = g.handle.take() {
                h.shutdown().await?;
            }
            return Ok(());
        }
        let cfg = gateway_config(s, &self.gateway_tls_dir())?;
        if let Some(mut h) = g.handle.take() {
            // On failure the old listener is already stopped; the handle is dropped.
            h.reconfigure(cfg).await?;
            g.handle = Some(h);
            return Ok(());
        }
        g.handle = Some(Gateway::start(cfg, self.gateway_deps()).await?);
        Ok(())
    }

    pub async fn gateway_status(&self) -> GatewayStatus {
        let g = self.gateway.lock().await;
        gateway_status(&g)
    }

    /// Where the gateway's self-signed certificate is kept.
    pub fn gateway_tls_dir(&self) -> PathBuf {
        self.dirs.config().join(GATEWAY_TLS_DIR)
    }

    /// Replaces the LAN certificate with a new one on the running listener. Clients that pinned
    /// the old fingerprint must pin the new one. Refused when the gateway is not running.
    pub async fn rotate_gateway_tls(&self) -> Result<GatewayStatus> {
        let mut g = self.gateway.lock().await;
        let handle = g
            .handle
            .as_mut()
            .ok_or_else(|| RaptorError::Conflict("the gateway is not running".into()))?;
        handle.rotate_tls().await?;
        Ok(gateway_status(&g))
    }

    /// Probes the hardware and caches the result for the rest of the session.
    pub async fn refresh_hardware(&self) -> Result<HardwareInfo> {
        let _probe = self.hardware_probe_lock.lock().await;
        let hw = self.runtime.detect_hardware().await?;
        *lock(&self.hardware) = Some(hw.clone());
        Ok(hw)
    }

    /// The hardware detected earlier in this session, if any.
    pub fn cached_hardware(&self) -> Option<HardwareInfo> {
        lock(&self.hardware).clone()
    }

    /// The cached hardware, probing once if nothing is cached yet.
    pub async fn hardware_for_use(&self) -> Result<HardwareInfo> {
        match self.cached_hardware() {
            Some(hw) => Ok(hw),
            None => {
                let _probe = self.hardware_probe_lock.lock().await;
                if let Some(hw) = self.cached_hardware() {
                    return Ok(hw);
                }
                let hw = self.runtime.detect_hardware().await?;
                *lock(&self.hardware) = Some(hw.clone());
                Ok(hw)
            }
        }
    }

    /// The recommendation for a library model at `ctx` tokens: architecture-aware (hybrid KV
    /// layers, recurrent state, offloadable layers), and calibrated by llama.cpp's own
    /// projection on a GPU runtime. The projection loads no weights and starts no server. If it
    /// fails, the GGUF estimate is kept and the rationale says why. The runtime must be selected.
    pub async fn recommend_for_entry(
        &self,
        entry: &ModelEntry,
        path: &Path,
        ctx: u32,
    ) -> Result<Recommendation> {
        self.recommend_for_entry_with(entry, path, ctx, 0).await
    }

    /// Like [`AppState::recommend_for_entry`], keeping `extra_device_bytes` of VRAM free for a
    /// model loaded with this one (a DFlash draft, see [`AppState::planned_draft_bytes`]).
    pub async fn recommend_for_entry_with(
        &self,
        entry: &ModelEntry,
        path: &Path,
        ctx: u32,
        extra_device_bytes: u64,
    ) -> Result<Recommendation> {
        let meta = model_meta(entry)?;
        let (arch, mut facts) = model_arch(entry, &meta);
        let arch = ModelArch {
            extra_device_bytes,
            ..arch
        };
        let mut rec = self
            .runtime
            .recommend_config_with_arch(&meta, &arch, ctx, None)?;
        let gpu = self
            .runtime
            .status()
            .runtime
            .is_some_and(|r| r.backend.is_gpu());
        if gpu {
            let probe = match rec.config.clone().with_model(path) {
                Ok(cfg) => self.runtime.probe_fit(&cfg).await,
                Err(e) => Err(e),
            };
            match probe {
                Ok(projection) => {
                    rec = self.runtime.recommend_config_with_arch(
                        &meta,
                        &arch,
                        ctx,
                        Some(&projection),
                    )?;
                }
                Err(e) => facts.push(format!(
                    "llama.cpp projection unavailable ({e}); the memory estimate is the GGUF one"
                )),
            }
        }
        facts.extend(rec.rationale);
        rec.rationale = facts;
        Ok(rec)
    }

    /// Loads a library model. The runtime must already be installed.
    ///
    /// The launch configuration is the known-good one for this machine, model and runtime
    /// when it exists and matches the requested context. Otherwise it is the recommendation.
    /// After the model is Ready, one benchmark run is recorded in performance memory.
    pub async fn load_model(
        &self,
        model_id: &str,
        context: Option<u32>,
        preset: Option<&str>,
    ) -> std::result::Result<LoadModelResult, LoadFailure> {
        self.load_model_ext(model_id, context, preset, LoadExtras::default())
            .await
    }

    /// Like [`AppState::load_model`], with the KV-cache type and slot count of one of the
    /// [`ContextOption`]s. A cache type needs a context too: together they are how a larger window
    /// is chosen.
    pub async fn load_model_ext(
        &self,
        model_id: &str,
        context: Option<u32>,
        preset: Option<&str>,
        extras: LoadExtras,
    ) -> std::result::Result<LoadModelResult, LoadFailure> {
        *lock(&self.load_phase) = Some(LoadPhase {
            started: std::time::Instant::now(),
            benchmarking: false,
        });
        let result = self
            .load_model_inner(model_id, context, preset, extras)
            .await;
        *lock(&self.load_phase) = None;
        result
    }

    /// Where the load that is running now has got to, read from llama-server's own log.
    pub fn load_progress(&self) -> crate::load_progress::LoadProgress {
        use crate::load_progress::{stage_from_log, LoadProgress, LoadStage};
        let Some(phase) = *lock(&self.load_phase) else {
            return LoadProgress::idle();
        };
        let stage = if phase.benchmarking {
            LoadStage::Benchmark
        } else {
            let lines = self.runtime.log_tail(600);
            stage_from_log(lines.iter().map(|l| l.text.as_str()))
        };
        LoadProgress::at(true, stage, phase.started.elapsed().as_millis() as u64)
    }

    async fn load_model_inner(
        &self,
        model_id: &str,
        context: Option<u32>,
        preset: Option<&str>,
        extras: LoadExtras,
    ) -> std::result::Result<LoadModelResult, LoadFailure> {
        let library = self.models.library();
        let entry = library.get(model_id)?;
        if entry.gguf.architecture.as_deref() == Some(crate::tuning::DFLASH_ARCH) {
            return Err(RaptorError::Unsupported(
                "this is a DFlash draft model: it cannot run on its own, and it is used \
                 automatically when you load the model it was made for"
                    .into(),
            )
            .into());
        }
        let path = library.file_path(model_id)?;
        // Checked before anything is stopped: a header without the sizing fields fails here.
        model_meta(&entry)?;
        let applied_preset: Option<Preset> = match preset {
            Some(name) => Some(self.models.presets().get(model_id, name)?),
            None => None,
        };
        // An explicit context wins over the preset's; the preset's wins over the settings.
        let ctx = context
            .or(applied_preset.as_ref().map(|p| p.n_ctx))
            .unwrap_or(self.settings.snapshot().context_target);
        if ctx > crate::settings::MAX_CONTEXT_TARGET {
            return Err(RaptorError::InvalidInput(format!(
                "context target cannot exceed {} tokens",
                crate::settings::MAX_CONTEXT_TARGET
            ))
            .into());
        }
        let settings = self.settings.snapshot();

        // A restarted app has forgotten its selection; an installed runtime is picked up again.
        // Nothing is downloaded here, so a missing runtime still fails below with the message.
        if let Err(e) = self.restore_installed_runtime().await {
            tracing::warn!(error = %e, "could not select the installed runtime");
        }
        let _guard = self.runtime_lock.lock().await;
        let status = self.runtime.status();
        let runtime = status.runtime.clone().ok_or_else(|| {
            RaptorError::Conflict(
                "the llama.cpp runtime is not installed; install it from Settings first".into(),
            )
        })?;
        let hw = self.hardware_for_use().await?;
        if status.server.state != ServerState::Stopped {
            // Loading replaces the current model, so the old process is stopped first.
            self.runtime.unload().await?;
        }
        // Nothing counts as loaded until the new server reports Ready.
        *lock(&self.loaded) = None;
        *lock(&self.load_report) = None;

        // ROCm or Vulkan for this model (the server is stopped, so the runtime can change). Every
        // later step reads the runtime in use, so it is read again here.
        let backend_notes = self.apply_backend_choice(model_id, &entry, &hw).await?;
        let runtime = self.runtime.status().runtime.unwrap_or(runtime);

        let draft_bytes = self.planned_draft_bytes(
            &entry,
            runtime.backend,
            extras.slots.unwrap_or(settings.parallel_slots),
            extras.speculation.unwrap_or(true),
        );
        let Recommendation {
            config: recommended,
            rationale,
            estimate,
        } = self
            .recommend_for_entry_with(&entry, &path, ctx, draft_bytes)
            .await?;
        let mut notes = Vec::new();
        let known = self.known_good_for(&entry.sha256, &path, &runtime, &hw, ctx, &mut notes)?;
        let (config, config_source, mut rationale) = match known {
            Some(kg) => {
                let mut r = perf::known_good_rationale(&kg.record);
                r.push("The memory estimate describes the recommended configuration.".into());
                (kg.config, ConfigSource::KnownGood, r)
            }
            None => (
                recommended.with_model(path)?,
                ConfigSource::Recommended,
                rationale,
            ),
        };
        rationale.extend(notes);
        rationale.extend(backend_notes);
        if crate::backend_choice::mtp_unverified(runtime.backend)
            && entry.gguf.nextn_predict_layers.is_some_and(|n| n > 0)
        {
            rationale.push(crate::backend_choice::MTP_ON_VULKAN_NOTE.to_string());
        }
        if let Some(p) = &applied_preset {
            rationale.push(format!(
                "Preset \"{}\" supplied the context length ({} tokens).",
                p.name, p.n_ctx
            ));
        }
        let mut config = config;
        config.allow_cpu_fallback = settings.allow_cpu_fallback;
        apply_embedding_mode(&mut config, &settings, model_id, &entry);
        config.parallel_slots = Some(settings.parallel_slots);
        if let Some(n) = extras.slots {
            config.parallel_slots = Some(n);
        }
        if let Some(kv) = extras.kv_cache {
            // A larger window chosen from `context_options`: llama.cpp projected that every layer
            // fits with this cache type, so every layer is requested and the recommendation (made
            // for the default window and cache) does not cap it.
            config.kv_cache = kv;
            config.flash_attn = true;
            if let Some(c) = context {
                config.ctx_size = c;
                config.batch_size = config.batch_size.min(c);
                config.ubatch_size = config.ubatch_size.min(config.batch_size);
            }
            if let Some(n) = offload_layers_of(&entry) {
                config.gpu_layers = n;
            }
            rationale.push(format!(
                "Context window {} tokens with a {} KV cache and {} slot(s), chosen to fit the GPU (llama.cpp's projection).",
                config.ctx_size,
                kv.llama_arg(),
                config.parallel_slots.unwrap_or(0)
            ));
        }
        let allow_fallback = config.allow_cpu_fallback;

        // The speed plan: the fastest variants first (speculative decoding, and for a sparse
        // Mixture-of-experts model that spills, all layers on the GPU), each falling back to a more
        // conservative one if the server cannot start with it. See `tuning`.
        let mut facts = self.speed_facts(&entry, &config, &hw, runtime.backend);
        facts.allow_speculation = extras.speculation.unwrap_or(true);
        // A spilling sparse MoE at a 32K+ window with no cache chosen: 4-bit, its fastest.
        if extras.kv_cache.is_none() {
            if let Some(kv) = crate::tuning::long_window_cache(&config, &facts) {
                config.kv_cache = kv;
                config.flash_attn = true;
                rationale.push(format!(
                    "{} KV cache: this Mixture-of-experts model spills into shared memory, and at a {}+ token window the 4-bit cache was fastest when measured (Ornith 46.9 tok/s vs 24.0 with 16-bit and 10.8 with 8-bit).",
                    kv.llama_arg(),
                    crate::tuning::LONG_WINDOW_CTX
                ));
            }
        }
        // What tuning measured on this machine for this model and runtime build wins over the
        // general rules, with the untuned plan kept as the fallback if it cannot start.
        let untuned = config.clone();
        let saved_tuning = lock(&self.tuned).get(&self.tuned_key(&hw, &entry.sha256, &runtime));
        let tuned = saved_tuning.as_ref().filter(|t| {
            if t.setting_for(config.ctx_size).long {
                t.matches_long(config.kv_cache, config.flash_attn)
            } else {
                t.matches(config.kv_cache, config.flash_attn)
            }
        });
        let mut tuned_applied = false;
        if let (Some(t), false) = (&tuned, config.embeddings) {
            // The long-prompt setting (opt-in deep tune) applies to windows as large as the depth
            // it was measured at; smaller windows use the quick-tune setting.
            let s = t.setting_for(config.ctx_size);
            config.gpu_layers = s.gpu_layers.min(facts.offload_layers);
            config.ubatch_size = s.ubatch_size.min(config.batch_size);
            tuned_applied = true;
            let gain = (s.baseline_seconds / s.seconds - 1.0) * 100.0;
            rationale.push(if s.long {
                format!(
                    "Tuned on this machine for long prompts (measured with {} tokens of context, used for windows of that size or more): {} layers on the GPU, micro-batch {} ({:.0} prompt tok/s, {:.1} generation tok/s measured; {:.0}% faster than the short-prompt setting at that depth).",
                    raptor_runtime::tune::DEEP_DEPTH,
                    config.gpu_layers,
                    config.ubatch_size,
                    s.prompt_tps,
                    s.gen_tps,
                    gain
                )
            } else {
                format!(
                    "Tuned on this machine: {} layers on the GPU, micro-batch {} ({:.0} prompt tok/s, {:.1} generation tok/s measured; {:.0}% faster than the untuned setup for a long prompt and a reply).",
                    config.gpu_layers,
                    config.ubatch_size,
                    s.prompt_tps,
                    s.gen_tps,
                    gain
                )
            });
        }
        if saved_tuning.is_some() && tuned.is_none() {
            rationale.push(
                "Saved speed settings were measured with different cache or flash-attention settings, so the current load uses its fresh recommendation. Run Optimize speed to refresh the profile.".into(),
            );
        }
        let probe = lock(&self.gpu_memory).clone();
        let memory_before = probe.read().await.ok();
        facts.layers_measured = tuned_applied;
        facts.requested_ctx = entry.gguf.context_length.map_or(ctx, |train| {
            ctx.min(u32::try_from(train).unwrap_or(u32::MAX))
        });
        let mut attempts = crate::tuning::plan(&config, &facts);
        if tuned_applied {
            for a in crate::tuning::plan(&untuned, &facts) {
                if !attempts.iter().any(|b| b.config == a.config) {
                    attempts.push(a);
                }
            }
        }
        let attempt_count = attempts.len();
        let mut chosen = None;
        for (i, attempt) in attempts.into_iter().enumerate() {
            match self.runtime.load_model(attempt.config.clone()).await {
                Ok(_) => {
                    chosen = Some(attempt);
                    break;
                }
                Err(e) => {
                    let failure = self.load_failure(e);
                    // A missing GPU device is not something a different setup can fix.
                    if i + 1 == attempt_count || failure.backend_failure.is_some() {
                        return Err(failure);
                    }
                    rationale.push(format!(
                        "A faster setup did not start ({}); trying a more conservative one.",
                        failure.error
                    ));
                }
            }
        }
        let Some(chosen) = chosen else {
            return Err(self.load_failure(RaptorError::Runtime(
                "no launch variant could be started".into(),
            )));
        };
        rationale.extend(chosen.notes);
        let config = chosen.config;
        if config.ctx_size < crate::settings::MIN_CONTEXT_TARGET {
            rationale.push(format!(
                "Only {} context tokens fit for this model and machine, below the 65,536-token coding-agent target. The model's trained limit, available memory, and selected GPU offload constrain the effective window.",
                config.ctx_size
            ));
        }
        // Did the load fit in dedicated VRAM, or did the driver spill into shared system memory?
        let gpu_memory = match (memory_before, probe.read().await.ok()) {
            (Some(before), Some(after)) if runtime.backend.is_gpu() => {
                let r = raptor_runtime::gpumem::spill_between(before, after);
                let expected = r.spilled
                    && crate::tuning::is_sparse_moe(facts.experts)
                    && facts.model_bytes > facts.vram_total;
                if r.spilled && expected {
                    rationale.push(format!(
                        "{} MiB of the model is in shared system memory, as planned for a sparse Mixture-of-experts model larger than VRAM.",
                        r.shared_growth_mib
                    ));
                } else if r.spilled {
                    rationale.push(format!(
                        "WARNING: {} MiB of this load went into shared system memory because VRAM is full. The model works but generation can be several times slower. Choose a smaller context window or cache.",
                        r.shared_growth_mib
                    ));
                } else {
                    rationale.push(format!(
                        "GPU memory: {} MiB dedicated in use, nothing spilled into shared system memory.",
                        r.dedicated_mib
                    ));
                }
                Some(GpuMemoryReport {
                    dedicated_mib: r.dedicated_mib,
                    shared_growth_mib: r.shared_growth_mib,
                    spilled: r.spilled,
                    expected,
                })
            }
            _ => None,
        };
        let status_after_load = self.runtime.status();
        let cpu_fallback = allow_fallback
            && runtime.backend.is_gpu()
            && status_after_load
                .device_preflight
                .as_ref()
                .is_some_and(|p| p.failure.is_some());
        if cpu_fallback {
            rationale.push(
                "CPU FALLBACK: the GPU backend had no device. This model runs on the CPU because allow_cpu_fallback is on; it is not GPU-accelerated.".into(),
            );
        }
        let loaded = LoadedModel {
            model_id: model_id.to_string(),
            model_sha256: entry.sha256.clone(),
            backend: runtime.backend,
            runtime_id: runtime.id.clone(),
            runtime_version: runtime.version.clone(),
            config,
        };
        *lock(&self.loaded) = Some(loaded.clone());
        self.load_seq.fetch_add(1, Ordering::SeqCst);
        if let Some(phase) = lock(&self.load_phase).as_mut() {
            phase.benchmarking = true;
        }
        let benchmark = self.benchmark(&loaded).await;
        let thinking = self.refresh_thinking().await;
        let result = LoadModelResult {
            status: self.runtime.status(),
            rationale,
            estimate,
            model_id: model_id.to_string(),
            config_source,
            benchmark,
            preset: applied_preset.map(|p| p.name),
            cpu_fallback,
            gpu_memory,
            thinking,
        };
        *lock(&self.load_report) = Some(StoredLoadReport {
            result: result.clone(),
            ctx_size: loaded.config.ctx_size,
            gpu_layers: loaded.config.gpu_layers,
        });
        Ok(result)
    }

    /// The facts the speed plan and the tuner need about `entry` on this machine.
    fn speed_facts(
        &self,
        entry: &ModelEntry,
        config: &LaunchConfig,
        hw: &HardwareInfo,
        backend: Backend,
    ) -> crate::tuning::Facts {
        crate::tuning::Facts {
            windows: cfg!(windows),
            gpu_backend: backend.is_gpu(),
            embeddings: config.embeddings,
            allow_speculation: true,
            experts: entry.gguf.expert_count.zip(entry.gguf.expert_used_count),
            // MTP is unverified on Vulkan, so a Vulkan load plans n-gram speculation instead.
            has_mtp_head: entry.gguf.nextn_predict_layers.is_some_and(|n| n > 0)
                && !crate::backend_choice::mtp_unverified(backend),
            dflash_draft: self.dflash_draft_for(entry, backend),
            layers_measured: false,
            requested_ctx: 0,
            offload_layers: offload_layers_of(entry).unwrap_or(config.gpu_layers),
            model_bytes: entry.size_bytes,
            vram_total: hw.gpus.iter().map(|g| g.vram_total).max().unwrap_or(0),
            ram_total: hw.ram_total,
            ram_avail: hw.ram_avail,
            cpu_threads: hw.cpu.threads,
        }
    }

    /// VRAM to keep free for the DFlash draft the speed plan will load with `entry`, or 0 when it
    /// will not load one (speculation off, the model's own MTP head wins, more request slots than
    /// `tuning::DFLASH_MAX_SLOTS`, or no matching draft). The draft's file size plus 64 MiB for its
    /// cache and compute: measured, the GPU model buffers grew by 547 MiB for a 551 MiB draft file.
    fn planned_draft_bytes(
        &self,
        entry: &ModelEntry,
        backend: Backend,
        slots: u32,
        speculation: bool,
    ) -> u64 {
        let own_head = entry.gguf.nextn_predict_layers.is_some_and(|n| n > 0)
            && !crate::backend_choice::mtp_unverified(backend);
        if !speculation || own_head || slots > crate::tuning::DFLASH_MAX_SLOTS {
            return 0;
        }
        self.dflash_draft_for(entry, backend)
            .and_then(|p| std::fs::metadata(p).ok())
            .map_or(0, |m| m.len() + 64 * 1024 * 1024)
    }

    /// A DFlash draft model in the library that can serve `entry` (see `tuning::dflash_matches`).
    /// Headers are read from the files, so entries indexed before the vocabulary size was
    /// recorded still match. None on Vulkan: like MTP, draft decoding was only measured on ROCm.
    fn dflash_draft_for(&self, entry: &ModelEntry, backend: Backend) -> Option<PathBuf> {
        if crate::backend_choice::mtp_unverified(backend)
            || entry.gguf.architecture.as_deref() == Some(crate::tuning::DFLASH_ARCH)
        {
            return None;
        }
        let library = self.models.library();
        let candidates: Vec<ModelEntry> = library
            .list()
            .ok()?
            .into_iter()
            .filter(|e| e.gguf.architecture.as_deref() == Some(crate::tuning::DFLASH_ARCH))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let target = raptor_models::gguf::parse_file(&library.file_path(&entry.id).ok()?).ok()?;
        candidates.into_iter().find_map(|c| {
            let path = library.file_path(&c.id).ok()?;
            let draft = raptor_models::gguf::parse_file(&path).ok()?;
            crate::tuning::dflash_matches(&target, &draft).then_some(path)
        })
    }

    fn tuned_key(&self, hw: &HardwareInfo, model_sha256: &str, runtime: &RuntimeSummary) -> String {
        let k = perf::key_for(hw, model_sha256, &runtime.id);
        crate::tuned::key(&k.machine, &k.model_hash, &k.runtime_id, &runtime.version)
    }

    /// Steps done and total of the tuning run in progress (0, 0 when idle).
    pub fn tune_progress(&self) -> (u32, u32) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.tune_steps.0.load(Relaxed),
            self.tune_steps.1.load(Relaxed),
        )
    }

    /// Measures a candidate with the real server and the same fixed streamed request used by
    /// performance history. One warmup is discarded; reported values are medians of three runs.
    /// The server is always stopped before this method returns, including on request failure.
    async fn measure_tune_candidate(
        &self,
        base: &LaunchConfig,
        candidate: &raptor_runtime::tune::Measured,
    ) -> Result<crate::tuned::TuneMetrics> {
        let mut config = base.clone();
        config.gpu_layers = candidate.gpu_layers;
        config.ubatch_size = candidate.ubatch.min(config.batch_size);
        self.runtime.load_model(config).await?;

        let measured = async {
            let status = self.runtime.status();
            if status.server.state != ServerState::Ready {
                return Err(RaptorError::Runtime(
                    "the benchmark server did not remain Ready".into(),
                ));
            }
            if candidate.gpu_layers > 0
                && !matches!(
                    status.gpu_verdict,
                    Some(raptor_runtime::verify::GpuVerdict::Verified { .. })
                )
            {
                return Err(RaptorError::Runtime(format!(
                    "GPU offload was not verified for the {}-layer benchmark candidate",
                    candidate.gpu_layers
                )));
            }
            let model = self.upstream.models().await?.loaded.ok_or_else(|| {
                RaptorError::Runtime("the benchmark server reports no model".into())
            })?;

            // Warm caches and kernel paths, but do not mix the warmup into the comparison.
            perf::sample(self.upstream.as_ref(), self.token_counter.clone(), &model).await?;
            let mut ttft = Vec::with_capacity(3);
            let mut prompt = Vec::with_capacity(3);
            let mut generation = Vec::with_capacity(3);
            let mut exact = true;
            let mut prompt_tokens = None;
            let mut output_tokens = None;
            for _ in 0..3 {
                let (sample, tokenizer_exact) =
                    perf::sample(self.upstream.as_ref(), self.token_counter.clone(), &model)
                        .await?;
                let after = self.runtime.status();
                let reasons = perf::assess(
                    &sample,
                    base.backend,
                    after.server.state == ServerState::Ready,
                    after.gpu_verdict.as_ref(),
                );
                if !reasons.is_empty() {
                    return Err(RaptorError::Runtime(format!(
                        "benchmark sample was unstable: {}",
                        reasons.join("; ")
                    )));
                }
                exact &= tokenizer_exact;
                prompt_tokens.get_or_insert(sample.prompt_tokens);
                output_tokens.get_or_insert(sample.gen_tokens);
                ttft.push(sample.ttft_ms);
                prompt.push(sample.prompt_tps);
                generation.push(sample.gen_tps);
            }
            Ok(crate::tuned::TuneMetrics {
                gpu_layers: candidate.gpu_layers,
                ubatch_size: candidate.ubatch,
                parallel_slots: base.parallel_slots,
                kv_cache: base.kv_cache,
                flash_attn: base.flash_attn,
                speculative: base.speculative.is_some(),
                prompt_tokens: prompt_tokens.unwrap_or(0),
                output_tokens: output_tokens.unwrap_or(0),
                ttft_ms: median(ttft),
                prompt_tps: median(prompt),
                gen_tps: median(generation),
                samples: 3,
                prompt_tokens_exact: exact,
            })
        }
        .await;
        let stopped = self.runtime.unload().await;
        match (measured, stopped) {
            (Ok(metrics), Ok(())) => Ok(metrics),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(RaptorError::Runtime(format!(
                "the comparison finished, but the benchmark server did not stop: {error}"
            ))),
        }
    }

    /// Forgets the tuning of a model, so the general rules apply again. True when there was one.
    pub async fn forget_tuning(&self, model_id: &str) -> Result<bool> {
        let entry = self.models.library().get(model_id)?;
        let hw = self.hardware_for_use().await?;
        let runtime = self.runtime.status().runtime.ok_or_else(|| {
            RaptorError::Conflict("the llama.cpp runtime is not installed".into())
        })?;
        let key = self.tuned_key(&hw, &entry.sha256, &runtime);
        lock(&self.tuned).remove(&key)
    }

    /// Measures the model at the layer counts and micro-batch sizes worth trying on this machine
    /// with llama-bench, keeps the fastest for a long prompt plus a reply, and saves it so the next
    /// load uses it. The model must not be loaded: the benchmark needs the GPU. A layer count that
    /// fails (out of memory, say) is reported and skipped; the run fails only if nothing could be
    /// measured. Nothing is saved then.
    ///
    /// `deep` adds a second pass: every setup the first pass measured is measured again with
    /// [`raptor_runtime::tune::DEEP_DEPTH`] tokens in the context, and the fastest there is saved
    /// as the long-prompt setting, used for windows of that size or more. A setup that fails at
    /// that depth is reported; if none works, only the quick-tune result is saved.
    pub async fn tune_model(&self, model_id: &str, deep: bool) -> Result<TuneReport> {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = self.runtime_lock.lock().await;
        if self.runtime.status().server.state != ServerState::Stopped {
            return Err(RaptorError::Conflict(
                "unload the model before tuning: the benchmark needs the GPU".into(),
            ));
        }
        let library = self.models.library();
        let entry = library.get(model_id)?;
        let path = library.file_path(model_id)?;
        let hw = self.hardware_for_use().await?;
        // Tuned values are kept per runtime, so the model is tuned on the backend it will use.
        self.apply_backend_choice(model_id, &entry, &hw).await?;
        let runtime = self.runtime.status().runtime.ok_or_else(|| {
            RaptorError::Conflict(
                "the llama.cpp runtime is not installed; install it from Settings first".into(),
            )
        })?;
        let settings = self.settings.snapshot();
        let ctx = settings.context_target;
        let mut base = self
            .recommend_for_entry(&entry, &path, ctx)
            .await?
            .config
            .with_model(path.clone())?;
        let recommended_layers = base.gpu_layers;
        let recommended_config = base.clone();
        base.parallel_slots = Some(settings.parallel_slots);
        let mut facts = self.speed_facts(&entry, &base, &hw, runtime.backend);
        facts.allow_speculation = true;
        facts.requested_ctx = entry.gguf.context_length.map_or(ctx, |train| {
            ctx.min(u32::try_from(train).unwrap_or(u32::MAX))
        });
        if let Some(kv) = crate::tuning::long_window_cache(&base, &facts) {
            base.kv_cache = kv;
            base.flash_attn = true;
        }
        // The untuned baseline is exactly the first launch attempt for a normal load. The sweep
        // varies only layer count and micro-batch around those same cache/speculation/slot flags.
        if let Some(first) = crate::tuning::plan(&base, &facts).into_iter().next() {
            base = first.config;
        }

        let mut layer_counts = vec![base.gpu_layers, recommended_layers];
        if crate::tuning::oversubscription_applies(&base, &facts)
            || crate::tuning::oversubscription_applies(&recommended_config, &facts)
        {
            layer_counts.push(facts.offload_layers);
        }
        layer_counts.sort_unstable();
        layer_counts.dedup();
        self.tune_steps.0.store(0, Relaxed);
        self.tune_steps.1.store(layer_counts.len() as u32, Relaxed);
        let mut results: Vec<raptor_runtime::tune::Measured> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        for ngl in &layer_counts {
            match self
                .runtime
                .tune_measure(&path, *ngl, base.ctx_size, base.kv_cache, base.flash_attn)
                .await
            {
                Ok(m) => results.extend(m),
                Err(e) => errors.push(format!("{ngl} layers on the GPU: {e}")),
            }
            self.tune_steps.0.fetch_add(1, Relaxed);
        }
        let Some(best) = raptor_runtime::tune::best(&results).cloned() else {
            return Err(RaptorError::Runtime(format!(
                "nothing could be measured: {}",
                errors.join("; ")
            )));
        };
        // The untuned setup: the recommended layers and the default micro-batch.
        let baseline = results
            .iter()
            .find(|m| m.gpu_layers == base.gpu_layers && m.ubatch == base.ubatch_size)
            .or_else(|| results.iter().find(|m| m.gpu_layers == base.gpu_layers))
            .cloned();
        let baseline_seconds = baseline.as_ref().map_or(best.seconds, |b| b.seconds);

        let mut long_results: Vec<raptor_runtime::tune::Measured> = Vec::new();
        if deep {
            let depth = raptor_runtime::tune::DEEP_DEPTH;
            self.tune_steps.1.fetch_add(results.len() as u32, Relaxed);
            for m in &results {
                match self
                    .runtime
                    .bench_depth(
                        &path,
                        m.gpu_layers,
                        m.ubatch,
                        base.kv_cache,
                        depth,
                        base.flash_attn,
                    )
                    .await
                {
                    Ok(p) => long_results.push(raptor_runtime::tune::Measured::at_depth(
                        m.gpu_layers,
                        m.ubatch,
                        &p,
                    )),
                    Err(e) => errors.push(format!(
                        "{} layers, micro-batch {} at {depth} tokens of context: {e}",
                        m.gpu_layers, m.ubatch
                    )),
                }
                self.tune_steps.0.fetch_add(1, Relaxed);
            }
        }
        let long_best = raptor_runtime::tune::best(&long_results).cloned();
        // The quick-tune setup measured at the same depth, to say what the long one gains over it.
        let long_baseline_seconds = long_best.as_ref().map(|lb| {
            long_results
                .iter()
                .find(|m| m.gpu_layers == best.gpu_layers && m.ubatch == best.ubatch)
                .map_or(lb.seconds, |m| m.seconds)
        });
        let long_improvement_percent = long_best
            .as_ref()
            .zip(long_baseline_seconds)
            .map(|(lb, base_s)| (base_s / lb.seconds - 1.0) * 100.0);

        // llama-bench gives repeatable prompt/decode throughput but has no first-token latency.
        // Measure the default and winner through the same streaming server path and workload.
        let comparison_count = if baseline.is_some()
            && baseline
                .as_ref()
                .is_some_and(|b| b.gpu_layers != best.gpu_layers || b.ubatch != best.ubatch)
        {
            2
        } else if baseline.is_some() {
            1
        } else {
            1
        };
        self.tune_steps.1.fetch_add(comparison_count, Relaxed);
        let baseline_live = if let Some(candidate) = baseline.as_ref() {
            let result = self.measure_tune_candidate(&base, candidate).await;
            self.tune_steps.0.fetch_add(1, Relaxed);
            match result {
                Ok(metrics) => Some(metrics),
                Err(e) => {
                    errors.push(format!("default streamed comparison: {e}"));
                    None
                }
            }
        } else {
            None
        };
        let same_candidate = baseline
            .as_ref()
            .is_some_and(|b| b.gpu_layers == best.gpu_layers && b.ubatch == best.ubatch);
        let best_live = if same_candidate {
            baseline_live.clone()
        } else {
            let result = self.measure_tune_candidate(&base, &best).await;
            self.tune_steps.0.fetch_add(1, Relaxed);
            match result {
                Ok(metrics) => Some(metrics),
                Err(e) => {
                    errors.push(format!("optimized streamed comparison: {e}"));
                    None
                }
            }
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let key = self.tuned_key(&hw, &entry.sha256, &runtime);
        // A quick tune keeps a long-prompt setting from an earlier deep tune: it is still the
        // measured answer for long windows. A deep tune that found nothing keeps it too.
        let kept_long = lock(&self.tuned).get(&key).and_then(|e| e.long);
        let long = match (&long_best, long_baseline_seconds) {
            (Some(lb), Some(base_s)) => Some(crate::tuned::LongTuned {
                gpu_layers: lb.gpu_layers,
                ubatch_size: lb.ubatch,
                prompt_tps: lb.prompt_tps,
                gen_tps: lb.gen_tps,
                seconds: lb.seconds,
                baseline_seconds: base_s,
                depth: raptor_runtime::tune::DEEP_DEPTH,
                profile: Some(crate::tuned::TuneProfile {
                    kv_cache: base.kv_cache,
                    flash_attn: base.flash_attn,
                }),
            }),
            _ => kept_long,
        };
        lock(&self.tuned).put(
            &key,
            crate::tuned::TunedEntry {
                gpu_layers: best.gpu_layers,
                ubatch_size: best.ubatch,
                prompt_tps: best.prompt_tps,
                gen_tps: best.gen_tps,
                seconds: best.seconds,
                baseline_seconds,
                measured_at: now,
                profile: Some(crate::tuned::TuneProfile {
                    kv_cache: base.kv_cache,
                    flash_attn: base.flash_attn,
                }),
                baseline_live: baseline_live.clone(),
                best_live: best_live.clone(),
                long,
            },
        )?;
        let improvement_percent = (baseline_seconds / best.seconds - 1.0) * 100.0;
        Ok(TuneReport {
            model_id: model_id.to_string(),
            runtime_id: runtime.id,
            runtime_version: runtime.version,
            backend: runtime.backend,
            // `ctx` is the user's requested target. `recommend_for_entry` may reduce it to
            // fit available VRAM, so report the effective context used by both comparison runs.
            context_size: base.ctx_size,
            gpu_name: hw.gpu_of(None).map(|g| g.name.clone()),
            gpu_driver: hw.gpu_of(None).and_then(|g| g.driver.clone()),
            results,
            best,
            baseline,
            baseline_live,
            best_live,
            improvement_percent,
            errors,
            long_depth: if long_results.is_empty() {
                0
            } else {
                raptor_runtime::tune::DEEP_DEPTH
            },
            long_results,
            long_best,
            long_improvement_percent,
        })
    }

    /// The context windows the GPU can hold for `model_id`: the default, a larger one with an
    /// 8-bit KV cache, and the maximum with a 4-bit cache and one slot. Sizes come from llama.cpp's
    /// own memory projection (`llama-fit-params`), not an estimate. KV-cache type does not change
    /// generation or prompt speed on this GPU (measured within 1%), so the trade is memory against
    /// a little accuracy. Larger windows are offered only when the model fits entirely in VRAM at
    /// the default window: pushing layers to the CPU costs far more speed than any window is worth
    /// (granite: 20.5 tok/s on the GPU, 1-4 with 25 layers on the CPU).
    pub async fn context_options(&self, model_id: &str) -> Result<Vec<ContextOption>> {
        use raptor_runtime::config::{headroom, KvCacheType};
        use raptor_runtime::context_max::Overhead;
        const MIB: u64 = 1024 * 1024;
        let library = self.models.library();
        let entry = library.get(model_id)?;
        let path = library.file_path(model_id)?;
        let hw = self.hardware_for_use().await?;
        let runtime = self.runtime.status().runtime.ok_or_else(|| {
            RaptorError::Conflict("the llama.cpp runtime is not installed".into())
        })?;
        let settings = self.settings.snapshot();
        let meta = model_meta(&entry)?;
        let base = self
            .recommend_for_entry(&entry, &path, settings.context_target)
            .await?
            .config;
        let facts = self.speed_facts(&entry, &base, &hw, runtime.backend);
        let vram_total = hw.gpus.iter().map(|g| g.vram_total).max().unwrap_or(0);
        let budget_mib = vram_total.saturating_sub(headroom(vram_total)) / MIB;

        let mut options = vec![ContextOption {
            id: "default",
            label: "Default".into(),
            ctx: base.ctx_size,
            kv_cache: base.kv_cache,
            slots: settings.parallel_slots,
            speculation: true,
            projected_mib: None,
            budget_mib,
            capped_by_model: base.ctx_size >= meta.n_ctx_train,
            note: "The size set in Settings, with the cache type the app picks.".into(),
        }];
        // A sparse MoE that spills runs with every layer requested on the GPU (see `tuning::plan`),
        // so the "does not fit" rule below does not apply to it; it gets the 4-bit "Long" window
        // instead of the 8-bit one, which is its slowest cache.
        if crate::tuning::oversubscription_applies(&base, &facts) {
            options.extend(long_option(options[0].ctx, meta.n_ctx_train, budget_mib));
            options[0].note.push_str(
                " This model is larger than VRAM and runs partly from shared system memory, so only the cache type that was fastest when measured is offered for a longer window.",
            );
            return Ok(options);
        }
        if !runtime.backend.is_gpu() || base.gpu_layers < facts.offload_layers {
            options[0].note.push_str(
                " Larger windows are not offered: this model does not fit entirely in VRAM at the default size, and moving layers to the CPU slows generation far more than a bigger window is worth.",
            );
            return Ok(options);
        }
        // The recommendation carries no model path until `with_model`; the probe needs one.
        let mut probe_base = base.clone().with_model(&path)?;
        probe_base.gpu_layers = facts.offload_layers;
        let cap = meta.n_ctx_train;
        let larger = |prev: u32, ctx: u32| u64::from(ctx) * 4 > u64::from(prev) * 5;
        let mtp = facts.has_mtp_head;
        let with_mtp = if mtp {
            Overhead::BASE.plus(Overhead::MTP)
        } else {
            Overhead::BASE
        };
        let large = self
            .runtime
            .max_context(
                &probe_base,
                KvCacheType::Q8_0,
                settings.parallel_slots,
                budget_mib,
                cap,
                with_mtp,
            )
            .await
            .map_err(|e| tracing::warn!(error = %e, "large context option not available"))
            .ok()
            .filter(|f| larger(options[0].ctx, f.ctx));
        if let Some(f) = &large {
            options.push(ContextOption {
                id: "large",
                label: "Large (8-bit cache)".into(),
                ctx: f.ctx,
                kv_cache: f.kv_cache,
                slots: f.slots,
                speculation: true,
                projected_mib: Some(f.projected_mib),
                budget_mib,
                capped_by_model: f.capped_by_model,
                note: "An 8-bit KV cache halves the cache memory. Measured on Qwen3.8-27B: perplexity within noise of 16-bit (+0.18%), and at most about 6% slower generation at a 32K context.".into(),
            });
        }
        let prev = large.as_ref().map_or(options[0].ctx, |f| f.ctx);
        let maximum = self
            .runtime
            .max_context(
                &probe_base,
                KvCacheType::Q4_0,
                1,
                budget_mib,
                cap,
                Overhead::BASE,
            )
            .await
            .map_err(|e| tracing::warn!(error = %e, "maximum context option not available"));
        if let Ok(f) = maximum {
            if larger(prev, f.ctx) {
                options.push(ContextOption {
                    id: "maximum",
                    label: "Maximum (4-bit cache)".into(),
                    ctx: f.ctx,
                    kv_cache: f.kv_cache,
                    slots: f.slots,
                    speculation: !mtp,
                    projected_mib: Some(f.projected_mib),
                    budget_mib,
                    capped_by_model: f.capped_by_model,
                    note: format!(
                        "A 4-bit cache and one slot give the most room. Measured on Qwen3.8-27B at a 2K context, perplexity was within noise of 16-bit (+0.15%); accuracy at very long contexts has not been measured. A long prompt takes a long time to process.{}",
                        if mtp {
                            " This model's multi-token-prediction speedup is turned off for this window, because its draft context needs memory: generation is about half as fast (measured 47 -> 21 tok/s)."
                        } else {
                            ""
                        }
                    ),
                });
            }
        }
        Ok(options)
    }

    /// The saved context-length matrix for a model on this machine and runtime build, if any.
    pub async fn last_context_matrix(
        &self,
        model_id: &str,
    ) -> Result<Option<crate::ctxbench::MatrixReport>> {
        let entry = self.models.library().get(model_id)?;
        let hw = self.hardware_for_use().await?;
        let Some(runtime) = self.runtime.status().runtime else {
            return Ok(None);
        };
        Ok(lock(&self.matrix).get(&self.tuned_key(&hw, &entry.sha256, &runtime)))
    }

    /// Measures prefill speed, decode speed, the start latency of a 512-token turn and peak GPU
    /// memory at 0 / 4K / 16K / 32K / 64K / 128K / 192K tokens of context, up to `max_depth`, with the
    /// cheapest KV-cache type that fits each depth. The model must not be loaded: the benchmark needs
    /// the GPU. A depth that fails or does not fit is kept as a row with the reason; the run fails only
    /// if nothing could be measured. Uses the tuned layers and micro-batch when the model was tuned.
    /// llama-bench runs plain decoding, so speculative decoding is not part of these speeds.
    pub async fn context_matrix(
        &self,
        model_id: &str,
        max_depth: u32,
    ) -> Result<crate::ctxbench::MatrixReport> {
        use raptor_runtime::config::{headroom, KvCacheType};
        use raptor_runtime::context_max::Overhead;
        use raptor_runtime::ctxmatrix::{
            cumulative_prompt_seconds, plan_depths, turn_ms, MatrixRow,
        };
        use std::sync::atomic::Ordering::Relaxed;
        const MIB: u64 = 1024 * 1024;
        let _guard = self.runtime_lock.lock().await;
        if self.runtime.status().server.state != ServerState::Stopped {
            return Err(RaptorError::Conflict(
                "unload the model first: the benchmark needs the GPU".into(),
            ));
        }
        let library = self.models.library();
        let entry = library.get(model_id)?;
        let path = library.file_path(model_id)?;
        let hw = self.hardware_for_use().await?;
        self.apply_backend_choice(model_id, &entry, &hw).await?;
        let runtime = self.runtime.status().runtime.ok_or_else(|| {
            RaptorError::Conflict(
                "the llama.cpp runtime is not installed; install it from Settings first".into(),
            )
        })?;
        let settings = self.settings.snapshot();
        let meta = model_meta(&entry)?;
        let base = self
            .recommend_for_entry(&entry, &path, settings.context_target)
            .await?
            .config;
        let facts = self.speed_facts(&entry, &base, &hw, runtime.backend);
        let key = self.tuned_key(&hw, &entry.sha256, &runtime);
        let tuned = lock(&self.tuned).get(&key);
        let (ngl, ubatch) = match &tuned {
            Some(t) => (t.gpu_layers.min(facts.offload_layers), t.ubatch_size),
            None => (base.gpu_layers, base.ubatch_size),
        };
        let cap = meta.n_ctx_train;

        // Which cache types hold which context, from llama.cpp's projection. A model that does not
        // fit entirely in VRAM cannot be projected this way, so it is measured as it is and any
        // failure or spill is reported per depth.
        let vram_total = hw.gpus.iter().map(|g| g.vram_total).max().unwrap_or(0);
        let budget_mib = vram_total.saturating_sub(headroom(vram_total)) / MIB;
        let mut fits: Vec<(KvCacheType, u32)> = Vec::new();
        // Projection assumes the weights sit in VRAM. A model larger than the budget (a sparse
        // Mixture-of-experts model that spills by design) cannot be projected, so it is measured as it is.
        let weights_fit = entry.size_bytes / MIB < budget_mib;
        if ngl >= facts.offload_layers && weights_fit {
            let probe_base = base.clone().with_model(&path)?;
            for kv in [KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0] {
                if let Ok(f) = self
                    .runtime
                    .max_context(
                        &probe_base,
                        kv,
                        settings.parallel_slots,
                        budget_mib,
                        cap,
                        Overhead::BASE,
                    )
                    .await
                {
                    fits.push((kv, f.ctx));
                }
            }
        } else {
            fits.push((KvCacheType::Q8_0, cap));
        }
        let (plan, skipped) = plan_depths(max_depth, cap, &fits);
        self.tune_steps.0.store(0, Relaxed);
        self.tune_steps.1.store(plan.len() as u32, Relaxed);

        let mut rows: Vec<MatrixRow> = Vec::new();
        for (depth, kv) in plan {
            let probe = lock(&self.gpu_memory).clone();
            let (result, peak) = raptor_runtime::gpumem::sample_while(
                &*probe,
                std::time::Duration::from_secs(2),
                self.runtime
                    .bench_depth(&path, ngl, ubatch, kv, depth, base.flash_attn),
            )
            .await;
            let mut row = MatrixRow {
                depth,
                kv_cache: kv,
                prompt_tps: None,
                gen_tps: None,
                turn_ms: None,
                peak_dedicated_mib: peak.map(|p| p.peak_dedicated_mib),
                peak_shared_growth_mib: peak.map(|p| p.peak_shared_growth_mib),
                spilled: peak.is_some_and(|p| p.spilled),
                note: None,
            };
            match result {
                Ok(p) => {
                    row.prompt_tps = Some(p.prompt_tps);
                    row.gen_tps = Some(p.gen_tps);
                    row.turn_ms = Some(turn_ms(p.prompt_tps));
                }
                Err(e) => row.note = Some(e.to_string()),
            }
            rows.push(row);
            self.tune_steps.0.fetch_add(1, Relaxed);
        }
        for (depth, why) in skipped {
            rows.push(MatrixRow {
                depth,
                kv_cache: KvCacheType::Q4_0,
                prompt_tps: None,
                gen_tps: None,
                turn_ms: None,
                peak_dedicated_mib: None,
                peak_shared_growth_mib: None,
                spilled: false,
                note: Some(why),
            });
        }
        rows.sort_by_key(|r| r.depth);
        if !rows.iter().any(|r| r.prompt_tps.is_some()) {
            let why = rows
                .iter()
                .filter_map(|r| r.note.clone())
                .next()
                .unwrap_or_else(|| "no depth could be measured".into());
            return Err(RaptorError::Runtime(format!(
                "nothing could be measured: {why}"
            )));
        }
        let points: Vec<(u32, f64)> = rows
            .iter()
            .filter_map(|r| r.prompt_tps.map(|t| (r.depth, t)))
            .collect();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let report = crate::ctxbench::MatrixReport {
            model_id: model_id.to_string(),
            measured_at: now,
            gpu_layers: ngl,
            ubatch,
            max_depth,
            rows,
            prompt_seconds: cumulative_prompt_seconds(&points),
        };
        lock(&self.matrix).put(&key, report.clone())?;
        Ok(report)
    }

    /// The last successful load's report while its model is still serving, else None.
    pub fn current_load_report(&self) -> Option<StoredLoadReport> {
        self.current_model_id()?;
        lock(&self.load_report).clone()
    }

    /// Turns a failed runtime load into the structured form. A GPU backend with no device
    /// carries its typed diagnosis, which the engine keeps in its status.
    fn load_failure(&self, error: RaptorError) -> LoadFailure {
        let backend_failure = match error {
            RaptorError::Runtime(_) => self
                .runtime
                .status()
                .device_preflight
                .and_then(|p| p.failure),
            _ => None,
        };
        LoadFailure {
            error,
            backend_failure,
        }
    }

    /// The known-good configuration, when one exists for this machine and model and its
    /// context matches `ctx`. Problems reading performance memory are noted, not fatal.
    fn known_good_for(
        &self,
        model_sha256: &str,
        path: &Path,
        runtime: &RuntimeSummary,
        hw: &HardwareInfo,
        ctx: u32,
        notes: &mut Vec<String>,
    ) -> Result<Option<KnownGood>> {
        let key = perf::key_for(hw, model_sha256, &runtime.id);
        let guard = lock(&self.perf);
        let store = match &*guard {
            Ok(s) => s,
            Err(msg) => {
                notes.push(format!(
                    "Performance memory is unavailable ({msg}), so the recommended configuration is used."
                ));
                return Ok(None);
            }
        };
        let Some(kg) = perf::known_good(store, &key, path, runtime.backend)? else {
            return Ok(None);
        };
        if kg.config.ctx_size != ctx {
            notes.push(format!(
                "A known-good configuration exists for a {}-token context; {ctx} tokens was requested, so the recommended configuration is used.",
                kg.config.ctx_size
            ));
            return Ok(None);
        }
        Ok(Some(kg))
    }

    /// Runs the fixed benchmark on the loaded server and records the result. Never fails:
    /// a failed or unstable run is returned with its reasons and stored as unstable.
    async fn benchmark(&self, loaded: &LoadedModel) -> BenchmarkOutcome {
        let mut notes = Vec::new();
        let measured: Result<(perf::BenchSample, bool)> = async {
            let model =
                self.upstream.models().await?.loaded.ok_or_else(|| {
                    RaptorError::Runtime("the server reports no loaded model".into())
                })?;
            perf::sample(self.upstream.as_ref(), self.token_counter.clone(), &model).await
        }
        .await;
        let status = self.runtime.status();
        let still_ready = status.server.state == ServerState::Ready;
        let mut reasons = Vec::new();
        let sample = match measured {
            Ok((sample, exact)) => {
                if !exact {
                    notes.push(
                        "Prompt tokens were estimated, because the model's tokenizer was unavailable."
                            .into(),
                    );
                }
                reasons = perf::assess(
                    &sample,
                    loaded.backend,
                    still_ready,
                    status.gpu_verdict.as_ref(),
                );
                Some(sample)
            }
            Err(e) => {
                reasons.push(format!("benchmark failed: {e}"));
                if !still_ready {
                    reasons.push("the server left the Ready state during the benchmark".into());
                }
                None
            }
        };
        let stable = reasons.is_empty();

        let (hw, memory) = match self.refresh_hardware().await {
            Ok(fresh) => {
                let memory = perf::memory_used(&fresh);
                (Some(fresh), memory)
            }
            Err(e) => {
                notes.push(format!("memory was not sampled: {e}"));
                (self.cached_hardware(), (0, 0))
            }
        };
        let Some(hw) = hw else {
            notes.push("Hardware is unknown, so this run cannot be keyed to the machine and was not stored.".into());
            return BenchmarkOutcome {
                stable,
                recorded: "not_saved".into(),
                entry: None,
                reasons,
                gpu_verdict: status.gpu_verdict,
                notes,
            };
        };
        let mut record = perf::build_record(
            &hw,
            &loaded.model_sha256,
            &loaded.runtime_id,
            &loaded.runtime_version,
            ConfigSnapshot::from(&loaded.config),
            sample.as_ref(),
            memory,
            stable,
            perf::now_secs(),
        );
        record.quant_label = self
            .models
            .library()
            .get(&loaded.model_id)
            .ok()
            .and_then(|entry| entry.gguf.quant_label);
        let (recorded, note) = {
            let mut guard = lock(&self.perf);
            perf::store_record(&mut guard, record.clone())
        };
        notes.extend(note);
        if recorded == "kept_stable" {
            notes.push(
                "The stored known-good result was kept; this unstable run was not stored.".into(),
            );
        }
        BenchmarkOutcome {
            stable,
            recorded,
            entry: Some(perf::entry_view(&record)),
            reasons,
            gpu_verdict: status.gpu_verdict,
            notes,
        }
    }

    pub async fn unload_model(&self) -> Result<EngineStatus> {
        let _guard = self.runtime_lock.lock().await;
        self.runtime.unload().await?;
        *lock(&self.loaded) = None;
        *lock(&self.load_report) = None;
        Ok(self.runtime.status())
    }

    /// The loaded model's id, or None when no model is serving. A crashed server serves
    /// nothing, so it reports None until a retry loads a model again.
    pub fn current_model_id(&self) -> Option<String> {
        match self.runtime.status().server.state {
            ServerState::Stopped => {
                *lock(&self.loaded) = None;
                None
            }
            ServerState::Starting | ServerState::Ready | ServerState::Degraded { .. } => {
                lock(&self.loaded).as_ref().map(|l| l.model_id.clone())
            }
            ServerState::Crashed { .. } => None,
        }
    }

    /// Runs the benchmark again on the model that is loaded and ready.
    pub async fn perf_run_benchmark(&self) -> Result<BenchmarkOutcome> {
        let _guard = self.runtime_lock.lock().await;
        let loaded = lock(&self.loaded).clone().ok_or_else(|| {
            RaptorError::Conflict("no model is loaded; load one before benchmarking".into())
        })?;
        if self.runtime.status().server.state != ServerState::Ready {
            return Err(RaptorError::Conflict(
                "the model server is not ready; wait for it or load the model again".into(),
            ));
        }
        Ok(self.benchmark(&loaded).await)
    }

    /// Stored benchmark records, optionally only for one model.
    pub fn perf_history(&self, model_id: Option<&str>) -> Result<Vec<PerfEntry>> {
        let sha = match model_id {
            Some(id) => Some(self.models.library().get(id)?.sha256),
            None => None,
        };
        let mut entries: Vec<PerfEntry> = self
            .store_records()?
            .iter()
            .filter(|r| sha.as_deref().is_none_or(|s| r.model_hash == s))
            .map(perf::entry_view)
            .collect();
        entries.sort_by(|a, b| {
            b.record
                .measured_at
                .cmp(&a.record.measured_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(entries)
    }

    /// Compares two stored records by id. `a` is the baseline.
    pub fn perf_compare(&self, a: &str, b: &str) -> Result<PerfComparison> {
        let records = self.store_records()?;
        let find = |id: &str| -> Result<PerfEntry> {
            records
                .iter()
                .map(perf::entry_view)
                .find(|e| e.id == id)
                .ok_or_else(|| RaptorError::NotFound(format!("no performance record {id}")))
        };
        perf::compare_entries(&find(a)?, &find(b)?)
    }

    /// The fastest stable record for `model_id` on this machine with the installed runtime.
    pub async fn perf_best_known(&self, model_id: &str) -> Result<Option<PerfEntry>> {
        let sha = self.models.library().get(model_id)?.sha256;
        let runtime = self.runtime.status().runtime.ok_or_else(|| {
            RaptorError::Conflict("no runtime is selected yet; run runtime_ensure first".into())
        })?;
        let hw = self.hardware_for_use().await?;
        let key = perf::key_for(&hw, &sha, &runtime.id);
        let guard = lock(&self.perf);
        let store = guard
            .as_ref()
            .map_err(|m| RaptorError::Integrity(m.clone()))?;
        Ok(store.best_known(&key).map(perf::entry_view))
    }

    fn store_records(&self) -> Result<Vec<BenchRecord>> {
        match &*lock(&self.perf) {
            Ok(s) => Ok(s.records().to_vec()),
            Err(m) => Err(RaptorError::Integrity(m.clone())),
        }
    }

    /// The first-run checklist. It never downloads or installs anything. It runs the
    /// selected llama-server with `--list-devices` (no model is loaded) so the GPU row
    /// reflects what the runtime can actually use.
    pub async fn setup_status(&self) -> SetupStatus {
        let mut notes = self.startup_notes.clone();
        let hardware = match self.hardware_for_use().await {
            Ok(hw) => Some(hw),
            Err(e) => {
                notes.push(format!("hardware detection failed: {e}"));
                None
            }
        };
        let status = self.runtime.status();
        let device_preflight = match status.runtime {
            Some(_) => match self.runtime.preflight_devices().await {
                Ok(pf) => Some(pf),
                Err(e) => {
                    notes.push(format!("the device check could not run: {e}"));
                    None
                }
            },
            None => None,
        };
        let installed = match self.runtime.installed_runtimes() {
            Ok(list) => list
                .into_iter()
                .map(|r| (r.id, r.version, r.backend))
                .collect(),
            Err(e) => {
                notes.push(format!("installed runtimes could not be listed: {e}"));
                Vec::new()
            }
        };
        let available = self.available_runtime(hardware.as_ref(), &mut notes);
        let models_installed = match self.models.library().list() {
            Ok(list) => list.len(),
            Err(e) => {
                notes.push(format!("the model library could not be read: {e}"));
                0
            }
        };
        let perf_store_error = lock(&self.perf).as_ref().err().cloned();
        setup::build(SetupInput {
            hardware,
            selected: status.runtime,
            installed,
            available,
            models_installed,
            storage_free_bytes: setup::free_bytes(&self.dirs.models()),
            perf_store_error,
            notes,
            device_preflight,
        })
    }

    /// The compatible runtime the index offers for this machine. Nothing is installed.
    fn available_runtime(
        &self,
        hw: Option<&HardwareInfo>,
        notes: &mut Vec<String>,
    ) -> Option<AvailableRuntime> {
        let hw = hw?;
        let index = match self.load_runtime_index() {
            Ok(i) => i,
            Err(e) => {
                notes.push(e.to_string());
                return None;
            }
        };
        let pin = self.settings.snapshot().pinned_runtime;
        let candidates = match candidates_for_pin(index.manifests, pin.as_ref()) {
            Ok(c) => c,
            Err(e) => {
                notes.push(e.to_string());
                return None;
            }
        };
        match select_runtime(hw, &candidates) {
            Ok(sel) => Some(AvailableRuntime {
                id: sel.manifest.id,
                version: sel.manifest.version,
                backend: sel.backend,
                degraded: sel.degraded,
                reason: sel.reason,
            }),
            Err(e) => {
                notes.push(format!("no runtime in the index fits this machine: {e}"));
                None
            }
        }
    }

    /// Stops background work and the model server. Called once on app exit so no llama-server
    /// process or status loop is left behind. Running downloads are paused, so their partial
    /// files stay consistent for `models_recover_downloads`.
    pub async fn shutdown(&self) {
        self.background.cancel();
        if let Ok(chats) = self.chats.lock() {
            for token in chats.values() {
                token.cancel();
            }
        }
        let handle = self.gateway.lock().await.handle.take();
        if let Some(h) = handle {
            if let Err(e) = h.shutdown().await {
                tracing::warn!("gateway shutdown: {e}");
            }
        }
        match self.models.downloads().list_jobs() {
            Ok(jobs) => {
                for job in jobs
                    .into_iter()
                    .filter(|j| matches!(j.state, JobState::Running))
                {
                    if let Err(e) = self.models.downloads().pause(&job.key) {
                        tracing::warn!("could not pause download {}: {e}", job.key);
                    }
                }
            }
            Err(e) => tracing::warn!("download list at exit: {e}"),
        }
        if let Err(e) = self.runtime.unload().await {
            tracing::warn!("llama-server stop: {e}");
        }
    }

    /// Newest `n` llama-server output lines, oldest first, as JSON objects
    /// `{"stream", "text"}`. Lines come from the engine's in-memory buffer.
    pub fn server_log_lines(&self, n: usize) -> Vec<String> {
        self.runtime
            .log_tail(n)
            .into_iter()
            .map(|l| serde_json::json!({ "stream": l.stream, "text": l.text }).to_string())
            .collect()
    }

    /// Download queue for this session: every job the download manager knows, with the
    /// outcome details recorded when it finished.
    pub fn download_queue(&self) -> Result<Vec<QueueEntry>> {
        let outcomes = lock(&self.download_outcomes);
        let mut out: Vec<QueueEntry> = self
            .models
            .downloads()
            .list_jobs()?
            .into_iter()
            .map(|j| {
                let key = j.key.clone();
                let progress = DownloadProgress {
                    key: key.clone(),
                    state: j.state,
                    downloaded: j.downloaded,
                    total: j.total,
                    attempts: j.attempts,
                };
                let finished = finished_view(&key, j.state, outcomes.get(&key));
                QueueEntry {
                    key,
                    repo: j.repo,
                    file_name: j.file_name,
                    progress,
                    finished,
                }
            })
            .collect();
        out.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(out)
    }

    /// Records the final details of a download. Called once the job reaches a final state.
    pub fn record_download_finished(&self, finished: DownloadFinished) {
        lock(&self.download_outcomes).insert(finished.key.clone(), finished);
    }

    /// Forgets the recorded outcome of a file that is started again.
    pub fn clear_download_outcome(&self, key: &str) {
        lock(&self.download_outcomes).remove(key);
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The `finished` field for a queue row. Only final states have one.
fn finished_view(
    key: &str,
    state: raptor_models::JobState,
    recorded: Option<&DownloadFinished>,
) -> Option<DownloadFinished> {
    use raptor_models::JobState;
    let fallback = |status: &str| DownloadFinished {
        key: key.to_string(),
        status: status.to_string(),
        path: None,
        sha256: None,
        error: None,
    };
    match state {
        JobState::Completed => Some(recorded.cloned().unwrap_or_else(|| fallback("completed"))),
        JobState::Failed => Some(recorded.cloned().unwrap_or_else(|| fallback("failed"))),
        JobState::Cancelled => Some(recorded.cloned().unwrap_or_else(|| fallback("cancelled"))),
        _ => None,
    }
}

/// Engine dependencies for the real system. llama-server runs under [`JobSpawner`], so it
/// cannot outlive the app; the log file is written next to the other data.
fn system_engine_deps(dirs: &DataDirs, progress: Arc<AtomicU64>) -> Result<EngineDeps> {
    let runner: Arc<dyn CommandRunner> = Arc::new(SystemCommandRunner);
    let sink = FileLogSink::rolling(&dirs.logs())?;
    Ok(EngineDeps {
        probe: Arc::new(SystemProbe::new(runner.clone())),
        fetcher: Arc::new(ReqwestFetcher::new()?.with_progress(progress)),
        spawner: Arc::new(JobSpawner),
        http: Arc::new(ReqwestClient::new()?),
        ports: Arc::new(TcpPortCheck),
        sink: Arc::new(sink),
        restart: RestartPolicy::default(),
        timing: Timing::default(),
        runner,
        hip_sdk: HipSdkSearch::system(),
    })
}

/// Parses a runtime index. Entries whose sha256 is missing or null are removed before the
/// crate parses the rest, and reported as refused. Any other invalid entry fails the index.
pub fn parse_runtime_index(text: &str) -> Result<RuntimeIndex> {
    let mut root: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| RaptorError::InvalidInput(format!("bad runtime index: {e}")))?;
    let mut refused = Vec::new();
    if let Some(list) = root.get_mut("runtimes").and_then(|v| v.as_array_mut()) {
        list.retain(|entry| {
            if entry.get("sha256").is_some_and(|v| v.is_string()) {
                return true;
            }
            refused.push(describe_refused(entry));
            false
        });
    }
    let manifests = parse_manifest_index(&root.to_string())?;
    Ok(RuntimeIndex { manifests, refused })
}

fn describe_refused(entry: &serde_json::Value) -> String {
    let field = |name: &str| entry.get(name).and_then(|v| v.as_str()).unwrap_or("?");
    format!(
        "{} {} ({}): no verified sha256 in the runtime index, so it was refused",
        field("id"),
        field("version"),
        field("backend")
    )
}

/// The manifests `runtime_ensure` may choose from. A pinned runtime narrows the list to that
/// one version, so a newer install is never promoted over it. Errors when the pinned version
/// is missing from the index.
pub fn candidates_for_pin(
    manifests: Vec<Manifest>,
    pin: Option<&PinnedRuntime>,
) -> Result<Vec<Manifest>> {
    let Some(pin) = pin else {
        return Ok(manifests);
    };
    let kept: Vec<Manifest> = manifests
        .into_iter()
        .filter(|m| m.id == pin.id && m.version == pin.version)
        .collect();
    if kept.is_empty() {
        return Err(RaptorError::NotFound(format!(
            "pinned runtime {} {} is not in the runtime index; unpin it to follow the newest version",
            pin.id, pin.version
        )));
    }
    Ok(kept)
}

fn gateway_status(g: &GatewayControl) -> GatewayStatus {
    let listener = g.handle.as_ref().map(|h| h.status());
    GatewayStatus {
        running: g.handle.is_some(),
        addr: g.handle.as_ref().map(|h| h.addr().to_string()),
        metrics: g.handle.as_ref().map(|h| h.metrics()),
        last_error: g.last_error.clone(),
        lan: listener.as_ref().is_some_and(|l| l.lan),
        tls: listener.as_ref().is_some_and(|l| l.tls),
        tls_fingerprint_sha256: listener
            .as_ref()
            .and_then(|l| l.tls_fingerprint_sha256.clone()),
        insecure: listener.as_ref().is_some_and(|l| l.insecure),
    }
}

/// Directory holding the gateway's self-signed certificate, under the config directory.
pub const GATEWAY_TLS_DIR: &str = "gateway-tls";

/// Gateway configuration derived from settings. LAN mode always serves TLS with a
/// self-signed certificate persisted under `tls_dir`, so the same certificate (and pin) is
/// reused across restarts. Localhost stays plain HTTP. There is no insecure option in the
/// shell. LAN without a key is refused by `GatewayConfig::validate`.
pub fn gateway_config(s: &Settings, tls_dir: &Path) -> Result<GatewayConfig> {
    let api_keys = match &s.api_key {
        Some(k) => vec![k.record.clone()],
        None => Vec::new(),
    };
    let lan = s.lan_enabled;
    let cfg = GatewayConfig {
        bind: if lan {
            BindMode::Lan
        } else {
            BindMode::Localhost
        },
        port: s.gateway_port,
        api_keys,
        tls: lan.then(|| TlsConfig::SelfSigned {
            dir: tls_dir.to_path_buf(),
            subject_alt_names: Vec::new(),
        }),
        allow_insecure_lan: false,
        strip_history_reasoning: s.strip_history_reasoning,
        default_effort: s.thinking_effort,
        ..GatewayConfig::default()
    };
    cfg.validate()?;
    Ok(cfg)
}

/// The architecture facts for the memory model, from the GGUF header, with the rationale lines
/// that report them. A header without layer counts keeps the dense estimate, and says so.
pub fn model_arch(entry: &ModelEntry, meta: &ModelMeta) -> (ModelArch, Vec<String>) {
    let g = &entry.gguf;
    let mut facts = Vec::new();
    let info = raptor_models::gguf::arch_info(g.architecture.as_deref());
    if let Some(note) = &info.note {
        facts.push(match &info.name {
            Some(name) => format!("architecture '{name}': {note}"),
            None => format!("architecture: {note}"),
        });
    }
    let hints = raptor_models::runtime_hints(g);
    if hints.offload_layers == 0 {
        facts.push(
            "the header lacks layer counts, so the memory estimate assumes every layer holds KV"
                .into(),
        );
        return (ModelArch::dense(meta), facts);
    }
    let slots = SERVER_DEFAULT_SLOTS;
    let arch = ModelArch {
        kv_layers: hints.kv_layers,
        // The header hint is for one slot; llama-server runs its default slot count.
        recurrent_bytes: hints
            .recurrent_bytes_per_slot
            .saturating_mul(u64::from(slots)),
        offload_layers: hints.offload_layers,
        extra_device_bytes: 0,
    };
    let main = g.main_layer_count().unwrap_or(u64::from(meta.n_layers));
    let recurrent = g.recurrent_layer_count().unwrap_or(0);
    if recurrent > 0 {
        facts.push(format!(
            "hybrid model: {} of {main} layers hold KV, {recurrent} hold recurrent state ({slots} slots)",
            hints.kv_layers
        ));
    } else {
        facts.push(format!("dense attention: all {main} layers hold KV"));
    }
    if g.is_moe() {
        facts.push(format!(
            "mixture of experts: {} experts, {} used per token",
            g.expert_count.unwrap_or(0),
            g.expert_used_count.unwrap_or(0)
        ));
    }
    (arch, facts)
}

/// Launch-relevant facts from the GGUF header. Missing fields are reported, never guessed.
pub fn model_meta(entry: &ModelEntry) -> Result<ModelMeta> {
    let g = &entry.gguf;
    let name = entry.file_name.clone();
    let need = |v: Option<u64>, what: &str| -> Result<u32> {
        let v = v.ok_or_else(|| {
            RaptorError::Unsupported(format!(
                "{name} has no {what} in its GGUF header, so it cannot be sized for loading"
            ))
        })?;
        u32::try_from(v)
            .map_err(|_| RaptorError::Unsupported(format!("{what} of {name} is out of range")))
    };
    Ok(ModelMeta {
        size_bytes: entry.size_bytes,
        n_layers: need(g.block_count, "block count")?,
        n_ctx_train: need(g.context_length, "context length")?,
        n_kv_heads: need(g.head_count_kv, "KV head count")?,
        head_dim: need(g.head_dim(), "head dimension")?,
    })
}

#[cfg(test)]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::ApiKeyEntry;

    const BUNDLED: &str = include_str!("../resources/runtime-manifest.json");

    #[test]
    fn auto_prefers_installed_native_runtime_and_uses_vulkan_as_fallback() {
        let mut hw = crate::testing::fixed_hardware();
        assert_eq!(
            auto_backend_for(&hw, true, true, true, true),
            Some(Backend::Hip)
        );
        hw.gpus[0].vendor = raptor_runtime::hardware::GpuVendor::Nvidia;
        assert_eq!(
            auto_backend_for(&hw, false, true, true, true),
            Some(Backend::Cuda)
        );
        hw.gpus[0].vendor = raptor_runtime::hardware::GpuVendor::Intel;
        assert_eq!(
            auto_backend_for(&hw, true, false, true, true),
            Some(Backend::Sycl)
        );
        assert_eq!(
            auto_backend_for(&hw, false, false, false, true),
            Some(Backend::Vulkan)
        );
        assert_eq!(auto_backend_for(&hw, false, false, false, false), None);
    }

    #[test]
    fn the_long_window_for_a_spilling_moe_is_32k_with_a_4_bit_cache_and_one_slot() {
        use raptor_runtime::config::KvCacheType;
        let o = long_option(8_192, 262_144, 14_673).expect("a longer window than the default");
        assert_eq!(o.id, "long");
        assert_eq!(o.ctx, 32_768);
        assert_eq!(o.kv_cache, KvCacheType::Q4_0);
        assert_eq!(o.slots, 1);
        assert!(o.speculation && !o.capped_by_model);
        assert!(o.note.contains("46.9") && o.note.contains("not measured"));
        assert_ne!(o.kv_cache, KvCacheType::Q8_0, "8-bit is the slowest here");
    }

    #[test]
    fn the_long_window_is_capped_by_the_model_and_skipped_when_not_longer() {
        let capped = long_option(8_192, 16_384, 14_673).unwrap();
        assert_eq!(capped.ctx, 16_384);
        assert!(capped.capped_by_model);
        // Not more than 25% longer than what is already the default: nothing to offer.
        assert!(long_option(32_768, 262_144, 14_673).is_none());
        assert!(long_option(28_000, 262_144, 14_673).is_none());
        assert!(long_option(16_384, 16_384, 14_673).is_none());
    }

    #[test]
    fn default_settings_produce_a_localhost_gateway_config() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = gateway_config(&Settings::default(), dir.path()).unwrap();
        assert_eq!(cfg.bind, BindMode::Localhost);
        assert_eq!(cfg.port, 8790);
        assert!(cfg.api_keys.is_empty());
        assert!(cfg.tls.is_none(), "localhost stays plain HTTP");
    }

    #[test]
    fn the_gateway_default_thinking_effort_follows_the_setting() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = gateway_config(&Settings::default(), dir.path()).unwrap();
        assert_eq!(cfg.default_effort, raptor_gateway::Effort::Medium);
        let high = Settings {
            thinking_effort: raptor_gateway::Effort::High,
            ..Settings::default()
        };
        assert_eq!(
            gateway_config(&high, dir.path()).unwrap().default_effort,
            raptor_gateway::Effort::High
        );
    }

    #[test]
    fn history_reasoning_is_stripped_by_default_and_the_setting_turns_it_off() {
        let dir = tempfile::tempdir().unwrap();
        let on = gateway_config(&Settings::default(), dir.path()).unwrap();
        assert!(on.strip_history_reasoning);
        let off = Settings {
            strip_history_reasoning: false,
            ..Settings::default()
        };
        assert!(
            !gateway_config(&off, dir.path())
                .unwrap()
                .strip_history_reasoning
        );
        // An old settings.json without the field keeps the default (on).
        let old: Settings = serde_json::from_str(r#"{"gateway_port": 9001}"#).unwrap();
        assert!(old.strip_history_reasoning);
    }

    #[test]
    fn lan_without_a_key_is_refused_with_the_gateway_error() {
        let dir = tempfile::tempdir().unwrap();
        let s = Settings {
            lan_enabled: true,
            ..Settings::default()
        };
        assert!(matches!(
            gateway_config(&s, dir.path()),
            Err(RaptorError::Unauthorized(_))
        ));
    }

    #[test]
    fn lan_with_a_key_serves_self_signed_tls_from_the_config_dir_and_no_insecure_option() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, secret) = ApiKeyEntry::create("k1".into(), 1).unwrap();
        let s = Settings {
            lan_enabled: true,
            api_key: Some(entry),
            ..Settings::default()
        };
        let cfg = gateway_config(&s, dir.path()).unwrap();
        assert_eq!(cfg.bind, BindMode::Lan);
        assert!(!cfg.allow_insecure_lan);
        assert_eq!(cfg.api_keys.len(), 1);
        assert!(cfg.api_keys[0].verify(&secret));
        match &cfg.tls {
            Some(TlsConfig::SelfSigned { dir: d, .. }) => assert_eq!(d, dir.path()),
            other => panic!("expected self-signed TLS, got {other:?}"),
        }
    }

    #[test]
    fn lan_tls_certificate_is_stable_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let names: Vec<String> = Vec::new();
        let first = raptor_gateway::tls::ensure_self_signed(dir.path(), &names).unwrap();
        let again = raptor_gateway::tls::ensure_self_signed(dir.path(), &names).unwrap();
        assert_eq!(
            first.fingerprint_sha256, again.fingerprint_sha256,
            "the pin survives restarts"
        );
        assert_eq!(first.fingerprint_sha256.len(), 64);
    }

    #[test]
    fn gateway_status_without_a_listener_reports_no_tls() {
        let g = GatewayControl::default();
        let s = gateway_status(&g);
        assert!(!s.running && !s.tls && !s.lan && !s.insecure);
        assert!(s.tls_fingerprint_sha256.is_none());
    }

    fn hybrid_entry(architecture: &str) -> raptor_models::ModelEntry {
        raptor_models::ModelEntry {
            id: "local/q.gguf".into(),
            file_name: "q.gguf".into(),
            location: "q.gguf".into(),
            managed: false,
            size_bytes: 1,
            sha256: String::new(),
            added_at: 0,
            source: raptor_models::ModelSource::LocalReference {
                original_path: "q.gguf".into(),
            },
            gguf: raptor_models::GgufMeta {
                version: 3,
                architecture: Some(architecture.into()),
                context_length: Some(262_144),
                block_count: Some(65),
                nextn_predict_layers: Some(1),
                full_attention_interval: Some(4),
                ssm_conv_kernel: Some(4),
                ssm_state_size: Some(128),
                ssm_inner_size: Some(4096),
                head_count_kv: Some(4),
                key_length: Some(256),
                ..Default::default()
            },
        }
    }

    #[test]
    fn model_arch_reports_hybrid_kv_layers_and_scales_state_to_server_slots() {
        let entry = hybrid_entry("qwen35");
        let meta = model_meta(&entry).unwrap();
        let (arch, facts) = model_arch(&entry, &meta);
        // 64 main layers, every 4th is attention; the NextN layer is excluded from KV.
        assert_eq!(arch.kv_layers, 16);
        assert_eq!(arch.offload_layers, 66);
        let one_slot = raptor_models::runtime_hints(&entry.gguf).recurrent_bytes_per_slot;
        assert!(one_slot > 0);
        assert_eq!(arch.recurrent_bytes, one_slot * 4);
        assert!(
            facts
                .iter()
                .any(|f| f
                    == "hybrid model: 16 of 64 layers hold KV, 48 hold recurrent state (4 slots)"),
            "{facts:?}"
        );
        // A known architecture adds no support note.
        assert!(
            !facts.iter().any(|f| f.contains("tested list")),
            "{facts:?}"
        );
    }

    #[test]
    fn model_arch_marks_an_unknown_architecture_and_keeps_dense_without_layer_counts() {
        let entry = hybrid_entry("zzfake-arch");
        let meta = model_meta(&entry).unwrap();
        let (_, facts) = model_arch(&entry, &meta);
        assert!(
            facts
                .iter()
                .any(|f| f == "architecture 'zzfake-arch': not in Raptor's tested list; llama-server will decide when you load it"),
            "{facts:?}"
        );

        let mut bare = hybrid_entry("llama");
        bare.gguf.block_count = None;
        let meta = ModelMeta {
            size_bytes: 1,
            n_layers: 32,
            n_ctx_train: 4096,
            n_kv_heads: 8,
            head_dim: 128,
        };
        let (arch, facts) = model_arch(&bare, &meta);
        assert_eq!(arch, ModelArch::dense(&meta));
        assert!(
            facts.iter().any(|f| f.contains("lacks layer counts")),
            "{facts:?}"
        );
    }

    #[test]
    fn model_meta_reports_missing_header_fields_instead_of_guessing() {
        let entry = raptor_models::ModelEntry {
            id: "org/repo/m.gguf".into(),
            file_name: "m.gguf".into(),
            location: "m.gguf".into(),
            managed: true,
            size_bytes: 1,
            sha256: String::new(),
            added_at: 0,
            source: raptor_models::ModelSource::LocalCopy {
                original_path: "x".into(),
            },
            gguf: raptor_models::GgufMeta {
                version: 3,
                kv_count: 0,
                architecture: Some("llama".into()),
                name: None,
                file_type: None,
                quant_label: None,
                context_length: Some(4096),
                block_count: None,
                embedding_length: None,
                head_count: None,
                head_count_kv: Some(8),
                key_length: Some(128),
                tokenizer_model: None,
                has_chat_template: false,
                ..Default::default()
            },
        };
        let err = model_meta(&entry).unwrap_err();
        assert!(
            matches!(err, RaptorError::Unsupported(ref m) if m.contains("block count")),
            "{err}"
        );
    }

    #[test]
    fn bundled_runtime_manifest_is_valid_and_every_entry_is_verified() {
        let index = parse_runtime_index(BUNDLED).unwrap();
        assert!(index.refused.is_empty(), "{:?}", index.refused);
        assert_eq!(index.manifests.len(), 8);
        for m in &index.manifests {
            assert!(m
                .url
                .starts_with("https://github.com/ggml-org/llama.cpp/releases/download/"));
            assert_eq!(m.sha256.len(), 64, "{}", m.id);
        }
    }

    #[test]
    fn entries_without_a_verified_sha256_are_refused_not_installed() {
        let text = r#"{"schema_version": 1, "runtimes": [
            {"id": "llama-cpu-win", "version": "b1", "llama_cpp_build": "b1", "backend": "cpu", "os": "windows",
             "url": "https://example.invalid/a.zip", "sha256": null},
            {"id": "llama-vulkan-win", "version": "b1", "llama_cpp_build": "b1", "backend": "vulkan", "os": "windows",
             "url": "https://example.invalid/b.zip"},
            {"id": "llama-hip-win", "version": "b1", "llama_cpp_build": "b1", "backend": "hip", "os": "windows",
             "gfx_archs": ["gfx1200"], "url": "https://example.invalid/c.zip",
             "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}
        ]}"#;
        let index = parse_runtime_index(text).unwrap();
        assert_eq!(index.manifests.len(), 1);
        assert_eq!(index.manifests[0].id, "llama-hip-win");
        assert_eq!(index.refused.len(), 2);
        assert!(index.refused[0].contains("llama-cpu-win"));
        assert!(index.refused[1].contains("llama-vulkan-win"));
    }

    #[test]
    fn invalid_hash_text_still_fails_the_whole_index() {
        let text = r#"{"schema_version": 1, "runtimes": [
            {"id": "llama-cpu-win", "version": "b1", "llama_cpp_build": "b1", "backend": "cpu", "os": "windows",
             "url": "https://example.invalid/a.zip", "sha256": "not-hex"}]}"#;
        assert!(parse_runtime_index(text).is_err());
    }

    #[test]
    fn pinned_runtime_keeps_only_the_pinned_version_candidates() {
        let index = parse_runtime_index(BUNDLED).unwrap();
        let pin = PinnedRuntime {
            id: "llama-cpu-win".into(),
            version: "b12340".into(),
        };
        let kept = candidates_for_pin(index.manifests.clone(), Some(&pin)).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "llama-cpu-win");

        let missing = PinnedRuntime {
            id: "llama-cpu-win".into(),
            version: "b1".into(),
        };
        assert!(matches!(
            candidates_for_pin(index.manifests.clone(), Some(&missing)),
            Err(RaptorError::NotFound(_))
        ));
        // Without a pin every entry stays available, so the newest can be chosen.
        assert_eq!(
            candidates_for_pin(index.manifests.clone(), None)
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn finished_view_only_reports_final_states() {
        use raptor_models::JobState;
        assert!(finished_view("k", JobState::Running, None).is_none());
        assert!(finished_view("k", JobState::Paused, None).is_none());
        let done = finished_view("k", JobState::Completed, None).unwrap();
        assert_eq!(done.status, "completed");
        let recorded = DownloadFinished {
            key: "k".into(),
            status: "completed".into(),
            path: Some("C:/m/a.gguf".into()),
            sha256: Some("ab".into()),
            error: None,
        };
        let with_detail = finished_view("k", JobState::Completed, Some(&recorded)).unwrap();
        assert_eq!(with_detail.path.as_deref(), Some("C:/m/a.gguf"));
    }
}
