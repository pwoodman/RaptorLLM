//! raptor-runtime: hardware detection, llama.cpp runtime install and selection,
//! llama-server lifecycle, GPU verification, memory-aware configuration,
//! performance memory, and token counting. See docs/specs/runtime.md.
//!
//! The [`RuntimeEngine`] facade ties the modules together. Every external
//! effect (processes, HTTP, downloads, hardware probes) sits behind a trait so
//! the whole crate is testable without a GPU or network.

pub mod config;
pub mod context_max;
pub mod ctxmatrix;
pub mod devices;
pub mod fit;
pub mod gpumem;
pub mod hardware;
pub mod hash;
pub mod http;
pub mod install;
pub mod manifest;
pub mod perfmem;
pub mod process;
pub mod sys;
pub mod tokenizer;
pub mod tune;
pub mod verify;

#[cfg(test)]
mod testing;

use crate::config::{
    recommend, recommend_with_arch, LaunchConfig, ModelArch, ModelMeta, Recommendation,
};
use crate::devices::{launch_env, DevicePreflight, HipSdkSearch};
use crate::fit::{FitProjection, FIT_PARAMS_BINARY};
use crate::hardware::{HardwareInfo, HardwareProbe, SystemProbe};
use crate::http::{HttpClient, ReqwestClient};
use crate::install::{Fetcher, InstalledRuntime, ReqwestFetcher, RuntimeInstaller};
use crate::manifest::{select_runtime, Backend, Manifest, Selection};
use crate::process::{
    FileLogSink, LogLine, LogSink, ManagerDeps, RestartPolicy, ServerManager, ServerState,
    ServerStatus, Spawner, SystemSpawner, Timing,
};
use crate::sys::{CommandRunner, PortCheck, SystemCommandRunner, TcpPortCheck};
use crate::verify::{parse_server_log, verify_gpu, GpuVerdict};
use raptor_core::{DataDirs, RaptorError, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio_util::sync::CancellationToken;

/// Everything the engine needs from the outside world.
pub struct EngineDeps {
    pub probe: Arc<dyn HardwareProbe>,
    pub fetcher: Arc<dyn Fetcher>,
    pub spawner: Arc<dyn Spawner>,
    pub http: Arc<dyn HttpClient>,
    pub ports: Arc<dyn PortCheck>,
    pub sink: Arc<dyn LogSink>,
    pub restart: RestartPolicy,
    pub timing: Timing,
    /// Runs `llama-server --list-devices` for the device preflight.
    pub runner: Arc<dyn CommandRunner>,
    /// Where the HIP SDK bin folders are searched (Windows HIP runtimes).
    pub hip_sdk: HipSdkSearch,
}

/// The installed runtime in use.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeSummary {
    pub id: String,
    pub version: String,
    pub backend: Backend,
    pub degraded: bool,
    pub reason: String,
    pub binary: PathBuf,
}

/// Snapshot of the engine. `gpu_verdict` is present once llama-server has produced log output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineStatus {
    pub hardware_detected: bool,
    pub runtime: Option<RuntimeSummary>,
    pub server: ServerStatus,
    pub gpu_verdict: Option<GpuVerdict>,
    /// Latest device preflight for the selected runtime: set by a GPU load attempt
    /// or [`RuntimeEngine::preflight_devices`]. A `failure` inside means the GPU
    /// backend has no usable device.
    pub device_preflight: Option<DevicePreflight>,
}

#[derive(Default)]
struct EngineState {
    hardware: Option<HardwareInfo>,
    runtime: Option<ActiveRuntime>,
    preflight: Option<DevicePreflight>,
}

/// The runtime version in use, cached so launches need no disk read.
struct ActiveRuntime {
    backend: Backend,
    degraded: bool,
    reason: String,
    installed: InstalledRuntime,
}

impl ActiveRuntime {
    fn from_selection(sel: &Selection, installed: InstalledRuntime) -> Self {
        Self {
            backend: sel.backend,
            degraded: sel.degraded,
            reason: sel.reason.clone(),
            installed,
        }
    }

    fn summary(&self) -> RuntimeSummary {
        RuntimeSummary {
            id: self.installed.id.clone(),
            version: self.installed.version.clone(),
            backend: self.backend,
            degraded: self.degraded,
            reason: self.reason.clone(),
            binary: self.installed.binary.clone(),
        }
    }
}

/// Facade over the runtime subsystem.
pub struct RuntimeEngine {
    probe: Arc<dyn HardwareProbe>,
    installer: RuntimeInstaller,
    server: ServerManager,
    runner: Arc<dyn CommandRunner>,
    hip_sdk: HipSdkSearch,
    state: Mutex<EngineState>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl RuntimeEngine {
    /// Builds the engine from explicit dependencies. Runtimes live under `dirs`.
    pub fn new(dirs: &DataDirs, deps: EngineDeps) -> Self {
        Self {
            probe: deps.probe,
            installer: RuntimeInstaller::new(dirs, deps.fetcher),
            server: ServerManager::new(
                ManagerDeps {
                    spawner: deps.spawner,
                    http: deps.http,
                    ports: deps.ports,
                    sink: deps.sink,
                },
                deps.restart,
                deps.timing,
            ),
            runner: deps.runner,
            hip_sdk: deps.hip_sdk,
            state: Mutex::new(EngineState::default()),
        }
    }

    /// Engine wired to the real system: PowerShell/Linux probes, reqwest downloads,
    /// real child processes, and a log file at `logs/llama-server.log`.
    pub fn with_system_deps(dirs: &DataDirs) -> Result<Self> {
        let runner: Arc<dyn CommandRunner> = Arc::new(SystemCommandRunner);
        let sink = FileLogSink::rolling(&dirs.logs())?;
        Ok(Self::new(
            dirs,
            EngineDeps {
                probe: Arc::new(SystemProbe::new(runner.clone())),
                fetcher: Arc::new(ReqwestFetcher::new()?),
                spawner: Arc::new(SystemSpawner),
                http: Arc::new(ReqwestClient::new()?),
                ports: Arc::new(TcpPortCheck),
                sink: Arc::new(sink),
                restart: RestartPolicy::default(),
                timing: Timing::default(),
                runner,
                hip_sdk: HipSdkSearch::system(),
            },
        ))
    }

    /// Probes the hardware and caches the result for later calls.
    pub async fn detect_hardware(&self) -> Result<HardwareInfo> {
        let hw = self.probe.probe().await?;
        let mut state = lock(&self.state);
        state.hardware = Some(hw.clone());
        // Sampling memory after a load must retain the device diagnosis for that running
        // process, including the reason it fell back to CPU.
        if self.server.status().state == ServerState::Stopped {
            state.preflight = None;
        }
        Ok(hw)
    }

    /// Selects the best runtime for this machine from `manifests` and installs it
    /// if needed. Returns the selection, including `degraded` when only CPU is possible.
    pub async fn ensure_runtime(
        &self,
        manifests: &[Manifest],
        cancel: &CancellationToken,
    ) -> Result<Selection> {
        let cached = lock(&self.state).hardware.clone();
        let hw = match cached {
            Some(hw) => hw,
            None => self.detect_hardware().await?,
        };
        let selection = select_runtime(&hw, manifests)?;
        if selection.degraded {
            tracing::warn!(reason = %selection.reason, "no GPU runtime compatible; using CPU");
        }
        let installed = self.installer.install(&selection.manifest, cancel).await?;
        let mut state = lock(&self.state);
        state.runtime = Some(ActiveRuntime::from_selection(&selection, installed));
        state.preflight = None;
        Ok(selection)
    }

    /// Uses an llama-server binary this engine did not install: a developer build or
    /// a test fixture. Nothing is copied, downloaded or hash-checked, and the runtime
    /// is reported as `local`. The shell never calls this. Refused while the server runs.
    pub fn use_local_runtime(&self, binary: &Path, backend: Backend) -> Result<RuntimeSummary> {
        self.refuse_while_running("switch the runtime")?;
        if !binary.is_file() {
            return Err(RaptorError::NotFound(format!(
                "no llama-server binary at {}",
                binary.display()
            )));
        }
        let dir = binary
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let active = ActiveRuntime {
            backend,
            degraded: false,
            reason: format!("local {} runtime at {}", backend.label(), binary.display()),
            installed: InstalledRuntime {
                id: format!("local-{}", backend.label()),
                version: "local".into(),
                backend,
                sha256: String::new(),
                dir,
                binary: binary.to_path_buf(),
            },
        };
        let summary = active.summary();
        let mut state = lock(&self.state);
        state.runtime = Some(active);
        state.preflight = None;
        Ok(summary)
    }

    /// Downloads, verifies and installs `manifest` without selecting it. This adds a second
    /// backend (Vulkan next to ROCm) while the one in use stays as it is. The SHA-256 is checked
    /// before anything is extracted, exactly as for [`RuntimeEngine::ensure_runtime`].
    pub async fn install_runtime(
        &self,
        manifest: &Manifest,
        cancel: &CancellationToken,
    ) -> Result<InstalledRuntime> {
        self.installer.install(manifest, cancel).await
    }

    /// Makes the newest installed runtime of `backend` the one in use. Used to run one model on
    /// Vulkan and another on ROCm. Nothing is downloaded: when no runtime of that backend is
    /// installed this is an error that says so, never a quiet switch to another backend. Each
    /// runtime id contributes the version marked current (so a rollback is respected). Refused
    /// while the server runs.
    pub fn activate_installed(&self, backend: Backend) -> Result<RuntimeSummary> {
        self.refuse_while_running("switch the runtime")?;
        let mut best: Option<InstalledRuntime> = None;
        for id in self
            .installer
            .list_installed()?
            .into_iter()
            .filter(|r| r.backend == backend)
            .map(|r| r.id)
        {
            let Some(cur) = self.installer.current(&id)? else {
                continue;
            };
            let newer = best.as_ref().is_none_or(|b| {
                crate::manifest::version_key(&cur.version)
                    > crate::manifest::version_key(&b.version)
            });
            if newer {
                best = Some(cur);
            }
        }
        let installed = best.ok_or_else(|| {
            RaptorError::NotFound(format!("no {} runtime is installed", backend.label()))
        })?;
        let active = ActiveRuntime {
            backend,
            degraded: false,
            reason: format!(
                "{} {} ({}) chosen for this model",
                installed.id,
                installed.version,
                backend.label()
            ),
            installed,
        };
        let summary = active.summary();
        let mut state = lock(&self.state);
        state.runtime = Some(active);
        state.preflight = None;
        Ok(summary)
    }

    /// Asks llama-server which devices the selected runtime can use and returns the
    /// result, also stored for [`RuntimeEngine::status`]. Starts no model. For CPU
    /// runtimes the preflight never reports a failure.
    pub async fn preflight_devices(&self) -> Result<DevicePreflight> {
        let (backend, binary) = self.selected_runtime()?;
        let (pf, _) = self.device_check(backend, &binary).await?;
        lock(&self.state).preflight = Some(pf.clone());
        Ok(pf)
    }

    fn selected_runtime(&self) -> Result<(Backend, PathBuf)> {
        let state = lock(&self.state);
        let active = state.runtime.as_ref().ok_or_else(|| {
            RaptorError::Conflict("no runtime selected; call ensure_runtime first".into())
        })?;
        Ok((active.backend, active.installed.binary.clone()))
    }

    /// Runs the device preflight. Returns the result and the environment the
    /// server needs (HIP SDK folders on PATH). `Err` only when the binary cannot run.
    async fn device_check(
        &self,
        backend: Backend,
        binary: &Path,
    ) -> Result<(DevicePreflight, Vec<(String, String)>)> {
        let sdk_bin_dirs = if backend == Backend::Hip {
            self.hip_sdk.bin_dirs()
        } else {
            Vec::new()
        };
        let env = launch_env(&sdk_bin_dirs, std::env::var("PATH").ok().as_deref())?;
        let pf = devices::preflight(&*self.runner, binary, backend, sdk_bin_dirs, &env).await?;
        Ok((pf, env))
    }

    /// Makes the previously installed version of the runtime in use current again
    /// and returns its summary. The version it replaces becomes the rollback target,
    /// so calling this again reverts the rollback. Refused while llama-server is
    /// running, because the server would keep the binary it started with.
    ///
    /// The cached selection from `ensure_runtime` is dropped and replaced with the
    /// restored version. Calling `ensure_runtime` again re-selects the newest
    /// installed version, which undoes a rollback.
    pub fn rollback(&self) -> Result<RuntimeSummary> {
        self.refuse_while_running("roll back the runtime")?;
        let (id, degraded) = {
            let state = lock(&self.state);
            let active = state.runtime.as_ref().ok_or_else(|| {
                RaptorError::Conflict("no runtime selected; call ensure_runtime first".into())
            })?;
            (active.installed.id.clone(), active.degraded)
        };
        let restored = self.installer.rollback(&id)?;
        let reason = format!("rolled back to version {}", restored.version);
        let active = ActiveRuntime {
            backend: restored.backend,
            degraded,
            reason,
            installed: restored,
        };
        let summary = active.summary();
        let mut state = lock(&self.state);
        state.runtime = Some(active);
        state.preflight = None;
        Ok(summary)
    }

    /// Removes every installed version of runtime `runtime_id`. Models and
    /// downloads are never touched. Refused while llama-server is running on
    /// that runtime. If it was the runtime in use, the cached selection is dropped.
    pub fn uninstall(&self, runtime_id: &str) -> Result<()> {
        let in_use = lock(&self.state)
            .runtime
            .as_ref()
            .is_some_and(|a| a.installed.id == runtime_id);
        if in_use {
            self.refuse_while_running("uninstall the runtime in use")?;
        }
        self.installer.uninstall(runtime_id)?;
        if in_use {
            let mut state = lock(&self.state);
            state.runtime = None;
            state.preflight = None;
        }
        Ok(())
    }

    /// Lists all installed runtimes on disk.
    pub fn installed_runtimes(&self) -> Result<Vec<InstalledRuntime>> {
        self.installer.list_installed()
    }

    pub fn install_stage(&self) -> crate::install::InstallStage {
        self.installer.stage()
    }

    /// The newest `n` llama-server log lines, oldest first.
    pub fn log_tail(&self, n: usize) -> Vec<LogLine> {
        self.server.log_tail(n)
    }

    /// Recommends a launch configuration for `model` using the detected hardware
    /// and the selected runtime. Call [`RuntimeEngine::ensure_runtime`] first.
    /// The returned config has no model path yet; set it with `with_model`.
    pub fn recommend_config(&self, model: &ModelMeta, ctx_target: u32) -> Result<Recommendation> {
        let state = lock(&self.state);
        let hw = state.hardware.as_ref().ok_or_else(|| {
            RaptorError::Conflict("hardware not detected; call detect_hardware first".into())
        })?;
        let active = state.runtime.as_ref().ok_or_else(|| {
            RaptorError::Conflict("no runtime selected; call ensure_runtime first".into())
        })?;
        recommend(hw, model, ctx_target, active.backend)
    }

    /// Like [`RuntimeEngine::recommend_config`], for a model whose architecture is
    /// known (KV layers, recurrent state, offloadable layers). `projection` is an
    /// optional full-offload projection from [`RuntimeEngine::probe_fit`] for the
    /// same context; without it the estimate is the GGUF-based one.
    pub fn recommend_config_with_arch(
        &self,
        model: &ModelMeta,
        arch: &ModelArch,
        ctx_target: u32,
        projection: Option<&FitProjection>,
    ) -> Result<Recommendation> {
        let state = lock(&self.state);
        let hw = state.hardware.as_ref().ok_or_else(|| {
            RaptorError::Conflict("hardware not detected; call detect_hardware first".into())
        })?;
        let active = state.runtime.as_ref().ok_or_else(|| {
            RaptorError::Conflict("no runtime selected; call ensure_runtime first".into())
        })?;
        recommend_with_arch(hw, model, arch, ctx_target, active.backend, projection)
    }

    /// Asks llama.cpp for its own memory projection for `config` with every layer on
    /// the GPU, using `llama-fit-params --fit-print` from the selected runtime. It
    /// loads no weights and starts no server. `config.backend` must be the selected
    /// backend. Errors (missing tool, no GPU row) are returned, not papered over.
    pub async fn probe_fit(&self, config: &LaunchConfig) -> Result<FitProjection> {
        let (backend, binary) = self.selected_runtime()?;
        if config.backend != backend {
            return Err(RaptorError::InvalidInput(format!(
                "config is for the {} backend but the selected runtime is {}",
                config.backend.label(),
                backend.label()
            )));
        }
        let dir = binary.parent().unwrap_or_else(|| Path::new("."));
        let fit_binary = dir.join(FIT_PARAMS_BINARY);
        let sdk_bin_dirs = if backend == Backend::Hip {
            self.hip_sdk.bin_dirs()
        } else {
            Vec::new()
        };
        let env = launch_env(&sdk_bin_dirs, std::env::var("PATH").ok().as_deref())?;
        crate::fit::probe(&*self.runner, &fit_binary, config, &env).await
    }

    /// The largest context for `base` (every layer on the GPU) that fits `budget_mib` with `kv`
    /// and `slots`, from llama.cpp's own memory projections. See [`context_max`].
    pub async fn max_context(
        &self,
        base: &LaunchConfig,
        kv: crate::config::KvCacheType,
        slots: u32,
        budget_mib: u64,
        cap: u32,
        overhead: crate::context_max::Overhead,
    ) -> Result<crate::context_max::ContextFit> {
        let (backend, binary) = self.selected_runtime()?;
        let dir = binary.parent().unwrap_or_else(|| Path::new("."));
        let sdk_bin_dirs = if backend == Backend::Hip {
            self.hip_sdk.bin_dirs()
        } else {
            Vec::new()
        };
        let env = launch_env(&sdk_bin_dirs, std::env::var("PATH").ok().as_deref())?;
        crate::context_max::find(
            &*self.runner,
            dir,
            base,
            kv,
            slots,
            budget_mib,
            cap,
            overhead,
            &env,
        )
        .await
    }

    /// Measures one context depth of `model` with llama-bench from the selected runtime. Refused
    /// while llama-server runs. See [`crate::tune::measure_depth`] and [`ctxmatrix`].
    pub async fn bench_depth(
        &self,
        model: &Path,
        gpu_layers: u32,
        ubatch: u32,
        kv: crate::config::KvCacheType,
        depth: u32,
        flash_attn: bool,
    ) -> Result<crate::tune::DepthPoint> {
        self.refuse_while_running("benchmark a model")?;
        let (backend, binary) = self.selected_runtime()?;
        let dir = binary.parent().unwrap_or_else(|| Path::new("."));
        let bench = dir.join(crate::tune::BENCH_BINARY);
        if !bench.is_file() {
            return Err(RaptorError::Unsupported(format!(
                "{} is not in this runtime, so the model cannot be benchmarked",
                crate::tune::BENCH_BINARY
            )));
        }
        let sdk_bin_dirs = if backend == Backend::Hip {
            self.hip_sdk.bin_dirs()
        } else {
            Vec::new()
        };
        let env = launch_env(&sdk_bin_dirs, std::env::var("PATH").ok().as_deref())?;
        crate::tune::measure_depth(
            &*self.runner,
            &bench,
            model,
            gpu_layers,
            ubatch,
            kv.llama_arg(),
            depth,
            flash_attn,
            &env,
        )
        .await
    }

    /// Measures `model` with llama-bench from the selected runtime at `gpu_layers`, across the
    /// micro-batch candidates. Starts no server and needs the GPU free, so it is refused while
    /// llama-server runs. Errors (a missing llama-bench, out of memory) are returned as they are.
    pub async fn tune_measure(
        &self,
        model: &Path,
        gpu_layers: u32,
        context_size: u32,
        kv_cache: crate::config::KvCacheType,
        flash_attn: bool,
    ) -> Result<Vec<crate::tune::Measured>> {
        self.refuse_while_running("tune a model")?;
        let (backend, binary) = self.selected_runtime()?;
        let dir = binary.parent().unwrap_or_else(|| Path::new("."));
        let bench = dir.join(crate::tune::BENCH_BINARY);
        if !bench.is_file() {
            return Err(RaptorError::Unsupported(format!(
                "{} is not in this runtime, so the model cannot be benchmarked",
                crate::tune::BENCH_BINARY
            )));
        }
        let sdk_bin_dirs = if backend == Backend::Hip {
            self.hip_sdk.bin_dirs()
        } else {
            Vec::new()
        };
        let env = launch_env(&sdk_bin_dirs, std::env::var("PATH").ok().as_deref())?;
        crate::tune::measure(
            &*self.runner,
            &bench,
            model,
            gpu_layers,
            context_size,
            kv_cache,
            flash_attn,
            &env,
        )
        .await
    }

    fn refuse_while_running(&self, action: &str) -> Result<()> {
        if self.server.status().state != ServerState::Stopped {
            return Err(RaptorError::Conflict(format!(
                "unload the model before you {action}"
            )));
        }
        Ok(())
    }

    /// Starts llama-server with `config` on the selected runtime. Returns once the
    /// server is Ready, with the GPU verdict taken from its log. A GPU backend that
    /// is not verified is reported as such and never shown as active.
    ///
    /// A GPU backend first runs the device preflight. If llama-server initializes no
    /// device of that backend, the load fails with the typed diagnosis, unless
    /// `config.allow_cpu_fallback` is set. A silent CPU run never happens.
    pub async fn load_model(&self, config: LaunchConfig) -> Result<EngineStatus> {
        let (backend, binary) = self.selected_runtime()?;
        if config.backend != backend {
            return Err(RaptorError::InvalidInput(format!(
                "config is for the {} backend but the selected runtime is {}",
                config.backend.label(),
                backend.label()
            )));
        }
        let mut extra_env = Vec::new();
        if backend.is_gpu() {
            let (pf, env) = self.device_check(backend, &binary).await?;
            extra_env = env;
            let failure = pf.failure.clone();
            lock(&self.state).preflight = Some(pf);
            if let Some(f) = failure {
                if !config.allow_cpu_fallback {
                    tracing::error!(cause = ?f.cause, "GPU backend has no device; load refused");
                    return Err(RaptorError::Runtime(f.full_text()));
                }
                tracing::warn!(cause = ?f.cause, "GPU backend has no device; CPU fallback allowed by the caller");
            }
        } else {
            lock(&self.state).preflight = None;
        }
        self.server
            .start_with_env(config, &binary, extra_env)
            .await?;
        let status = self.status();
        if let Some(GpuVerdict::Verified { .. }) = status.gpu_verdict {
            tracing::info!("GPU offload verified from llama-server log");
        } else {
            tracing::warn!(verdict = ?status.gpu_verdict, "GPU offload NOT verified");
        }
        Ok(status)
    }

    /// Stops llama-server (idempotent).
    pub async fn unload(&self) -> Result<()> {
        self.server.stop().await
    }

    /// Current state. The GPU verdict is computed from the server log whenever
    /// the server is not stopped.
    pub fn status(&self) -> EngineStatus {
        let state = lock(&self.state);
        let runtime = state.runtime.as_ref().map(ActiveRuntime::summary);
        let server = self.server.status();
        let gpu_verdict = match (&state.runtime, &server.state) {
            (Some(active), s) if *s != ServerState::Stopped => Some(verify_gpu(
                active.backend,
                &parse_server_log(&self.server.log_text()),
            )),
            _ => None,
        };
        EngineStatus {
            hardware_detected: state.hardware.is_some(),
            runtime,
            server,
            gpu_verdict,
            device_preflight: state.preflight.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GIB;
    use crate::hardware::{CpuInfo, GpuInfo, GpuVendor, Os};
    use crate::hash::sha256_hex;
    use crate::testing::{
        make_zip, FakeFetcher, FakeHttp, FakeLaunch, FakePorts, FakeProbe, FakeRunner, FakeSpawner,
        MemorySink, Step,
    };
    use std::time::Duration;

    const URL: &str = "https://example.invalid/llama-hip-win-b7100.zip";
    /// What `llama-server --list-devices` prints on the RX 9060 XT machine.
    const RX9060_LIST: &str =
        "Available devices:\n  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n";

    fn manifest_for(zip: &[u8], backend: Backend, gfx: &[&str]) -> Manifest {
        Manifest {
            id: format!("llama-{}-win", backend.label()),
            version: "b7100".into(),
            llama_cpp_build: "b7100".into(),
            backend,
            os: Os::Windows,
            gfx_archs: gfx.iter().map(|s| s.to_string()).collect(),
            url: URL.into(),
            sha256: sha256_hex(zip),
            min_driver: None,
            signature: None,
        }
    }

    fn rx9060_hw() -> HardwareInfo {
        HardwareInfo {
            os: Os::Windows,
            cpu: CpuInfo {
                brand: "Ryzen".into(),
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
                vram_free: None,
                arch: Some("gfx1200".into()),
                driver: Some("32.0.12019.1024".into()),
            }],
        }
    }

    fn cpu_hw() -> HardwareInfo {
        HardwareInfo {
            gpus: vec![],
            ..rx9060_hw()
        }
    }

    fn model() -> ModelMeta {
        ModelMeta {
            size_bytes: 8 * GIB,
            n_layers: 40,
            n_ctx_train: 32768,
            n_kv_heads: 8,
            head_dim: 128,
        }
    }

    const HIP_LOG: &[&str] = &[
        "ggml_cuda_init: found 1 ROCm devices:",
        "  Device 0: AMD Radeon RX 9060 XT, gfx1200 (0x1200), VMM: no, Wave Size: 32",
        "load_tensors: offloaded 41/41 layers to GPU",
        "load_tensors:        ROCm0 model buffer size =  4168.09 MiB",
        "main: server is listening on http://127.0.0.1:8080",
    ];

    fn engine(
        hw: HardwareInfo,
        zip: Vec<u8>,
        script: Vec<FakeLaunch>,
        dirs: &DataDirs,
    ) -> (RuntimeEngine, Arc<FakeSpawner>) {
        let spawner = Arc::new(FakeSpawner::new(script));
        let http = FakeHttp::new()
            .route("/health", vec![Step::Status(503), Step::Status(200)])
            .route("/v1/models", vec![Step::Status(200)]);
        let eng = RuntimeEngine::new(
            dirs,
            EngineDeps {
                probe: Arc::new(FakeProbe { hw }),
                fetcher: Arc::new(FakeFetcher::default().serve(URL, zip)),
                spawner: spawner.clone(),
                http: Arc::new(http),
                ports: Arc::new(FakePorts { free: true }),
                sink: Arc::new(MemorySink::default()),
                restart: RestartPolicy::default(),
                timing: Timing {
                    startup_timeout: Duration::from_secs(30),
                    poll_interval: Duration::from_millis(200),
                    health_interval: Duration::from_secs(60),
                    stop_grace: Duration::from_secs(1),
                    failures_before_degraded: 2,
                },
                runner: Arc::new(FakeRunner::default().with_ok_for_all(RX9060_LIST)),
                hip_sdk: HipSdkSearch::default(),
            },
        );
        (eng, spawner)
    }

    fn hip_zip() -> Vec<u8> {
        make_zip(&[("llama-b7100/llama-server.exe", b"MZ fake binary")])
    }

    #[tokio::test(start_paused = true)]
    async fn end_to_end_hip_install_recommend_load_verified() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = hip_zip();
        let manifests = vec![
            manifest_for(&zip, Backend::Hip, &["gfx1200"]),
            manifest_for(&zip, Backend::Cpu, &[]),
        ];
        let (eng, spawner) = engine(rx9060_hw(), zip, vec![FakeLaunch::lines(HIP_LOG)], &dirs);
        let cancel = CancellationToken::new();

        let before = eng.status();
        assert!(!before.hardware_detected);
        assert!(
            eng.recommend_config(&model(), 8192).is_err(),
            "recommend needs hardware and runtime first"
        );

        let sel = eng.ensure_runtime(&manifests, &cancel).await.unwrap();
        assert_eq!(sel.backend, Backend::Hip);
        assert!(!sel.degraded);

        let rec = eng.recommend_config(&model(), 8192).unwrap();
        assert_eq!(rec.config.gpu_layers, 40);
        let cfg = rec.config.with_model(dir.path().join("m.gguf")).unwrap();

        let status = eng.load_model(cfg).await.unwrap();
        assert_eq!(status.server.state, ServerState::Ready);
        assert!(
            matches!(status.gpu_verdict, Some(GpuVerdict::Verified { .. })),
            "{:?}",
            status.gpu_verdict
        );
        assert_eq!(status.runtime.as_ref().unwrap().backend, Backend::Hip);
        assert_eq!(spawner.spawn_count(), 1);
        let spec = spawner.last_spec().unwrap();
        assert!(spec.program.ends_with("llama-server.exe"));

        eng.unload().await.unwrap();
        assert_eq!(eng.status().server.state, ServerState::Stopped);
        assert_eq!(eng.status().gpu_verdict, None);
    }

    #[tokio::test(start_paused = true)]
    async fn no_gpu_selects_cpu_and_verdict_is_cpu_only() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = make_zip(&[("llama-server.exe", b"cpu build")]);
        let manifests = vec![
            manifest_for(&zip, Backend::Hip, &["gfx1200"]),
            manifest_for(&zip, Backend::Cpu, &[]),
        ];
        let cpu_log = &[
            "load_backend: loaded CPU backend",
            "llama_context: flash_attn = 0",
            "main: server is listening",
        ];
        let (eng, _) = engine(cpu_hw(), zip, vec![FakeLaunch::lines(cpu_log)], &dirs);
        let cancel = CancellationToken::new();
        let sel = eng.ensure_runtime(&manifests, &cancel).await.unwrap();
        assert!(sel.degraded);
        assert_eq!(sel.backend, Backend::Cpu);
        let rec = eng.recommend_config(&model(), 4096).unwrap();
        let status = eng
            .load_model(rec.config.with_model(dir.path().join("m.gguf")).unwrap())
            .await
            .unwrap();
        assert!(matches!(
            status.gpu_verdict,
            Some(GpuVerdict::CpuOnly { .. })
        ));
        eng.unload().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn load_rejects_config_for_a_different_backend() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = hip_zip();
        let manifests = vec![
            manifest_for(&zip, Backend::Hip, &["gfx1200"]),
            manifest_for(&zip, Backend::Cpu, &[]),
        ];
        let (eng, spawner) = engine(rx9060_hw(), zip, vec![], &dirs);
        eng.ensure_runtime(&manifests, &CancellationToken::new())
            .await
            .unwrap();
        let cpu_cfg = LaunchConfig::builder(dir.path().join("m.gguf"), Backend::Cpu)
            .build()
            .unwrap();
        let err = eng.load_model(cpu_cfg).await.unwrap_err();
        assert!(matches!(err, RaptorError::InvalidInput(_)));
        assert_eq!(spawner.spawn_count(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn ensure_runtime_is_idempotent_and_does_not_redownload() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = hip_zip();
        let manifests = vec![
            manifest_for(&zip, Backend::Hip, &["gfx1200"]),
            manifest_for(&zip, Backend::Cpu, &[]),
        ];
        let spawner = Arc::new(FakeSpawner::new(vec![]));
        let fetcher = Arc::new(FakeFetcher::default().serve(URL, zip));
        let eng = RuntimeEngine::new(
            &dirs,
            EngineDeps {
                probe: Arc::new(FakeProbe { hw: rx9060_hw() }),
                fetcher: fetcher.clone(),
                spawner,
                http: Arc::new(FakeHttp::new()),
                ports: Arc::new(FakePorts { free: true }),
                sink: Arc::new(MemorySink::default()),
                restart: RestartPolicy::default(),
                timing: Timing::default(),
                runner: Arc::new(FakeRunner::default()),
                hip_sdk: HipSdkSearch::default(),
            },
        );
        let cancel = CancellationToken::new();
        eng.ensure_runtime(&manifests, &cancel).await.unwrap();
        eng.ensure_runtime(&manifests, &cancel).await.unwrap();
        assert_eq!(fetcher.calls(), 1);
        let rt = eng.status().runtime.unwrap();
        assert_eq!(rt.version, "b7100");
        assert!(rt.binary.ends_with("llama-server.exe"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_model_can_switch_between_installed_backends_without_downloading() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = hip_zip();
        let hip = manifest_for(&zip, Backend::Hip, &["gfx1200"]);
        let vulkan = manifest_for(&zip, Backend::Vulkan, &[]);
        let (eng, _) = engine(rx9060_hw(), zip, vec![], &dirs);
        let cancel = CancellationToken::new();
        eng.ensure_runtime(std::slice::from_ref(&hip), &cancel)
            .await
            .unwrap();

        // Vulkan is not installed yet: an error that names it, and the ROCm selection stays.
        let err = eng.activate_installed(Backend::Vulkan).unwrap_err();
        assert!(matches!(err, RaptorError::NotFound(_)));
        assert!(err.to_string().contains("vulkan"), "{err}");
        assert_eq!(eng.status().runtime.unwrap().backend, Backend::Hip);

        eng.installer.install(&vulkan, &cancel).await.unwrap();
        let summary = eng.activate_installed(Backend::Vulkan).unwrap();
        assert_eq!(summary.id, "llama-vulkan-win");
        assert_eq!(summary.backend, Backend::Vulkan);
        assert!(!summary.degraded);
        assert_eq!(eng.status().runtime.unwrap().backend, Backend::Vulkan);

        let back = eng.activate_installed(Backend::Hip).unwrap();
        assert_eq!(back.id, "llama-hip-win");
        assert!(eng.activate_installed(Backend::Cuda).is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn switching_backend_is_refused_while_a_model_runs() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let zip = hip_zip();
        let hip = manifest_for(&zip, Backend::Hip, &["gfx1200"]);
        let vulkan = manifest_for(&zip, Backend::Vulkan, &[]);
        let (eng, _) = engine(rx9060_hw(), zip, vec![FakeLaunch::lines(HIP_LOG)], &dirs);
        let cancel = CancellationToken::new();
        eng.ensure_runtime(std::slice::from_ref(&hip), &cancel)
            .await
            .unwrap();
        eng.installer.install(&vulkan, &cancel).await.unwrap();
        let cfg = launch_config(&dir, &eng);
        eng.load_model(cfg).await.unwrap();
        let err = eng.activate_installed(Backend::Vulkan).unwrap_err();
        assert!(matches!(err, RaptorError::Conflict(_)), "{err}");
        assert_eq!(eng.status().runtime.unwrap().backend, Backend::Hip);
        eng.unload().await.unwrap();
    }

    const URL_OLD: &str = "https://example.invalid/llama-hip-win-b7000.zip";

    /// Two installed Hip versions: b7000 first, then b7100, which is current.
    async fn two_versions(
        dirs: &DataDirs,
        script: Vec<FakeLaunch>,
    ) -> (RuntimeEngine, Arc<FakeSpawner>) {
        let old_zip = make_zip(&[("llama-server.exe", b"old build")]);
        let new_zip = hip_zip();
        let mut old = manifest_for(&old_zip, Backend::Hip, &["gfx1200"]);
        old.version = "b7000".into();
        old.url = URL_OLD.into();
        old.sha256 = sha256_hex(&old_zip);
        let new = manifest_for(&new_zip, Backend::Hip, &["gfx1200"]);
        let spawner = Arc::new(FakeSpawner::new(script));
        let fetcher = FakeFetcher::default()
            .serve(URL_OLD, old_zip)
            .serve(URL, new_zip);
        let http = FakeHttp::new()
            .route("/health", vec![Step::Status(200)])
            .route("/v1/models", vec![Step::Status(200)]);
        let eng = RuntimeEngine::new(
            dirs,
            EngineDeps {
                probe: Arc::new(FakeProbe { hw: rx9060_hw() }),
                fetcher: Arc::new(fetcher),
                spawner: spawner.clone(),
                http: Arc::new(http),
                ports: Arc::new(FakePorts { free: true }),
                sink: Arc::new(MemorySink::default()),
                restart: RestartPolicy::default(),
                timing: Timing::default(),
                runner: Arc::new(FakeRunner::default().with_ok_for_all(RX9060_LIST)),
                hip_sdk: HipSdkSearch::default(),
            },
        );
        let cancel = CancellationToken::new();
        eng.ensure_runtime(std::slice::from_ref(&old), &cancel)
            .await
            .unwrap();
        eng.ensure_runtime(&[new, old], &cancel).await.unwrap();
        (eng, spawner)
    }

    fn launch_config(dir: &tempfile::TempDir, eng: &RuntimeEngine) -> LaunchConfig {
        eng.recommend_config(&model(), 8192)
            .unwrap()
            .config
            .with_model(dir.path().join("m.gguf"))
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn rollback_switches_status_and_can_be_undone() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let (eng, _) = two_versions(&dirs, vec![]).await;
        assert_eq!(eng.status().runtime.unwrap().version, "b7100");

        let back = eng.rollback().unwrap();
        assert_eq!(back.version, "b7000");
        let status = eng.status().runtime.unwrap();
        assert_eq!(status.version, "b7000");
        assert!(status.binary.ends_with("llama-server.exe"));

        let forward = eng.rollback().unwrap();
        assert_eq!(forward.version, "b7100", "rollback is reversible");
        assert_eq!(eng.status().runtime.unwrap().version, "b7100");
    }

    #[tokio::test(start_paused = true)]
    async fn rollback_without_selection_or_previous_version_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let (eng, _) = engine(rx9060_hw(), hip_zip(), vec![], &dirs);
        assert!(matches!(eng.rollback(), Err(RaptorError::Conflict(_))));

        let zip = hip_zip();
        let manifests = vec![manifest_for(&zip, Backend::Hip, &["gfx1200"])];
        let (eng, _) = engine(rx9060_hw(), zip, vec![], &dirs);
        eng.ensure_runtime(&manifests, &CancellationToken::new())
            .await
            .unwrap();
        assert!(matches!(eng.rollback(), Err(RaptorError::NotFound(_))));
        assert_eq!(eng.status().runtime.unwrap().version, "b7100");
    }

    #[tokio::test(start_paused = true)]
    async fn rollback_is_refused_while_the_server_runs() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let (eng, _) = two_versions(&dirs, vec![FakeLaunch::lines(HIP_LOG)]).await;
        let cfg = launch_config(&dir, &eng);
        eng.load_model(cfg).await.unwrap();
        assert!(matches!(eng.rollback(), Err(RaptorError::Conflict(_))));
        assert_eq!(eng.status().runtime.unwrap().version, "b7100");
        eng.unload().await.unwrap();
        assert_eq!(eng.rollback().unwrap().version, "b7000");
    }

    #[tokio::test(start_paused = true)]
    async fn uninstall_removes_runtime_and_never_touches_models() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let model_file = dirs.models().join("keep.gguf");
        std::fs::write(&model_file, b"weights").unwrap();
        let (eng, _) = two_versions(&dirs, vec![]).await;

        eng.uninstall("llama-hip-win").unwrap();
        assert!(!dirs.runtimes().join("llama-hip-win").exists());
        assert!(eng.status().runtime.is_none(), "stale selection dropped");
        assert_eq!(std::fs::read(&model_file).unwrap(), b"weights");
        assert!(matches!(
            eng.uninstall("llama-hip-win"),
            Err(RaptorError::NotFound(_))
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn uninstall_rejects_traversal_and_is_refused_for_a_running_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let victim = dirs.models().join("keep.gguf");
        std::fs::write(&victim, b"x").unwrap();
        let (eng, _) = two_versions(&dirs, vec![FakeLaunch::lines(HIP_LOG)]).await;
        for bad in ["..", "../models", "a/b", "C:x", ""] {
            assert!(
                matches!(eng.uninstall(bad), Err(RaptorError::InvalidInput(_))),
                "{bad:?}"
            );
        }
        assert!(victim.exists());

        eng.load_model(launch_config(&dir, &eng)).await.unwrap();
        assert!(matches!(
            eng.uninstall("llama-hip-win"),
            Err(RaptorError::Conflict(_))
        ));
        assert!(dirs.runtimes().join("llama-hip-win").exists());
        eng.unload().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn log_tail_returns_newest_lines_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let (eng, _) = two_versions(&dirs, vec![FakeLaunch::lines(HIP_LOG)]).await;
        assert!(eng.log_tail(5).is_empty());
        eng.load_model(launch_config(&dir, &eng)).await.unwrap();
        // The log pump runs on its own task; wait for the fake's lines to arrive.
        for _ in 0..100 {
            if eng.log_tail(usize::MAX).len() >= HIP_LOG.len() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let tail = eng.log_tail(2);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[1].text, *HIP_LOG.last().unwrap());
        assert!(eng.log_tail(0).is_empty());
        eng.unload().await.unwrap();
    }

    const HIP_NO_DEVICE: &str = "Available devices:\n  (none)\n";
    const CPU_LOG: &[&str] = &[
        "load_tensors:   CPU_Mapped model buffer size =  4168.09 MiB",
        "main: server is listening on http://127.0.0.1:18080",
    ];

    /// Engine with a developer runtime folder (`llama-server.exe` and `ggml-hip.dll`)
    /// selected through `use_local_runtime`.
    fn local_engine(
        dir: &tempfile::TempDir,
        dirs: &DataDirs,
        backend: Backend,
        runner: Arc<FakeRunner>,
        sdk: HipSdkSearch,
        script: Vec<FakeLaunch>,
    ) -> (RuntimeEngine, Arc<FakeSpawner>, PathBuf) {
        let rt = dir.path().join("runtime");
        std::fs::create_dir_all(&rt).unwrap();
        let bin = rt.join("llama-server.exe");
        std::fs::write(&bin, b"MZ fake binary").unwrap();
        std::fs::write(rt.join("ggml-hip.dll"), b"MZ ... hipblas.dll ...").unwrap();
        let spawner = Arc::new(FakeSpawner::new(script));
        let http = FakeHttp::new()
            .route("/health", vec![Step::Status(200)])
            .route("/v1/models", vec![Step::Status(200)]);
        let eng = RuntimeEngine::new(
            dirs,
            EngineDeps {
                probe: Arc::new(FakeProbe { hw: rx9060_hw() }),
                fetcher: Arc::new(FakeFetcher::default()),
                spawner: spawner.clone(),
                http: Arc::new(http),
                ports: Arc::new(FakePorts { free: true }),
                sink: Arc::new(MemorySink::default()),
                restart: RestartPolicy::default(),
                timing: Timing::default(),
                runner,
                hip_sdk: sdk,
            },
        );
        eng.use_local_runtime(&bin, backend).unwrap();
        (eng, spawner, bin)
    }

    /// SDK root whose `bin` folder holds `libhipblas.dll` only (the HIP SDK 7.x layout).
    fn sdk_with_libhipblas(dir: &tempfile::TempDir) -> HipSdkSearch {
        let hip = dir.path().join("hip-sdk");
        std::fs::create_dir_all(hip.join("bin")).unwrap();
        std::fs::write(hip.join("bin").join("libhipblas.dll"), b"x").unwrap();
        HipSdkSearch {
            roots: vec![],
            hip_path: Some(hip),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn gpu_load_without_devices_is_refused_with_the_hipblas_diagnosis() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let runner = Arc::new(FakeRunner::default().with_ok_for_all(HIP_NO_DEVICE));
        let (eng, spawner, _) = local_engine(
            &dir,
            &dirs,
            Backend::Hip,
            runner,
            sdk_with_libhipblas(&dir),
            vec![],
        );
        let cfg = LaunchConfig::builder(dir.path().join("m.gguf"), Backend::Hip)
            .build()
            .unwrap();
        let err = eng.load_model(cfg).await.unwrap_err();
        let RaptorError::Runtime(msg) = err else {
            panic!("expected a runtime error, got {err:?}");
        };
        assert!(
            msg.contains("libhipblas.dll") && msg.contains("hipblas.dll"),
            "{msg}"
        );
        assert_eq!(
            spawner.spawn_count(),
            0,
            "no process may start on a refused load"
        );
        let status = eng.status();
        assert_eq!(status.server.state, ServerState::Stopped);
        let failure = status.device_preflight.unwrap().failure.unwrap();
        assert_eq!(failure.cause, devices::FailureCause::HipblasNameMismatch);
    }

    #[tokio::test(start_paused = true)]
    async fn gpu_load_with_explicit_cpu_fallback_runs_and_reports_cpu_only() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let runner = Arc::new(FakeRunner::default().with_ok_for_all(HIP_NO_DEVICE));
        let (eng, spawner, _) = local_engine(
            &dir,
            &dirs,
            Backend::Hip,
            runner,
            HipSdkSearch::default(),
            vec![FakeLaunch::lines(CPU_LOG)],
        );
        let cfg = LaunchConfig::builder(dir.path().join("m.gguf"), Backend::Hip)
            .allow_cpu_fallback(true)
            .build()
            .unwrap();
        eng.load_model(cfg).await.unwrap();
        // The log pump runs on its own task; wait for the fake's lines to arrive.
        for _ in 0..100 {
            if eng.log_tail(usize::MAX).len() >= CPU_LOG.len() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let status = eng.status();
        assert_eq!(spawner.spawn_count(), 1);
        match status.gpu_verdict {
            Some(GpuVerdict::CpuOnly { reason }) => assert!(reason.contains("WARNING"), "{reason}"),
            other => panic!("expected CpuOnly, got {other:?}"),
        }
        assert!(status.device_preflight.unwrap().failure.is_some());
        eng.unload().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn preflight_devices_reports_without_starting_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let runner = Arc::new(FakeRunner::default().with_ok_for_all(RX9060_LIST));
        let (eng, spawner, _) = local_engine(
            &dir,
            &dirs,
            Backend::Hip,
            runner,
            HipSdkSearch::default(),
            vec![],
        );
        assert!(eng.status().device_preflight.is_none());
        let pf = eng.preflight_devices().await.unwrap();
        assert!(pf.has_usable_device());
        assert_eq!(pf.devices[0].id, "ROCm0");
        assert_eq!(spawner.spawn_count(), 0);
        assert!(eng.status().device_preflight.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn hip_sdk_bin_folder_is_prepended_to_the_child_path() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let sdk = sdk_with_libhipblas(&dir);
        let bin = sdk.hip_path.clone().unwrap().join("bin");
        let runner = Arc::new(FakeRunner::default().with_ok_for_all(RX9060_LIST));
        let (eng, spawner, _) = local_engine(
            &dir,
            &dirs,
            Backend::Hip,
            runner.clone(),
            sdk,
            vec![FakeLaunch::lines(HIP_LOG)],
        );
        let cfg = LaunchConfig::builder(dir.path().join("m.gguf"), Backend::Hip)
            .build()
            .unwrap();
        eng.load_model(cfg).await.unwrap();
        let path_of = |env: &[(String, String)]| {
            env.iter()
                .find(|(k, _)| k == "PATH")
                .map(|(_, v)| std::env::split_paths(v).next().unwrap())
                .unwrap()
        };
        assert_eq!(path_of(&runner.last_env()), bin);
        assert_eq!(path_of(&spawner.last_spec().unwrap().env), bin);
        eng.unload().await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cpu_runtime_skips_the_device_preflight() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        // The runner has no answers at all: a preflight would fail the load.
        let runner = Arc::new(FakeRunner::default());
        let (eng, spawner, _) = local_engine(
            &dir,
            &dirs,
            Backend::Cpu,
            runner,
            HipSdkSearch::default(),
            vec![FakeLaunch::lines(CPU_LOG)],
        );
        let cfg = LaunchConfig::builder(dir.path().join("m.gguf"), Backend::Cpu)
            .build()
            .unwrap();
        let status = eng.load_model(cfg).await.unwrap();
        assert_eq!(spawner.spawn_count(), 1);
        assert!(status.device_preflight.is_none());
        assert!(matches!(
            status.gpu_verdict,
            Some(GpuVerdict::CpuOnly { .. })
        ));
        eng.unload().await.unwrap();
    }

    #[test]
    fn use_local_runtime_rejects_a_missing_binary() {
        let dir = tempfile::tempdir().unwrap();
        let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
        dirs.ensure().unwrap();
        let eng = RuntimeEngine::new(
            &dirs,
            EngineDeps {
                probe: Arc::new(FakeProbe { hw: rx9060_hw() }),
                fetcher: Arc::new(FakeFetcher::default()),
                spawner: Arc::new(FakeSpawner::new(vec![])),
                http: Arc::new(FakeHttp::new()),
                ports: Arc::new(FakePorts { free: true }),
                sink: Arc::new(MemorySink::default()),
                restart: RestartPolicy::default(),
                timing: Timing::default(),
                runner: Arc::new(FakeRunner::default()),
                hip_sdk: HipSdkSearch::default(),
            },
        );
        let err = eng
            .use_local_runtime(&dir.path().join("nope.exe"), Backend::Hip)
            .unwrap_err();
        assert!(matches!(err, RaptorError::NotFound(_)));
    }
}
