//! Real-hardware gate. Ignored by default: it needs an AMD GPU with the HIP SDK,
//! a llama.cpp runtime folder, and a GGUF model. Nothing here runs in CI.
//!
//! Environment (Git Bash):
//!   RAPTOR_TEST_RUNTIME_DIR  folder containing llama-server.exe
//!   RAPTOR_TEST_MODEL        path to a .gguf model
//!
//! Runs (one at a time: both use port 18090):
//!   RAPTOR_TEST_RUNTIME_DIR=.../exp RAPTOR_TEST_MODEL=.../m.gguf \
//!     cargo test -p raptor-runtime --test gate_real_hardware -- --ignored --nocapture gate_gpu_runtime_verified
//!   RAPTOR_TEST_RUNTIME_DIR=.../bin RAPTOR_TEST_MODEL=.../m.gguf \
//!     cargo test -p raptor-runtime --test gate_real_hardware -- --ignored --nocapture gate_stock_runtime_fails_loudly
//!
//! The first test expects GPU offload to be verified from the log and a chat
//! completion to work. The second expects a loud device failure with the typed
//! diagnosis and no process started, never a silent CPU run.

use raptor_core::DataDirs;
use raptor_core::RaptorError;
use raptor_runtime::config::{recommend_with_arch, LaunchConfig, ModelArch, ModelMeta, MIB};
use raptor_runtime::devices::{self, FailureCause, HipSdkSearch};
use raptor_runtime::fit::{self, FIT_PARAMS_BINARY};
use raptor_runtime::hardware::{HardwareProbe, SystemProbe};
use raptor_runtime::http::ReqwestClient;
use raptor_runtime::install::ReqwestFetcher;
use raptor_runtime::manifest::Backend;
use raptor_runtime::process::{
    LogLine, LogSink, ManagerDeps, RestartPolicy, ServerManager, ServerState, SystemSpawner, Timing,
};
use raptor_runtime::sys::{SystemCommandRunner, TcpPortCheck};
use raptor_runtime::verify::{parse_server_log, verify_gpu, GpuVerdict};
use raptor_runtime::{EngineDeps, RuntimeEngine};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PORT: u16 = 18090;
static GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Counts log lines and bytes so the per-request log growth can be measured.
#[derive(Default)]
struct CountingSink {
    lines: AtomicUsize,
    bytes: AtomicUsize,
}

impl LogSink for CountingSink {
    fn record(&self, line: &LogLine) {
        self.lines.fetch_add(1, Ordering::SeqCst);
        self.bytes.fetch_add(line.text.len() + 1, Ordering::SeqCst);
    }
}

fn runtime_binary() -> PathBuf {
    let dir = std::env::var_os("RAPTOR_TEST_RUNTIME_DIR")
        .map(PathBuf::from)
        .expect("set RAPTOR_TEST_RUNTIME_DIR to the folder containing llama-server.exe");
    let bin = dir.join("llama-server.exe");
    assert!(bin.is_file(), "no llama-server.exe in {}", dir.display());
    bin
}

fn model_path() -> PathBuf {
    let model = std::env::var_os("RAPTOR_TEST_MODEL")
        .map(PathBuf::from)
        .expect("set RAPTOR_TEST_MODEL to a .gguf file");
    assert!(model.is_file(), "no model at {}", model.display());
    model
}

fn gpu_config(model: PathBuf) -> LaunchConfig {
    LaunchConfig::builder(model, Backend::Hip)
        .gpu_layers(999)
        .ctx_size(4096)
        .port(PORT)
        .build()
        .expect("valid launch config")
}

/// Device preflight with the same PATH the server gets. Asserts a usable ROCm device.
async fn device_env(binary: &Path) -> Vec<(String, String)> {
    let sdk = HipSdkSearch::system();
    let bins = sdk.bin_dirs();
    let current = std::env::var("PATH").ok();
    let env = devices::launch_env(&bins, current.as_deref()).expect("PATH");
    println!("HIP SDK bin folders first on PATH: {:?}", bins.first());
    let pf = devices::preflight(&SystemCommandRunner, binary, Backend::Hip, bins, &env)
        .await
        .expect("llama-server --list-devices runs");
    println!("devices: {:?}", pf.devices);
    assert!(pf.has_usable_device(), "no ROCm device: {:?}", pf.failure);
    env
}

/// Architecture of Qwen3.8-27B-UD-IQ3_S as read from its GGUF header in the fixture
/// log (block_count 65, full_attention_interval 4, 4 KV heads of 256, 66 offloadable
/// layers, recurrent state 598.5 MiB at 4 slots). The gate supplies these by hand.
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

/// The gate is about the real GPU path, so the verified run must not reach the
/// server with any CPU-only shortcut: the device check runs first and must pass.
/// Starts llama-server with `cfg`, checks the log verdict, runs one chat, stops it.
async fn run_gpu_gate(
    cfg: LaunchConfig,
    binary: &Path,
    env: Vec<(String, String)>,
    require_full: bool,
) {
    // Real server through ServerManager and SystemSpawner.
    let sink = Arc::new(CountingSink::default());
    let manager = ServerManager::new(
        ManagerDeps {
            spawner: Arc::new(SystemSpawner),
            http: Arc::new(ReqwestClient::new().unwrap()),
            ports: Arc::new(TcpPortCheck),
            sink: sink.clone(),
        },
        RestartPolicy::default(),
        Timing {
            startup_timeout: Duration::from_secs(600),
            ..Timing::default()
        },
    );
    let started = Instant::now();
    let status = manager
        .start_with_env(cfg, binary, env)
        .await
        .expect("server reaches Ready");
    println!("ready after {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(status.state, ServerState::Ready);
    let pid = status.pid.expect("pid while ready");

    // Verdict from the server's own log.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let verdict = verify_gpu(Backend::Hip, &parse_server_log(&manager.log_text()));
    println!("verdict: {verdict:?}");
    match &verdict {
        GpuVerdict::Verified {
            offloaded_layers: Some(done),
            total_layers: Some(total),
            ..
        } if require_full => assert_eq!(done, total, "all layers must be offloaded"),
        GpuVerdict::Verified { .. } => {
            println!("partial offload is reported as verified, not as full");
        }
        other => panic!("GPU offload not verified: {other:?}"),
    }

    // One short chat completion, with log growth and throughput measured.
    let before = (
        sink.lines.load(Ordering::SeqCst),
        sink.bytes.load(Ordering::SeqCst),
    );
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(600))
        .build()
        .unwrap();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Reply with one word: ok"}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": false
    });
    let t = Instant::now();
    let text = client
        .post(format!("http://127.0.0.1:{PORT}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("chat request sent")
        .text()
        .await
        .expect("chat response read");
    let wall = t.elapsed().as_secs_f64();
    let json: serde_json::Value = serde_json::from_str(&text).expect("JSON response");
    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let server_tps = json["timings"]["predicted_per_second"].as_f64();
    tokio::time::sleep(Duration::from_secs(1)).await;
    let after = (
        sink.lines.load(Ordering::SeqCst),
        sink.bytes.load(Ordering::SeqCst),
    );
    println!(
        "chat: {completion_tokens} tokens in {wall:.2} s = {:.1} tok/s wall; server predicted_per_second = {server_tps:?}",
        completion_tokens as f64 / wall
    );
    println!(
        "log growth for this request: {} lines, {} bytes",
        after.0 - before.0,
        after.1 - before.1
    );
    assert!(completion_tokens > 0, "no tokens generated: {text}");

    // Stop, and check the process is really gone.
    manager.stop().await.expect("stop");
    assert_eq!(manager.status().state, ServerState::Stopped);
    assert!(!process_alive(pid), "llama-server pid {pid} still running");
    println!("stopped; pid {pid} gone");
}

/// Verified GPU run with a fixed launch config (ctx 4096, every layer requested).
#[tokio::test]
#[ignore = "needs a GPU, a llama.cpp runtime folder and a model; see the file header"]
async fn gate_gpu_runtime_verified() {
    let _guard = GATE.lock().await;
    let binary = runtime_binary();
    let env = device_env(&binary).await;
    run_gpu_gate(gpu_config(model_path()), &binary, env, true).await;
}

/// The recommended config for ctx 8192. Hardware comes from the probe, the
/// architecture from the fixture above. Step 1 estimates from the GGUF figures, step
/// 2 asks llama-fit-params for llama.cpp's projection with all layers on the GPU, and
/// step 3 recommends again with that projection. The recommendation must be full
/// offload (66 of 66) and must load and verify on the GPU.
#[tokio::test]
#[ignore = "needs a GPU, a llama.cpp runtime folder and a model; see the file header"]
async fn gate_recommended_config_full_offload() {
    let _guard = GATE.lock().await;
    let binary = runtime_binary();
    let model = model_path();
    let env = device_env(&binary).await;
    let hw = SystemProbe::new(Arc::new(SystemCommandRunner))
        .probe()
        .await
        .expect("hardware probe");
    let (meta, arch) = (qwen35_meta(), qwen35_arch());

    let first = recommend_with_arch(&hw, &meta, &arch, 8192, Backend::Hip, None).expect("estimate");
    let probe_cfg = first.config.with_model(model.clone()).expect("model path");
    let fit_binary = binary
        .parent()
        .expect("runtime folder")
        .join(FIT_PARAMS_BINARY);
    let started = Instant::now();
    let projection = fit::probe(&SystemCommandRunner, &fit_binary, &probe_cfg, &env)
        .await
        .expect("llama-fit-params projection");
    println!(
        "llama.cpp projection in {:.2} s: {projection:?} (total {} MiB)",
        started.elapsed().as_secs_f64(),
        projection.total_mib()
    );

    let rec = recommend_with_arch(&hw, &meta, &arch, 8192, Backend::Hip, Some(&projection))
        .expect("calibrated recommendation");
    for line in &rec.rationale {
        println!("  rationale: {line}");
    }
    println!(
        "recommended: -ngl {} -c {} vram estimate {} MiB of budget {} MiB",
        rec.config.gpu_layers,
        rec.config.ctx_size,
        rec.estimate.vram_bytes / MIB,
        rec.estimate.vram_budget / MIB
    );
    assert_eq!(rec.config.gpu_layers, 66, "full offload of 66 layers");
    assert_eq!(rec.config.ctx_size, 8192);
    // The recommendation keeps the shell's default port; the gate uses its own.
    let mut cfg = rec.config.with_model(model).expect("model path");
    cfg.port = PORT;
    run_gpu_gate(cfg, &binary, env, true).await;
}

/// Baseline for comparison: the config the shell produced before this change
/// (65 of 66 layers at ctx 8192), run on the same machine in the same session.
#[tokio::test]
#[ignore = "needs a GPU, a llama.cpp runtime folder and a model; see the file header"]
async fn gate_baseline_65_of_66_layers() {
    let _guard = GATE.lock().await;
    let binary = runtime_binary();
    let env = device_env(&binary).await;
    let cfg = LaunchConfig::builder(model_path(), Backend::Hip)
        .gpu_layers(65)
        .ctx_size(8192)
        .port(PORT)
        .build()
        .expect("valid launch config");
    run_gpu_gate(cfg, &binary, env, false).await;
}

/// The stock zip's ggml-hip.dll needs hipblas.dll. HIP SDK 7.1 ships only
/// libhipblas.dll, so with 7.1 as the only SDK the GPU load must fail loudly and
/// start no process. (SDK 7.2 ships hipblas.dll; with 7.2 installed the stock
/// zip verifies on the GPU, which the search order handles.)
///
/// The SDK root is RAPTOR_TEST_HIP_SDK, default `C:\Program Files\AMD\ROCm\7.1`.
#[tokio::test]
#[ignore = "needs a GPU, a llama.cpp runtime folder and a model; see the file header"]
async fn gate_stock_runtime_fails_loudly() {
    let _guard = GATE.lock().await;
    let binary = runtime_binary();
    let model = model_path();
    let sdk_root = std::env::var_os("RAPTOR_TEST_HIP_SDK")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\AMD\ROCm\7.1"));
    let dir = tempfile::tempdir().unwrap();
    let dirs = DataDirs::new(dir.path().join("RaptorLLM"));
    dirs.ensure().unwrap();
    let engine = RuntimeEngine::new(
        &dirs,
        EngineDeps {
            probe: Arc::new(SystemProbe::new(Arc::new(SystemCommandRunner))),
            fetcher: Arc::new(ReqwestFetcher::new().unwrap()),
            spawner: Arc::new(SystemSpawner),
            http: Arc::new(ReqwestClient::new().unwrap()),
            ports: Arc::new(TcpPortCheck),
            sink: Arc::new(CountingSink::default()),
            restart: RestartPolicy::default(),
            timing: Timing::default(),
            runner: Arc::new(SystemCommandRunner),
            hip_sdk: HipSdkSearch {
                roots: vec![],
                hip_path: Some(sdk_root.clone()),
            },
        },
    );
    println!("HIP SDK used for the search: {}", sdk_root.display());
    engine
        .use_local_runtime(&binary, Backend::Hip)
        .expect("runtime folder accepted");

    let result = engine.load_model(gpu_config(model)).await;
    // Never leave a server running, even when the assertions below fail.
    if result.is_ok() {
        let _ = engine.unload().await;
    }
    let err = result.expect_err("a stock runtime without hipblas.dll must not load");
    let RaptorError::Runtime(msg) = err else {
        panic!("expected a runtime error, got {err:?}");
    };
    println!("load error: {msg}");

    let status = engine.status();
    assert_eq!(
        status.server.state,
        ServerState::Stopped,
        "no process may run"
    );
    assert!(
        status.gpu_verdict.is_none(),
        "no verdict for a refused load"
    );
    let failure = status
        .device_preflight
        .and_then(|p| p.failure)
        .expect("typed BackendLoadFailure");
    println!("cause: {:?}", failure.cause);
    assert_eq!(failure.cause, FailureCause::HipblasNameMismatch);
    assert!(msg.contains("libhipblas.dll") && msg.contains("hipblas.dll"));
}

/// True when a process with this pid is still running (Windows: tasklist).
fn process_alive(pid: u32) -> bool {
    if cfg!(windows) {
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .expect("tasklist runs");
        String::from_utf8_lossy(&out.stdout).contains(&format!("\"{pid}\""))
    } else {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}
