//! End-to-end gate for the shell path on real hardware. It drives the same functions the Tauri
//! commands call: model import, `recommend_config`, `load_model` (device preflight, launch,
//! GPU verdict, benchmark), `perf_run_benchmark`, a streamed chat through `begin_chat`, the
//! performance record, and `unload_model`. Afterwards no llama-server from the runtime
//! directory may still be running.
//!
//! Ignored by default: it needs a GPU, a llama.cpp build and a large local model.
//!
//! Environment:
//!   RAPTOR_TEST_TUNE_ONLY     when set, run only the full-window test and require a 1.50x result.
//!   RAPTOR_TEST_CHAT_ONLY     when set, skip the full-window tuning sweep after checking chat.
//!   RAPTOR_TEST_RUNTIME_DIR  directory that holds `llama-server.exe` (an extracted llama.cpp build).
//!   RAPTOR_TEST_MODEL        a `.gguf` file. It is referenced in place: never copied, moved or changed.
//!   RAPTOR_TEST_GPU_NAME, RAPTOR_TEST_GPU_VRAM_MIB and RAPTOR_TEST_GPU_FREE_MIB (optional together)
//!                             seed real device facts when Windows WMI/CIM is unavailable. Source
//!                             these from `llama-server --list-devices`; device/load/offload checks
//!                             still run against the real runtime. RAM defaults to 64/52 GiB and can
//!                             be overridden with RAPTOR_TEST_RAM_GIB / RAPTOR_TEST_RAM_FREE_GIB.
//!
//! Run: `cargo test -p raptor-llm --test gate_real_app -- --ignored --nocapture`

#![cfg(windows)]

use raptor_core::{ChatMessage, DataDirs, Result as CoreResult, Role};
use raptor_gateway::{Completion, Endpoint, Upstream};
use raptor_lib::commands::chat::{begin_chat, ChatEvent, ChatOptions};
use raptor_lib::settings::{Settings, SettingsStore};
use raptor_lib::state::{model_meta, AppState, ShellDeps};
use raptor_runtime::hardware::{CpuInfo, GpuInfo, GpuVendor, HardwareInfo, HardwareProbe, Os};
use raptor_runtime::manifest::Backend;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;

struct FixedProbe(HardwareInfo);

#[async_trait::async_trait]
impl HardwareProbe for FixedProbe {
    async fn probe(&self) -> CoreResult<HardwareInfo> {
        Ok(self.0.clone())
    }
}

fn manual_hardware() -> Option<HardwareInfo> {
    let (name, total_mib, free_mib) = (
        std::env::var("RAPTOR_TEST_GPU_NAME").ok()?,
        std::env::var("RAPTOR_TEST_GPU_VRAM_MIB")
            .ok()?
            .parse::<u64>()
            .ok()?,
        std::env::var("RAPTOR_TEST_GPU_FREE_MIB")
            .ok()?
            .parse::<u64>()
            .ok()?,
    );
    let ram_gib = std::env::var("RAPTOR_TEST_RAM_GIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(64);
    let ram_free_gib = std::env::var("RAPTOR_TEST_RAM_FREE_GIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(ram_gib.saturating_sub(12));
    Some(HardwareInfo {
        os: Os::Windows,
        cpu: CpuInfo {
            brand: "Windows host (manual real-hardware gate facts)".into(),
            cores: 8,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
            features: Vec::new(),
        },
        ram_total: ram_gib * 1024 * 1024 * 1024,
        ram_avail: ram_free_gib * 1024 * 1024 * 1024,
        gpus: vec![GpuInfo {
            vendor: GpuVendor::Amd,
            name,
            vram_total: total_mib * MIB,
            vram_free: Some(free_mib * MIB),
            arch: std::env::var("RAPTOR_TEST_GPU_ARCH").ok(),
            driver: std::env::var("RAPTOR_TEST_GPU_DRIVER").ok(),
        }],
    })
}

fn required_path(var: &str) -> PathBuf {
    match std::env::var(var) {
        Ok(v) if !v.trim().is_empty() => PathBuf::from(v),
        _ => panic!("set {var} to run the real-hardware gate (see the header of this file)"),
    }
}

/// Every llama-server.exe whose executable lives under `dir`, found by WMI. Empty when none.
fn llama_servers_under(dir: &Path) -> Vec<String> {
    let script = "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | ForEach-Object { $_.ExecutablePath }";
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .expect("powershell runs");
    let want = dir.display().to_string().to_lowercase();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|p| !p.is_empty() && p.to_lowercase().starts_with(&want))
        .map(str::to_string)
        .collect()
}

/// The shell path, step by step. Returns the first failure as text, so the caller can always
/// unload the server before reporting it.
async fn exercise(state: &AppState, model: &Path, bin: &Path) -> Result<(), String> {
    let t = Instant::now();
    let entry = state
        .models
        .library()
        .import_reference(model)
        .map_err(|e| format!("import by path failed: {e}"))?;
    println!(
        "import: id={} size={} MiB sha256={}.. ({:?})",
        entry.id,
        entry.size_bytes / MIB,
        &entry.sha256[..entry.sha256.len().min(16)],
        t.elapsed()
    );
    if std::env::var_os("RAPTOR_TEST_TUNE_ONLY").is_some() {
        return qualify(state, &entry.id).await;
    }
    println!(
        "runtime: {} backend={:?} binary={}",
        state
            .runtime
            .status()
            .runtime
            .map_or("none".into(), |r| format!("{} (local build)", r.id)),
        Backend::Hip,
        bin.display()
    );

    let hw = state
        .hardware_for_use()
        .await
        .map_err(|e| format!("hardware: {e}"))?;
    for g in &hw.gpus {
        println!(
            "gpu: {} arch={:?} vram={} MiB driver={:?}",
            g.name,
            g.arch,
            g.vram_total / MIB,
            g.driver
        );
    }

    model_meta(&entry).map_err(|e| format!("model header: {e}"))?;
    let rec = state
        .recommend_for_entry(&entry, model, 8192)
        .await
        .map_err(|e| format!("recommend_for_entry: {e}"))?;
    println!(
        "recommendation: gpu_layers={} ctx={} kv={:?} flash_attn={} est_vram={} MiB",
        rec.config.gpu_layers,
        rec.config.ctx_size,
        rec.config.kv_cache,
        rec.config.flash_attn,
        rec.estimate.vram_bytes / MIB
    );
    for line in &rec.rationale {
        println!("  rationale: {line}");
    }

    let started = Instant::now();
    let loaded = state.load_model(&entry.id, None, None).await.map_err(|f| {
        format!(
            "load_model: {} (backend_failure: {:?})",
            f.error, f.backend_failure
        )
    })?;
    let ready = started.elapsed();
    let window = state.upstream.context_window().await.unwrap_or(0);
    println!("runtime reports {window} context tokens per request");
    if window < 65_536 {
        return Err(format!(
            "the runtime reports {window} tokens per request, below 65,536"
        ));
    }
    println!(
        "load: ready and benchmarked in {ready:?}, config_source={:?}, cpu_fallback={}",
        loaded.config_source, loaded.cpu_fallback
    );
    if let Some(pf) = &loaded.status.device_preflight {
        let names: Vec<String> = pf.devices.iter().map(|d| d.name.clone()).collect();
        println!(
            "preflight devices: {names:?} failure={:?}",
            pf.failure.is_some()
        );
    }
    let verdict = loaded.status.gpu_verdict.clone();
    println!("gpu verdict: {verdict:?}");
    println!(
        "benchmark: stable={} recorded={} reasons={:?}",
        loaded.benchmark.stable, loaded.benchmark.recorded, loaded.benchmark.reasons
    );
    if let Some(e) = &loaded.benchmark.entry {
        println!(
            "benchmark record: ttft={:.1} ms prompt={:.1} tok/s gen={:.2} tok/s samples={} stable={}",
            e.record.ttft_ms, e.record.prompt_tps, e.record.gen_tps, e.record.samples, e.record.stable
        );
    }
    if loaded.cpu_fallback {
        return Err("the load fell back to CPU; this gate needs the GPU".into());
    }
    if !verdict.as_ref().is_some_and(|v| v.is_verified()) {
        return Err(format!("GPU offload is not verified: {verdict:?}"));
    }

    let rerun = state
        .perf_run_benchmark()
        .await
        .map_err(|e| format!("perf_run_benchmark: {e}"))?;
    println!(
        "re-run benchmark: stable={} recorded={} reasons={:?}",
        rerun.stable, rerun.recorded, rerun.reasons
    );
    if let Some(e) = &rerun.entry {
        println!(
            "re-run record: ttft={:.1} ms prompt={:.1} tok/s gen={:.2} tok/s samples={} stable={}",
            e.record.ttft_ms,
            e.record.prompt_tps,
            e.record.gen_tps,
            e.record.samples,
            e.record.stable
        );
    }

    let (tx, rx) = mpsc::channel::<ChatEvent>();
    let sent = Instant::now();
    begin_chat(
        state,
        "gate-chat-1".into(),
        vec![ChatMessage::new(
            Role::User,
            "Name three colours, one per line.",
        )],
        ChatOptions {
            max_tokens: Some(64),
            ..Default::default()
        },
        move |event| {
            let _ = tx.send(event);
        },
    )
    .await
    .map_err(|e| format!("chat_start: {e}"))?;
    let mut reply = String::new();
    let mut first_chunk: Option<Duration> = None;
    let mut chunks = 0usize;
    let cancelled = loop {
        match rx.recv_timeout(Duration::from_secs(300)) {
            Ok(ChatEvent::Chunk { text }) => {
                first_chunk.get_or_insert_with(|| sent.elapsed());
                chunks += 1;
                reply.push_str(&text);
            }
            Ok(ChatEvent::Reasoning { .. }) => {}
            Ok(ChatEvent::Done { cancelled, stats }) => {
                println!("chat stats: {stats:?}");
                break cancelled;
            }
            Ok(ChatEvent::Error { error }) => return Err(format!("chat error event: {error}")),
            Err(e) => return Err(format!("chat stream stalled: {e}")),
        }
    };
    let total = sent.elapsed();
    println!(
        "chat: {chunks} chunks, first chunk after {first_chunk:?}, done after {total:?}, cancelled={cancelled}"
    );
    println!("chat reply: {reply:?}");
    if cancelled {
        return Err("the chat stream was reported as cancelled".into());
    }
    if reply.trim().is_empty() {
        return Err("the chat stream produced no text".into());
    }

    // Cross-check of the benchmark: the same generation, not streamed, so llama-server's own
    // `timings` and `usage` come back in one JSON object. The server log lines with the
    // "tokens per second" figures are printed too.
    let model_name = state
        .upstream
        .models()
        .await
        .map_err(|e| format!("models: {e}"))?
        .loaded
        .unwrap_or_default();
    let direct = state
        .upstream
        .call(
            Endpoint::ChatCompletions,
            serde_json::json!({
                "model": model_name,
                "messages": [
                    { "role": "user", "content": "obsolete ".repeat(window as usize + 512) },
                    { "role": "assistant", "content": "Earlier draft." },
                    { "role": "user", "content": "Name three colours, one per line." }
                ],
                "chat_template_kwargs": { "enable_thinking": false },
                "stream": false,
                "max_tokens": 64,
                "temperature": 0,
                "seed": 42,
            }),
        )
        .await
        .map_err(|e| format!("direct completion: {e}"))?;
    if let Completion::Json(v) = direct {
        println!("server timings: {}", v["timings"]);
        println!("server usage: {}", v["usage"]);
        println!("server chat context: {}", v["raptor"]);
        if v["raptor"]["omitted_history_turns"] != 1
            || v["usage"]["prompt_tokens"].as_u64().is_none_or(|n| n > 256)
            || v["usage"]["completion_tokens"].as_u64().unwrap_or(0) == 0
        {
            return Err("the over-budget chat request did not compact and generate".into());
        }
    } else {
        return Err("the direct completion was not a JSON reply".into());
    }
    for line in state.server_log_lines(400) {
        if line.contains("tokens per second") || line.contains("eval time") {
            println!("server log: {}", line.chars().take(300).collect::<String>());
        }
    }

    let history = state
        .perf_history(Some(&entry.id))
        .map_err(|e| format!("perf_history: {e}"))?;
    println!("performance records for this model: {}", history.len());
    if history.is_empty() {
        return Err("no performance record was saved".into());
    }
    if std::env::var_os("RAPTOR_TEST_CHAT_ONLY").is_some() {
        return Ok(());
    }

    // Run the app's own short optimizer and its matched streamed comparison after unloading.
    state
        .unload_model()
        .await
        .map_err(|e| format!("unload before tune: {e}"))?;
    qualify(state, &entry.id).await
}

async fn qualify(state: &AppState, model_id: &str) -> Result<(), String> {
    let tuned = state
        .tune_model(model_id, false)
        .await
        .map_err(|e| format!("Raptor speed tune: {e}"))?;
    println!(
        "Full-window tuning evidence: {}",
        serde_json::to_string_pretty(&tuned).unwrap()
    );
    if tuned.saved {
        let loaded = state
            .load_model(model_id, None, None)
            .await
            .map_err(|f| f.error.to_string())?;
        if !loaded.tuning_applied {
            return Err(
                "The verified settings were not reproduced by the normal model loader".into(),
            );
        }
    } else if std::env::var_os("RAPTOR_TEST_TUNE_ONLY").is_some() {
        return Err("No candidate proved a 50% boost at the required context".into());
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "real hardware: set RAPTOR_TEST_RUNTIME_DIR and RAPTOR_TEST_MODEL"]
async fn shell_path_loads_benchmarks_chats_and_unloads_on_real_hardware() {
    let runtime_dir = required_path("RAPTOR_TEST_RUNTIME_DIR");
    let model = required_path("RAPTOR_TEST_MODEL");
    let bin = runtime_dir.join("llama-server.exe");
    assert!(
        bin.is_file(),
        "no llama-server.exe in {}",
        runtime_dir.display()
    );
    assert!(model.is_file(), "no model file at {}", model.display());

    let tmp = tempfile::tempdir().expect("temp data dir");
    let dirs = DataDirs::new(tmp.path().join("RaptorLLM"));
    dirs.ensure().expect("data dirs");
    let settings = Arc::new(SettingsStore::with_settings(
        dirs.config().join("settings.json"),
        Settings::default(),
    ));
    let mut deps = ShellDeps::production(&dirs).expect("production deps");
    if let Some(hw) = manual_hardware() {
        // The same physical device was confirmed by `llama-server --list-devices`.
        // Only replace WMI hardware discovery; runtime/device verification stays real.
        deps.engine.probe = Arc::new(FixedProbe(hw));
    }
    let state = AppState::from_parts(dirs.clone(), settings, vec![], deps).expect("app state");
    state
        .runtime
        .use_local_runtime(&bin, Backend::Hip)
        .expect("use the local llama.cpp build");

    let outcome = exercise(&state, &model, &bin).await;

    // Always stop the server, even when a step failed, then check that none is left behind.
    let unloaded = state.unload_model().await;
    let leftover = llama_servers_under(&runtime_dir);
    println!(
        "unload: {} | server state now {:?} | llama-server processes under the runtime dir: {}",
        match &unloaded {
            Ok(_) => "ok".to_string(),
            Err(e) => format!("error: {e}"),
        },
        state.runtime.status().server.state,
        leftover.len()
    );
    assert!(
        leftover.is_empty(),
        "llama-server still running: {leftover:?}"
    );
    unloaded.expect("unload");
    if let Err(e) = outcome {
        panic!("real-hardware gate failed: {e}");
    }
}
