//! Performance memory wiring. After each successful model load the shell runs a short,
//! deterministic benchmark through the same upstream chat uses, then records the result with
//! the exact launch configuration in [`PerfStore`]. Records are keyed by
//! (machine fingerprint, model sha256, runtime id).
//!
//! What is measured:
//! - time to first token (TTFT): request sent until the first generated text arrives.
//! - prompt tokens per second: prompt tokens divided by TTFT. This includes prefill and the
//!   first token, so it is a conservative figure.
//! - generation tokens per second: tokens after the first, divided by the time after TTFT.
//! - memory: system RAM in use and VRAM in use, read from the hardware probe after the run.
//!
//! Prompt tokens come from the model's tokenizer (exact while the server is ready). Generation
//! tokens come from the server's usage report when it sends one, else from the number of
//! streamed text deltas.

use futures_util::StreamExt;
use raptor_core::DataDirs;
use raptor_core::{RaptorError, Result, TokenCounter};
use raptor_gateway::{Completion, Endpoint, Upstream};
use raptor_runtime::config::LaunchConfig;
use raptor_runtime::hardware::HardwareInfo;
use raptor_runtime::hash::sha256_hex;
use raptor_runtime::manifest::Backend;
use raptor_runtime::perfmem::{
    compare, machine_fingerprint, BenchRecord, Comparison, ConfigSnapshot, PerfStore, RecordKey,
    RecordOutcome,
};
use raptor_runtime::verify::GpuVerdict;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// File name of the store, inside the config directory.
pub const PERF_STORE_FILE: &str = "perf-memory.json";
/// Generation length of the benchmark. Short enough to keep loads quick.
pub const BENCH_MAX_TOKENS: u32 = 64;
/// Upper bound on one benchmark run. A run that exceeds it is recorded as unstable.
pub const BENCH_TIMEOUT: Duration = Duration::from_secs(120);
const BENCH_PROMPT_REPEATS: usize = 24;
const BENCH_SEED: u64 = 42;

/// Where the performance store lives.
pub fn store_path(dirs: &DataDirs) -> PathBuf {
    dirs.config().join(PERF_STORE_FILE)
}

/// Opens the store. A corrupt or unreadable file is kept as an error text: performance memory
/// then reports the problem and never overwrites the file.
pub fn open_store(dirs: &DataDirs) -> std::result::Result<PerfStore, String> {
    PerfStore::open(store_path(dirs)).map_err(|e| e.to_string())
}

/// The fixed benchmark prompt. It is identical on every machine and run, so results compare.
pub fn bench_prompt() -> String {
    let mut prompt = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
        .repeat(BENCH_PROMPT_REPEATS);
    prompt.push_str("Reply with the numbers from one to twenty, separated by commas.");
    prompt
}

/// What one streamed SSE payload contributes to the measurement.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Payload {
    /// Non-empty text delta.
    pub content: Option<String>,
    /// Non-empty reasoning delta (`reasoning_content`) of a thinking model. These tokens are
    /// generated and counted by the server, so they start the timing like any other token.
    pub reasoning: Option<String>,
    /// Generated-token count from a usage report, when the server sends one.
    pub completion_tokens: Option<u64>,
    /// The `[DONE]` terminator.
    pub done: bool,
}

/// Reads one SSE payload. Malformed payloads contribute nothing rather than failing the run.
pub fn parse_payload(data: &str) -> Payload {
    let trimmed = data.trim();
    if trimmed == "[DONE]" {
        return Payload {
            done: true,
            ..Payload::default()
        };
    }
    let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
        return Payload::default();
    };
    let text = |field: &str| {
        v["choices"][0]["delta"][field]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    Payload {
        content: text("content"),
        reasoning: text("reasoning_content"),
        completion_tokens: v["usage"]["completion_tokens"].as_u64(),
        done: false,
    }
}

/// Timings of one streamed generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    pub ttft: Duration,
    pub total: Duration,
    pub gen_tokens: u64,
    /// True when `gen_tokens` came from the server's usage report.
    pub usage_reported: bool,
}

/// Streams one fixed generation through `upstream` and times it. Fails when the stream
/// errors, is not a stream, or produces no text.
pub async fn measure(upstream: &dyn Upstream, model: &str, prompt: &str) -> Result<Timing> {
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "stream": true,
        "stream_options": { "include_usage": true },
        "temperature": 0,
        "seed": BENCH_SEED,
        "max_tokens": BENCH_MAX_TOKENS,
    });
    let start = Instant::now();
    let mut stream = match upstream.call(Endpoint::ChatCompletions, body).await? {
        Completion::Stream(s) => s,
        Completion::Json(_) => {
            return Err(RaptorError::Runtime(
                "benchmark expected a streamed reply".into(),
            ))
        }
    };
    let mut first: Option<Instant> = None;
    let mut deltas = 0u64;
    let mut usage = None;
    while let Some(item) = stream.next().await {
        let p = parse_payload(&item?);
        // The first generated token of any kind starts the clock. The server counts reasoning
        // tokens in its usage, so the window must start with them, or the rate is inflated.
        if p.content.is_some() || p.reasoning.is_some() {
            first.get_or_insert_with(Instant::now);
            deltas += 1;
        }
        if p.completion_tokens.is_some() {
            usage = p.completion_tokens;
        }
        if p.done {
            break;
        }
    }
    let first = first.ok_or_else(|| RaptorError::Runtime("benchmark produced no text".into()))?;
    Ok(Timing {
        ttft: first.duration_since(start),
        total: start.elapsed(),
        gen_tokens: usage.unwrap_or(deltas),
        usage_reported: usage.is_some(),
    })
}

/// One benchmark run, in the units the store keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchSample {
    pub ttft_ms: f64,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
}

/// Converts raw timings to the stored units. Division uses a floor of one microsecond, so a
/// very fast run stays finite.
pub fn sample_from(t: &Timing, prompt_tokens: u64) -> BenchSample {
    let ttft = t.ttft.as_secs_f64().max(1e-6);
    let decode = t.total.saturating_sub(t.ttft).as_secs_f64().max(1e-6);
    let decoded = t.gen_tokens.saturating_sub(1);
    BenchSample {
        ttft_ms: ttft * 1000.0,
        prompt_tps: prompt_tokens as f64 / ttft,
        gen_tps: decoded as f64 / decode,
        prompt_tokens,
        gen_tokens: t.gen_tokens,
    }
}

/// Runs the fixed benchmark against the loaded model. Prompt tokens are counted off the async
/// runtime, because the tokenizer call is blocking. Returns the sample and whether the prompt
/// count came from the model's tokenizer.
pub async fn sample(
    upstream: &dyn Upstream,
    counter: Arc<dyn TokenCounter>,
    model: &str,
) -> Result<(BenchSample, bool)> {
    let prompt = bench_prompt();
    let counted = prompt.clone();
    let (prompt_tokens, exact) = tokio::task::spawn_blocking(move || {
        let n = counter.count(&counted);
        (n as u64, counter.is_exact())
    })
    .await
    .map_err(|e| RaptorError::Internal(format!("token count task failed: {e}")))?;
    let timing = tokio::time::timeout(BENCH_TIMEOUT, measure(upstream, model, &prompt))
        .await
        .map_err(|_| {
            RaptorError::Runtime(format!(
                "benchmark timed out after {} s",
                BENCH_TIMEOUT.as_secs()
            ))
        })??;
    Ok((sample_from(&timing, prompt_tokens), exact))
}

/// Why a run is not stable. An empty list means stable. The rules are the ones the store
/// relies on: GPU runs need a verified GPU offload, and the server must still be Ready.
pub fn assess(
    sample: &BenchSample,
    backend: Backend,
    still_ready: bool,
    verdict: Option<&GpuVerdict>,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !still_ready {
        reasons.push("the server left the Ready state during the benchmark".into());
    }
    if backend.is_gpu() {
        match verdict {
            Some(GpuVerdict::Verified { .. }) => {}
            Some(GpuVerdict::CpuOnly { reason }) => {
                reasons.push(format!("the model ran on the CPU: {reason}"));
            }
            Some(GpuVerdict::Unverified { reason }) => {
                reasons.push(format!("GPU offload is not verified: {reason}"));
            }
            None => reasons.push("GPU offload is not verified: no server log yet".into()),
        }
    }
    if sample.gen_tokens < 2 {
        reasons.push("too few generated tokens to measure speed".into());
    }
    let finite = [sample.ttft_ms, sample.prompt_tps, sample.gen_tps]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0);
    if !finite {
        reasons.push("the measured numbers are not finite".into());
    }
    reasons
}

/// System RAM in use and VRAM in use (bytes), from one hardware probe. VRAM is zero when the
/// platform does not report free VRAM.
pub fn memory_used(hw: &HardwareInfo) -> (u64, u64) {
    let ram = hw.ram_total.saturating_sub(hw.ram_avail);
    let vram = hw
        .gpus
        .iter()
        .filter_map(|g| g.vram_free.map(|free| g.vram_total.saturating_sub(free)))
        .sum();
    (vram, ram)
}

/// The store key for this machine, model and runtime.
pub fn key_for(hw: &HardwareInfo, model_sha256: &str, runtime_id: &str) -> RecordKey {
    RecordKey {
        machine: machine_fingerprint(hw),
        model_hash: model_sha256.to_string(),
        runtime_id: runtime_id.to_string(),
    }
}

/// Builds the record for one run. A failed run has no sample and zero numbers, and is stable
/// only if it is stable, which it never is.
#[allow(clippy::too_many_arguments)]
pub fn build_record(
    hw: &HardwareInfo,
    model_sha256: &str,
    runtime_id: &str,
    runtime_version: &str,
    config: ConfigSnapshot,
    sample: Option<&BenchSample>,
    memory: (u64, u64),
    stable: bool,
    measured_at: u64,
) -> BenchRecord {
    let (ttft_ms, prompt_tps, gen_tps, prompt_tokens, gen_tokens) = sample
        .map(|s| {
            (
                s.ttft_ms,
                s.prompt_tps,
                s.gen_tps,
                s.prompt_tokens,
                s.gen_tokens,
            )
        })
        .unwrap_or((0.0, 0.0, 0.0, 0, 0));
    BenchRecord {
        machine: machine_fingerprint(hw),
        model_hash: model_sha256.to_string(),
        runtime_id: runtime_id.to_string(),
        runtime_version: runtime_version.to_string(),
        gpu_name: hw.gpu_of(None).map(|g| g.name.clone()),
        gpu_driver: hw.gpu_of(None).and_then(|g| g.driver.clone()),
        quant_label: None,
        config,
        ttft_ms,
        prompt_tps,
        gen_tps,
        prompt_tokens,
        gen_tokens,
        vram_bytes: memory.0,
        ram_bytes: memory.1,
        stable,
        samples: 1,
        measured_at,
    }
}

/// Unix seconds now, or zero if the clock is before 1970.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A stored record as the UI sees it. `id` is stable for the same record and is what
/// `perf_compare` takes.
#[derive(Debug, Clone, Serialize)]
pub struct PerfEntry {
    pub id: String,
    pub record: BenchRecord,
}

/// The id of a record: a short hash over its key, runtime version and configuration.
pub fn entry_id(r: &BenchRecord) -> String {
    let config = serde_json::to_string(&r.config).unwrap_or_default();
    let canon = format!(
        "{}|{}|{}|{}|{config}",
        r.machine, r.model_hash, r.runtime_id, r.runtime_version
    );
    sha256_hex(canon.as_bytes())[..16].to_string()
}

/// Wraps a record with its id.
pub fn entry_view(r: &BenchRecord) -> PerfEntry {
    PerfEntry {
        id: entry_id(r),
        record: r.clone(),
    }
}

/// Result of the benchmark that follows a load, or of `perf_run_benchmark`.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkOutcome {
    /// True when the run met every stability rule.
    pub stable: bool,
    /// What the store did with this run: `inserted`, `merged`, `kept_stable` or `not_saved`.
    pub recorded: String,
    /// This run, whether or not the store kept it. Absent only when the hardware is unknown,
    /// because the record cannot be keyed to the machine then.
    pub entry: Option<PerfEntry>,
    /// Why the run is unstable, or the error that stopped it. Empty when stable.
    pub reasons: Vec<String>,
    pub gpu_verdict: Option<GpuVerdict>,
    /// Non-fatal notes, such as a store that could not be saved.
    pub notes: Vec<String>,
}

/// Puts one run into the store and saves it. A store that is unavailable reports
/// `not_saved` and says why. An unstable run never replaces a stable one (the store enforces it).
pub fn store_record(
    slot: &mut std::result::Result<PerfStore, String>,
    record: BenchRecord,
) -> (String, Option<String>) {
    let store = match slot {
        Ok(s) => s,
        Err(msg) => {
            return (
                "not_saved".into(),
                Some(format!("performance memory is unavailable: {msg}")),
            )
        }
    };
    let outcome = match store.record(record) {
        Ok(o) => o,
        Err(e) => return ("not_saved".into(), Some(e.to_string())),
    };
    if let Err(e) = store.save() {
        return (
            "not_saved".into(),
            Some(format!("could not save performance memory: {e}")),
        );
    }
    let label = match outcome {
        RecordOutcome::Inserted => "inserted",
        RecordOutcome::Merged { .. } => "merged",
        RecordOutcome::KeptStable => "kept_stable",
    };
    (label.into(), None)
}

/// A stored configuration that is known to be stable, ready to launch.
#[derive(Debug, Clone)]
pub struct KnownGood {
    pub record: BenchRecord,
    pub config: LaunchConfig,
}

/// The fastest stable record for `key`, as a launch configuration for `model_path`.
pub fn known_good(
    store: &PerfStore,
    key: &RecordKey,
    model_path: &Path,
    backend: Backend,
) -> Result<Option<KnownGood>> {
    let Some(record) = store.best_known(key) else {
        return Ok(None);
    };
    let c = record.config;
    let config = LaunchConfig::builder(model_path, backend)
        .gpu_layers(c.gpu_layers)
        .ctx_size(c.ctx_size)
        .batch_size(c.batch_size)
        .ubatch_size(c.ubatch_size)
        .threads(c.threads)
        .flash_attn(c.flash_attn)
        .kv_cache(c.kv_cache)
        .build()?;
    if c.uses_draft_model {
        return Ok(None);
    }
    let mut config = config;
    config.speculative = c.speculative;
    Ok(Some(KnownGood {
        record: record.clone(),
        config,
    }))
}

/// Explains a known-good choice in the words the UI shows.
pub fn known_good_rationale(record: &BenchRecord) -> Vec<String> {
    vec![
        format!(
            "Known-good configuration: {} benchmark run(s) on this machine with {} {}, {:.1} tokens/s generation and {:.0} ms to first token.",
            record.samples, record.runtime_id, record.runtime_version, record.gen_tps, record.ttft_ms
        ),
        format!(
            "GPU layers {}, flash attention {}, KV cache {}.",
            record.config.gpu_layers,
            if record.config.flash_attn { "on" } else { "off" },
            record.config.kv_cache.llama_arg()
        ),
    ]
}

/// Comparison of two stored records, as `perf_compare` returns it.
#[derive(Debug, Clone, Serialize)]
pub struct PerfComparison {
    pub base: PerfEntry,
    pub candidate: PerfEntry,
    pub comparison: Comparison,
}

/// Compares two records by id. `base` must have a non-zero speed.
pub fn compare_entries(base: &PerfEntry, candidate: &PerfEntry) -> Result<PerfComparison> {
    let comparison = compare(&base.record, &candidate.record)?;
    Ok(PerfComparison {
        base: base.clone(),
        candidate: candidate.clone(),
        comparison,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::fakes::FakeUpstream;
    use crate::testing::{fixed_hardware, FixedTokenizer};
    use raptor_runtime::config::{KvCacheType, GIB};
    use std::sync::Mutex;

    fn payloads(items: &[&str]) -> Vec<raptor_core::Result<String>> {
        items.iter().map(|s| Ok(s.to_string())).collect()
    }

    fn upstream_with(items: Vec<raptor_core::Result<String>>) -> FakeUpstream {
        FakeUpstream {
            streamed: Mutex::new(Some(items)),
            ..Default::default()
        }
    }

    fn delta(text: &str) -> String {
        json!({"choices": [{"delta": {"content": text}}]}).to_string()
    }

    #[test]
    fn payloads_are_read_for_text_usage_and_done() {
        assert_eq!(
            parse_payload(&delta("hi")),
            Payload {
                content: Some("hi".into()),
                ..Payload::default()
            }
        );
        assert!(parse_payload("[DONE]").done);
        let usage = json!({"choices": [], "usage": {"completion_tokens": 7}}).to_string();
        assert_eq!(parse_payload(&usage).completion_tokens, Some(7));
        assert_eq!(
            parse_payload(&delta("")).content,
            None,
            "empty deltas are ignored"
        );
        assert_eq!(parse_payload("not json"), Payload::default());
    }

    #[test]
    fn reasoning_deltas_are_read_apart_from_content() {
        let r = json!({"choices": [{"delta": {"reasoning_content": "hmm"}}]}).to_string();
        let p = parse_payload(&r);
        assert_eq!(p.reasoning.as_deref(), Some("hmm"));
        assert_eq!(p.content, None);
    }

    #[tokio::test]
    async fn reasoning_tokens_start_the_clock_and_are_counted_with_the_usage() {
        // A thinking model: reasoning first, a visible answer later. The usage counts all of it,
        // so the first token of any kind must be the start of the window.
        let r = |t: &str| json!({"choices": [{"delta": {"reasoning_content": t}}]}).to_string();
        let usage = json!({"choices": [], "usage": {"completion_tokens": 4}}).to_string();
        let up = upstream_with(payloads(&[&r("a"), &r("b"), &delta("c"), &usage, "[DONE]"]));
        let t = measure(&up, "m", "prompt").await.unwrap();
        assert_eq!(t.gen_tokens, 4);
        assert!(t.usage_reported);
        assert!(t.ttft <= t.total);
    }

    #[tokio::test]
    async fn measure_counts_deltas_when_no_usage_is_reported() {
        let up = upstream_with(payloads(&[&delta("a"), &delta("b"), &delta("c"), "[DONE]"]));
        let t = measure(&up, "m", "prompt").await.unwrap();
        assert_eq!(t.gen_tokens, 3);
        assert!(!t.usage_reported);
        assert!(t.ttft <= t.total);
    }

    #[tokio::test]
    async fn measure_prefers_the_usage_report() {
        let usage = json!({"choices": [], "usage": {"completion_tokens": 9}}).to_string();
        let up = upstream_with(payloads(&[&delta("a"), &delta("b"), &usage, "[DONE]"]));
        let t = measure(&up, "m", "prompt").await.unwrap();
        assert_eq!(t.gen_tokens, 9);
        assert!(t.usage_reported);
    }

    #[tokio::test]
    async fn measure_fails_on_stream_errors_and_empty_output() {
        let failing = upstream_with(vec![Err(RaptorError::Network("reset".into()))]);
        assert!(matches!(
            measure(&failing, "m", "p").await,
            Err(RaptorError::Network(_))
        ));
        let empty = upstream_with(payloads(&["[DONE]"]));
        let err = measure(&empty, "m", "p").await.unwrap_err();
        assert!(err.to_string().contains("no text"), "{err}");
    }

    #[test]
    fn sample_units_are_finite_and_consistent() {
        let t = Timing {
            ttft: Duration::from_millis(250),
            total: Duration::from_millis(1250),
            gen_tokens: 11,
            usage_reported: true,
        };
        let s = sample_from(&t, 500);
        assert!((s.ttft_ms - 250.0).abs() < 1e-6);
        assert!((s.prompt_tps - 2000.0).abs() < 1e-6);
        // 10 decoded tokens over 1 second.
        assert!((s.gen_tps - 10.0).abs() < 1e-6);
        let instant = Timing {
            ttft: Duration::ZERO,
            total: Duration::ZERO,
            gen_tokens: 1,
            usage_reported: false,
        };
        let s = sample_from(&instant, 0);
        assert!(s.gen_tps.is_finite() && s.prompt_tps.is_finite());
    }

    #[tokio::test]
    async fn sample_counts_the_prompt_with_the_tokenizer() {
        let up = upstream_with(payloads(&[&delta("a"), &delta("b"), "[DONE]"]));
        let (s, exact) = sample(&up, Arc::new(FixedTokenizer::exact(300)), "m")
            .await
            .unwrap();
        assert_eq!(s.prompt_tokens, 300);
        assert!(exact);
        assert_eq!(s.gen_tokens, 2);
    }

    #[test]
    fn stability_requires_a_verified_gpu_and_a_ready_server() {
        let s = BenchSample {
            ttft_ms: 300.0,
            prompt_tps: 900.0,
            gen_tps: 60.0,
            prompt_tokens: 500,
            gen_tokens: 40,
        };
        let verified = GpuVerdict::Verified {
            evidence: vec![],
            offloaded_layers: Some(41),
            total_layers: Some(41),
        };
        assert!(assess(&s, Backend::Hip, true, Some(&verified)).is_empty());

        let cpu = GpuVerdict::CpuOnly {
            reason: "no layers offloaded".into(),
        };
        let reasons = assess(&s, Backend::Hip, true, Some(&cpu));
        assert!(
            reasons.iter().any(|r| r.contains("ran on the CPU")),
            "{reasons:?}"
        );

        assert!(!assess(&s, Backend::Hip, true, None).is_empty());
        assert!(
            assess(&s, Backend::Cpu, true, None).is_empty(),
            "CPU runs need no GPU proof"
        );
        assert!(!assess(&s, Backend::Cpu, false, None).is_empty());

        let thin = BenchSample { gen_tokens: 1, ..s };
        assert!(!assess(&thin, Backend::Cpu, true, None).is_empty());
    }

    #[test]
    fn memory_is_read_from_the_probe_snapshot() {
        let hw = fixed_hardware();
        let (vram, ram) = memory_used(&hw);
        assert_eq!(vram, 16 * GIB - hw.gpus[0].vram_free.unwrap());
        assert_eq!(ram, hw.ram_total - hw.ram_avail);
    }

    #[test]
    fn entry_ids_are_stable_and_change_with_the_config() {
        let hw = fixed_hardware();
        let cfg = ConfigSnapshot {
            gpu_layers: 40,
            ctx_size: 8192,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 8,
            flash_attn: true,
            kv_cache: KvCacheType::F16,
            speculative: None,
            uses_draft_model: false,
        };
        let a = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            cfg,
            None,
            (0, 0),
            false,
            1,
        );
        let b = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            cfg,
            None,
            (0, 0),
            false,
            1,
        );
        assert_eq!(entry_id(&a), entry_id(&b));
        let mut other = cfg;
        other.ctx_size = 4096;
        let c = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            other,
            None,
            (0, 0),
            false,
            1,
        );
        assert_ne!(entry_id(&a), entry_id(&c));
        assert_eq!(entry_id(&a).len(), 16);
    }

    #[test]
    fn failed_runs_are_stored_as_unstable_zeros() {
        let hw = fixed_hardware();
        let cfg = ConfigSnapshot {
            gpu_layers: 0,
            ctx_size: 4096,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 4,
            flash_attn: false,
            kv_cache: KvCacheType::F16,
            speculative: None,
            uses_draft_model: false,
        };
        let r = build_record(&hw, "h", "llama-cpu-win", "b1", cfg, None, (0, 0), false, 5);
        assert!(!r.stable);
        assert_eq!(r.gen_tps, 0.0);
        let mut slot: std::result::Result<PerfStore, String> = Err("corrupt".into());
        let (outcome, note) = store_record(&mut slot, r);
        assert_eq!(outcome, "not_saved");
        assert!(note.unwrap().contains("corrupt"));
    }

    #[test]
    fn unstable_runs_never_replace_a_stable_record_and_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = Ok(PerfStore::open(dir.path().join("perf.json")).unwrap());
        let hw = fixed_hardware();
        let cfg = ConfigSnapshot {
            gpu_layers: 40,
            ctx_size: 8192,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 8,
            flash_attn: true,
            kv_cache: KvCacheType::F16,
            speculative: None,
            uses_draft_model: false,
        };
        let s = BenchSample {
            ttft_ms: 300.0,
            prompt_tps: 900.0,
            gen_tps: 60.0,
            prompt_tokens: 500,
            gen_tokens: 40,
        };
        let good = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            cfg,
            Some(&s),
            (1, 2),
            true,
            1,
        );
        assert_eq!(store_record(&mut slot, good).0, "inserted");
        let bad = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            cfg,
            None,
            (0, 0),
            false,
            2,
        );
        assert_eq!(store_record(&mut slot, bad).0, "kept_stable");
        let store = slot.unwrap();
        let key = key_for(&hw, "h", "llama-hip-win");
        let best = store.best_known(&key).unwrap();
        assert!((best.gen_tps - 60.0).abs() < 1e-9, "stable record kept");
        assert_eq!(store.records().len(), 1);
    }

    #[test]
    fn known_good_becomes_a_launch_config_with_the_stored_settings() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = PerfStore::open(dir.path().join("perf.json")).unwrap();
        let hw = fixed_hardware();
        let cfg = ConfigSnapshot {
            gpu_layers: 33,
            ctx_size: 6144,
            batch_size: 1024,
            ubatch_size: 256,
            threads: 6,
            flash_attn: true,
            kv_cache: KvCacheType::Q8_0,
            speculative: None,
            uses_draft_model: false,
        };
        let s = BenchSample {
            ttft_ms: 200.0,
            prompt_tps: 800.0,
            gen_tps: 55.0,
            prompt_tokens: 400,
            gen_tokens: 30,
        };
        let r = build_record(
            &hw,
            "h",
            "llama-hip-win",
            "b7100",
            cfg,
            Some(&s),
            (1, 2),
            true,
            1,
        );
        store.record(r).unwrap();
        let key = key_for(&hw, "h", "llama-hip-win");
        let kg = known_good(&store, &key, Path::new("C:/m/tiny.gguf"), Backend::Hip)
            .unwrap()
            .expect("stable record found");
        assert_eq!(kg.config.gpu_layers, 33);
        assert_eq!(kg.config.ctx_size, 6144);
        assert_eq!(kg.config.kv_cache, KvCacheType::Q8_0);
        assert_eq!(ConfigSnapshot::from(&kg.config), cfg);
        assert!(known_good_rationale(&kg.record)[0].contains("Known-good"));

        let other_key = key_for(&hw, "other-model", "llama-hip-win");
        assert!(
            known_good(&store, &other_key, Path::new("C:/m/x.gguf"), Backend::Hip)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn comparison_of_two_entries_reports_percentages() {
        let hw = fixed_hardware();
        let cfg = |ctx| ConfigSnapshot {
            gpu_layers: 40,
            ctx_size: ctx,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 8,
            flash_attn: true,
            kv_cache: KvCacheType::F16,
            speculative: None,
            uses_draft_model: false,
        };
        let mk = |ctx, gen| {
            let s = BenchSample {
                ttft_ms: 300.0,
                prompt_tps: 900.0,
                gen_tps: gen,
                prompt_tokens: 500,
                gen_tokens: 40,
            };
            entry_view(&build_record(
                &hw,
                "h",
                "llama-hip-win",
                "b7100",
                cfg(ctx),
                Some(&s),
                (1, 1),
                true,
                1,
            ))
        };
        let a = mk(4096, 50.0);
        let b = mk(8192, 60.0);
        assert_ne!(a.id, b.id);
        let c = compare_entries(&a, &b).unwrap();
        assert!((c.comparison.gen_tps_delta_pct - 20.0).abs() < 1e-9);
    }
}
