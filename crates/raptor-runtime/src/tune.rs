//! Per-model speed tuning. `llama-bench` (shipped in every llama.cpp runtime zip) measures prompt
//! and generation speed for a model at a GPU layer count and micro-batch size, and this module
//! turns those measurements into a choice.
//!
//! Why measure: on the RX 9060 XT the best micro-batch is model-specific (Ornith 2.6x faster
//! prompt processing at 256 than at 512, flint best at 512, dense models within 8%), and for a
//! sparse Mixture-of-experts model that does not fit in VRAM the best layer count is "all of them".
//! A fixed rule cannot know that, a measurement does.
//!
//! The score is the time of a representative coding request: a long prompt plus a reply. Lower is
//! better, and it weighs both speeds the way an agent session does.

use crate::sys::CommandRunner;
use raptor_core::{RaptorError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Tokens of prompt in the representative request.
pub const SCORE_PROMPT_TOKENS: f64 = 4096.0;
/// Tokens of reply in the representative request.
pub const SCORE_REPLY_TOKENS: f64 = 256.0;
/// Prompt length measured by llama-bench. Long enough that the micro-batch matters.
pub const BENCH_PROMPT_TOKENS: u32 = 2048;
/// Reply length measured by llama-bench.
pub const BENCH_REPLY_TOKENS: u32 = 64;
/// Micro-batch sizes tried. 1024 was dropped: across five measured models it never won, and
/// leaving it out cuts each sweep by a third (see docs/EXPERIMENTS.md).
pub const UBATCH_CANDIDATES: [u32; 2] = [256, 512];
/// Context depth of the opt-in deep tune: the "long prompt" setting is measured with this many
/// tokens already in the context, and is used for context windows of at least this size.
pub const DEEP_DEPTH: u32 = 16_384;
/// A model that cannot be benchmarked in this long is reported as failed, not waited on forever.
const BENCH_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// Existing context tokens needed so the measured prompt and generated reply reach the effective
/// window used by the app.
fn bench_depth(context_size: u32) -> u32 {
    context_size.saturating_sub(BENCH_PROMPT_TOKENS + BENCH_REPLY_TOKENS)
}

#[cfg(windows)]
pub const BENCH_BINARY: &str = "llama-bench.exe";
#[cfg(not(windows))]
pub const BENCH_BINARY: &str = "llama-bench";

/// One measured combination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measured {
    pub gpu_layers: u32,
    pub ubatch: u32,
    /// Prompt processing, tokens per second.
    pub prompt_tps: f64,
    /// Generation, tokens per second.
    pub gen_tps: f64,
    /// Time of the representative request, seconds. Lower is better.
    pub seconds: f64,
}

impl Measured {
    /// The measurement of `point` (taken with `depth` tokens in the context) for these layers and
    /// micro-batch, scored like the empty-context one.
    pub fn at_depth(gpu_layers: u32, ubatch: u32, point: &DepthPoint) -> Self {
        Self {
            gpu_layers,
            ubatch,
            prompt_tps: point.prompt_tps,
            gen_tps: point.gen_tps,
            seconds: request_seconds(point.prompt_tps, point.gen_tps),
        }
    }
}

/// Time of the representative request at the measured speeds.
pub fn request_seconds(prompt_tps: f64, gen_tps: f64) -> f64 {
    SCORE_PROMPT_TOKENS / prompt_tps + SCORE_REPLY_TOKENS / gen_tps
}

/// One row of `llama-bench -o json`.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    ubatch: u32,
    depth: u32,
    n_prompt: u32,
    n_gen: u32,
    tps: f64,
}

fn parse_rows(json: &str) -> Result<Vec<Row>> {
    // llama-bench prints backend load lines before the JSON array on some builds.
    let start = json
        .find('[')
        .ok_or_else(|| RaptorError::Runtime("llama-bench printed no JSON result".into()))?;
    let v: serde_json::Value = serde_json::from_str(json[start..].trim())
        .map_err(|e| RaptorError::Runtime(format!("llama-bench output is not valid JSON: {e}")))?;
    let rows = v
        .as_array()
        .ok_or_else(|| RaptorError::Runtime("llama-bench JSON is not a list".into()))?;
    Ok(rows
        .iter()
        .filter_map(|r| {
            Some(Row {
                ubatch: u32::try_from(r.get("n_ubatch")?.as_u64()?).ok()?,
                depth: r
                    .get("n_depth")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|d| u32::try_from(d).ok())
                    .unwrap_or(0),
                n_prompt: u32::try_from(r.get("n_prompt")?.as_u64()?).ok()?,
                n_gen: u32::try_from(r.get("n_gen")?.as_u64()?).ok()?,
                tps: r
                    .get("avg_ts")?
                    .as_f64()
                    .filter(|t| t.is_finite() && *t > 0.0)?,
            })
        })
        .collect())
}

/// Pairs the prompt-processing and generation rows of each micro-batch into [`Measured`].
/// A micro-batch with only one of the two is dropped: half a measurement cannot be scored.
pub fn parse_bench(json: &str, gpu_layers: u32) -> Result<Vec<Measured>> {
    parse_bench_at_depth(json, gpu_layers, 0)
}

/// Parses a sweep measured with `depth` tokens already in the context.
pub fn parse_bench_at_depth(json: &str, gpu_layers: u32, depth: u32) -> Result<Vec<Measured>> {
    let rows = parse_rows(json)?;
    let mut out = Vec::new();
    for ub in UBATCH_CANDIDATES {
        let pp = rows
            .iter()
            .find(|r| r.ubatch == ub && r.depth == depth && r.n_prompt > 0 && r.n_gen == 0)
            .map(|r| r.tps);
        let tg = rows
            .iter()
            .find(|r| r.ubatch == ub && r.depth == depth && r.n_gen > 0 && r.n_prompt == 0)
            .map(|r| r.tps);
        if let (Some(pp), Some(tg)) = (pp, tg) {
            out.push(Measured {
                gpu_layers,
                ubatch: ub,
                prompt_tps: pp,
                gen_tps: tg,
                seconds: request_seconds(pp, tg),
            });
        }
    }
    if out.is_empty() {
        return Err(RaptorError::Runtime(
            "llama-bench reported no complete prompt and generation measurement".into(),
        ));
    }
    Ok(out)
}

/// The fastest measurement, by the time of the representative request.
pub fn best(measured: &[Measured]) -> Option<&Measured> {
    measured.iter().min_by(|a, b| {
        a.seconds
            .partial_cmp(&b.seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Runs llama-bench for `model` at `gpu_layers` across every micro-batch candidate.
/// `bench` is the llama-bench binary of the selected runtime. The model is loaded once.
/// An error is returned as is: the caller decides whether a candidate that failed (out of
/// memory, say) ends the search or is skipped.
pub async fn measure(
    runner: &dyn CommandRunner,
    bench: &Path,
    model: &Path,
    gpu_layers: u32,
    context_size: u32,
    kv_cache: crate::config::KvCacheType,
    flash_attn: bool,
    env: &[(String, String)],
) -> Result<Vec<Measured>> {
    let ubs = UBATCH_CANDIDATES
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    // Match the requested effective window while leaving room for the measured prompt and reply.
    // llama-bench exposes existing context through -d rather than -c.
    let depth = bench_depth(context_size);
    let cache = kv_cache.llama_arg();
    let args: Vec<String> = [
        "-m",
        &model.to_string_lossy(),
        "-ngl",
        &gpu_layers.to_string(),
        "-ub",
        &ubs,
        "-b",
        "2048",
        "-p",
        &BENCH_PROMPT_TOKENS.to_string(),
        "-n",
        &BENCH_REPLY_TOKENS.to_string(),
        "-d",
        &depth.to_string(),
        "-ctk",
        cache,
        "-ctv",
        cache,
        "-fa",
        if flash_attn { "1" } else { "0" },
        "-r",
        "3",
        "-o",
        "json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let out = runner
        .run_with_env(&bench.to_string_lossy(), &args, env, BENCH_TIMEOUT)
        .await?;
    parse_bench_at_depth(&out, gpu_layers, depth)
}

/// Prefill and decode speed with `depth` tokens already in the context.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DepthPoint {
    pub depth: u32,
    /// Prefill speed of a [`crate::ctxmatrix::TURN_TOKENS`]-token turn at this depth.
    pub prompt_tps: f64,
    pub gen_tps: f64,
}

/// Reads the prefill and decode rows measured at `depth` from `llama-bench -o json`.
pub fn parse_depth(json: &str, depth: u32) -> Result<DepthPoint> {
    let rows = parse_rows(json)?;
    let pp = rows
        .iter()
        .find(|r| r.depth == depth && r.n_prompt > 0 && r.n_gen == 0)
        .map(|r| r.tps);
    let tg = rows
        .iter()
        .find(|r| r.depth == depth && r.n_gen > 0 && r.n_prompt == 0)
        .map(|r| r.tps);
    match (pp, tg) {
        (Some(prompt_tps), Some(gen_tps)) => Ok(DepthPoint {
            depth,
            prompt_tps,
            gen_tps,
        }),
        _ => Err(RaptorError::Runtime(format!(
            "llama-bench reported no complete measurement at a depth of {depth} tokens"
        ))),
    }
}

/// Measures one depth with llama-bench in its own process: the cache type, layers and micro-batch
/// given, a 512-token turn for prefill and 64 generated tokens for decode, after `depth` tokens
/// of context. The depth itself is prefilled first, so a deep run takes minutes.
#[allow(clippy::too_many_arguments)]
pub async fn measure_depth(
    runner: &dyn CommandRunner,
    bench: &Path,
    model: &Path,
    gpu_layers: u32,
    ubatch: u32,
    kv: &str,
    depth: u32,
    flash_attn: bool,
    env: &[(String, String)],
) -> Result<DepthPoint> {
    let args: Vec<String> = [
        "-m",
        &model.to_string_lossy(),
        "-ngl",
        &gpu_layers.to_string(),
        "-ub",
        &ubatch.to_string(),
        "-b",
        "2048",
        "-p",
        &crate::ctxmatrix::TURN_TOKENS.to_string(),
        "-n",
        &BENCH_REPLY_TOKENS.to_string(),
        "-d",
        &depth.to_string(),
        "-fa",
        if flash_attn { "1" } else { "0" },
        "-ctk",
        kv,
        "-ctv",
        kv,
        "-r",
        "1",
        "-o",
        "json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let out = runner
        .run_with_env(&bench.to_string_lossy(), &args, env, BENCH_TIMEOUT)
        .await?;
    parse_depth(&out, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeRunner;

    #[test]
    fn depth_rows_are_paired_and_the_zero_depth_pairing_ignores_deeper_rows() {
        let json = r#"[
          {"n_ubatch": 256, "n_depth": 0, "n_prompt": 512, "n_gen": 0, "avg_ts": 584.0},
          {"n_ubatch": 256, "n_depth": 0, "n_prompt": 0, "n_gen": 64, "avg_ts": 21.2},
          {"n_ubatch": 256, "n_depth": 16384, "n_prompt": 512, "n_gen": 0, "avg_ts": 499.0},
          {"n_ubatch": 256, "n_depth": 16384, "n_prompt": 0, "n_gen": 64, "avg_ts": 19.5}
        ]"#;
        let deep = parse_depth(json, 16384).unwrap();
        assert_eq!((deep.prompt_tps, deep.gen_tps), (499.0, 19.5));
        let base = parse_depth(json, 0).unwrap();
        assert_eq!((base.prompt_tps, base.gen_tps), (584.0, 21.2));
        assert!(parse_depth(json, 4096).is_err(), "no rows at that depth");
        // The sweep parser (micro-batch search) must not be confused by depth rows.
        assert_eq!(parse_bench(json, 66).unwrap()[0].prompt_tps, 584.0);
    }

    #[tokio::test]
    async fn measure_depth_passes_the_depth_cache_type_and_layers_to_llama_bench() {
        let json = r#"[{"n_ubatch": 256, "n_depth": 4096, "n_prompt": 512, "n_gen": 0, "avg_ts": 540.0},
                       {"n_ubatch": 256, "n_depth": 4096, "n_prompt": 0, "n_gen": 64, "avg_ts": 20.5}]"#;
        let runner = FakeRunner::default().with_ok_for_all(json);
        let p = measure_depth(
            &runner,
            Path::new("C:/rt/llama-bench.exe"),
            Path::new("C:/m.gguf"),
            66,
            256,
            "q8_0",
            4096,
            true,
            &[],
        )
        .await
        .unwrap();
        assert_eq!(p.depth, 4096);
        assert!((p.gen_tps - 20.5).abs() < 1e-9);
    }

    fn row(ub: u32, n_prompt: u32, n_gen: u32, ts: f64) -> String {
        format!(r#"{{"n_ubatch": {ub}, "n_prompt": {n_prompt}, "n_gen": {n_gen}, "avg_ts": {ts}}}"#)
    }

    fn sample() -> String {
        // The measured Ornith numbers at all 41 layers.
        let rows = [
            row(256, 2048, 0, 2234.7),
            row(256, 0, 64, 68.4),
            row(512, 2048, 0, 866.2),
            row(512, 0, 64, 68.3),
        ];
        format!(
            "load_backend: loaded ROCm backend\n[\n{}\n]\n",
            rows.join(",\n")
        )
    }

    #[test]
    fn a_measurement_at_depth_is_scored_like_one_at_an_empty_context() {
        let point = DepthPoint {
            depth: DEEP_DEPTH,
            prompt_tps: 500.0,
            gen_tps: 20.0,
        };
        let m = Measured::at_depth(66, 512, &point);
        assert_eq!((m.gpu_layers, m.ubatch), (66, 512));
        assert!((m.seconds - request_seconds(500.0, 20.0)).abs() < 1e-9);
    }

    #[test]
    fn the_deep_depth_is_the_window_size_that_selects_the_long_prompt_setting() {
        assert_eq!(DEEP_DEPTH, 16_384);
    }

    #[test]
    fn quick_tune_depth_reserves_the_same_prompt_and_reply_inside_the_effective_context() {
        assert_eq!(bench_depth(65_536), 63_424);
        assert_eq!(bench_depth(BENCH_PROMPT_TOKENS + BENCH_REPLY_TOKENS), 0);
        assert_eq!(bench_depth(1024), 0);
    }

    #[test]
    fn the_score_is_the_time_of_a_long_prompt_plus_a_reply() {
        let s = request_seconds(1000.0, 50.0);
        assert!((s - (4.096 + 5.12)).abs() < 1e-9);
        assert!(request_seconds(2000.0, 50.0) < s);
        assert!(request_seconds(1000.0, 60.0) < s);
    }

    #[test]
    fn prompt_and_generation_rows_are_paired_per_micro_batch() {
        let m = parse_bench(&sample(), 41).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!((m[0].ubatch, m[0].gpu_layers), (256, 41));
        assert!((m[0].prompt_tps - 2234.7).abs() < 1e-9 && (m[0].gen_tps - 68.4).abs() < 1e-9);
        // The measured Ornith result: micro-batch 256 wins by a wide margin.
        assert_eq!(best(&m).unwrap().ubatch, 256);
        assert!(m[0].seconds < m[1].seconds * 0.7);
    }

    #[test]
    fn an_incomplete_measurement_is_dropped_and_none_at_all_is_an_error() {
        let only_prompt = format!("[{}]", row(256, 2048, 0, 100.0));
        assert!(parse_bench(&only_prompt, 10).is_err());
        let one_complete = format!(
            "[{},{},{}]",
            row(512, 2048, 0, 700.0),
            row(512, 0, 64, 30.0),
            row(256, 2048, 0, 5.0)
        );
        let m = parse_bench(&one_complete, 10).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].ubatch, 512);
        assert!(parse_bench("no json here", 10).is_err());
        assert!(parse_bench("[not json]", 10).is_err());
        let zero = format!("[{},{}]", row(256, 2048, 0, 0.0), row(256, 0, 64, 10.0));
        assert!(
            parse_bench(&zero, 10).is_err(),
            "a zero speed is not a measurement"
        );
    }

    #[tokio::test]
    async fn measure_runs_llama_bench_with_the_sweep_arguments_and_the_given_environment() {
        let runner = FakeRunner::default().with_ok_for_all(&sample());
        let env = vec![("PATH".to_string(), "C:/sdk/bin".to_string())];
        let m = measure(
            &runner,
            Path::new("C:/rt/llama-bench.exe"),
            Path::new("C:/m.gguf"),
            41,
            BENCH_PROMPT_TOKENS + BENCH_REPLY_TOKENS,
            crate::config::KvCacheType::Q4_0,
            true,
            &env,
        )
        .await
        .unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(runner.last_env(), env);
        let args = runner.last_args();
        assert!(args.windows(2).any(|a| a == ["-d", "0"]));
        assert!(args.windows(2).any(|a| a == ["-ctk", "q4_0"]));
        assert!(args.windows(2).any(|a| a == ["-ctv", "q4_0"]));
    }

    #[tokio::test]
    async fn a_failing_benchmark_is_an_error_not_a_zero() {
        let runner = FakeRunner::default().with_err("C:/rt/llama-bench.exe", "out of memory");
        let err = measure(
            &runner,
            Path::new("C:/rt/llama-bench.exe"),
            Path::new("m.gguf"),
            99,
            BENCH_PROMPT_TOKENS + BENCH_REPLY_TOKENS,
            crate::config::KvCacheType::F16,
            true,
            &[],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("out of memory"));
    }
}
