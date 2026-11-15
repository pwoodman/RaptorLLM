//! The largest context window that fits the GPU for a given KV-cache type and slot count.
//!
//! The memory a context needs is close to linear in its length, but not exactly: with a
//! quantized KV cache the compute buffer also grows (Qwen3.8-27B, q8_0: 266 MiB at 8,192 tokens,
//! 1,202 MiB at 262,144). So the answer is not computed from the GGUF. It is extrapolated from two
//! of llama.cpp's own projections (`llama-fit-params`, no weights loaded, about a second each) and
//! then confirmed with a third at the predicted length, shrinking it if the confirmation is over.
//!
//! Measured on the RX 9060 XT, KV-cache type does not change speed (f16, q8_0 and q4_0 within 1%),
//! so the choice between them is memory against quality: q8_0 halves the cache, q4_0 quarters it.

use crate::config::{KvCacheType, LaunchConfig};
use crate::fit::{FitProjection, FIT_PARAMS_BINARY};
use crate::sys::CommandRunner;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::path::Path;

/// Contexts are rounded down to a multiple of this.
pub const STEP: u32 = 1024;
/// The smallest context worth offering.
pub const FLOOR: u32 = 4096;
/// Extra shrink applied to an over-budget confirmation, to converge in few probes.
const SHRINK_MARGIN: f64 = 1.15;
/// Probes allowed after the prediction.
const MAX_REFINEMENTS: usize = 3;

/// Device memory the launch uses beyond llama.cpp's projection, measured on the RX 9060 XT
/// (dedicated GPU memory with the model loaded, minus the 586 MiB the desktop uses, against the
/// projection for the same launch): granite 257 MiB; Qwen3.8-27B with its MTP head 1,531 MiB at an
/// 8,192-token window and 2,335 MiB at 166,912, which is a fixed part plus about 5.2 KiB per
/// context token for the draft context. Ignoring it put the maximum window 2.3 GB over VRAM and
/// generation fell from 47 to 10 tok/s as the driver spilled into shared memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overhead {
    pub fixed_mib: u64,
    pub per_token_mib: f64,
}

impl Overhead {
    /// What every launch uses beyond the projection (driver and runtime workspaces).
    pub const BASE: Overhead = Overhead {
        fixed_mib: 300,
        per_token_mib: 0.0,
    };
    /// Adds the draft context of speculative decoding with the model's own MTP head.
    pub const MTP: Overhead = Overhead {
        fixed_mib: 1_250,
        per_token_mib: 0.0052,
    };

    pub fn plus(self, other: Overhead) -> Overhead {
        Overhead {
            fixed_mib: self.fixed_mib + other.fixed_mib,
            per_token_mib: self.per_token_mib + other.per_token_mib,
        }
    }

    /// Total device memory at `ctx` tokens for a projection of `projected_mib`.
    pub fn total(self, projected_mib: u64, ctx: u32) -> u64 {
        projected_mib + self.fixed_mib + (self.per_token_mib * f64::from(ctx)).ceil() as u64
    }
}

/// A context the GPU can hold, with the evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextFit {
    pub kv_cache: KvCacheType,
    pub slots: u32,
    pub ctx: u32,
    /// Projected device memory at `ctx` including the measured overhead, MiB.
    pub projected_mib: u64,
    /// What it had to fit in, MiB.
    pub budget_mib: u64,
    /// True when `ctx` is the model's trained maximum: the GPU could hold more.
    pub capped_by_model: bool,
}

fn round_down(ctx: u64) -> u32 {
    u32::try_from(ctx / u64::from(STEP) * u64::from(STEP)).unwrap_or(u32::MAX)
}

/// MiB per context token between two projections. Never negative.
pub fn slope(p1: (u32, u64), p2: (u32, u64)) -> f64 {
    if p2.0 <= p1.0 {
        return 0.0;
    }
    (p2.1.saturating_sub(p1.1)) as f64 / f64::from(p2.0 - p1.0)
}

/// The context predicted to fit `budget_mib`, from two projections `(ctx, total_mib)`.
/// `None` when even the first (smaller) one is over budget, or the result is below [`FLOOR`].
pub fn extrapolate(p1: (u32, u64), p2: (u32, u64), budget_mib: u64, cap: u32) -> Option<u32> {
    if p1.1 > budget_mib {
        return None;
    }
    let s = slope(p1, p2);
    let ctx = if s <= 0.0 {
        u64::from(cap)
    } else {
        let extra = (budget_mib - p1.1) as f64 / s;
        (f64::from(p1.0) + extra).min(f64::from(cap)) as u64
    };
    let ctx = round_down(ctx.min(u64::from(cap)));
    (ctx >= FLOOR.min(cap)).then_some(ctx)
}

/// A shorter context to try after `used_mib` at `ctx` came out over `budget_mib`.
pub fn shrink(ctx: u32, used_mib: u64, budget_mib: u64, slope_mib_per_token: f64) -> u32 {
    let over = used_mib.saturating_sub(budget_mib) as f64;
    let drop = if slope_mib_per_token > 0.0 {
        (over / slope_mib_per_token * SHRINK_MARGIN).ceil() as u64
    } else {
        u64::from(STEP)
    };
    round_down(u64::from(ctx).saturating_sub(drop.max(u64::from(STEP))))
}

/// Arguments for a projection at exactly the launch's context, cache types and slot count, with
/// every layer on the GPU.
pub fn probe_args_exact(cfg: &LaunchConfig, slots: u32) -> Vec<String> {
    vec![
        "-m".into(),
        cfg.model_path.to_string_lossy().into_owned(),
        "-ngl".into(),
        "999".into(),
        "-c".into(),
        cfg.ctx_size.to_string(),
        "-b".into(),
        cfg.batch_size.to_string(),
        "-ub".into(),
        cfg.ubatch_size.to_string(),
        "-fa".into(),
        if cfg.flash_attn { "on" } else { "off" }.into(),
        "-ctk".into(),
        cfg.kv_cache.llama_arg().into(),
        "-ctv".into(),
        cfg.kv_cache.llama_arg().into(),
        "-np".into(),
        slots.to_string(),
        "--fit-print".into(),
        "on".into(),
    ]
}

async fn project(
    runner: &dyn CommandRunner,
    tool: &Path,
    base: &LaunchConfig,
    ctx: u32,
    slots: u32,
    env: &[(String, String)],
) -> Result<FitProjection> {
    let mut cfg = base.clone();
    cfg.ctx_size = ctx;
    cfg.batch_size = cfg.batch_size.min(ctx);
    cfg.ubatch_size = cfg.ubatch_size.min(cfg.batch_size);
    let program = tool.to_string_lossy().into_owned();
    let out = runner
        .run_with_env(
            &program,
            &probe_args_exact(&cfg, slots),
            env,
            std::time::Duration::from_secs(120),
        )
        .await?;
    crate::fit::parse_fit_projection(&out, ctx)
        .ok_or_else(|| RaptorError::Runtime("llama-fit-params printed no GPU memory row".into()))
}

/// Finds the largest context for `base` (every layer on the GPU) with `kv` and `slots` that fits
/// `budget_mib`, at most `cap` tokens (the model's trained length). `tool_dir` holds
/// `llama-fit-params`. Errors when the tool is missing or even [`FLOOR`] tokens do not fit.
#[allow(clippy::too_many_arguments)]
pub async fn find(
    runner: &dyn CommandRunner,
    tool_dir: &Path,
    base: &LaunchConfig,
    kv: KvCacheType,
    slots: u32,
    budget_mib: u64,
    cap: u32,
    overhead: Overhead,
    env: &[(String, String)],
) -> Result<ContextFit> {
    if base.model_path.as_os_str().is_empty() {
        return Err(RaptorError::InvalidInput(
            "the context probe needs the model path (use LaunchConfig::with_model)".into(),
        ));
    }
    let tool = tool_dir.join(FIT_PARAMS_BINARY);
    if !tool.is_file() {
        return Err(RaptorError::Unsupported(format!(
            "{FIT_PARAMS_BINARY} is not in this runtime, so the context limit cannot be projected"
        )));
    }
    let mut base = base.clone();
    base.kv_cache = kv;
    base.flash_attn = true;
    let low = FLOOR.min(cap).max(2048);
    let high = cap.min(131_072).max(low + STEP);
    let p1 = project(runner, &tool, &base, low, slots, env).await?;
    let p2 = project(runner, &tool, &base, high, slots, env).await?;
    let points = (
        (low, overhead.total(p1.total_mib(), low)),
        (high, overhead.total(p2.total_mib(), high)),
    );
    let Some(mut ctx) = extrapolate(points.0, points.1, budget_mib, cap) else {
        return Err(RaptorError::Runtime(format!(
            "not even {low} tokens fit in {budget_mib} MiB with a {} KV cache",
            kv.llama_arg()
        )));
    };
    let s = slope(points.0, points.1);
    for _ in 0..=MAX_REFINEMENTS {
        let p = project(runner, &tool, &base, ctx, slots, env).await?;
        let used = overhead.total(p.total_mib(), ctx);
        if used <= budget_mib {
            return Ok(ContextFit {
                kv_cache: kv,
                slots,
                ctx,
                projected_mib: used,
                budget_mib,
                capped_by_model: ctx >= round_down(u64::from(cap)),
            });
        }
        ctx = shrink(ctx, used, budget_mib, s);
        if ctx < FLOOR.min(cap) {
            break;
        }
    }
    Err(RaptorError::Runtime(format!(
        "no context of at least {} tokens fits in {budget_mib} MiB with a {} KV cache",
        FLOOR.min(cap),
        kv.llama_arg()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Backend;
    use crate::testing::FakeRunner;

    const ZERO: Overhead = Overhead {
        fixed_mib: 0,
        per_token_mib: 0.0,
    };

    // The measured Qwen3.8-27B q8_0, 1 slot rows: (ctx, model + context + compute MiB).
    const Q8_8K: (u32, u64) = (8192, 11307);
    const Q8_131K: (u32, u64) = (131072, 15747);

    #[test]
    fn the_slope_is_mib_per_token_and_never_negative() {
        let s = slope(Q8_8K, Q8_131K);
        assert!((s - (15747.0 - 11307.0) / (131072.0 - 8192.0)).abs() < 1e-9);
        assert_eq!(slope(Q8_131K, Q8_8K), 0.0, "reversed points give no slope");
        assert_eq!(
            slope((8192, 100), (16384, 50)),
            0.0,
            "memory cannot shrink with context"
        );
    }

    #[test]
    fn the_mtp_overhead_is_the_measured_fixed_part_plus_a_cost_per_context_token() {
        let o = Overhead::BASE.plus(Overhead::MTP);
        // Measured gap for Qwen3.8-27B: about 1,531 MiB at 8,192 tokens, 2,335 at 166,912.
        let at_small = o.total(0, 8192);
        let at_large = o.total(0, 166_912);
        assert!((1_500..=1_650).contains(&at_small), "{at_small}");
        assert!((2_300..=2_500).contains(&at_large), "{at_large}");
        assert_eq!(Overhead::BASE.total(100, 4096), 400);
    }

    #[test]
    fn the_prediction_extrapolates_to_the_budget_and_rounds_down_to_a_step() {
        // 14,675 MiB is 90% of the 16,304 MiB card.
        let ctx = extrapolate(Q8_8K, Q8_131K, 14_675, 262_144).unwrap();
        assert!(ctx.is_multiple_of(STEP));
        // Measured: q8_0 needs 13,283 MiB at 65k and 15,747 MiB at 131k, so the answer sits between.
        assert!(ctx > 65_536 && ctx < 131_072, "{ctx}");
    }

    #[test]
    fn the_prediction_is_capped_by_the_model_and_refuses_when_nothing_fits() {
        assert_eq!(extrapolate(Q8_8K, Q8_131K, 100_000, 32_768), Some(32_768));
        assert_eq!(
            extrapolate(Q8_8K, Q8_131K, 11_000, 262_144),
            None,
            "even the smallest is over budget"
        );
        // A flat projection means the cap is the limit.
        assert_eq!(
            extrapolate((8192, 500), (65536, 500), 1000, 65_536),
            Some(65_536)
        );
    }

    #[test]
    fn shrinking_always_moves_down_by_at_least_one_step() {
        let s = slope(Q8_8K, Q8_131K);
        let smaller = shrink(131_072, 15_747, 14_675, s);
        assert!(smaller < 131_072 && smaller.is_multiple_of(STEP));
        // The margin overshoots the minimum so the next probe is likely to fit.
        let exact = 131_072.0 - (15_747.0 - 14_675.0) / s;
        assert!(f64::from(smaller) < exact);
        assert_eq!(shrink(8192, 5000, 4999, 0.0), 8192 - STEP);
    }

    fn cfg() -> LaunchConfig {
        LaunchConfig::builder("C:/m.gguf", Backend::Hip)
            .ctx_size(8192)
            .build()
            .unwrap()
    }

    #[test]
    fn the_probe_asks_for_the_cache_type_slots_and_context_exactly() {
        let mut c = cfg();
        c.kv_cache = KvCacheType::Q8_0;
        let args = probe_args_exact(&c, 2);
        let pair = |k: &str| {
            args.iter()
                .position(|a| a == k)
                .map(|i| args[i + 1].as_str())
        };
        assert_eq!(pair("-ctk"), Some("q8_0"));
        assert_eq!(pair("-ctv"), Some("q8_0"));
        assert_eq!(pair("-np"), Some("2"));
        assert_eq!(pair("-c"), Some("8192"));
        assert_eq!(pair("-ngl"), Some("999"));
        assert_eq!(pair("--fit-print"), Some("on"));
    }

    #[tokio::test]
    async fn a_config_without_a_model_path_is_refused_before_running_anything() {
        let mut c = cfg();
        c.model_path = std::path::PathBuf::new();
        let runner = FakeRunner::default().with_ok_for_all(
            "ROCm0 1 1 1 
",
        );
        let dir = tempfile::tempdir().unwrap();
        let err = find(
            &runner,
            dir.path(),
            &c,
            KvCacheType::Q8_0,
            1,
            14_000,
            65_536,
            ZERO,
            &[],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("model path"), "{err}");
    }

    #[tokio::test]
    async fn a_missing_projection_tool_is_reported_not_guessed() {
        let runner = FakeRunner::default().with_ok_for_all("ROCm0 10616 500 266 \n");
        let dir = tempfile::tempdir().unwrap();
        let err = find(
            &runner,
            dir.path(),
            &cfg(),
            KvCacheType::Q8_0,
            1,
            14_000,
            262_144,
            ZERO,
            &[],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cannot be projected"), "{err}");
    }

    #[tokio::test]
    async fn a_flat_projection_gives_the_model_maximum_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FIT_PARAMS_BINARY), b"x").unwrap();
        let runner = FakeRunner::default().with_ok_for_all("ROCm0 10616 500 266 \nHost 1 0 1\n");
        let fit = find(
            &runner,
            dir.path(),
            &cfg(),
            KvCacheType::Q8_0,
            1,
            14_000,
            65_536,
            ZERO,
            &[],
        )
        .await
        .unwrap();
        assert_eq!(fit.ctx, 65_536);
        assert!(fit.capped_by_model);
        assert_eq!(fit.projected_mib, 11_382);
    }

    #[tokio::test]
    async fn a_model_that_does_not_fit_at_all_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FIT_PARAMS_BINARY), b"x").unwrap();
        let runner = FakeRunner::default().with_ok_for_all("ROCm0 20000 500 266 \n");
        let err = find(
            &runner,
            dir.path(),
            &cfg(),
            KvCacheType::F16,
            1,
            14_000,
            65_536,
            ZERO,
            &[],
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("not even"), "{err}");
    }
}
