//! llama.cpp's own memory projection, used to calibrate `config::recommend`.
//!
//! llama.cpp b12340 projects device memory itself. Two places show it:
//!
//! - the server log at load time: `common_memory_breakdown_print` rows of the form
//!   `| - ROCm0 (RX 9060 XT) | 16304 = 16146 + (11617 = 10616 + 854 + 146) + -11459 |`
//!   (total = free + (model + context + compute) + unaccounted, all MiB);
//! - `llama-fit-params --fit-print on`, a dry run that prints the same breakdown on
//!   stdout as `ROCm0 <model> <context> <compute>` without serving. Measured on the
//!   test machine: about 0.6 s, no server started.
//!
//! The probe is optional. It runs with every layer on the GPU at the requested
//! context, so the figures are a full-offload projection. The server is never
//! given `--fit` decisions: the launch keeps the explicit `-ngl` and `-c`.

use crate::config::LaunchConfig;
use crate::sys::CommandRunner;
use raptor_core::{RaptorError, Result};
use serde::Serialize;
use std::path::Path;
use std::time::Duration;

/// Slots the server uses by default (`-np` auto resolves to 4 on b12340, see the
/// gate fixture log). The recurrent-state buffer scales with the slot count, so
/// the probe passes the same value to get the same KV and recurrent figures.
pub const SERVER_DEFAULT_SLOTS: u32 = 4;

/// Name of the dry-run tool shipped next to `llama-server` in a llama.cpp build.
#[cfg(windows)]
pub const FIT_PARAMS_BINARY: &str = "llama-fit-params.exe";
/// Name of the dry-run tool shipped next to `llama-server` in a llama.cpp build.
#[cfg(not(windows))]
pub const FIT_PARAMS_BINARY: &str = "llama-fit-params";

const PROBE_TIMEOUT: Duration = Duration::from_secs(120);

/// Device memory llama.cpp projects for a full-offload launch at `ctx` tokens, MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FitProjection {
    /// Context size the projection was computed for.
    pub ctx: u32,
    /// Model weights on the device.
    pub model_mib: u64,
    /// KV cache plus recurrent state on the device.
    pub context_mib: u64,
    /// Compute buffers on the device.
    pub compute_mib: u64,
}

impl FitProjection {
    /// Total projected device use (model + context + compute), MiB.
    pub fn total_mib(&self) -> u64 {
        self.model_mib + self.context_mib + self.compute_mib
    }
}

/// Parses a projection from llama.cpp output. Accepts the compact `--fit-print`
/// rows (`ROCm0 <model> <context> <compute>`) and the server's log table rows
/// (`| - ROCm0 (..) | total = free + (model = ... ) ... |`). `ctx` is the context
/// the projection was computed for, which the caller knows. Returns the first GPU
/// device row. Host and CPU rows are skipped. `None` when no row parses.
pub fn parse_fit_projection(text: &str, ctx: u32) -> Option<FitProjection> {
    text.lines()
        .find_map(|l| parse_compact_row(l).or_else(|| parse_log_row(l)))
        .map(|(model, context, compute)| FitProjection {
            ctx,
            model_mib: model,
            context_mib: context,
            compute_mib: compute,
        })
}

fn is_gpu_device(name: &str) -> bool {
    !name.is_empty() && !name.starts_with("Host") && !name.starts_with("CPU")
}

/// `ROCm0 10616 1110 527`
fn parse_compact_row(line: &str) -> Option<(u64, u64, u64)> {
    let mut it = line.split_whitespace();
    let name = it.next()?;
    let vals: Vec<&str> = it.collect();
    if vals.len() != 3 || !is_gpu_device(name) {
        return None;
    }
    let nums: Option<Vec<u64>> = vals.iter().map(|v| v.parse().ok()).collect();
    let n = nums?;
    Some((n[0], n[1], n[2]))
}

/// `| - ROCm0 (RX 9060 XT) | 16304 = 16146 + (11617 = 10616 + 854 + 146) + -11459 |`
fn parse_log_row(line: &str) -> Option<(u64, u64, u64)> {
    let cells: Vec<&str> = line.split('|').collect();
    if cells.len() < 3 {
        return None;
    }
    let name = cells[1].trim().trim_start_matches('-').trim();
    let name = name.split_whitespace().next().unwrap_or("");
    let value = cells[2];
    if !is_gpu_device(name) || !value.contains('(') || !value.contains(" = ") {
        return None;
    }
    let nums: Vec<i64> = value
        .split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '+' | '='))
        .filter_map(|t| t.parse::<i64>().ok())
        .collect();
    // total, free, projected, model, context, compute, unaccounted
    if nums.len() != 7 {
        return None;
    }
    let (proj, model, context, compute) = (nums[2], nums[3], nums[4], nums[5]);
    // llama rounds the projected total, so allow a couple of MiB of drift.
    if (proj - (model + context + compute)).abs() > 2 {
        return None;
    }
    if [model, context, compute].iter().any(|v| *v < 0) {
        return None;
    }
    Some((model as u64, context as u64, compute as u64))
}

/// Arguments for the dry run: the launch's model, batch, flash attention and
/// context, with every layer on the GPU and the server's default slot count.
/// Caches stay F16 so the projection is the largest KV case; quantized KV needs
/// less, so calibrating against F16 is conservative.
pub fn probe_args(cfg: &LaunchConfig) -> Vec<String> {
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
        "-np".into(),
        SERVER_DEFAULT_SLOTS.to_string(),
        "--fit-print".into(),
        "on".into(),
    ]
}

/// Runs `llama-fit-params` for `cfg` and parses its projection. `fit_binary` is
/// the dry-run tool in the same folder as the selected `llama-server`. Errors when
/// the tool is missing, fails, or prints no GPU row, so the caller can say so.
pub async fn probe(
    runner: &dyn CommandRunner,
    fit_binary: &Path,
    cfg: &LaunchConfig,
    env: &[(String, String)],
) -> Result<FitProjection> {
    if !fit_binary.is_file() {
        return Err(RaptorError::Unsupported(format!(
            "{} is not in this runtime, so llama.cpp's memory projection is unavailable",
            fit_binary.display()
        )));
    }
    let program = fit_binary.to_string_lossy().into_owned();
    let out = runner
        .run_with_env(&program, &probe_args(cfg), env, PROBE_TIMEOUT)
        .await?;
    parse_fit_projection(&out, cfg.ctx_size).ok_or_else(|| {
        RaptorError::Runtime(format!(
            "{program} printed no GPU memory row; projection not used"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real llama-fit-params stdout, RX 9060 XT, Qwen3.8-27B-UD-IQ3_S, ctx 8192, 4 slots.
    const COMPACT_8192: &str = "ROCm0 10616 1110 527 \nHost 520 0 42 \n";

    /// Real server log row, b12340, ctx 4096, 4 slots (fixture real_b12340_hip_gpu_lv4.log).
    const LOG_4096: &str = "0.00.640.691 I common_memory_breakdown_print: |   - ROCm0 (RX 9060 XT) | 16304 = 16146 + (11617 = 10616 +     854 +     146) +      -11459 |\n\
0.00.640.692 I common_memory_breakdown_print: |   - Host               |                    565 =   520 +       0 +      44                |\n";

    #[test]
    fn parses_compact_fit_print_rows_and_skips_host() {
        let p = parse_fit_projection(COMPACT_8192, 8192).unwrap();
        assert_eq!(
            p,
            FitProjection {
                ctx: 8192,
                model_mib: 10616,
                context_mib: 1110,
                compute_mib: 527
            }
        );
        assert_eq!(p.total_mib(), 12253);
    }

    #[test]
    fn parses_server_log_breakdown_row() {
        let p = parse_fit_projection(LOG_4096, 4096).unwrap();
        assert_eq!(p.model_mib, 10616);
        assert_eq!(p.context_mib, 854);
        assert_eq!(p.compute_mib, 146);
        assert_eq!(p.total_mib(), 11616);
    }

    #[test]
    fn rejects_text_without_a_gpu_row() {
        assert!(parse_fit_projection("Host 520 0 42\nno devices\n", 4096).is_none());
        assert!(parse_fit_projection("", 4096).is_none());
    }

    #[test]
    fn rejects_inconsistent_log_row() {
        let bad =
            "| - ROCm0 (RX) | 16304 = 16146 + (11617 = 1000 +     854 +     146) +      -11459 |";
        assert!(parse_fit_projection(bad, 4096).is_none());
    }

    #[test]
    fn probe_args_mirror_launch_and_force_full_offload() {
        let cfg = LaunchConfig::builder("D:/m/a.gguf", crate::manifest::Backend::Hip)
            .ctx_size(8192)
            .kv_cache(crate::config::KvCacheType::Q8_0)
            .build()
            .unwrap();
        let args = probe_args(&cfg);
        let pairs: Vec<(&str, &str)> = args
            .chunks(2)
            .map(|c| (c[0].as_str(), c[1].as_str()))
            .collect();
        assert!(pairs.contains(&("-ngl", "999")));
        assert!(pairs.contains(&("-c", "8192")));
        assert!(pairs.contains(&("-np", "4")));
        assert!(pairs.contains(&("--fit-print", "on")));
        // Calibration is F16: the probe never passes cache types.
        assert!(!args.iter().any(|a| a.starts_with("--cache-type")));
    }
}
