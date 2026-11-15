//! Performance memory commands. Records come from the benchmark that runs after each model
//! load, and from `perf_run_benchmark`. Nothing here changes a stored record except through
//! the store's own rules (an unstable run never replaces a stable one).

use crate::perf::{BenchmarkOutcome, PerfComparison, PerfEntry};
use crate::state::AppState;
use raptor_core::Result;
use tauri::State;

/// Stored benchmark records, newest first. `model` (a library id) limits them to one model.
#[tauri::command]
pub async fn perf_history(
    state: State<'_, AppState>,
    model: Option<String>,
) -> Result<Vec<PerfEntry>> {
    state.perf_history(model.as_deref())
}

/// Compares two records by id. `a` is the baseline; deltas are in percent, and a negative
/// `ttft_delta_pct` means the candidate answers faster.
#[tauri::command]
pub async fn perf_compare(
    state: State<'_, AppState>,
    a: String,
    b: String,
) -> Result<PerfComparison> {
    state.perf_compare(&a, &b)
}

/// Runs the fixed benchmark again on the loaded model and records the result.
#[tauri::command]
pub async fn perf_run_benchmark(state: State<'_, AppState>) -> Result<BenchmarkOutcome> {
    state.perf_run_benchmark().await
}

/// The fastest stable record for `model` on this machine with the installed runtime, or null.
#[tauri::command]
pub async fn perf_best_known(
    state: State<'_, AppState>,
    model: String,
) -> Result<Option<PerfEntry>> {
    state.perf_best_known(&model).await
}
