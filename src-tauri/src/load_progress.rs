//! Model loading progress. llama-server prints no percentage while it loads weights, so the
//! progress is the stage it has reached, read from its own log lines. Nothing is estimated:
//! the bar advances only when the log shows the next stage.

use serde::Serialize;

/// The stages of a load, in order. `Benchmark` is the app's own step after the server is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStage {
    Starting,
    CheckingFit,
    LoadingWeights,
    BuildingContext,
    WarmingUp,
    Benchmark,
}

pub const STAGE_COUNT: usize = 6;

impl LoadStage {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn label(self) -> &'static str {
        match self {
            LoadStage::Starting => "Starting the inference server",
            LoadStage::CheckingFit => "Checking that the model fits in memory",
            LoadStage::LoadingWeights => "Loading the model weights",
            LoadStage::BuildingContext => "Building the context and KV cache",
            LoadStage::WarmingUp => "Warming up",
            LoadStage::Benchmark => "Measuring speed",
        }
    }
}

/// What the UI shows during a load. `active` is false when no load is running.
#[derive(Debug, Clone, Serialize)]
pub struct LoadProgress {
    pub active: bool,
    pub stage: LoadStage,
    /// 1-based position of the stage, out of `stage_count`.
    pub step: usize,
    pub stage_count: usize,
    pub label: &'static str,
    pub elapsed_ms: u64,
}

impl LoadProgress {
    pub fn idle() -> Self {
        Self::at(false, LoadStage::Starting, 0)
    }

    pub fn at(active: bool, stage: LoadStage, elapsed_ms: u64) -> Self {
        Self {
            active,
            stage,
            step: stage.index() + 1,
            stage_count: STAGE_COUNT,
            label: stage.label(),
            elapsed_ms,
        }
    }
}

/// The furthest stage the log shows for the load that started last. `lines` are oldest first.
/// Lines from earlier loads are ignored: only those after the last `loading model` line count.
pub fn stage_from_log<'a>(lines: impl IntoIterator<Item = &'a str>) -> LoadStage {
    let mut stage = LoadStage::Starting;
    for line in lines {
        if line.contains("load_model: loading model") {
            stage = LoadStage::Starting;
        } else if line.contains("fitting params to device memory") {
            stage = stage.max(LoadStage::CheckingFit);
        } else if line.contains("load_tensors: loading model tensors") {
            stage = stage.max(LoadStage::LoadingWeights);
        } else if line.contains("llama_context: constructing")
            || line.contains("offloaded ") && line.contains("layers to GPU")
        {
            stage = stage.max(LoadStage::BuildingContext);
        } else if line.contains("warming up the model") {
            stage = stage.max(LoadStage::WarmingUp);
        }
    }
    stage
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOAD: &[&str] = &[
        "srv    load_model: loading model 'm.gguf'",
        "cmn  common_fit_a: fitting params to device memory ...",
        "llama_model_loader: loaded meta data with 50 key-value pairs",
        "load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
        "load_tensors: offloaded 66/66 layers to GPU",
        "llama_context: constructing llama_context",
        "cmn  common_init_: warming up the model with an empty run - please wait ...",
        "srv  server_on_li: listening on http://127.0.0.1:8080",
    ];

    #[test]
    fn the_stage_follows_the_log_and_never_goes_backwards() {
        assert_eq!(stage_from_log([]), LoadStage::Starting);
        assert_eq!(
            stage_from_log(LOAD[..1].iter().copied()),
            LoadStage::Starting
        );
        assert_eq!(
            stage_from_log(LOAD[..2].iter().copied()),
            LoadStage::CheckingFit
        );
        assert_eq!(
            stage_from_log(LOAD[..4].iter().copied()),
            LoadStage::LoadingWeights
        );
        assert_eq!(
            stage_from_log(LOAD[..6].iter().copied()),
            LoadStage::BuildingContext
        );
        assert_eq!(stage_from_log(LOAD.iter().copied()), LoadStage::WarmingUp);
    }

    #[test]
    fn an_earlier_load_in_the_log_does_not_count() {
        let mut lines: Vec<&str> = LOAD.to_vec();
        lines.push("srv    load_model: loading model 'second.gguf'");
        assert_eq!(stage_from_log(lines), LoadStage::Starting);
    }

    #[test]
    fn steps_are_one_based_and_labelled() {
        let p = LoadProgress::at(true, LoadStage::Benchmark, 1500);
        assert_eq!((p.step, p.stage_count), (6, STAGE_COUNT));
        assert!(p.active && !p.label.is_empty());
        assert!(!LoadProgress::idle().active);
    }
}
