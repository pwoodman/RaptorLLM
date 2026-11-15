//! raptor-models: Hugging Face discovery, GGUF metadata, quant guidance, pinned verified
//! downloads (single files and split shard sets), the local model library, model comparison
//! and per-model presets. See docs/specs/models.md.
//!
//! Depends only on `raptor-core`. Network and disk-space effects sit behind traits
//! ([`hf::HfApi`], [`download::HttpFetch`], [`download::DiskSpace`]) so every test runs offline.

pub mod arch;
pub mod compare;
pub mod download;
pub mod gguf;
pub mod hf;
pub mod library;
mod manager;
pub mod presets;
pub mod quant;
pub mod shard;
mod util;

#[cfg(test)]
mod testing;

pub use manager::{ModelManager, UpdateCheck, UpdateStatus};

pub use arch::{runtime_hints, ArchHints};
pub use compare::{Comparison, ComparisonRow, ModelProfile, RankedCandidate, Task, TaskTag};
pub use download::{
    DownloadConfig, DownloadHandle, DownloadManager, DownloadOutcome, DownloadProgress,
    DownloadRequest, JobState, Provenance, RecoveryReport,
};
pub use gguf::{ArchInfo, ArchSupport, GgufMeta};
pub use hf::{HfModelSummary, HfRepoInfo, HubRateLimit, RepoFile, RepoSnapshot, SearchSort};
pub use library::{
    Library, ModelDetails, ModelEntry, ModelFile, ModelSource, ScanItem, ScanStatus, StorageUsage,
};
pub use presets::{Preset, PresetStore};
pub use quant::{CompatRating, HardwareSummary, QuantOption, QuantRecommendation};
pub use shard::{ShardName, ShardSet};
