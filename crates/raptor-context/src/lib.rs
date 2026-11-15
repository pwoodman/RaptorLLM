//! raptor-context: the Context Intelligence Engine. See `docs/specs/context.md` and this
//! crate's README for the public API and status.
//!
//! Pipeline: `scan` (walk, filter, hash) -> `parse` (symbols, imports, calls) -> `index`
//! (chunks, incremental update, store) -> `graph` (import / call / test edges) ->
//! `retrieve` (hybrid ranking) -> `pack` + `budget` (protected-first packing) ->
//! `provenance` (manifest). `api::ContextEngine` is the facade over all of it.

pub mod api;
pub mod budget;
pub mod git;
pub mod graph;
pub mod index;
pub mod memory;
pub mod pack;
pub mod parse;
mod parse_ts;
pub mod postings;
pub mod provenance;
pub mod recovery;
pub mod retrieve;
pub mod scan;
pub mod summary;
pub mod text;
pub mod vector;

pub use api::{
    ContextEngine, ContextRequest, ContextResponse, DiffInput, EngineDeps, ImpactReport,
    IndexStatus, LanguageExtractor, RecoverRequest, RecoverResponse, RecoveredItem, SearchRequest,
    SemanticStatus, SummaryOutcome, SymbolHit, SymbolQuery,
};
pub use budget::{Category, TaskKind};
pub use git::{GitCli, GitProbe, GitState, NoGit};
pub use index::UpdateReport;
pub use memory::{Added, MemoryEntry, MemoryKind, ProjectMemory, Summarizer};
pub use parse::{Backend, Language};
pub use provenance::{CategoryUsage, ContextManifest, ExcludedItem, ManifestItem};
pub use retrieve::{Embedder, Hit, NoEmbedder};
pub use summary::{ModelSummarizer, TextGenerator};
pub use vector::{EmbeddingBackend, HashingEmbedder, VectorIndex};

#[cfg(test)]
mod resolve_tests;

#[cfg(test)]
mod task_tests;

#[cfg(test)]
mod status_tests;

#[cfg(test)]
mod semantic_tests;
