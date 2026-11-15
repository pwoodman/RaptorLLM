//! `ContextEngine`: the facade the gateway and MCP tools call.
//!
//! Every request and response is serde-serialisable and independent of any agent
//! harness. Each public method corresponds to one tool: `build_context`, `search`,
//! `symbols`, `impact`, `recover`, `index_status`, plus the memory operations.

use crate::budget::{available_tokens, Category, TaskKind};
use crate::git::{GitProbe, GitState};
use crate::index::{FileEntry, Index, UpdateReport, INDEX_VERSION};
use crate::memory::{
    detect_loops, extract_decisions, looks_like_error, mentioned_sources, Added, MemoryKind,
    MemoryStore, ProjectMemory, SourceRef, Summarizer, PROJECT_MEMORY_VERSION,
};
use crate::pack::{assemble_messages, pack, Candidate};
use crate::parse::{default_extractors, Backend, Language, SymbolKind};
use crate::provenance::{ContextManifest, ExcludedItem, LineRange, ManifestItem};
use crate::recovery::{is_safe_relative, make_handle, parse_handle, stale_error};
use crate::retrieve::{recency_ranking, search, Embedder, Hit, NoEmbedder, SearchParams};
use crate::scan::sha256_hex;
use crate::summary::failed_attempts;
use crate::text::term_set;
use crate::vector::{EmbeddingBackend, VectorIndex, DEFAULT_MAX_VECTORS, VECTOR_VERSION};
use raptor_core::{
    ChatMessage, DataDirs, EstimatingCounter, RaptorError, Result, Role, TokenCounter,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A diff the caller wants treated as active, protected context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffInput {
    pub path: String,
    pub patch: String,
}

/// Input to [`ContextEngine::build_context`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextRequest {
    /// The active task, in the caller's words.
    pub task: String,
    pub task_kind: TaskKind,
    /// System prompt or harness instructions. Protected.
    pub instructions: Option<String>,
    /// The conversation so far. System messages become instructions; the rest is history.
    pub messages: Vec<ChatMessage>,
    /// Files being edited, included in full. Protected.
    pub active_files: Vec<String>,
    pub diffs: Vec<DiffInput>,
    /// Failing test or error output the caller already has. Protected.
    pub errors: Vec<String>,
    /// Files the caller wants favoured in retrieval.
    pub pinned_files: Vec<String>,
    /// Session whose memory is loaded, invalidated and included.
    pub session_id: Option<String>,
    /// Model context window, in tokens.
    pub model_ctx: usize,
    /// Tokens reserved for the model's reply.
    pub reserved_output: usize,
    /// Upper bound on retrieved code chunks.
    pub max_retrieved: usize,
}

impl Default for ContextRequest {
    fn default() -> Self {
        Self {
            task: String::new(),
            task_kind: TaskKind::default(),
            instructions: None,
            messages: Vec::new(),
            active_files: Vec::new(),
            diffs: Vec::new(),
            errors: Vec::new(),
            pinned_files: Vec::new(),
            session_id: None,
            model_ctx: 4096,
            reserved_output: 512,
            max_retrieved: 12,
        }
    }
}

/// Output of [`ContextEngine::build_context`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResponse {
    /// Messages to send to the model: one system message with the context, then kept history.
    pub messages: Vec<ChatMessage>,
    pub manifest: ContextManifest,
}

/// Input to [`ContextEngine::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchRequest {
    pub query: String,
    pub task_kind: TaskKind,
    pub limit: usize,
    pub active_files: Vec<String>,
    pub pinned_files: Vec<String>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            task_kind: TaskKind::default(),
            limit: 10,
            active_files: Vec::new(),
            pinned_files: Vec::new(),
        }
    }
}

/// Input to [`ContextEngine::symbols`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SymbolQuery {
    /// Case-insensitive substring of the symbol name. Empty matches all.
    pub name_contains: String,
    pub kind: Option<SymbolKind>,
    pub limit: usize,
}

impl Default for SymbolQuery {
    fn default() -> Self {
        Self {
            name_contains: String::new(),
            kind: None,
            limit: 50,
        }
    }
}

/// One declaration found by [`ContextEngine::symbols`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolHit {
    pub path: String,
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: u32,
    pub end_line: u32,
    pub parent: Option<String>,
}

/// Downstream and upstream relations of one file, from [`ContextEngine::impact`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactReport {
    pub file: String,
    pub imports: Vec<String>,
    pub dependents: Vec<String>,
    /// Files that depend on this one transitively (up to three hops), with distance.
    pub transitive_dependents: BTreeMap<String, usize>,
    pub related_tests: Vec<String>,
}

/// Summary of the index and git state, from [`ContextEngine::index_status`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStatus {
    pub root: String,
    pub index_version: u32,
    pub files: usize,
    pub chunks: usize,
    pub symbols: usize,
    pub imports: usize,
    pub excluded: usize,
    pub graph_files: usize,
    pub graph_edges: usize,
    pub git_repo: bool,
    pub changed_files: usize,
    pub git_warning: Option<String>,
    /// Per-language extraction backend and how many indexed files fell back to regex.
    #[serde(default)]
    pub extractors: Vec<LanguageExtractor>,
    /// Semantic retrieval: whether it is configured, and whether it is degraded.
    #[serde(default)]
    pub semantic: SemanticStatus,
    /// Entries in this workspace's long-term project memory.
    #[serde(default)]
    pub project_memory_entries: usize,
    /// Project memory entries that are stale and not served.
    #[serde(default)]
    pub project_memory_stale: usize,
}

/// State of semantic retrieval, from [`IndexStatus::semantic`].
///
/// Semantic retrieval never fails a build: when the backend is down, the errors appear here
/// and as manifest warnings, and retrieval continues on the lexical and graph signals.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticStatus {
    /// True when an [`EmbeddingBackend`] is configured.
    pub enabled: bool,
    pub model_id: Option<String>,
    pub dimension: usize,
    /// Vectors held in the store.
    pub vectors: usize,
    /// Store capacity in vectors.
    pub capacity: usize,
    /// Indexed chunks with no usable vector (capacity reached, backend failure, or pending).
    pub chunks_without_vector: usize,
    /// The most recent backend error, from indexing or from embedding a query.
    pub last_error: Option<String>,
    /// Why the stored vectors were discarded at open (other model, or corrupt), if they were.
    pub store_notice: Option<String>,
}

/// Result of [`ContextEngine::summarize_session_report`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryOutcome {
    /// Ids of the entries recorded, summary first when there is one.
    pub ids: Vec<String>,
    /// Problems the caller should show. Empty when everything was recorded.
    pub warnings: Vec<String>,
}

/// Extraction status of one language, from [`IndexStatus::extractors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageExtractor {
    pub language: Language,
    /// `tree-sitter` when a grammar is bundled, otherwise `regex`.
    pub extractor: Backend,
    /// Indexed files of this language.
    pub files: usize,
    /// Files that should have used tree-sitter but fell back to regex (syntax errors).
    pub regex_fallback_files: usize,
}

/// Input to [`ContextEngine::recover`]: a handle to redeem, or a query to search with.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecoverRequest {
    pub handle: Option<String>,
    pub query: Option<String>,
    pub task_kind: TaskKind,
    /// Tokens the query results may use.
    pub extra_budget: usize,
}

/// A chunk returned by [`ContextEngine::recover`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveredItem {
    pub handle: String,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    pub tokens: usize,
}

/// Output of [`ContextEngine::recover`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoverResponse {
    pub items: Vec<RecoveredItem>,
    pub tokens_used: usize,
    pub warnings: Vec<String>,
}

/// Collaborators the engine is built from. Tests inject fakes; production uses the defaults.
pub struct EngineDeps {
    pub counter: Arc<dyn TokenCounter>,
    pub git: Box<dyn GitProbe>,
    pub embedder: Arc<dyn Embedder>,
    pub summarizer: Option<Box<dyn Summarizer>>,
}

impl EngineDeps {
    /// Deps with the given counter, the `git` CLI probe, no embedder and no summarizer.
    pub fn new(counter: Arc<dyn TokenCounter>) -> Self {
        Self {
            counter,
            git: Box::new(crate::git::GitCli),
            embedder: Arc::new(NoEmbedder),
            summarizer: None,
        }
    }

    /// Deps using the conservative estimating counter (no model tokenizer).
    pub fn estimating() -> Self {
        Self::new(Arc::new(EstimatingCounter))
    }
}

/// Collected inputs for one build, before packing.
#[derive(Default)]
struct Acc {
    cands: Vec<Candidate>,
    excluded: Vec<ExcludedItem>,
    warnings: Vec<String>,
}

/// Semantic errors kept between calls. Indexing and query embedding are reported separately.
#[derive(Debug, Default)]
struct SemanticErrors {
    sync: Option<String>,
    query: Option<String>,
}

/// The context engine for one workspace.
pub struct ContextEngine {
    store_dir: PathBuf,
    index: Index,
    counter: Arc<dyn TokenCounter>,
    git: Box<dyn GitProbe>,
    embedder: Arc<dyn Embedder>,
    summarizer: Option<Box<dyn Summarizer>>,
    git_state: GitState,
    git_warning: Option<String>,
    backend: Option<Arc<dyn EmbeddingBackend>>,
    vectors: VectorIndex,
    vector_path: PathBuf,
    semantic_errors: Mutex<SemanticErrors>,
    project: ProjectMemory,
    project_path: PathBuf,
}

/// Text of each chunk of an indexed file, in chunk order. `None` when the file no longer
/// matches the hash it was indexed with (it changed after the scan); the next update covers it.
fn chunk_texts(root: &Path, entry: &FileEntry) -> Result<Option<Vec<String>>> {
    let text = std::fs::read_to_string(root.join(&entry.record.path))?;
    if sha256_hex(text.as_bytes()) != entry.record.hash {
        return Ok(None);
    }
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    Ok(Some(
        entry
            .chunks
            .iter()
            .map(|c| slice_lines(&lines, c.start_line, c.end_line))
            .collect(),
    ))
}

fn read_lines(root: &Path, path: &str) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(root.join(path))?;
    Ok(text.lines().map(str::to_string).collect())
}

/// Lines `start..=end` (1-based, clamped) joined with newlines.
fn slice_lines(lines: &[String], start: u32, end: u32) -> String {
    let s = (start.max(1) as usize - 1).min(lines.len());
    let e = (end as usize).min(lines.len()).max(s);
    lines[s..e].join("\n")
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn kind_name(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Decision => "decision",
        MemoryKind::Convention => "convention",
        MemoryKind::Summary => "summary",
        MemoryKind::CompletedWork => "completed work",
        MemoryKind::FailedAttempt => "failed attempt",
    }
}

fn overlap_count(text: &str, qterms: &[String]) -> usize {
    let here = term_set(text);
    here.iter()
        .filter(|t| qterms.iter().any(|q| q == *t))
        .count()
}

impl ContextEngine {
    /// Opens the engine for `workspace`, loading any persisted index for it.
    pub fn open(data: &DataDirs, workspace: &Path, deps: EngineDeps) -> Result<Self> {
        let root = std::path::absolute(workspace)?;
        if !root.is_dir() {
            return Err(RaptorError::InvalidInput(format!(
                "workspace is not a directory: {}",
                root.display()
            )));
        }
        let key = hex::encode(Sha256::digest(root.display().to_string().as_bytes()));
        let store_dir = data.context_index().join(&key[..16]);
        let store = store_dir.join(format!("index-v{INDEX_VERSION}.json"));
        let index = Index::open(
            &root,
            Some(store),
            default_extractors()?,
            Arc::clone(&deps.embedder),
        )?;
        let vector_path = store_dir.join(format!("vectors-v{VECTOR_VERSION}.bin"));
        let project_path = store_dir.join(format!("project-memory-v{PROJECT_MEMORY_VERSION}.json"));
        let project = ProjectMemory::load_or_new(&project_path)?;
        let mut engine = Self {
            store_dir,
            index,
            counter: deps.counter,
            git: deps.git,
            embedder: deps.embedder,
            summarizer: deps.summarizer,
            git_state: GitState::default(),
            git_warning: None,
            backend: None,
            vectors: VectorIndex::empty("", 0, DEFAULT_MAX_VECTORS),
            vector_path,
            semantic_errors: Mutex::new(SemanticErrors::default()),
            project,
            project_path,
        };
        engine.refresh_git();
        Ok(engine)
    }

    /// Configures semantic retrieval with `backend`, or turns it off with `None`.
    ///
    /// Loads the vector store for the backend's model (discarding one built for another
    /// model), then re-scans so that missing vectors are computed. A backend that fails does
    /// not fail this call: the failure is reported in [`IndexStatus::semantic`].
    pub fn set_embedding_backend(
        &mut self,
        backend: Option<Arc<dyn EmbeddingBackend>>,
    ) -> Result<()> {
        match &backend {
            Some(b) => {
                if b.dimension() == 0 {
                    return Err(RaptorError::InvalidInput(
                        "embedding dimension must be greater than zero".into(),
                    ));
                }
                self.vectors = VectorIndex::open(
                    &self.vector_path,
                    b.model_id(),
                    b.dimension(),
                    DEFAULT_MAX_VECTORS,
                );
            }
            None => self.vectors = VectorIndex::empty("", 0, DEFAULT_MAX_VECTORS),
        }
        self.backend = backend;
        if let Ok(mut errors) = self.semantic_errors.lock() {
            *errors = SemanticErrors::default();
        }
        self.update()?;
        Ok(())
    }

    fn refresh_git(&mut self) {
        match self.git.state(self.index.root()) {
            Ok(state) => {
                self.git_state = state;
                self.git_warning = None;
            }
            Err(e) => {
                self.git_state = GitState::default();
                self.git_warning = Some(format!(
                    "git state unavailable, recency uses file times only: {e}"
                ));
            }
        }
    }

    /// Re-scans the workspace and saves the index if anything was parsed or removed.
    ///
    /// Files whose size and mtime are unchanged are not read (see [`UpdateReport::stat_skipped`]).
    /// Use [`ContextEngine::update_full`] to read and hash every file.
    pub fn update(&mut self) -> Result<UpdateReport> {
        self.refresh(true)
    }

    /// Like [`ContextEngine::update`], but reads and hashes every file. This catches a content
    /// change that kept the file's size and mtime, which the fast path cannot see.
    pub fn update_full(&mut self) -> Result<UpdateReport> {
        self.refresh(false)
    }

    fn refresh(&mut self, trust_stat: bool) -> Result<UpdateReport> {
        let report = self.index.update_with(trust_stat)?;
        if !report.reparsed.is_empty() || !report.removed.is_empty() {
            self.index.save()?;
        }
        self.sync_vectors(&report.removed);
        self.refresh_project_memory()?;
        self.refresh_git();
        Ok(report)
    }

    /// Embeds the chunks that have no vector yet and persists the store. Never returns an
    /// error: a backend or disk failure is recorded in the semantic status instead.
    fn sync_vectors(&mut self, removed: &[String]) {
        let Some(backend) = self.backend.clone() else {
            return;
        };
        for path in removed {
            self.vectors.remove_file(path);
        }
        let root = self.index.root().to_path_buf();
        let todo: Vec<String> = self
            .index
            .files()
            .iter()
            .filter(|(p, f)| self.vectors.needs_file(p, &f.record.hash, f.chunks.len()))
            .map(|(p, _)| p.clone())
            .collect();
        let mut error: Option<String> = None;
        for path in &todo {
            let Some(entry) = self.index.file(path) else {
                continue;
            };
            match chunk_texts(&root, entry) {
                Ok(Some(texts)) => {
                    let stats =
                        self.vectors
                            .sync_file(backend.as_ref(), path, &entry.record.hash, &texts);
                    if let Some(e) = stats.error {
                        error.get_or_insert(format!("{path}: {e}"));
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    error.get_or_insert(format!("{path}: could not read for embedding: {e}"));
                }
            }
        }
        self.vectors.prune();
        if let Err(e) = self.vectors.save(&self.vector_path) {
            error.get_or_insert(format!("vector store not saved: {e}"));
        }
        if let Ok(mut errors) = self.semantic_errors.lock() {
            errors.sync = error;
        }
    }

    /// Marks project memory stale when a cited file changed, and saves when that changes anything.
    fn refresh_project_memory(&mut self) -> Result<()> {
        let has_sources = self.project.current().any(|e| !e.sources.is_empty());
        if has_sources
            && !self
                .project
                .invalidate_stale(&self.index.hashes())
                .is_empty()
        {
            self.project.save(&self.project_path)?;
        }
        Ok(())
    }

    /// Records a long-term project memory entry (decision, convention, summary, ...) that
    /// cites `paths`. The entry goes stale when one of those files changes. Decisions and
    /// conventions are never evicted, but are refused past the project's protected cap.
    pub fn remember_project(
        &mut self,
        kind: MemoryKind,
        text: &str,
        paths: &[String],
    ) -> Result<Added> {
        self.update()?;
        let hashes = self.index.hashes();
        let mut sources = Vec::with_capacity(paths.len());
        for p in paths {
            if !is_safe_relative(p) {
                return Err(RaptorError::InvalidInput(format!(
                    "unsafe source path: {p}"
                )));
            }
            sources.push(SourceRef {
                path: p.clone(),
                hash: hashes.get(p).cloned().unwrap_or_default(),
            });
        }
        let added = self.project.add(kind, text, sources)?;
        self.project.save(&self.project_path)?;
        Ok(added)
    }

    /// Every project memory entry, with stale ones flagged. The index is brought up to date
    /// first, so an entry whose cited file changed shows as stale now, not at the next build.
    pub fn project_memory(&mut self) -> Result<Vec<crate::memory::MemoryEntry>> {
        self.update()?;
        let hashes = self.index.hashes();
        self.project.invalidate_stale(&hashes);
        Ok(self.project.entries().to_vec())
    }

    /// Removes a project memory entry by id. This is how a decision or convention is retired.
    pub fn forget_project_memory(&mut self, id: &str) -> Result<()> {
        self.project.forget(id)?;
        self.project.save(&self.project_path)
    }

    fn set_query_error(&self, error: Option<String>) {
        if let Ok(mut errors) = self.semantic_errors.lock() {
            errors.query = error;
        }
    }

    fn semantic_status(&self) -> SemanticStatus {
        let Some(backend) = self.backend.as_ref() else {
            return SemanticStatus::default();
        };
        let last_error = match self.semantic_errors.lock() {
            Ok(e) => e.sync.clone().or_else(|| e.query.clone()),
            Err(_) => Some("semantic error state is unavailable".into()),
        };
        let missing = self.vectors.missing_chunks(
            self.index
                .files()
                .iter()
                .map(|(p, f)| (p.as_str(), f.record.hash.as_str(), f.chunks.len())),
        );
        SemanticStatus {
            enabled: true,
            model_id: Some(backend.model_id().to_string()),
            dimension: backend.dimension(),
            vectors: self.vectors.len(),
            capacity: self.vectors.capacity(),
            chunks_without_vector: missing,
            last_error,
            store_notice: self.vectors.notice().map(str::to_string),
        }
    }

    /// Warnings for the manifest when semantic retrieval is degraded.
    fn semantic_warnings(&self) -> Vec<String> {
        let s = self.semantic_status();
        if !s.enabled {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(e) = &s.last_error {
            out.push(format!(
                "semantic retrieval degraded: {e}; affected chunks rank on lexical and graph signals only"
            ));
        }
        if s.chunks_without_vector > 0 {
            out.push(format!(
                "{} chunk(s) have no embedding (store capacity {}); they rank without the semantic signal",
                s.chunks_without_vector, s.capacity
            ));
        }
        out
    }

    /// The persisted vectors, when a backend is configured.
    fn vector_store(&self) -> Option<&VectorIndex> {
        self.backend.as_ref().map(|_| &self.vectors)
    }

    /// Per language: the backend its extractor uses, and how many files fell back to regex.
    fn language_extractors(&self) -> Vec<LanguageExtractor> {
        self.index
            .extractor_backends()
            .into_iter()
            .map(|(language, declared)| {
                let mut files = 0usize;
                let mut regex_fallback_files = 0usize;
                for entry in self.index.files().values() {
                    if entry.record.language != language {
                        continue;
                    }
                    files += 1;
                    if declared == Backend::TreeSitter && entry.backend == Backend::Regex {
                        regex_fallback_files += 1;
                    }
                }
                LanguageExtractor {
                    language,
                    extractor: declared,
                    files,
                    regex_fallback_files,
                }
            })
            .collect()
    }

    /// Counts and state of the index.
    pub fn index_status(&self) -> IndexStatus {
        let graph = self.index.graph();
        IndexStatus {
            root: self.index.root().display().to_string(),
            index_version: INDEX_VERSION,
            files: self.index.files().len(),
            chunks: self.index.chunk_count(),
            symbols: self.index.symbol_count(),
            imports: self.index.import_count(),
            excluded: self.index.excluded().len(),
            graph_files: graph.file_count(),
            graph_edges: graph.edge_count(),
            git_repo: self.git_state.is_repo,
            changed_files: self.git_state.changed.len(),
            git_warning: self.git_warning.clone(),
            extractors: self.language_extractors(),
            semantic: self.semantic_status(),
            project_memory_entries: self.project.entries().len(),
            project_memory_stale: self.project.stale().count(),
        }
    }

    /// Hybrid search over the indexed code. Does not re-scan; call [`ContextEngine::update`] first if needed.
    ///
    /// If the query cannot be embedded, the search still runs on the lexical and graph signals.
    pub fn search(&self, req: &SearchRequest) -> Result<Vec<Hit>> {
        let active = safe_set(&req.active_files)?;
        let pinned = safe_set(&req.pinned_files)?;
        let recency = recency_ranking(&self.index, &self.git_state);
        let qvec = self.query_vector(&req.query);
        let params = SearchParams {
            query: &req.query,
            kind: req.task_kind,
            limit: req.limit.max(1),
            active: &active,
            pinned: &pinned,
            recency: &recency,
            query_vector: qvec.as_deref(),
            vectors: self.vector_store(),
        };
        Ok(search(&self.index, &params))
    }

    /// Embeds a query. Returns `None` when no embedder is configured or embedding failed; a
    /// failure is recorded in the semantic status and never propagated.
    fn query_vector(&self, text: &str) -> Option<Vec<f32>> {
        if text.trim().is_empty() {
            return None;
        }
        if let Some(backend) = self.backend.as_ref() {
            let outcome = backend.embed_batch(&[text]).and_then(|mut vs| {
                if vs.len() == 1 && vs[0].len() == backend.dimension() {
                    Ok(vs.remove(0))
                } else {
                    Err(RaptorError::Runtime(format!(
                        "backend returned an unexpected query vector (expected width {})",
                        backend.dimension()
                    )))
                }
            });
            return match outcome {
                Ok(v) => {
                    self.set_query_error(None);
                    Some(v)
                }
                Err(e) => {
                    self.set_query_error(Some(format!("query embedding failed: {e}")));
                    None
                }
            };
        }
        self.embedder.dimensions()?;
        match self.embedder.embed(text) {
            Ok(v) => Some(v),
            Err(e) => {
                self.set_query_error(Some(format!("query embedding failed: {e}")));
                None
            }
        }
    }

    /// Declarations whose names contain the query, exact matches first.
    pub fn symbols(&self, q: &SymbolQuery) -> Vec<SymbolHit> {
        let needle = q.name_contains.to_lowercase();
        let mut out: Vec<(bool, SymbolHit)> = Vec::new();
        for (path, entry) in self.index.files() {
            for s in &entry.facts.symbols {
                if q.kind.is_some_and(|k| k != s.kind) || !s.name.to_lowercase().contains(&needle) {
                    continue;
                }
                let exact = s.name.eq_ignore_ascii_case(&q.name_contains);
                out.push((
                    exact,
                    SymbolHit {
                        path: path.clone(),
                        name: s.name.clone(),
                        kind: s.kind,
                        start_line: s.start_line,
                        end_line: s.end_line,
                        parent: s.parent.clone(),
                    },
                ));
            }
        }
        out.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.path.cmp(&b.1.path))
                .then_with(|| a.1.start_line.cmp(&b.1.start_line))
        });
        out.into_iter()
            .take(q.limit.max(1))
            .map(|(_, h)| h)
            .collect()
    }

    /// Imports, dependents, transitive dependents and tests of one file.
    pub fn impact(&self, file: &str) -> Result<ImpactReport> {
        if self.index.file(file).is_none() {
            return Err(RaptorError::NotFound(format!("{file} is not in the index")));
        }
        let g = self.index.graph();
        Ok(ImpactReport {
            file: file.to_string(),
            imports: g.imports_of(file),
            dependents: g.dependents_of(file),
            transitive_dependents: g.transitive_dependents(file, 3),
            related_tests: g.related_tests(file),
        })
    }

    /// Redeems a handle, or searches for a query within `extra_budget` tokens.
    pub fn recover(&mut self, req: RecoverRequest) -> Result<RecoverResponse> {
        self.update()?;
        match (req.handle, req.query) {
            (Some(handle), _) => {
                let item = self.redeem(&handle)?;
                let tokens_used = item.tokens;
                Ok(RecoverResponse {
                    items: vec![item],
                    tokens_used,
                    warnings: Vec::new(),
                })
            }
            (None, Some(query)) => self.recover_query(&query, req.task_kind, req.extra_budget),
            (None, None) => Err(RaptorError::InvalidInput(
                "recover needs a handle or a query".into(),
            )),
        }
    }

    fn redeem(&self, handle: &str) -> Result<RecoveredItem> {
        let parsed = parse_handle(handle)?;
        let file = self
            .index
            .file(&parsed.path)
            .ok_or_else(|| RaptorError::NotFound(format!("{} is not in the index", parsed.path)))?;
        if !file.record.hash.starts_with(&parsed.hash_prefix) {
            return Err(stale_error(&parsed.path));
        }
        let lines = read_lines(self.index.root(), &parsed.path)?;
        let text = slice_lines(&lines, parsed.start, parsed.end);
        let tokens = self.counter.count(&text);
        Ok(RecoveredItem {
            handle: handle.to_string(),
            path: parsed.path,
            start_line: parsed.start,
            end_line: parsed.end,
            text,
            tokens,
        })
    }

    fn recover_query(
        &self,
        query: &str,
        kind: TaskKind,
        extra_budget: usize,
    ) -> Result<RecoverResponse> {
        let active = BTreeSet::new();
        let pinned = BTreeSet::new();
        let recency = recency_ranking(&self.index, &self.git_state);
        let qvec = self.query_vector(query);
        let params = SearchParams {
            query,
            kind,
            limit: 50,
            active: &active,
            pinned: &pinned,
            recency: &recency,
            query_vector: qvec.as_deref(),
            vectors: self.vector_store(),
        };
        let mut items = Vec::new();
        let mut used = 0usize;
        let mut skipped = 0usize;
        for hit in search(&self.index, &params) {
            let lines = read_lines(self.index.root(), &hit.path)?;
            let text = slice_lines(&lines, hit.start_line, hit.end_line);
            let tokens = self.counter.count(&text);
            if used + tokens > extra_budget {
                skipped += 1;
                continue;
            }
            used += tokens;
            items.push(RecoveredItem {
                handle: make_handle(&hit.path, hit.start_line, hit.end_line, &hit.hash),
                path: hit.path,
                start_line: hit.start_line,
                end_line: hit.end_line,
                text,
                tokens,
            });
        }
        let warnings = if skipped > 0 {
            vec![format!(
                "{skipped} matching chunk(s) did not fit the extra budget"
            )]
        } else {
            Vec::new()
        };
        Ok(RecoverResponse {
            items,
            tokens_used: used,
            warnings,
        })
    }

    /// Builds the context for one agent turn: a system message with the protected and
    /// retrieved context, plus the conversation history that fits, and the manifest.
    pub fn build_context(&mut self, req: ContextRequest) -> Result<ContextResponse> {
        let available = available_tokens(req.model_ctx, req.reserved_output)?;
        self.update()?;
        let mut acc = Acc::default();
        if let Some(w) = self.git_warning.clone() {
            acc.warnings.push(w);
        }
        let qtext = format!("{}\n{}", req.task, req.errors.join("\n"));
        let qterms: Vec<String> = term_set(&qtext).into_iter().collect();

        add_instructions(&req, &mut acc);
        add_task(&req, &mut acc);
        let active = self.add_active(&req.active_files, &mut acc)?;
        add_diffs(&req, &mut acc);
        let last_error = add_errors(&req, &mut acc);
        self.add_memory(req.session_id.as_deref(), &qterms, &mut acc)?;
        self.add_project_memory(&qterms, &mut acc);
        self.add_retrieved(&req, &active, &qtext, &qterms, &mut acc)?;
        self.add_graph(&active, &mut acc)?;
        add_history(&req, &qterms, last_error, &mut acc);
        acc.warnings.extend(detect_loops(&req.messages));
        acc.warnings.extend(self.semantic_warnings());

        let Acc {
            cands,
            excluded,
            mut warnings,
        } = acc;
        let packed = pack(
            cands,
            excluded,
            available,
            req.task_kind,
            self.counter.as_ref(),
        );
        if packed.over_budget_by > 0 {
            warnings.push(format!(
                "protected context exceeds the budget by {} tokens; nothing protected was dropped",
                packed.over_budget_by
            ));
        }
        let messages = assemble_messages(&packed.selected);
        let included = packed
            .selected
            .iter()
            .map(|s| ManifestItem {
                id: s.cand.id.clone(),
                source: s.cand.source.clone(),
                category: s.cand.category,
                hash: s.cand.hash.clone(),
                range: s.cand.range,
                tokens: s.tokens,
                score: s.cand.score,
                why: s.cand.why.clone(),
                truncated: s.truncated,
            })
            .collect();
        let manifest = ContextManifest {
            budget_tokens: available,
            used_tokens: packed.used,
            over_budget_by: packed.over_budget_by,
            tokens_exact: self.counter.is_exact(),
            included,
            excluded: packed.excluded,
            utilization_by_category: packed.utilization,
            warnings,
        };
        Ok(ContextResponse { messages, manifest })
    }

    /// Adds the active files in full. Returns the set of active paths that are indexed.
    fn add_active(&self, paths: &[String], acc: &mut Acc) -> Result<BTreeSet<String>> {
        let mut active = BTreeSet::new();
        for raw in paths {
            if !is_safe_relative(raw) {
                return Err(RaptorError::InvalidInput(format!(
                    "unsafe active file path: {raw}"
                )));
            }
            if let Some(entry) = self.index.file(raw) {
                let lines = read_lines(self.index.root(), raw)?;
                let mut c = Candidate::new(
                    format!("active:{raw}"),
                    Category::ActiveFiles,
                    true,
                    lines.join("\n"),
                );
                c.source = raw.clone();
                c.hash = Some(entry.record.hash.clone());
                c.range = Some(LineRange {
                    start: 1,
                    end: lines.len().max(1) as u32,
                });
                c.heading = format!("### {raw} (active file, full)");
                c.why = "file is being edited, so it is included in full".into();
                acc.cands.push(c);
                active.insert(raw.clone());
            } else if let Some(ex) = self.index.excluded().iter().find(|e| e.path == *raw) {
                let why = format!("active file not included: {}", ex.reason.describe());
                acc.warnings.push(format!("{raw}: {why}"));
                acc.excluded.push(ExcludedItem {
                    id: format!("active:{raw}"),
                    source: raw.clone(),
                    category: Category::ActiveFiles,
                    tokens: 0,
                    why,
                    recover_handle: None,
                });
            } else {
                acc.warnings.push(format!(
                    "active file {raw} is not in the index (missing from the workspace?)"
                ));
            }
        }
        Ok(active)
    }

    fn add_memory(&self, session: Option<&str>, qterms: &[String], acc: &mut Acc) -> Result<()> {
        let Some(session) = session else {
            return Ok(());
        };
        let path = self.memory_path(session)?;
        let mut store = MemoryStore::load_or_new(&path, session)?;
        if !store.invalidate_stale(&self.index.hashes()).is_empty() {
            store.save(&path)?;
        }
        let mut stale = 0usize;
        for entry in &store.entries {
            let id = format!("memory:{}", entry.id);
            if entry.stale {
                stale += 1;
                acc.excluded.push(ExcludedItem {
                    id,
                    source: format!("memory {}", entry.id),
                    category: Category::Memory,
                    tokens: self.counter.count(&entry.text),
                    why: format!(
                        "stale: {}; not served as current",
                        entry
                            .stale_reason
                            .clone()
                            .unwrap_or_else(|| "a source file changed".into())
                    ),
                    recover_handle: None,
                });
                continue;
            }
            let mut c = Candidate::new(
                id,
                Category::Memory,
                entry.kind.is_protected(),
                entry.text.clone(),
            );
            c.source = format!("memory {} ({})", entry.id, kind_name(entry.kind));
            c.score = overlap_count(&entry.text, qterms) as f64 + entry.seq as f64 * 1e-6;
            c.why = format!("{} recorded earlier in this session", kind_name(entry.kind));
            c.heading = format!("### {} ({})", kind_name(entry.kind), entry.id);
            acc.cands.push(c);
        }
        if stale > 0 {
            acc.warnings.push(format!(
                "{stale} memory entr{} stale because a source file changed; not served",
                if stale == 1 { "y is" } else { "ies are" }
            ));
        }
        Ok(())
    }

    /// Adds the workspace's long-term memory. Protected kinds are included in full; stale
    /// entries are reported as excluded and not served.
    fn add_project_memory(&self, qterms: &[String], acc: &mut Acc) {
        let mut stale = 0usize;
        for entry in self.project.entries() {
            let id = format!("project:{}", entry.id);
            if entry.stale {
                stale += 1;
                acc.excluded.push(ExcludedItem {
                    id,
                    source: format!("project memory {}", entry.id),
                    category: Category::Memory,
                    tokens: self.counter.count(&entry.text),
                    why: format!(
                        "stale: {}; not served as current",
                        entry
                            .stale_reason
                            .clone()
                            .unwrap_or_else(|| "a source file changed".into())
                    ),
                    recover_handle: None,
                });
                continue;
            }
            let mut c = Candidate::new(
                id,
                Category::Memory,
                entry.kind.is_protected(),
                entry.text.clone(),
            );
            c.source = format!("project memory {} ({})", entry.id, kind_name(entry.kind));
            c.score = overlap_count(&entry.text, qterms) as f64 + entry.seq as f64 * 1e-6;
            c.why = format!("{} recorded for this project", kind_name(entry.kind));
            c.heading = format!("### project {} ({})", kind_name(entry.kind), entry.id);
            acc.cands.push(c);
        }
        if stale > 0 {
            acc.warnings.push(format!(
                "{stale} project memory entr{} stale because a source file changed; not served",
                if stale == 1 { "y is" } else { "ies are" }
            ));
        }
    }

    fn add_retrieved(
        &self,
        req: &ContextRequest,
        active: &BTreeSet<String>,
        qtext: &str,
        qterms: &[String],
        acc: &mut Acc,
    ) -> Result<()> {
        if req.max_retrieved == 0 {
            return Ok(());
        }
        let pinned = safe_set(&req.pinned_files)?;
        let recency = recency_ranking(&self.index, &self.git_state);
        let qvec = self.query_vector(qtext);
        let params = SearchParams {
            query: qtext,
            kind: req.task_kind,
            limit: req.max_retrieved.saturating_mul(2),
            active,
            pinned: &pinned,
            recency: &recency,
            query_vector: qvec.as_deref(),
            vectors: self.vector_store(),
        };
        let mut cache: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut taken = 0usize;
        for hit in search(&self.index, &params) {
            if taken >= req.max_retrieved {
                break;
            }
            if active.contains(&hit.path) {
                continue;
            }
            if !cache.contains_key(&hit.path) {
                cache.insert(hit.path.clone(), read_lines(self.index.root(), &hit.path)?);
            }
            let lines = cache.get(&hit.path).map(Vec::as_slice).unwrap_or_default();
            let label = hit.symbol.clone().unwrap_or_else(|| "block".into());
            let mut c = Candidate::new(
                format!("chunk:{}:{}-{}", hit.path, hit.start_line, hit.end_line),
                Category::RetrievedCode,
                false,
                slice_lines(lines, hit.start_line, hit.end_line),
            );
            c.source = hit.path.clone();
            c.hash = Some(hit.hash.clone());
            c.range = Some(LineRange {
                start: hit.start_line,
                end: hit.end_line,
            });
            c.score = hit.score;
            c.why = hit.why.join("; ");
            c.heading = format!(
                "### {}:{}-{} ({label})",
                hit.path, hit.start_line, hit.end_line
            );
            c.elidable = true;
            c.focus = qterms.to_vec();
            c.recover_handle = Some(make_handle(
                &hit.path,
                hit.start_line,
                hit.end_line,
                &hit.hash,
            ));
            acc.cands.push(c);
            taken += 1;
        }
        Ok(())
    }

    /// Signature views of files one hop from an active file.
    fn add_graph(&self, active: &BTreeSet<String>, acc: &mut Acc) -> Result<()> {
        if active.is_empty() {
            return Ok(());
        }
        let seeds: Vec<String> = active.iter().cloned().collect();
        for (path, dist) in self.index.graph().neighbourhood(&seeds, 1) {
            let Some(entry) = self.index.file(&path) else {
                continue;
            };
            let lines = read_lines(self.index.root(), &path)?;
            let sigs: Vec<String> = entry
                .facts
                .symbols
                .iter()
                .filter(|s| s.kind.is_chunk_worthy())
                .take(40)
                .filter_map(|s| {
                    lines
                        .get(s.start_line as usize - 1)
                        .map(|l| l.trim().chars().take(160).collect())
                })
                .collect();
            if sigs.is_empty() {
                continue;
            }
            let mut c = Candidate::new(
                format!("graph:{path}"),
                Category::GraphNeighbours,
                false,
                sigs.join("\n"),
            );
            c.source = path.clone();
            c.hash = Some(entry.record.hash.clone());
            c.score = 1.0 / (1.0 + dist as f64);
            c.why = format!("adjacent to an active file ({dist} hop); signatures only");
            c.heading = format!("### {path} (signatures)");
            acc.cands.push(c);
        }
        Ok(())
    }

    fn memory_path(&self, session: &str) -> Result<PathBuf> {
        MemoryStore::new(session)?;
        Ok(self
            .store_dir
            .join("memory")
            .join(format!("{session}.json")))
    }

    /// Records a memory entry for `session`. `paths` are the files it depends on.
    ///
    /// Refreshes the index first, so the recorded hashes match the files as they are now.
    pub fn remember(
        &mut self,
        session: &str,
        kind: MemoryKind,
        text: &str,
        paths: &[String],
    ) -> Result<String> {
        self.update()?;
        let path = self.memory_path(session)?;
        let mut store = MemoryStore::load_or_new(&path, session)?;
        let hashes = self.index.hashes();
        let mut sources = Vec::with_capacity(paths.len());
        for p in paths {
            if !is_safe_relative(p) {
                return Err(RaptorError::InvalidInput(format!(
                    "unsafe source path: {p}"
                )));
            }
            sources.push(SourceRef {
                path: p.clone(),
                hash: hashes.get(p).cloned().unwrap_or_default(),
            });
        }
        let id = store.add(kind, text, sources)?;
        store.save(&path)?;
        Ok(id)
    }

    /// Summarises `messages` and records the result for `session`. Returns the ids created.
    ///
    /// Fails if the summary fails. The structured entries are still saved in that case; see
    /// [`ContextEngine::summarize_session_report`], which returns the warning instead.
    pub fn summarize_session(
        &mut self,
        session: &str,
        messages: &[ChatMessage],
    ) -> Result<Vec<String>> {
        let outcome = self.summarize_session_report(session, messages)?;
        if outcome.warnings.is_empty() {
            Ok(outcome.ids)
        } else {
            Err(RaptorError::Runtime(outcome.warnings.join("; ")))
        }
    }

    /// Records the session's structure, then a model summary of it.
    ///
    /// Decisions, conventions and failed attempts are taken from `messages` directly, so the
    /// model never writes them. They are recorded even when the summary fails. The summary is
    /// the only entry that can be lost: its failure is returned as a warning, and the
    /// conversation the caller holds is not changed.
    pub fn summarize_session_report(
        &mut self,
        session: &str,
        messages: &[ChatMessage],
    ) -> Result<SummaryOutcome> {
        if self.summarizer.is_none() {
            return Err(RaptorError::Unsupported(
                "no summarizer is configured".into(),
            ));
        }
        self.update()?;
        let path = self.memory_path(session)?;
        let mut store = MemoryStore::load_or_new(&path, session)?;
        let hashes = self.index.hashes();
        let mut ids = Vec::new();
        let mut warnings = Vec::new();

        // The summary is generated first so that, when it succeeds, it is the first entry.
        let transcript: Vec<&str> = messages.iter().map(|m| m.content.as_str()).collect();
        let transcript = transcript.join("\n");
        let result = match self.summarizer.as_ref() {
            Some(s) => s.summarize(messages),
            None => Err(RaptorError::Unsupported(
                "no summarizer is configured".into(),
            )),
        };
        match result {
            Ok(summary) => match store.add(
                MemoryKind::Summary,
                &summary,
                mentioned_sources(&transcript, &hashes),
            ) {
                Ok(id) => ids.push(id),
                Err(e) => warnings.push(format!("summary not recorded: {e}")),
            },
            Err(e) => warnings.push(format!(
                "summary not recorded ({e}); decisions, conventions and failed attempts were saved, and the conversation is unchanged"
            )),
        }

        // Structured items come from the messages themselves, never from the summary.
        let mut structured: Vec<(MemoryKind, String)> = extract_decisions(messages);
        structured.extend(
            failed_attempts(messages)
                .into_iter()
                .map(|t| (MemoryKind::FailedAttempt, t)),
        );
        for (kind, text) in structured {
            if store.current().any(|e| e.kind == kind && e.text == text) {
                continue;
            }
            let sources = mentioned_sources(&text, &hashes);
            match store.add(kind, &text, sources) {
                Ok(id) => ids.push(id),
                Err(e) => warnings.push(format!("not recorded: {e}")),
            }
        }
        store.save(&path)?;
        Ok(SummaryOutcome { ids, warnings })
    }
}

fn safe_set(paths: &[String]) -> Result<BTreeSet<String>> {
    paths
        .iter()
        .map(|p| {
            if is_safe_relative(p) {
                Ok(p.clone())
            } else {
                Err(RaptorError::InvalidInput(format!("unsafe file path: {p}")))
            }
        })
        .collect()
}

fn add_instructions(req: &ContextRequest, acc: &mut Acc) {
    if let Some(text) = req.instructions.as_deref().filter(|s| !s.trim().is_empty()) {
        let mut c = Candidate::new("instructions", Category::Instructions, true, text);
        c.why = "system instructions supplied by the caller".into();
        acc.cands.push(c);
    }
    for (i, m) in req
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::System)
    {
        let mut c = Candidate::new(
            format!("instructions:msg{i}"),
            Category::Instructions,
            true,
            m.content.clone(),
        );
        c.why = "system message in the conversation".into();
        acc.cands.push(c);
    }
}

fn add_task(req: &ContextRequest, acc: &mut Acc) {
    if req.task.trim().is_empty() {
        return;
    }
    let mut c = Candidate::new("task", Category::Task, true, req.task.clone());
    c.why = "active task statement supplied by the caller".into();
    acc.cands.push(c);
}

fn add_diffs(req: &ContextRequest, acc: &mut Acc) {
    for d in req.diffs.iter().filter(|d| !d.patch.trim().is_empty()) {
        let mut c = Candidate::new(
            format!("diff:{}", d.path),
            Category::Diffs,
            true,
            d.patch.clone(),
        );
        c.source = d.path.clone();
        c.heading = format!("### diff {}", d.path);
        c.why = "diff supplied by the caller, included in full".into();
        acc.cands.push(c);
    }
}

/// Adds failing output as protected context. Returns the message index of the latest
/// tool output that reports a failure, so history can leave it out.
fn add_errors(req: &ContextRequest, acc: &mut Acc) -> Option<usize> {
    for (i, e) in req
        .errors
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.trim().is_empty())
    {
        let mut c = Candidate::new(format!("error:{i}"), Category::ErrorsTests, true, e.clone());
        c.heading = format!("### error report {}", i + 1);
        c.why = "error or failing test output supplied by the caller".into();
        acc.cands.push(c);
    }
    let last = req
        .messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| m.role == Role::Tool && looks_like_error(&m.content))
        .map(|(i, _)| i);
    if let Some(i) = last {
        let mut c = Candidate::new(
            format!("error:msg{i}"),
            Category::ErrorsTests,
            true,
            req.messages[i].content.clone(),
        );
        c.heading = format!("### tool output (message {i})");
        c.why = "most recent tool output that reports a failure; kept until it is resolved".into();
        acc.cands.push(c);
    }
    last
}

fn add_history(req: &ContextRequest, qterms: &[String], last_error: Option<usize>, acc: &mut Acc) {
    let last_user = req.messages.iter().rposition(|m| m.role == Role::User);
    for (i, m) in req.messages.iter().enumerate() {
        if m.role == Role::System || Some(i) == last_error {
            continue;
        }
        let protected = Some(i) == last_user;
        let overlap = overlap_count(&m.content, qterms);
        let mut c = Candidate::new(
            format!("history:{i}"),
            Category::History,
            protected,
            m.content.clone(),
        );
        c.source = format!("history turn {i}");
        c.score = overlap as f64 / qterms.len().max(1) as f64 + i as f64 * 1e-6;
        c.why = if protected {
            "latest user turn: the active request".into()
        } else {
            format!(
                "earlier {} turn; shares {overlap} term(s) with the task",
                role_name(m.role)
            )
        };
        c.message = Some(i);
        c.role = Some(m.role);
        acc.cands.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_lines_clamps_bounds() {
        let lines: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(slice_lines(&lines, 2, 3), "b\nc");
        assert_eq!(slice_lines(&lines, 0, 1), "a");
        assert_eq!(slice_lines(&lines, 9, 12), "");
    }

    #[test]
    fn safe_set_rejects_traversal() {
        assert!(safe_set(&["../x".to_string()]).is_err());
        assert_eq!(safe_set(&["src/a.rs".to_string()]).expect("ok").len(), 1);
    }

    #[test]
    fn request_defaults_are_usable() {
        let r = ContextRequest::default();
        assert!(r.model_ctx > r.reserved_output);
        assert!(r.max_retrieved > 0);
    }

    #[test]
    fn request_deserialises_with_missing_fields() {
        let r: ContextRequest =
            serde_json::from_str(r#"{"task":"fix it","model_ctx":2048}"#).expect("parse");
        assert_eq!(r.task, "fix it");
        assert_eq!(r.model_ctx, 2048);
        assert_eq!(r.reserved_output, 512);
    }

    #[test]
    fn add_history_protects_latest_user_turn_and_skips_error_turn() {
        let req = ContextRequest {
            messages: vec![
                ChatMessage::new(Role::System, "be brief"),
                ChatMessage::new(Role::User, "first question"),
                ChatMessage::new(Role::Tool, "test failed: boom"),
                ChatMessage::new(Role::User, "second question"),
            ],
            ..ContextRequest::default()
        };
        let mut acc = Acc::default();
        add_history(&req, &["question".to_string()], Some(2), &mut acc);
        let ids: Vec<&str> = acc.cands.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["history:1", "history:3"]);
        assert!(acc.cands[1].protected);
        assert!(!acc.cands[0].protected);
        assert!(acc.cands.iter().all(|c| !c.why.is_empty()));
    }

    #[test]
    fn add_errors_picks_latest_failing_tool_output() {
        let req = ContextRequest {
            errors: vec!["cargo: 2 failed".into()],
            messages: vec![
                ChatMessage::new(Role::Tool, "error: first"),
                ChatMessage::new(Role::Tool, "ok"),
                ChatMessage::new(Role::Tool, "panicked at x"),
            ],
            ..ContextRequest::default()
        };
        let mut acc = Acc::default();
        let last = add_errors(&req, &mut acc);
        assert_eq!(last, Some(2));
        assert_eq!(acc.cands.len(), 2);
        assert!(acc
            .cands
            .iter()
            .all(|c| c.protected && c.category == Category::ErrorsTests));
    }

    #[test]
    fn add_instructions_collects_system_messages() {
        let req = ContextRequest {
            instructions: Some("house rules".into()),
            messages: vec![ChatMessage::new(Role::System, "sys msg")],
            ..ContextRequest::default()
        };
        let mut acc = Acc::default();
        add_instructions(&req, &mut acc);
        assert_eq!(acc.cands.len(), 2);
        assert!(acc
            .cands
            .iter()
            .all(|c| c.category == Category::Instructions && c.protected));
    }

    #[test]
    fn overlap_counts_shared_terms_once() {
        assert_eq!(
            overlap_count(
                "refund refund ledger",
                &["refund".to_string(), "x".to_string()]
            ),
            1
        );
    }
}
