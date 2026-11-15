# raptor-context

The Context Intelligence Engine. It indexes a repository, finds what matters for a task,
packs it under a model's context budget, and records why every item is in or out.
Design: `docs/specs/context.md`. Rules: `docs/specs/RULES.md`.

Pipeline: `scan` -> `parse` (tree-sitter, regex fallback) -> `index` (chunks, incremental update,
store, inverted index) -> `graph` -> `retrieve` (task-weighted hybrid ranking, with the persisted
`vector` store as one signal) -> `pack` + `budget` (protected-first) -> `provenance` (manifest).
`api::ContextEngine` is the facade. Every request and response type is serde-serialisable and maps
onto one MCP tool.

## Public API

Facade (`raptor_context::ContextEngine`), all errors are `raptor_core::RaptorError`:

- `ContextEngine::open(&DataDirs, workspace: &Path, EngineDeps) -> Result<Self>`
  Loads the persisted index, vector store and project memory for the workspace from
  `DataDirs::context_index()`.
- `EngineDeps { counter, git, embedder, summarizer }`, with `EngineDeps::new(counter)` and
  `EngineDeps::estimating()`. Collaborators are traits, so tests inject fakes.
- `set_embedding_backend(&mut self, Option<Arc<dyn EmbeddingBackend>>) -> Result<()>`: enables
  semantic retrieval (or disables it with `None`). A backend failure is reported, not returned.
- `update(&mut self) -> Result<UpdateReport>`: re-scan. Files whose size and mtime are unchanged are
  not read; changed files are hashed and parsed; missing vectors are embedded.
- `update_full(&mut self) -> Result<UpdateReport>`: as `update`, but reads and hashes every file.
- `build_context(&mut self, ContextRequest) -> Result<ContextResponse>`:
  `ContextResponse { messages: Vec<ChatMessage>, manifest: ContextManifest }`. Messages are one
  system message with the context grouped by category, then the history that fits.
- `search(&self, &SearchRequest) -> Result<Vec<Hit>>`: hybrid retrieval with reasons per hit.
- `symbols(&self, &SymbolQuery) -> Vec<SymbolHit>`
- `impact(&self, file) -> Result<ImpactReport>`: imports, dependents, transitive dependents, tests.
- `recover(&mut self, RecoverRequest) -> Result<RecoverResponse>`: redeem a `chunk:` handle,
  or search for a query within `extra_budget`.
- `index_status(&self) -> IndexStatus`: counts, git state, `extractors` (per language backend and
  regex fallbacks), `semantic: SemanticStatus` (model, vectors, chunks without a vector, last error,
  store notice), and `project_memory_entries` / `project_memory_stale`.
- `remember(&mut self, session, MemoryKind, text, paths) -> Result<String>`: session memory.
- `remember_project(&mut self, MemoryKind, text, paths) -> Result<Added>`: long-term memory for the
  workspace. `Added` carries the ids evicted to stay under the cap.
- `forget_project_memory(&mut self, id) -> Result<()>`: removes an entry, including a decision.
- `summarize_session(&mut self, session, &[ChatMessage]) -> Result<Vec<String>>`: needs a
  summarizer. Fails if any warning was raised; see `summarize_session_report`.
- `summarize_session_report(&mut self, session, &[ChatMessage]) -> Result<SummaryOutcome>`:
  `SummaryOutcome { ids, warnings }`. Structured entries are saved even when the summary fails.

Key types:

- `ContextRequest` (serde defaults): `task`, `task_kind`, `instructions`, `messages`,
  `active_files`, `diffs`, `errors`, `pinned_files`, `session_id`, `model_ctx`,
  `reserved_output`, `max_retrieved`.
- `ContextManifest`: `included: Vec<ManifestItem>`, `excluded: Vec<ExcludedItem>`,
  `utilization_by_category`, `used_tokens`, `budget_tokens`, `over_budget_by`, `tokens_exact`,
  `warnings`. `is_fully_explained()` checks that every item has a reason.
- `TaskKind`: `Debug`, `Architecture`, `Refactor`, `Feature`, `Ui`, `Security`, `Performance`.
- `Category`: `Instructions`, `Task`, `ActiveFiles`, `Diffs`, `ErrorsTests`, `Memory`,
  `RetrievedCode`, `GraphNeighbours`, `History`.
- `UpdateReport`: `reparsed`, `removed`, `unchanged`, plus `hashed` (files read) and
  `stat_skipped` (files not read).
- `Backend`: `TreeSitter` (`"tree-sitter"`) or `Regex` (`"regex"`). `Language` is re-exported.
- Traits: `GitProbe` (`GitCli`, `NoGit`), `Embedder` (`NoEmbedder`, per-chunk legacy hook),
  `EmbeddingBackend` (`HashingEmbedder` for tests), `Summarizer`, `TextGenerator` (for
  `ModelSummarizer`), `Extractor` (`parse::default_extractors()`), `TokenCounter` (from raptor-core).
- `ProjectMemory`: the on-disk store behind `remember_project`. `VectorIndex`: the vector store.

## Behaviour guarantees

- Protected items (instructions, task, active files, diffs, errors, decisions and conventions)
  are always included in full. If they exceed the budget, the manifest reports `over_budget_by`
  and a warning. Nothing protected is dropped.
- Unprotected items fill per-category caps set by `TaskKind`. Unused caps spill to waiting items.
  Oversized chunks are elided at structure level (signature, focus lines, `... N lines elided ...`
  markers), never cut mid-line.
- Every included and excluded item has a non-empty `why`. Excluded chunks carry a `recover_handle`.
- Output is deterministic: ordered maps, stable tie-breaks, no wall-clock values.
- Memory that cites a file (session or project) is marked stale when that file's hash changes.
  Stale entries are reported as excluded and never served as current. Stale decisions stay on disk
  until forgotten, but are not served.
- Semantic retrieval never fails a build. If the backend is down, `build_context` still succeeds,
  adds a manifest warning, and the status reports the error. Lexical and graph signals still run.
- Project memory eviction, when over 500 entries: stale entries go first, then failed attempts,
  summaries and completed work, oldest first. Decisions and conventions are never evicted. New
  decisions and conventions are refused past 200 and must be forgotten explicitly.
- Paths from callers are checked (`is_safe_relative`): no absolute paths, no `..`.

## Task kinds

`TaskKind` changes both the retrieval weights (`retrieve::weights`) and the budget shares
(`budget::shares`). Ten signals are fused by reciprocal-rank fusion; a zero weight switches one off.
The semantic signal is the `embedding` weight, so BM25 and vector scores are fused with the same
RRF rule, and each task kind sets how much the vector list counts.

| Kind | What it leans on |
| --- | --- |
| Debug | test code and error text among query matches (`diagnostics`), recent changes (`recency`), symbol matches |
| Architecture | the dependency overview: files many others import (`hubs`), graph neighbourhood; larger graph budget |
| Refactor | downstream impact: importers of active files and callers of their symbols (`downstream`), graph |
| Security | authentication, input and secret handling terms (`security`), graph |
| Feature, Ui, Performance | lexical and symbol matches, recency, graph in balance |

The tests in `src/task_tests.rs` prove each strategy against fixtures and check the budget
shares end to end through `build_context`.

## Behaviour of the scan and index

- `scan`: `.gitignore` and `.raptorignore` honoured; vendor directories pruned; secrets, lock files,
  binaries, generated and oversized files excluded with reasons; SHA-256 content hashes.
- Stat fast path: a file whose size and mtime match its record is not read. Its record counts as
  unchanged. Records are trusted only when the content was hashed at least 2 seconds after the mtime
  (`index::RACY_WINDOW_SECS`), because mtimes have one-second resolution.
- `index`: chunks per symbol plus line windows for uncovered code. Incremental `update()` parses
  only files whose content hash changed. The store is versioned JSON (version 2), written atomically.
- `vector`: `vectors-v1.bin` in the workspace's index directory. Vectors are keyed by the SHA-256 of
  the chunk text, so identical chunks share one. A file whose hash changes is re-keyed, and vectors
  nothing references are pruned. The store records the model id and dimension. A store built for
  another model is discarded and rebuilt. A corrupt store is discarded with a notice. Embedding is
  batched (32 texts per call), chunk text is cut at 2000 characters, and the store holds at most
  100,000 vectors, so memory stays under about 300 MB at 768 dimensions.
- `postings`: an inverted index over chunk terms, maintained incrementally, used for BM25 candidate
  selection.
- `memory` and `summary`: session memory per `session_id`. The summary model sees the transcript
  with `Decision:` and `Convention:` lines removed. Those lines and failed-attempt lines are recorded
  from the messages directly. Model replies are cut at 1200 characters.

## Status

Implemented and tested: 203 unit tests (`src/`), 9 evaluation tests (`tests/eval.rs`), 4 cross-file
evaluation tests (`tests/cross_file.rs`), and one ignored scale benchmark (`tests/scale.rs`). The
synonym eval (a query with no shared word finds the right file only with embeddings) is in
`src/semantic_tests.rs`, together with the degradation, incremental-embedding, restart and summary
tests.

- `parse`: tree-sitter extraction (`parse_ts.rs`) for Rust, Python, TypeScript (`.ts` and `.tsx`),
  JavaScript, Go, Java, C and C++, with the grammars pinned in `Cargo.toml`
  (`tree-sitter =0.25.10` and matching grammar crates). Symbols carry exact ranges and parents
  (Rust `impl`/`trait`, Python classes with decorators, TS classes and arrow functions, Go receivers,
  Java classes, C++ out-of-class definitions). Calls include Rust macro arguments. Test detection:
  Rust `#[test]`-style attributes, Python `test_*`, Go `Test*` in `_test.go`, JUnit annotations,
  Jest/Mocha/Vitest `it`/`test` calls.
- `parse` fallback: a file whose tree has syntax errors is extracted by the regex rules for that
  language and recorded as `regex`. A language without a grammar would use regex for every file.
  Both cases appear in `index_status().extractors`. Regex rules remain for these fallbacks only.
- Import resolution, with fixture tests per language in `src/resolve_tests.rs`:
  - Rust `crate::`, `self::` and `super::` are resolved against the file's module position, and
    `mod x;` links the child module. `use a::b::Type` finds the module when `Type` is not a file.
  - Python `from . import x` links the submodule `x` as well as the package; `from .pkg import y`
    links `pkg` and `pkg/y`.
  - TypeScript and JavaScript: relative paths with extensions (`./x.js` for `x.ts`), `index` files,
    parent directories (`../`), `require` and dynamic `import()`. Bare package names are not linked.
  - Go: import paths match package directories by the longest suffix.
  - Java: a class import links its file; a package wildcard links every file in the package directory.
  - C and C++: quoted includes resolve relative to the file, then from the root.
- `graph`: imports, defines, calls, tested-by edges. Queries: callers, callees, dependents,
  transitive dependents, related tests, k-hop neighbourhood, dependent counts. Edges are owned per
  file. A change re-resolves only files that reference a touched name or path segment. Lookups are
  keyed by module, suffix, file and name, so a common name costs the same as a rare one.
- `retrieve`: see Task kinds above. Lexical search reads the inverted index; file-level signals
  expand only the files they select. The vector signal scans every chunk with a vector, so its cost
  grows with the store, not the query. It was not measured at scale.
- `budget`, `pack`, `provenance`, `recovery`, `memory`: as in the design spec. Stale recovery
  handles are `RaptorError::Conflict` with a message starting `stale handle:`.

Measured (release build, Windows 11, MSVC, 20,000 small Rust files in 200 packages, 180,001 chunks,
100,001 symbols), from `cargo test -p raptor-context --test scale --release -- --ignored --nocapture`:

| Operation | Time | Files read |
| --- | --- | --- |
| Cold index (scan, parse, chunk, graph) | 5.7 s | 20,000 |
| No-op `update()` | 0.13 s | 0 |
| `update()` after editing one file | 0.32 s | 1 |
| `search` per query | 21-38 ms | none |
| `build_context` (includes a no-op `update()`) | 0.17 s | 0 |

The previous version of this README, before the stat fast path, reported 1.35 s for the no-op
update, 1.5 s for the one-file update and 1.4 s for `build_context`. Those figures were not
re-measured on the same run. Timings vary by 20-30% between runs on this machine.

The fast path does not cover the case where a file's content changes but its size and mtime do not.
`src/index.rs` (`same_size_and_mtime_edit_is_missed_by_fast_path_and_caught_by_full_rehash`)
proves that edge, and `update_full()` catches it. Touching a file without changing it costs one read.

Stubbed or limited, stated plainly:

- **Tree-sitter build needs a C compiler.** The grammars are compiled from C by the `cc` crate.
  This was verified on `x86_64-pc-windows-msvc` only. Linux CI was not run from this crate.
- **Regex fallback is weaker.** Python `from . import x` links only the package there, and call
  edges are name-based in both backends.
- **Crate-name prefixes are not dropped.** `use mycrate::a::B` from an integration test does not
  resolve, because the leading `mycrate` segment is not a path segment of the file. Crate-root paths
  resolve only when written as `crate::`, `super::` or `self::`, or as a file-path suffix.
- **Ambiguous names are not linked.** A call to a name defined in more than three files, with no
  local or imported definition, is dropped.
- **No embedding model is shipped, and this crate makes no network calls.** `EmbeddingBackend` and
  `TextGenerator` are the seams. The shell must supply the llama-server `/v1/embeddings` and chat
  implementations. The only backend in this crate is `HashingEmbedder`, which matches shared words
  and is not semantic. The synonym eval uses hand-made vectors in a fake.
- **The legacy `Embedder` hook is still present.** Its per-chunk vectors are held in the index
  store. `set_embedding_backend` takes precedence over it. Set both and the persisted vectors are used.
- **Vector capacity is a hard cap.** Past 100,000 vectors, the remaining chunks have no vector and
  rank without the semantic signal. The status and a manifest warning say how many.
- **Summaries need a model.** Without a summarizer, `summarize_session` returns `Unsupported`.
  The `ModelSummarizer` path is tested with a fake generator only.
- **Token counts are estimates by default.** `EstimatingCounter` is used unless the gateway
  passes a model-tokenizer `TokenCounter`. `tokens_exact` reports which one was used.
- **Section headings are not budgeted.** Each item is counted with its own heading. The few
  tokens for the `## Section` headings and separators are not.
- **Gitignored paths are not listed.** The walker prunes them. Only Raptor LLM-pruned vendor
  directories appear as exclusions.
- **Recovery handles exist only for chunks.** History turns belong to the caller and cannot be
  recovered from this crate.
- **Concurrency.** Mutating methods take `&mut self`. The caller must serialise access
  (for example with a `Mutex`).
- **Graph persistence.** The index store persists file facts. The graph is rebuilt from them on
  open, so edges are not stored separately.
- **Corrupt project memory is an error, not a reset.** It may hold decisions the user relies on, so
  the engine refuses to open until the file is moved aside. Index and vector stores are rebuilt.
