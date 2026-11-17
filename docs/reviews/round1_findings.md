# Goldfish Review Report — Round 1

**Target:** `D:/Projects/Apps/RaptorLLM/docs/ARCHITECTURE.md` (lines 1-50, full file)
**Purpose under review:** the repository's architecture definition. It should match the code and be able to guide a repo cleanup.

## Reviewer Status

| Reviewer | Lens | Standard | Status |
| --- | --- | --- | --- |
| `egm:egm-zoom-out` | zoom_out | zoom-out-standard.md | completed (3 Blockers, 4 Majors, 2 Nits raw) |
| `egm:egm-design` | design_critic | zoom-out-standard.md | completed (4 Blockers, 5 Majors, 3 Nits raw; did not read package.json, models.md, runtime.md or ui-contract.md) |
| `egm:egm-codebase-details` | codebase_details | re-implementation-standard.md | completed (7 Blockers, 9 Majors, 5 Nits raw) |

All three reviewers gave a FAIL verdict.

## Severity Summary (after deduplication)

| Severity | Count |
| --- | --- |
| 🔴 Blocker | 8 |
| 🟡 Major | 7 |
| 🟢 Nit | 2 |
| **Total** | **17** |

Deduplication rules:
- When reviewers rated a merged finding differently, it takes the highest rating, and the disagreement is noted.
- Each ID names the lead lens (the reviewer with the highest rating or the most evidence), and every contributing reviewer is listed.
- Code paths are relative to `D:/Projects/Apps/RaptorLLM/`.

---

## 🔴 Blockers

### zoom_out-1-1 — No purpose, scope, goals/non-goals or links to specs
- **Reviewers:** zoom_out (Blocker; also Nit 9), design_critic (Major M5), codebase_details (part of M9)
- **Lines:** 1-8 (whole document)
- **Discrepancy:**
  - The document opens with "Three top-level pages only" and never says what Raptor LLM is.
  - It does not mention:
    - running llama-server locally;
    - the OpenAI/Anthropic gateway;
    - context intelligence for coding agents;
    - the non-goals (no accounts, cloud sync, IDE or custom inference engine).
  - It has no status or last-verified date.
  - It does not link `docs/specs/{runtime,models,context,gateway,ui-contract}.md` or `RULES.md`.
- **Rationale:** Design decisions cannot be judged against goals that are never stated (zoom-out standard §0/§1). README.md also describes as implemented features that this document calls future, so the two contradict each other.
- **Evidence:** `README.md:3-4`, `README.md:10-19`, `docs/specs/RULES.md:3-4`.

### zoom_out-1-2 — No crate map, no dependency rule, no composition root
- **Reviewers:** zoom_out (Blocker), design_critic (Blocker B1), codebase_details (Major M1)
- **Lines:** 10-15
- **Discrepancy:**
  - "Process boundaries" lists four vague layers. It never names the crates (raptor-core, -runtime, -models, -context, -gateway, src-tauri) or what each one owns.
  - It omits the rule the code actually enforces:
    - subsystem crates depend only on `raptor-core`;
    - `src-tauri` is the only crate that depends on all of them and is the composition root;
    - the gateway defines trait seams (`Upstream`, `ModelCatalog`, `ContextService`) that the shell implements.
- **Rationale:** This is the main architecture fact a cleanup needs. Without it, the document cannot say where code belongs or flag violations.
- **Related issues:**
  - CI does not enforce the rule.
  - `RULES.md:7` wrongly says wiring also happens in `raptor-gateway`. Per its lib.rs header, the gateway depends on no other `raptor-*` crate.
- **Evidence:**
  - `crates/raptor-{gateway,context,runtime,models}/Cargo.toml:8` (only `raptor-core.workspace = true`)
  - `src-tauri/Cargo.toml:16-20`
  - `src-tauri/src/lib.rs:1-5`, `src-tauri/src/adapters.rs:1-7`, `src-tauri/src/state.rs:1-3`
  - `crates/raptor-gateway/src/lib.rs:3-7`
  - `.github/workflows/ci.yml:45-58` (fmt, clippy, test and build only)

### zoom_out-1-3 — Context Intelligence is called "future" but is built
- **Reviewers:** zoom_out (Blocker), design_critic (Blocker B3, part of M2), codebase_details (Blockers B1, B2)
- **Lines:** 15, 28-40
- **Discrepancy:**
  - **Line 15** says a stable local API and MCP tools are for "future versions". Both ship today:
    - `/raptor/context/{build,search,recover,impact,status}`;
    - `/mcp`, with six tools: `context_build`, `context_search`, `context_recover`, `repo_symbols`, `repo_impact`, `repo_read_file`.
  - **Line 30** says "Future pipeline". In fact the pipeline exists: scan, Tree-sitter parse with regex fallback, incremental index, graph, hybrid retrieval, protected-first pack and budget, provenance. It sits behind the `api::ContextEngine` facade.
  - The list leaves out shipped modules: `recovery.rs`, `memory.rs` (project memory, with Tauri commands) and `git.rs`.
  - Context Intelligence is not a separate "service". It is an in-process library crate, reached through `ContextAdapter`/`ContextService` and Tauri commands.
- **Rationale:** A cleanup guided by this text could treat working code as speculative scope creep, or build it again.
- **Altitude:** zoom_out notes that lines 31-38 duplicate `docs/specs/context.md` at step level, which guarantees drift. Replace them with:
  - the component boundary;
  - the pluggable traits (`Embedder`, `Summarizer`, `TextGenerator`, `GitProbe`, `LanguageExtractor`);
  - a link to the spec;
  - implemented/partial/planned labels.
- **Evidence:**
  - `crates/raptor-context/src/lib.rs:4-7, 9-40`
  - `scan.rs:91-92, 180-210, 312-316`
  - `parse.rs:4, 157-233`, `parse_ts.rs`
  - `retrieve.rs:1-80`, `pack.rs:5-6, 183-195`, `provenance.rs:52-65`
  - `crates/raptor-context/Cargo.toml:14-26`
  - `crates/raptor-gateway/src/context_api.rs:264-268`
  - `crates/raptor-gateway/src/mcp.rs:63-68, 114, 341-388`
  - `src-tauri/src/adapters.rs:601-612`, `src-tauri/src/lib.rs:208-220`
  - `src-tauri/src/semantic.rs:1-11`

### design_critic-1-4 — The gateway subsystem is missing
- **Reviewers:** design_critic (Blocker B2), zoom_out (Major 4), codebase_details (Majors M1, M2)
- **Lines:** 10-15 (whole document)
- **Discrepancy:**
  - `raptor-gateway` is never mentioned. It is an Axum server on 127.0.0.1:8790 serving:
    - OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/v1/models`;
    - Anthropic `/v1/messages` and `/v1/messages/count_tokens`;
    - `/mcp`, `/raptor/context/*`, `/raptor/metrics` and `/health`.
  - Its middleware is also absent: auth, per-key rate limit/queue, TLS, CORS, reasoning hygiene and thinking-effort injection.
  - Line 14 implies clients reach llama-server's own OpenAI API directly. In fact llama-server sits behind the gateway through `RuntimeUpstream`, which calls `/v1/*`, `/tokenize` and `/props`.
  - The document does not say whether the in-app chat path goes through the gateway, or whether llama-server's port is ever exposed.
- **Rationale:** The gateway is a core product surface (`README.md:4, 18`). An architecture definition without it is incomplete.
- **Evidence:**
  - `crates/raptor-gateway/src/lib.rs:1-36`
  - `server.rs:1-12, 83-111`
  - `openai/upstream.rs:41-43, 160, 206, 238`
  - `adapters/mod.rs:38-39`
  - `src-tauri/src/adapters.rs:3-4, 67, 153`

### codebase_details-1-5 — LAN security items listed as to-do are already enforced, and more strictly
- **Reviewers:** codebase_details (Blockers B3, B4), design_critic (Blocker B4), zoom_out (Major 5)
- **Lines:** 44-47
- **Discrepancy:**
  - Line 46 says "Add API-key enforcement before treating LAN mode as production-ready". The gateway already refuses LAN mode without at least one API key.
  - It also refuses LAN mode without TLS unless `allow_insecure_lan` is set.
  - Line 47 says "Never bind to 0.0.0.0 without a clear warning". The code does more than warn: it refuses outright. A warning plus an `insecure` status flag applies only when `allow_insecure_lan=true`.
- **Rationale:** Anyone rebuilding the gateway from this text would produce weaker behaviour than the real one, and the cleanup would treat finished work as open.
- **Evidence:**
  - `crates/raptor-gateway/src/server.rs:9-12, 45-51, 78-80, 121-130`
  - `auth.rs:3-4, 94-111`

### design_critic-1-6 — "Signature/hash verification" overclaims; runtime binaries have no independent signature
- **Reviewers:** design_critic (Blocker B4), codebase_details (Major M6), zoom_out (Major 5)
- **Lines:** 48-50
- **Discrepancy:**
  - `Manifest.signature` is "Reserved for detached signatures. Not verified yet".
  - Runtime update digests come from GitHub's own release metadata, over the same TLS channel as the download. The code itself says this "is not an independent signature", so whoever controls the source also controls the hash.
  - Updates need user confirmation; they are not automatic.
  - `RuntimeEngine::use_local_runtime` runs an unverified binary. It is a developer path the shell never calls.
  - Line 48 ("Validate all paths and download names") should name the validators that enforce it.
- **Rationale:** The wording gives a false sense of safety. The document should state the current trust model honestly (SHA-256 checked before extraction, digest sourced from GitHub, no signature) and record signing as an open risk.
- **Related issue:** `README.md:15` ("signed runtime manifest installation") makes the same false claim.
- **Evidence:**
  - `crates/raptor-runtime/src/manifest.rs:70-72`
  - `src-tauri/src/updates.rs:2-7`
  - `crates/raptor-runtime/src/lib.rs:216-218`
  - Validators: `crates/raptor-core/src/paths.rs:95` (`safe_file_name`), `crates/raptor-models/src/hf.rs:206-237` (`validate_repo_id`, `validate_commit_sha`, `safe_repo_path`), `download.rs:495` (`validate_request`)

### codebase_details-1-8 — The GPU verification signals described are not what the code does
- **Reviewers:** codebase_details (Blocker B7), zoom_out (Major 6), design_critic (Major M1)
- **Lines:** 19-24
- **Discrepancy:**
  - `verify_gpu` reads only the llama-server log: detected backend, initialized devices, offloaded layers, and GPU/CPU buffer sizes. A device preflight (`--list-devices`) runs before load.
  - Live telemetry (`apply_telemetry`, `RocmSmiTelemetry`) exists but is never called from `src-tauri`. It can downgrade a verdict, never upgrade one, and it is not wired in.
  - The "small inference smoke test" (line 24) does not exist: no matches for "smoke" under `crates/` or `src-tauri/src`.
  - Line 21 names only HIP, but verification also covers CUDA and Vulkan.
  - The real verdict types (`Verified`, `CpuOnly`, `Unverified`) and their decision rules are undocumented.
- **Rationale:** The principle ("no claim without positive evidence") matches the code. The listed mechanisms partly do not exist. Implemented checks should be split from planned ones.
- **Evidence:**
  - `crates/raptor-runtime/src/verify.rs:1-4, 45-60, 206-210, 233-246, 256-330, 383-442`
  - `crates/raptor-runtime/src/lib.rs:580-604`

---

## 🟡 Majors

### codebase_details-1-7 — Existing security controls are undocumented
- **Reviewers:** codebase_details (Major M7), design_critic (Blocker B4, "not covered" part), zoom_out (Major 5, TLS and hashed keys)
- **Lines:** 42-50
- **Discrepancy:** None of these existing controls is described:
  - keyless loopback, on by default;
  - an exact-origin CORS allowlist that rejects wildcards;
  - a 32 MiB body limit;
  - a per-key rate limit and queue;
  - an optional TLS listener with fingerprint display;
  - salted, hashed key storage;
  - `repo_read_file` confined to the workspace root and disabled when no root is set;
  - the workspace header;
  - llama-server's own bind (defaults to 127.0.0.1; a non-loopback host requires an api_key).
- **Rationale:** The security section should be a threat model with current controls and open risks, not a to-do list.
- **Evidence:**
  - `crates/raptor-gateway/src/server.rs:53, 59, 61-65, 72-73, 77, 98, 100`
  - `queue.rs`, `tls.rs`
  - `mcp.rs:575, 704` (traversal tests)
  - `context_api::WORKSPACE_HEADER`
  - `crates/raptor-runtime/src/config.rs:186, 257`

### codebase_details-1-9 — "Tested runtime profiles" has no mechanism; real selection, fallback and backend-choice design is missing
- **Reviewers:** codebase_details (Major M8), zoom_out (Major 6), design_critic (Major M1)
- **Lines:** 26
- **Discrepancy:**
  - No "profile" concept exists. The real mechanism:
    - a bundled `runtime-manifest.json` is seeded into `config/` on first run and never overwritten;
    - `select_runtime` matches on `os`, `gfx_archs` and `min_driver`, with a `degraded` CPU flag;
    - `HIP_SDK_MIN = (7, 2)`;
    - rollback, per-backend switch and uninstall are all refused while a model is loaded.
  - Also undocumented:
    - the CPU-fallback policy (`allow_cpu_fallback` defaults to false and turns the verdict CPU-only);
    - the measured per-model ROCm vs Vulkan backend choice;
    - the tuning/fallback plan.
- **Rationale:** These mechanisms make up the actual hardware-performance architecture.
- **Evidence:**
  - `src-tauri/src/lib.rs:38-67`
  - `crates/raptor-runtime/src/lib.rs:193-214, 262-394`
  - `crates/raptor-runtime/src/manifest.rs:3-5, 52-73`
  - `src-tauri/src/updates.rs:23-25`
  - `src-tauri/src/settings.rs:154, 689`, `src-tauri/src/state.rs:116`
  - `src-tauri/src/backend_choice.rs:1-12`, `src-tauri/src/tuning.rs:1-16`

### design_critic-1-10 — Process boundaries and lifecycle ownership are inaccurate and incomplete
- **Reviewers:** design_critic (Major M2), zoom_out (Major 7), codebase_details (Major M1 Job Object part, Nit N2)
- **Lines:** 12-15
- **Discrepancy:**
  - "Rust application core" fits no real component. The work is split across raptor-runtime, raptor-models and src-tauri.
  - The real topology has three processes:
    - the WebView UI;
    - the Tauri host, which runs the gateway and context engine in-process;
    - a child llama-server, inside a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
  - They are joined by Tauri IPC (UI to host; about 100 commands, plus events such as `EVENT_SERVER_STATE`) and loopback HTTP (host to llama-server).
  - Also missing:
    - background start-up and shutdown order;
    - the shell's ownership of the listener, streams and loaded-model record;
    - what "isolated" means: a separate process, with no sandbox.
- **Rationale:** Without these, there is no rule to check side-effect placement against during cleanup.
- **Evidence:**
  - `src-tauri/src/lib.rs:33, 69-102, 128-232, 242-248`
  - `src-tauri/src/job.rs:1-11, 72`
  - `src-tauri/src/state.rs:1-3`
  - `README.md:74`

### design_critic-1-11 — No state, storage, configuration, error or UI-contract architecture
- **Reviewers:** design_critic (Major M3), codebase_details (Major M9), zoom_out (Major 7, persistence part)
- **Lines:** whole document
- **Discrepancy:** The document does not describe:
  - the `DataDirs` layout (models, runtimes, logs, config, context index);
  - the JSON stores (settings, conversations, tuned.json, ctxbench.json);
  - the rolling `logs/llama-server.log`;
  - atomic writes (`fsutil::write_atomic`);
  - versioning and migration;
  - the error type (`raptor_core::RaptorError`);
  - the UI-to-shell command and event contract (`docs/specs/ui-contract.md`).
- **Rationale:** The re-implementation standard requires side effects, data locations, configuration and errors to be described. `docs/specs/context.md:9` already requires a versioned index under `DataDirs::context_index()`.
- **Evidence:**
  - `crates/raptor-core/src/paths.rs`
  - `src-tauri/src/settings.rs`, `conversations.rs`, `tuned.rs`, `ctxbench.rs`, `logging.rs`
  - `crates/raptor-runtime/src/lib.rs:164-168`

### design_critic-1-12 — Cross-cutting testing and honesty principles are missing
- **Reviewers:** design_critic (Major M4), codebase_details (Major M9, test-infra part)
- **Lines:** whole document
- **Discrepancy:** `RULES.md:13-14` sets architecture-level principles that ARCHITECTURE.md does not carry:
  - external effects go behind traits with fakes;
  - no network, GPU or llama-server in unit tests;
  - a separate real-hardware tier;
  - no silent fallbacks.
- **Rationale:** These principles shape module boundaries throughout the code. The architecture is currently split across two files.
- **Evidence:**
  - Traits with fakes: `ReleaseSource` (`src-tauri/src/updates.rs:43-47`), `ServerProbe` (`src-tauri/src/adapters.rs:46`), `Upstream`
  - Fakes: `crates/raptor-runtime/src/testing.rs`, `src-tauri/src/testing.rs:440` (`FakeGpuMemory`)
  - Real-hardware tier: `crates/raptor-runtime/tests/gate_real_hardware.rs`

### codebase_details-1-13 — Retrieval and scoring factors do not match the code
- **Reviewers:** codebase_details (Major M3)
- **Lines:** 34-35
- **Discrepancy:**
  - The document lists "task relevance, dependency distance, recency, test coverage, change status, explicit pins".
  - The code uses weighted reciprocal-rank fusion over ten signals: lexical, symbol, graph, recency, pin, embedding, downstream, diagnostics, security and hubs. Weights come from a per-`TaskKind` table with seven kinds.
  - "Test coverage" is not a signal. The diagnostics signal covers test code and error text.
  - Embeddings are optional: `NoEmbedder` is the default, and `HashingEmbedder` also exists.
- **Rationale:** The listed factors would lead a re-implementer to a different ranker.
- **Related issue:** the source comment at `retrieve.rs:3` ("Six ranked signals") is itself out of date.
- **Evidence:** `crates/raptor-context/src/retrieve.rs:3, 18-40, 42-80`, `crates/raptor-context/src/lib.rs:40`.

### codebase_details-1-14 — The budget and protected-item contract is unspecified and partly mismatched
- **Reviewers:** codebase_details (Majors M4, M5), design_critic (Nit N2, line 40 part)
- **Lines:** 36, 40
- **Discrepancy:**
  - "Model-tokenizer-derived budget" hides how it works:
    - any `TokenCounter` is accepted;
    - it falls back to `EstimatingCounter` when no tokenizer is available;
    - the manifest records whether counts were exact;
    - category shares are permille allocations of the unprotected pool.
  - The protected categories in `budget.rs` differ from line 40's list. For example, decisions and conventions from conversation memory are protected, but summaries are not.
  - The over-budget behaviour has a concrete contract the document should name: protected items are always included in full, and any overflow is reported as `over_budget_by`.
- **Rationale:** The invariant is sound, but it should cite the field and the module that enforce it.
- **Evidence:**
  - `crates/raptor-context/src/api.rs:263-282`
  - `budget.rs:14-24, 78, 127`
  - `pack.rs:5, 194`
  - `provenance.rs:63-65`

---

## 🟢 Nits

### design_critic-1-16 — Line 26 is at the wrong altitude and specific to one backend
- **Reviewers:** design_critic (Nit N2, plus Major M1 HIP-only part), codebase_details (Nit N3)
- **Lines:** 21, 26
- **Discrepancy:**
  - The RDNA 4 / `gfx1200` detail sits inside a general strategy section.
  - The code supports CUDA, Vulkan and CPU backends as well as HIP.
- **Rationale:** The GPU strategy should be written for all backends. Hardware-specific notes belong in a tested-hardware note or in `docs/EXPERIMENTS.md` (`RULES.md:23`).
- **Evidence:** `crates/raptor-runtime/src/verify.rs:44-50`, `manifest::Backend`.

### design_critic-1-17 — No diagrams; mixed concepts under one heading
- **Reviewers:** design_critic (Nit N3)
- **Lines:** 10-15, 30-38
- **Discrepancy:**
  - "Process boundaries" mixes process concepts with module concepts.
  - The numbered pipeline has no data-flow diagram.
- **Rationale:** A short request-flow diagram and a crate-dependency diagram would replace several paragraphs and make drift easy to spot.

---

## 🔴 Blocker (with severity disagreement)

### codebase_details-1-15 — Page descriptions do not match the UI
- **Reviewers:** codebase_details (Blockers B5, B6; Nits N1, N4), zoom_out (Nit 8), design_critic (Nit N1)
- **Severity note:** codebase_details rated this a Blocker because line 8 states something that does not exist. zoom_out and design_critic rated it a Nit. The report keeps the highest rating.
- **Lines:** 5-8
- **Discrepancy:**
  - The "three pages" claim is correct.
  - **Settings (line 8)** lists "appearance". No appearance setting exists; the only theme match is `styles.css`. It also lists "performance", which lives on the Models page. The real Settings sections are Hardware, Inference runtime, Dependencies, Server and API, Storage, Context engine, Diagnostics and About. API keys, TLS, CPU fallback and project memory are not mentioned.
  - **Models (line 7)** omits the Performance and Compare-and-rank tabs, presets, and load-failure / CPU-fallback handling.
  - **Chats (line 6)** undersells the page: quick-load, the parameter panel, the context meter, thinking blocks, search, and export/import are all missing.
- **Rationale:** Either keep the descriptions strictly high-level and cite `Page` (`src/state/ui.ts`) and `docs/specs/ui-contract.md` as the source of truth, or make them accurate.
- **Evidence:**
  - `src/pages/Settings.tsx:21-28`, `src/pages/Models.tsx:15-22`, `src/App.tsx:8-18`
  - `src/components/settings/` (ApiKeyPanel, GatewayTls, DiagnosticsPanel, ProjectMemoryView, CpuFallbackToggle)
  - `src/components/models/` (CompareTab, PresetsPanel, PerformancePanel, LoadFailureCard, CpuFallbackBanner)
  - `src/components/chat/*`, `src/lib/conversationFile.ts`

---

## Adjacent issues outside the target (for the cleanup, not counted)
- `README.md:15` claims "signed runtime manifest installation". This is false per `crates/raptor-runtime/src/manifest.rs:70-72`.
- `README.md:18` lists 4 MCP tools; `crates/raptor-gateway/src/mcp.rs` has 6.
- `docs/specs/RULES.md:7` says wiring happens in `raptor-gateway`. The gateway does no wiring.
- `crates/raptor-context/src/retrieve.rs:3` says "Six ranked signals"; the code has ten.
- The dependency rule is not enforced in CI (`.github/workflows/ci.yml:45-58`). Possible fixes: a `cargo metadata` check or cargo-deny bans.
