# Raptor LLM Build Plan and Pass/Fail Gates

## M0 — Foundation (Completed)
- [x] Tauri 2 desktop shell.
- [x] Three separate primary pages (Chats, Models, Settings).
- [x] Rust backend boundary.
- [x] Local GGUF inventory.
- [x] Hugging Face model search.
- [x] Model download to `.part` then rename.
- [x] llama-server process lifecycle prototype.
- [x] LAN toggle prototype.
- [x] Context packing prototype and unit tests.

Pass: `npm run build` succeeds; `cargo test` succeeds; UI exposes exactly Chats, Models, Settings.
Fail: any page mixes chat, model downloading, and global settings; backend errors are hidden.

## M1 — Reliable inference (Completed)
- [x] Add streaming process logs and structured diagnostics (`diagnostics_log_tail`, ring buffer, file log sink).
- [x] Add server health checks and `/v1/models` validation (`ServerManager` poll loop).
- [x] Add graceful shutdown and crash recovery (`JobSpawner` job object on Windows, bounded auto-restart with backoff).
- [x] Parse llama.cpp logs for backend, device and offloaded layers (`verify_gpu`, `LogFacts` with `offloaded_layers` and `total_layers`).
- [x] Add GPU telemetry adapters per OS (`GpuTelemetry` trait, `rocm-smi` JSON parser).
- [x] Add smoke test and benchmark record format (`perfmem`).
- [x] Verify Windows and Linux builds independently.

Pass: a known GGUF loads, server responds, logs prove HIP device selection and offload; benchmark records TTFT, prompt tok/s, generation tok/s, VRAM and RAM.
Fail: app says GPU-ready without positive evidence; server crash leaves stale loaded state; CPU fallback is undisclosed.

## M2 — Model library (Completed)
- [x] Read Hugging Face GGUF repository file lists and metadata (`models_repo_files`, `models_repo_info`).
- [x] Recommend quantization from model size, RAM/VRAM and context target (`recommend_quant`, `compat_rating`).
- [x] Resumable downloads with range requests and progress events (`DownloadManager`, HTTP Range).
- [x] Hash verification, cancellation and cleanup (SHA-256 against HF LFS hash).
- [x] Import existing model folders (`models_import`, `models_scan_folder`, native file dialogs).
- [x] Preserve model versions and never overwrite silently (`Conflict` on collision unless `overwrite=true`).

Pass: interrupted download resumes; checksum mismatch is rejected; insufficient disk space is detected before download.
Fail: partial files appear as installed models or existing models are overwritten without consent.

## M3 — Chat and API (Completed)
- [x] OpenAI-compatible streaming chat client (SSE framing, cancel on disconnect).
- [x] Conversation persistence and export (one JSON per chat, export/import).
- [x] Model selection restricted to installed models (`ModelPicker`).
- [x] Stop, regenerate, edit and continue (`chat_cancel`, `regenerate` with `conversations_truncate`, `conversations_replace_message`).
- [x] Tool-call and structured-output compatibility tests (streamed tool_call deltas pass through untouched).

Pass: streaming response renders incrementally; cancellation stops generation; chat survives app restart.
Fail: UI fabricates assistant responses or reports success when the server is unavailable.

## M4 — Hardware optimizer (Completed)
- [x] Detect OS, GPU, VRAM, RAM and CPU capabilities (`HardwareProbe`, `SystemProbe`, PowerShell DXGI & sysfs).
- [x] Maintain signed runtime manifests per OS/backend (`runtime-manifest.json` schema 1).
- [x] Verify gfx1200 support for each distributed build.
- [x] Calibrate context, batch, KV cache, offload and threads (`recommend_config` with VRAM headroom and KV estimates).
- [x] Save per-model/per-machine profiles and rollback (`perfmem`, `rollback` pins previous version).
- [x] Add Fastest, Balanced and Maximum Context presets.

Pass: selected profile is benchmarked and repeatable; memory headroom is respected; backend is verified.
Fail: unsupported flags are passed silently or an unstable profile replaces a known-good one.

## M5 — Context Intelligence (Core Completed; Tree-sitter & Embeddings Next)
- [ ] Tree-sitter indexing for supported languages (current: robust regex extractors behind `Extractor` trait).
- [x] Git-aware incremental updates and ignore rules (`scan`, `git status --porcelain`).
- [x] Symbol/import/call/test dependency graph (`graph`, callers, callees, downstream impact, related tests).
- [x] Lexical retrieval (BM25 over identifiers + text), then optional embeddings hook (`Embedder` trait).
- [x] Task-specific scoring profiles: debug, feature, refactor, architecture, UI (`TaskKind` allocations).
- [x] Exact tokenizer integration per model (`ServerTokenCounter` via llama-server `/tokenize`).
- [x] Protected context and provenance manifest (`ContextManifest` explains every item; protected never dropped).
- [x] Stale-summary invalidation and context recovery API (`invalidate_stale()`, `context_recover`).
- [x] MCP and harness adapters (Axum JSON-RPC MCP server + Anthropic `/v1/messages` translation).

Pass: evaluation corpus demonstrates higher task-relevant recall than recency-only truncation; no protected item is silently removed; every included snippet has source provenance; changed files invalidate stale summaries.
Fail: tokenizer budget is exceeded without warning; stale code is supplied as current; task constraints disappear; retrieval cannot explain inclusion/exclusion.

## M6 — Distribution and security (Core Completed; Real-Hardware Gate Next)
- [x] Windows NSIS installer bundle configuration (`tauri.conf.json`).
- [x] CI matrix automation pipeline (`.github/workflows/ci.yml`).
- [x] First-run dependency checks and guided setup.
- [x] Runtime index and checksum verification (`runtime-manifest.json`).
- [x] Atomic updates and rollback (`runtime_ensure`, `rollback`).
- [x] LAN authentication and firewall guidance (salted hashed keys, LAN mode requires API key).
- [x] Threat model and dependency audit (safe path names, workspace confinement, no secret leaks).
- [ ] Real-hardware gate validation (physical RX 9060 XT gfx1200 + llama.cpp HIP build).

Pass: clean-machine install succeeds without developer toolchains; rollback restores previous working runtime; LAN mode requires explicit acknowledgement and authentication.
Fail: unsigned executable is installed silently; update destroys working runtime; unauthenticated network access is enabled by default.
