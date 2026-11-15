# Raptor LLM Architecture

## Product structure

Three top-level pages only:
1. Chats — conversation experience and installed-model selection.
2. Models — Hugging Face discovery, GGUF download, installed model inventory and load/unload.
3. Settings — runtime paths, hardware, performance, context intelligence, server/API, storage and appearance.

## Process boundaries

- Tauri UI: presentation and user interaction.
- Rust application core: configuration, model library, process lifecycle, download orchestration.
- llama-server: isolated inference process; OpenAI-compatible HTTP API.
- Context Intelligence: independent Rust module/service boundary. Future versions should expose a stable local API and MCP tools.

## GPU strategy

Do not infer acceleration from the GPU name or a successful process launch. Verification must combine:
- llama.cpp build/backend metadata;
- runtime logs showing HIP backend initialization and device selection;
- actual offloaded layer count;
- live GPU utilization/VRAM telemetry where available;
- a small inference smoke test.

RDNA 4 `gfx1200` support depends on the exact driver, ROCm distribution, llama.cpp commit/build, and OS. Maintain tested runtime profiles per OS and driver family. Never silently claim CPU fallback is GPU acceleration.

## Context Intelligence design

Future pipeline:
1. Workspace scan with ignore rules and secret exclusions.
2. Parse source using Tree-sitter; extract files, symbols, imports, calls and tests.
3. Build incremental dependency graph with content hashes.
4. Retrieve candidates using lexical, symbol, graph and optional embedding retrieval.
5. Score by task relevance, dependency distance, recency, test coverage, change status and explicit pins.
6. Pack under a model-tokenizer-derived budget with protected instructions and active edits.
7. Emit provenance manifest: included/excluded items, score, source hash and reason.
8. Invalidate summaries and graph edges when files change.

Never summarize away exact source for a file currently being edited. Keep user/system instructions, task acceptance criteria, active diffs and unresolved test failures protected. Protected items can exceed a nominal budget; report this rather than silently deleting them.

## Security

- Localhost-only by default.
- LAN binding is opt-in.
- Add API-key enforcement before treating LAN mode as production-ready.
- Never bind to 0.0.0.0 without a clear warning.
- Validate all paths and download names.
- Add signed update manifests and checksums before automatic updates.
- Do not download or execute arbitrary runtime binaries without signature/hash verification.
