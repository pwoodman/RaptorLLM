# Raptor LLM Architecture

**Status:** describes the code at commit `14e78d8` (verified 2026-11-15). Detailed behaviour lives in
the specs; this file is the map and the rules. When the code and this file disagree, fix one of them
in the same change.

Related: [RULES.md](specs/RULES.md) (engineering rules), [runtime](specs/runtime.md),
[models](specs/models.md), [context](specs/context.md), [gateway](specs/gateway.md),
[UI contract](specs/ui-contract.md), [EXPERIMENTS.md](EXPERIMENTS.md) (measurements behind the
speed rules).

## What it is

Raptor LLM is a local-first desktop app for running GGUF models with llama.cpp on the user's own
GPU, and for serving them to coding agents through an OpenAI- and Anthropic-compatible gateway with
built-in code-context retrieval.

## Processes and how they talk

```
 WebView UI (React)  ──Tauri IPC (commands + events)──►  Tauri host process (Rust)
                                                          ├─ AppState (src-tauri)
                                                          ├─ Gateway (Axum, 127.0.0.1:8790) ◄── agents / API clients
                                                          ├─ Context engine (in-process library)
                                                          └─ loopback HTTP ──► llama-server (child process)
```

1. **WebView UI** (`src/`): presentation only. Talks to the host through Tauri commands and events,
   never to llama-server or the gateway directly.
2. **Tauri host** (`src-tauri/`): owns all state (`AppState`), runs the gateway and the context engine
   in-process, and supervises llama-server.
3. **llama-server**: a separate process for inference, bound to `127.0.0.1` by default (a
   non-loopback bind needs an API key). On Windows it runs in a Job Object with
   kill-on-close, so it cannot outlive the app (`src-tauri/src/job.rs`). "Isolated" means a separate
   process, not a sandbox.

The in-app chat and the gateway share one upstream (`RuntimeUpstream` in `src-tauri/src/adapters.rs`);
in-app chat calls it directly, external clients go through the gateway's HTTP routes. llama-server's
own port is never offered to clients.

## User interface

Three pages (`src/App.tsx`), each a thin view over Tauri commands; state lives in small stores under
`src/state/`:
- **Chats:** conversations, model picker, composer (thinking effort, Long answer), reply speed.
- **Models:** tabs Discover, Library, Downloads, Performance, Compare and rank
  (`src/pages/Models.tsx`); loading, presets and load-failure handling live here.
- **Settings:** Hardware, Inference runtime, Dependencies, Server and API, Storage, Context engine,
  Diagnostics, About (`src/pages/Settings.tsx`).

## Crates and the dependency rule

| Crate | Owns |
|---|---|
| `raptor-core` | Shared types: `RaptorError`, `DataDirs` (paths), chat types, `TokenCounter`, log files |
| `raptor-runtime` | llama.cpp runtimes (manifest, selection, install), launch config and memory recommendation, process lifecycle, GPU verification, fit projection, tuning and performance memory |
| `raptor-models` | GGUF header parsing, model library, Hugging Face discovery and downloads, presets, quant advice |
| `raptor-context` | Scan, Tree-sitter parse, index, graph, retrieval, budgeted packing, provenance, memory; facade `api::ContextEngine` |
| `raptor-gateway` | Axum server: OpenAI and Anthropic adapters, auth, rate limit and queue, TLS, CORS, request hygiene, thinking effort, MCP, `/raptor/*` routes |
| `src-tauri` (`raptor-llm`) | Composition root: `AppState`, Tauri commands, adapters implementing the gateway's traits, settings, conversations, speed plan (`tuning.rs`), long answers (`sections.rs`), updates |

**Rule:** a subsystem crate depends only on `raptor-core`; no subsystem crate depends on another.
Only `src-tauri` uses them all and wires them together. The gateway reaches the rest of the app only
through its traits (`Upstream`, `ContextService`), which `src-tauri/src/adapters.rs` implements.
Every subsystem puts its I/O behind traits (`HfApi`, `Spawner`, `CommandRunner`, `HardwareProbe`,
`GpuMemoryProbe`, ...) so it can be tested with fakes.

## Main flows

- **Load a model:** UI command → `AppState::load_model` → memory recommendation
  (`raptor-runtime::config::recommend_with_arch`, calibrated by llama.cpp's own projection) → speed
  plan (`tuning::plan`: speculative decoding mode, layer placement) → try each launch variant until
  one reaches Ready → GPU verification → one benchmark generation recorded in performance memory.
- **Chat in the app:** UI command `chat_start` → shared upstream → llama-server; text streams back as
  `chat://<requestId>` events.
- **Agent request:** HTTP → gateway auth, rate limit, hygiene → adapter (OpenAI or Anthropic shape) →
  shared upstream → llama-server; context routes and MCP tools call the context engine.

## GPU verification and runtime selection

A GPU run is reported only on positive evidence. Today that evidence is:
1. a device preflight (`llama-server --list-devices`) for the selected backend;
2. the llama-server log: backend initialisation, devices, offloaded layer count and GPU buffers
   (`raptor-runtime/src/verify.rs`), giving a verdict of `Verified`, `CpuOnly` or `Unverified`.

HIP, CUDA and Vulkan are verified the same way. A CPU run when a GPU was requested happens only when
the user turned on `allow_cpu_fallback` (off by default), and is reported as a warning. Runtimes come
from a seeded manifest (`src-tauri/resources/runtime-manifest.json`), selected by OS, GPU architecture
and minimum driver (`select_runtime`). The ROCm-or-Vulkan choice is per model
(`src-tauri/src/backend_choice.rs`). Rollback, switch and uninstall are refused while a model is
loaded.

Not done: live GPU telemetry (a `GpuTelemetry` trait exists but nothing calls it) and a dedicated
inference smoke test as part of the verdict.

## Context engine

Implemented, not future. An in-process library behind `raptor_context::api::ContextEngine`, exposed
to agents through the gateway's `/raptor/context/*` routes and MCP tools. The pipeline (scan with
ignore rules and secret exclusion, Tree-sitter parsing, incremental index and graph, weighted
rank fusion over several retrieval signals, budgeted packing, provenance) is specified in
[context.md](specs/context.md). The rules this architecture depends on:
- exact source is never summarised away for a file being edited;
- protected items (instructions, active edits and the categories in `raptor-context/src/budget.rs`)
  may exceed the budget, and the overflow is reported (`over_budget_by`), never silently cut;
- every pack carries a provenance manifest of what was included or excluded and why.

## State and storage

- Locations: `DataDirs` (`raptor-core/src/paths.rs`).
- Stores: JSON files for settings, conversations, saved tunings and performance memory, written with
  `fsutil::write_atomic`; files are versioned and older ones load with defaults.
- Logs: a rolling `llama-server.log` and the app log.
- Errors: one `RaptorError` type across crates, serialised to the UI as `{kind, message}`.
- The UI's view of commands and events is defined in [ui-contract.md](specs/ui-contract.md).

## Security

**Threat model:** one user on their own machine. Untrusted inputs: other devices on the LAN, model
files and their metadata, downloaded runtimes and updates, and repository content sent to the context
engine.

**Controls in place:**
- the gateway binds loopback by default, keyless only for loopback clients with a loopback `Host`
  header;
- LAN mode is refused without at least one API key, and without TLS unless `allow_insecure_lan` is
  set; a plain `0.0.0.0` bind is refused;
- API keys are stored salted and hashed; per-key rate limit and request queue; exact-origin CORS
  allowlist; a request body limit;
- llama-server binds loopback; a non-loopback bind needs an API key;
- paths and download names are validated; context tools stay inside the registered workspace.

**Known gaps** (do not describe these as done anywhere):
- runtime manifest signatures are not verified yet (`raptor-runtime/src/manifest.rs`);
- the update digest comes over the same channel as the download, so it is not an independent signature
  (`src-tauri/src/updates.rs`);
- `use_local_runtime` runs an unverified local binary; it is a developer path the app does not call.

## Testing

- Unit tests use the trait fakes; they never need a network, a GPU or llama-server.
- Real-hardware tests are a separate, gated tier (`raptor-runtime/tests/gate_real_hardware.rs`,
  `src-tauri/tests/gate_real_app.rs`).
- CI (`.github/workflows/ci.yml`) runs format, clippy with warnings as errors, and the tests. It does
  not yet check the crate dependency rule.
