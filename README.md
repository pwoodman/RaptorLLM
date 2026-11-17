# Raptor LLM — Local AI Runtime & Context Intelligence

Raptor LLM is a cross-platform desktop foundation for local LLM inference, model management,
OpenAI- and Anthropic-compatible API gateway with MCP support, and repository-aware Context Intelligence.

## Current status

Raptor LLM is organized as a high-performance Cargo workspace with dedicated, decoupled subsystems:
- **Three-page desktop UI (Tauri 2 + React + TypeScript)**:
  - **Chats**: Full streaming conversation interface with model picker, parameter tuning, context budget visualization, message regeneration with in-place history replacement, and export/import.
  - **Models**: Hugging Face search, file list inspection with SHA-256 verification, resumable range-based downloads with `.part` staging, local GGUF library inventory, model load/unload, and native file import dialogs.
  - **Settings**: Hardware probing, runtime installation and rollback, honest GPU offload verdict verification, API key rotation and digest authentication, LAN binding toggle, and workspace context management.
- **Independent Rust Subsystems**:
  - `raptor-core`: Shared error types, configuration structures, and token counting traits.
  - `raptor-runtime`: Hardware detection (PowerShell/DXGI on Windows, sysfs/lspci/rocminfo on Linux), hash-verified runtime installation and rollback, job-supervised `llama-server` process lifecycle, and evidence-based GPU verification (HIP/ROCm, CUDA, SYCL, Vulkan, CPU).
  - `raptor-models`: Hugging Face API discovery, pure GGUF header parser, quantization recommendation engine, resumable downloads with SHA-256 integrity checks, and local model library.
  - `raptor-context`: High-priority Context Intelligence engine with repo indexing, dependency graph, BM25 retrieval, task-specific dynamic token budget packing, protected instruction preservation, and provenance tracking.
  - `raptor-gateway`: Axum-based HTTP gateway providing OpenAI endpoints (`/v1/chat/completions`, `/v1/models`), Anthropic translation (`/v1/messages`), MCP tools surface (`context_build`, `context_search`, `repo_symbols`, `repo_read_file`), and Bearer API key authentication.
  - `src-tauri`: Desktop application shell bridging the subsystems with type-safe commands, native file/folder dialogs, and event channels.

## Requirements

- Node.js 20+
- Rust stable (1.80+)
- Windows 11 / 10, or Linux (with WebKit2GTK prerequisites)
- Windows GPU runtimes: AMD HIP/ROCm, NVIDIA CUDA, Intel SYCL, with Vulkan available as a cross-vendor fallback. A runtime is usable only when its bundled llama.cpp build can see a device; Raptor checks this before loading and verifies offload from the server log. Hardware-level validation results are tracked in [the Windows GPU matrix](docs/WINDOWS_GPU_VALIDATION.md). Raptor LLM never silently claims CPU fallback is GPU acceleration.

## Development

Install dependencies:
```bash
npm install
```

Run in development mode (starts Vite + Tauri 2 window):
```bash
npm run tauri dev
```

Run test suites:
```bash
# Rust workspace unit tests
cargo test --workspace

# UI test suite (Vitest)
npm test

# Production build
npm run build
```

## Architecture & Project Layout

```
RaptorLLM/
├── crates/
│   ├── raptor-core/      # Shared vocabulary, errors, and common primitives
│   ├── raptor-runtime/   # Hardware probe, runtime manifest, llama-server lifecycle & GPU verify
│   ├── raptor-models/    # Hugging Face client, GGUF metadata parser, downloads & library
│   ├── raptor-context/   # Context Intelligence: scanner, code graph, retrieval & packing
│   └── raptor-gateway/   # Axum API gateway (OpenAI, Anthropic translation, MCP server)
├── src-tauri/            # Tauri 2 desktop shell, command adapters, and background runners
├── src/                  # React + TypeScript frontend (Chats, Models, Settings)
├── docs/                 # Architecture, specifications, and build tracker
└── .github/workflows/    # CI automation workflow
```

## Security & Reliability

- **Localhost by default**: Serves on `127.0.0.1:8790`. LAN mode (`0.0.0.0`) is strictly opt-in and refused without an active API key.
- **Hashed Keys**: API keys are persisted only as salted SHA-256 digests; the plaintext key is displayed only once upon rotation.
- **Integrity Verification**: Model downloads verify HF LFS SHA-256 hashes before promoting files to the library. Runtime packages verify sha256 checksums before extraction.
- **Sandbox Protection**: MCP file reading is strictly confined to registered workspaces and validates against directory traversal.
- **Process Isolation**: Child `llama-server` processes run inside dedicated OS Job Objects on Windows to prevent orphaned processes upon app exit.
