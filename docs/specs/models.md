# raptor-models - Model Manager spec
Owns: Hugging Face discovery, GGUF metadata, downloads, local library, integrity, imports.

Modules
- `hf`: client behind trait `HfApi` (real reqwest impl + `FakeHf`): search GGUF repos (filter gguf, sort downloads/likes), list repo files (sizes, LFS sha256 from tree API), basic repo card info. Parse responses with pure fns tested on JSON fixtures. Handle rate limits/HTTP errors; optional HF token (never logged).
- `gguf`: pure GGUF header parser (read KV metadata only, never tensors): architecture, name, n_ctx_train, n_layers, embedding length, head counts, quantization (file_type), tokenizer model, chat-template presence. Works from a file AND from a byte prefix (HTTP range) so unsupported architectures are detected BEFORE a multi-GB download. `is_architecture_supported(arch)` against a maintained list. Truncated/corrupt/huge-length inputs must return Err - never panic or over-allocate (cap string/array lengths).
- `quant`: parse quant label from filenames (Q4_K_M, IQ3_XS, F16...), bits-per-weight table, `recommend_quant(options, ram, vram, ctx_target) -> ranked list with reasons`, `compat_rating(model, hw_summary) -> Great/Ok/Tight/WontFit` (plain input struct; do not depend on raptor-runtime).
- `download`: `DownloadManager`: queue with concurrency limit, HTTP Range resume, `.part` files in `DataDirs::downloads()`, pause/resume/cancel, progress via tokio broadcast/watch, disk-space preflight (trait `DiskSpace`), sha256 verify against HF LFS hash before promoting to library (atomic rename), mismatch => delete + `Integrity` error, never overwrite an existing model (`Conflict`) unless `overwrite=true`, restart recovery (scan `.part` + sidecar JSON). Transport behind trait `HttpFetch`; tests use a fake with range support and injected failures.
- `library`: model store under `DataDirs::models()` (publisher/repo/file), JSON index (or rusqlite with `bundled`), entries with gguf meta, sha256, size, added_at, source. Ops: list, get, delete (explicit), import file (copy/link, validate GGUF header), scan folders to import existing LM Studio (`~/.lmstudio/models`) and llama.cpp dirs, duplicate detection by hash/size, storage usage summary, partial files must never show as installed.
- `presets`: per-model presets (system prompt, sampling, ctx) CRUD in JSON.

Tests: >= 45 unit tests incl. resume after injected mid-stream failure, hash mismatch, path traversal in HF filenames, duplicate import, corrupt GGUF.
Facade: `ModelManager` struct re-exported from lib.rs.
