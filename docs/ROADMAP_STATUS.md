# Raptor LLM - roadmap status (manager-maintained)

Last reviewed: 2026-11-13. Legend: DONE = implemented and covered by automated tests (fakes/fixtures) - HW = still needs real-hardware/live-service verification - PART = partially done - TODO = not started.
"DONE" never means "verified on real GPU/network" unless stated. See docs/TRACKER.md for the build log and docs/EXPERIMENTS.md for what was tried (worked / rejected / untested).

## Release gates (roadmap section 6)
| Gate | Status | Evidence / blocker |
|---|---|---|
| Clean installation w/o dev toolchains | PART | NSIS installer builds (6.8 MB); uninstall script read (keeps data/models); never run on a clean machine |
| GPU detection (RX 9060 XT) | DONE | Real probe on dev machine: gfx1200, 16 GB; stale registry GPUs filtered |
| GPU inference (HIP init + offload proof) | DONE (runtime crate) | Real run on RX 9060 XT: 66/66 layers, 21 tok/s via gate test; stock zip needs HIP SDK >= 7.2 (7.1 refused with explanation). Full app path run live in the real Tauri window (runtime install, import, load, chat, unload) |
| Model download + verify | PART | Resume/hash/atomic-rename tested with fakes; never run vs live Hugging Face |
| Model loading without CLI | DONE (real app path, 1 model) | Shell gate on RX 9060 XT: import->recommend 66/66 @ctx 8192->Verified->21.4 tok/s->chat->unload; other models via fit probe only. Original note:  recommend_config + load path tested with fake spawner |
| Chat streams real responses | DONE (real window) | Streamed chat driven in the real Tauri/WebView2 window on RX 9060 XT (see TRACKER). Original note:  Streaming path + UI honesty tests; no real model run |
| OpenAI-compatible clients | PART | 129 gateway tests; smoke-tested /health, /v1/models, MCP on the running app |
| Hosting: local default, LAN opt-in | DONE (shell wiring pending) | LAN refused without key and TLS; foreign Host rejected; gateway TLS tested (handshake with pinned self-signed cert) |
| Context never silently drops protected instructions | DONE | pack.rs protected-first; tiny-budget test; eval suite |
| Repository retrieval across dependencies | DONE | eval: 2/2 relevant + test vs 0/2 recency; tree-sitter done (8 languages) |
| Memory invalidates stale summaries | DONE | eval test |
| Survives server crashes / interrupted downloads | PART | Fake-spawner crash/restart tests; resume tests; no real-process kill test |

## Critical
**A. Install/foundation:** Windows installer PART (NSIS built, uninstall script read, never run); Linux packages TODO (CI builds Linux, no packaging config verified); no-toolchain runtime install DONE (manifest w/ real sha256, b12340 prerelease - consider pinning stable); hardware detection DONE (Windows real, Linux fixtures); runtime checks PART (`setup_status` in wave 3); runtime/dependency update check + verified install DONE (HW-checked check; install run only in tests), app self-update integration DONE (signed GitHub Releases; first release awaits the Actions signing secret and publication); clean uninstall w/o deleting models DONE (runtime uninstall touches runtimes/ only); persistent config + crash recovery DONE (atomic settings, corrupt backup); offline operation PART (unverified).
**B. Inference/GPU:** llama.cpp integration DONE (process mgr); ROCm/HIP gfx1200 manifest entry DONE, runtime HW; offload verification DONE logic/HW; never silent CPU fallback DONE (recommend errors; verdict CpuOnly warns); load/unload DONE; GPU/RAM monitoring PART (rocm-smi telemetry exists, not surfaced live); load progress DONE (stage-based, real window); per-reply tok/s + TTFT in chat and API DONE (real window); VRAM-aware config + KV cache + flash-attn + ctx DONE (heuristic estimates, flags unverified vs pinned build); health checks/crash recovery DONE (fakes); known-good configs PART (perfmem built, wiring in wave 3).
**C. Chats:** streaming, history, create/rename/delete, stop, regenerate (in-place replace), markdown/code/highlight/copy, installed-model selector, system prompt + params (localStorage), context indicator (estimate), tok/s (estimate), persistence: DONE at UI-test level, none verified against a live backend.
**D. Models:** HF search, GGUF discovery/metadata, size/quant, recommendations, downloads w/ pause/resume/cancel, integrity, import, load/unload, delete, overwrite protection, storage mgmt: DONE with fakes; live HF HW. Gaps: split GGUF shards, revision pin.
**E. Settings/API:** exactly three pages DONE; localhost default DONE; LAN toggle + API auth DONE (TLS wave 3); OpenAI chat/models DONE; port/storage/runtime config DONE; logs/diagnostics DONE (in-memory tail); safe updates TODO.
**F. Context engine:** indexing, git-aware, symbols, import graph, relevance selection, budgeting, truncation, protection, memory, stale invalidation, provenance, harness-independent API, recovery: DONE (regex extractors; tree-sitter wave 3). Tokenizer-exact budgeting PART (wired in wave 3). Summarizer DONE in crate (shell generator pending).

## Important
Hardware optimisation: CPU features DONE; GPU profiles PART; hybrid inference DONE for sparse MoE (oversubscribed all-layer GPU load, measured) and PART for dense (layer split); ctx recommendation DONE; maximum-context windows (Large/Maximum from llama.cpp projections, real GPU) DONE; batch tuning DONE (per-model autotuner: layers x micro-batch via llama-bench, saved per machine/model/runtime); speculative decoding DONE (MTP + n-gram, real GPU); benchmark history/compare PART (wave 3); VRAM fragmentation TODO; quant recommendation DONE; unsupported-arch pre-download detection DONE.
Models advanced: compat ratings DONE; comparison TODO; coding/reasoning categories TODO; update notifications TODO; queue DONE; duplicates DONE; presets DONE (backend; UI not exposed); LM Studio/llama.cpp import DONE.
Context advanced: tree-sitter DONE (8 langs; Linux unverified); embeddings/semantic DONE (real GPU: embeddings enabled at load with pooling, 1,088-chunk workspace, semantic hits; summariser not yet run on hardware); lexical+RRF DONE; task strategies DONE (tests in wave 3); memory + decisions DONE; circular-attempt detection DONE; missing-context retrieval DONE (recovery); monorepo incremental DONE (20k files: cold 5.8s, 1-file update 1.5s); ignore rules DONE; utilisation by category DONE.
Harness compat: OpenCode/generic OpenAI DONE (untested vs real clients); Claude Code (/v1/messages) DONE, count_tokens wave 3; MCP tools DONE (sessions wave 3); tool calling/JSON/streaming tool calls DONE (pass-through tests); aliases DONE; concurrency/queue/per-client limits DONE; shared model sessions TODO.
Chat extras: attachments (text only), export/import, shortcuts, in-conversation search DONE (UI-tested only); folders, mermaid, image input, branching, templates, cross-conversation content search TODO.
Reliability: structured logs DONE (daily files, 7 days kept, app + llama-server); crash recovery DONE (fakes); runtime rollback DONE; load-failure diagnostics DONE; download recovery DONE; regression tests DONE (1115 Rust + 224 UI); hardware compat suite TODO; perf regression benchmarks TODO; DB migrations PART (perfmem schema version); config backup/restore TODO; graceful shutdown DONE; job object (no orphans) DONE (Windows).

## Nice-to-have: all TODO (deliberately deferred).

## Immediate next steps (reviewed 2026-11-13, priority order)
1. Installer on a clean machine/VM (built, never run) and the uninstall behaviour; then Linux packaging.
2. Summariser on real hardware; decide on a dedicated small embedding model instead of embedding with the 27B chat model (~3 chunks/s).
3. DONE: Ornith, granite and flint loaded and GPU-verified in the real app (see TRACKER). Next: quality/long-context checks and the no-`model`-field gateway case.
4. Live Hugging Face search and download.
5. Chat extras: folders, mermaid, image input, branching, prompt templates.
6. Reliability and performance: sustained/thermal run, Vulkan vs ROCm comparison (needs the 38 MB Vulkan runtime; AMD custom/nightly ROCm builds too), hardware compatibility suite, perf-regression benchmarks, config backup/restore, cache pre-warm/no-mmap for dense partial offload.
7. App self-update: implemented for signed GitHub Releases; configure the repository signing secret and publish the first updater-enabled release (see `docs/APP_UPDATES.md`).
Done since the last review: real-window run, runtime update path (b12340 -> b12360 live), logs, speed stats, load progress, splash/icon, quick-load from chat, discovery of known models, embeddings on GPU, project memory view, installer build.

