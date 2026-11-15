---
title: A dense model just over VRAM runs every layer on the GPU and spills; a separate draft model is only used with one or two request slots
author: PatrickW
date: 2026-11-15
cost: { minutes: 1, tokens: 2302 }
id: l-23b521ac
scope: repo
status: active
src: agent
---

## Decided
- A dense model within +-10% of VRAM (Windows, GPU) runs with every layer on the GPU and lets the driver spill, with the requested window restored; the split is the fallback. Measured (33B IQ4_XS, 16.4 GiB on a 16 GiB card): 53/51/50/49/48 layers 11.23/11.33/11.16/10.71/9.78 tok/s vs 8.9 for the old 45-layer split. Far over VRAM it flips (19.4 GiB, 22% over: 4.8 all layers vs 5.1-5.3 split), hence the 10% cap.
- The recommender keeps the KV cache type that puts the most layers on the GPU (cache type moved speed 1-6%, layers on the CPU cost far more).
- A DFlash draft is only loaded with <= 2 request slots: at 8 busy slots Long answer ran 11.0 tok/s with it vs 30.7 without. Its file size + 64 MiB is reserved in the plan.
## Tried and ruled out
- Letting the dense-spill rule override a saved tuning (a measured tuning wins).
- Asking Long-answer sections for a word count or to "end on a complete sentence": the model echoed both into the answer.
## In the code
- `src-tauri/src/tuning.rs` (`dense_spill_applies`, `DFLASH_MAX_SLOTS`, `Facts.layers_measured`), `crates/raptor-runtime/src/config.rs` (`recommend_with_arch`, `ModelArch.extra_device_bytes`), `src-tauri/src/state.rs` (`planned_draft_bytes`), `src-tauri/src/sections.rs` (`SECTION_MAX_TOKENS`).
