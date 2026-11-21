---
title: Models with their own draft head speculate two tokens at a time; longer drafts are slower on this AMD card except for copied text
author: Patrick
date: 2026-11-21
branch: codex/windows-gpu-ux-speed-review
cost: { minutes: 22, tokens: 64142 }
id: l-56471a43
scope: repo
status: active
src: agent
touches:
  - crates/raptor-runtime/src/config.rs
---

## Decided
- MTP models launch with a draft length of 2 (llama.cpp's default is 3). Qwen3.8-27B, stacked `draft-mtp,ngram-simple`, RX 9060 XT ROCm b12374-patched, 5 prompts x 2 reps at temperature 0.6: 36.0 tok/s mean at 2 vs 32.5 at 3 (+11%, faster on every prompt), 21.4 with no speculation.
## Tried and ruled out
- Longer MTP drafts (4/6/8). Acceptance falls 70% -> 31% and each extra token costs real verify time on AMD: prose fell 29 -> 17 tok/s at 8, below no speculation; only rewrites gained (40 -> 58). The +52-55% at length 6 reported for llama.cpp 1.0 is an RTX 3090 result from an NVIDIA-only small-batch kernel; do not copy it here.
## Watch out
- Measured on ROCm only, 8K context, thinking off. Vulkan's small-batch verify cost differs, so re-measure before assuming 2 there; the best length belongs in the autotuner eventually.
## In the code
- `crates/raptor-runtime/src/config.rs`: `MTP_DRAFT_N_MAX`, the `SpecMode::Mtp` branch in `LaunchConfig::args`. Harness: `scripts/bench_spec.py --spec draft-mtp,ngram-simple -- --spec-draft-n-max N`.
