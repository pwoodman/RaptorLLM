# Experiments log: what was tried, what works, what does not, what is untested

Read this before trying something. Update it in the same turn as the experiment, not later.

**Rules for entries**
- One line per finding: what, how it was measured, the numbers, the verdict.
- Verdicts: **WORKS** (measured, and applied or ready to apply), **REJECTED** (measured, worse or no gain), **BROKEN** (a bug or failure with the cause), **UNTESTED** (an idea or a gap, with what would settle it).
- Say where it was measured. Everything below is the dev machine unless noted: Windows 11, Ryzen 7 3800X (8C/16T), 64 GB RAM, Radeon RX 9060 XT 16 GB (gfx1200), HIP SDK 7.1 + 7.2, llama.cpp b12340 and b12360 (HIP).
- A result that is only a unit test or a fake is not "works on hardware". Say so.
- When something is retried and the verdict changes, edit the line and keep the old number in brackets.

Models used: Qwen3.8-27B IQ3_S (dense/hybrid, 10.4 GiB, has an MTP head), granite-4.2-30b IQ3_M (dense, 12.2 GiB), Ornith-1.5-35B-A3B IQ3_S (MoE 256 experts/8 used, 14.4 GiB, no MTP head), flint Q4_K_M (MoE 256 experts/6 used, 22.5 GiB, larger than VRAM).

---

## WORKS

### Speed
- **Speculative decoding, `--spec-type draft-mtp`** (the model's own multi-token-prediction head). Qwen3.8-27B: 21.5 -> 47.5 tok/s (2.2x) through the app; temperature 0.8 still 42-47 tok/s, 85-90% of drafted tokens accepted. Only for GGUFs with `nextn_predict_layers` > 0. Applied by default.
- **Speculative decoding, `--spec-type ngram-mod`** (no second model). granite 20.4 -> 25.9 on a first reply, 57.9 once a reply repeats text; Ornith 66.8 -> 77.8 first, 92 warmed; flint neutral on a first reply (33.5 vs 33.6), 41 warmed. Applied by default when there is no MTP head.
- **Sparse MoE larger than VRAM: request every layer on the GPU and let the Windows driver keep cold experts in shared memory.** flint 26.6 -> 35.0 tok/s (llama-bench) and prompt 474 -> 699; Ornith 47.8 -> 68.4 tok/s. Works because only 6-8 of 256 experts are read per token. Gated to Windows, <= 1/16 experts per token, model <= VRAM + half of RAM, 6 GiB RAM spare; falls back step by step. Applied.
- **Per-model autotuner** (layers x micro-batch via llama-bench, saved per machine/model/runtime). Results: Ornith 34% less time for a 4096-token prompt + 256-token reply (41 layers ub 256 vs 39 layers ub 512); flint 31% (all 50 layers ub 256 vs 30 layers ub 512); Qwen 3% (ub 256); granite 0% (default already best). Applied; saved in `config/tuned.json`.
- **Micro-batch is model-specific**: Ornith pp2048 2,235 tok/s at ub 256 vs 866 at 512; flint best at 512 in one sweep (695) and 256 in the in-app tune (585 vs 479, different candidate layer sets); dense models within +-8%. Hence the autotuner, not a fixed rule.
- **KV-cache type barely changes speed** (Qwen, llama-bench): within 1% at depth 0 and 16,384 (21.2 / 21.0 / 21.3 tok/s at 0; 19.6 / 19.3 / 19.4 at 16k) and up to ~6% slower generation at 32K (18.6 / 17.5 / 17.7). Quality: see the perplexity result above (unchanged within noise at 2K). So KV type is mostly a memory choice on this GPU.
- **Slot count (`-np`) costs memory, not speed**: Qwen needs ~450 MiB more at 4 slots than at 1 (recurrent state per slot). Measured with llama-fit-params.
- **Threads matter only when the CPU does work.** granite with 25 layers on the CPU: generation 1.2 / 2.3 / 3.1 / 3.6 / 4.0 / 4.1 tok/s at 2 / 4 / 6 / 8 / 12 / 16 threads, so logical cores (16) beat physical (8) by ~13%. With everything on the GPU (Qwen, flint) threads 1-16 are the same within noise.

### Context window, KV cache, slots, threads (applied 2026-11-14)
- **"Maximum context" is computed, not guessed**: two llama.cpp projections (`llama-fit-params`, ~1 s each, no weights loaded) are extrapolated and a third confirms the predicted length (`context_max.rs`). The Library's "Context window..." offers Default / Large (8-bit cache) / Maximum (4-bit cache, 1 slot). Qwen3.8-27B in the real app: default 8,192; Large 41,984 (MTP kept); Maximum 154,624 (MTP off). granite: 8,192 / 13,312 / 24,576.
- **Measured on hardware**: Qwen Large (42k, q8_0, 2 slots, MTP on) 66/66 layers, no spill, generation 29.0 tok/s (short prompt) and 25.3 tok/s with a 12k-token prompt (story prompt, temperature 0; code prompts give MTP ~47). Qwen Maximum (155k, q4_0, 1 slot, MTP off) 66/66, 21.3 / 19.9 tok/s, and an 82,640-token prompt completed in 224 s (408 prompt tok/s) with generation 13.6 tok/s at that depth. Answer quality at those lengths and with a 4-bit cache was not measured.
- **llama.cpp's projection under-counts real GPU use for an MTP model.** Dedicated memory minus the 586 MiB the desktop uses, against the projection for the same launch: granite +257 MiB; Qwen with its MTP head +1,531 MiB at 8,192 tokens and +2,335 at 166,912 (a fixed ~1.25 GiB plus ~5.2 KiB per context token for the MTP draft context). Without this the 167k window put 2.3 GB into shared memory and the driver spilled (see BROKEN). The solver now adds `Overhead::BASE` (300 MiB) and, for MTP models, `Overhead::MTP`.
- **MTP and a quantized cache coexist**: `draft-mtp` at ctx 8,192 to 41,984 with q8_0 gives 46.8-47.0 tok/s on the code prompt (f16 47.1, q4_0 45.9), np 2. The draft context is the memory cost, not speed.
- **Maximum turns MTP off on purpose**: it roughly doubles the window (Qwen ~80k -> 155k) at about half the generation speed (47 -> 21 tok/s). The option says so.
- **Threads** (partial offload only): with layers left on the CPU the app now uses every logical processor (granite 25 layers on CPU: 3.6 -> 4.1 tok/s). With everything on the GPU nothing changes.
- **Slots**: new setting "Requests at once", default 2 (was llama-server's 4), 1-8. Saves ~300 MiB at 2 and ~450 MiB at 1 on Qwen; speed unchanged.
- **Autotune results for the other models** (in-app, saved): granite 0% (default best), Qwen +3% (micro-batch 256), flint +31% (all 50 layers at 256: 14.5 s vs 18.9 s for a long prompt plus reply).

### Context-length matrix and KV-cache quality (2026-11-14)
- **Context-length matrix** (Library > "Benchmark context lengths", saved per model/machine/runtime; llama-bench, plain decoding, cheapest cache that fits each depth, peak memory sampled every 2 s). **Qwen3.8-27B, 12.4 min for 0-128K, no spill at any depth**:

  | context | cache | prompt tok/s | generation tok/s | wait for a 512-token turn | peak VRAM |
  |---|---|---|---|---|---|
  | empty | f16 | 601 | 21.5 | 0.9 s | 12.3 GB |
  | 4K | f16 | 574 | 21.2 | 0.9 s | 12.5 GB |
  | 16K | f16 | 510 | 20.1 | 1.0 s | 13.2 GB |
  | 32K | f16 | 445 | 18.8 | 1.2 s | 14.3 GB |
  | 64K | q8_0 | 344 | 12.2 | 1.5 s | 14.5 GB |
  | 128K | q4_0 | 240 | 11.1 | 2.1 s | 14.9 GB |

  Reading a whole prompt from empty (integrated from the measured speeds): 4K 7 s, 16K 30 s, 32K 64 s, 64K 149 s, 128K 380 s (6 min 20 s). Generation roughly halves by 64K; at 128K a long first prompt is the real cost.
- **Ornith-1.5-35B-A3B (sparse MoE, spills by design), quick run to 32K, 3.2 min**: generation 56.5 / 65.0 / 52.4 / **11.1** tok/s at empty / 4K / 16K / 32K, prompt 2,184 / 1,973 / 644 / 1,283 tok/s. Peak VRAM sits at 15.5-16.0 GB and shared-memory growth appears from 16K (366 MiB) and 32K (458 MiB, flagged). **Its usable window at full speed is about 16K**; at 32K generation falls to a sixth. This is exactly what spot checks at 4K hide.
- **KV-cache type at the same depth** (Qwen, llama-bench at 32,768): f16 18.64 / q8_0 17.45 / q4_0 17.69 tok/s generation, prompt 439 / 430 / 433. So the cache costs up to ~6% generation speed at 32K (at 16K and below it was within 1%). The 32K -> 64K drop in the matrix (18.8 -> 12.2) is therefore mostly depth, not the 8-bit cache.
- **KV-cache quality: perplexity is unchanged within noise** (user reported "q4 for KV has little to no degradation"; measured to check). Qwen3.8-27B, `llama-perplexity` on ~100 KB of this repo's docs (prose, tables, code), context 2,048, 24 chunks, flash attention on: f16/f16 **7.5646** +/- 0.165; q8_0/q8_0 7.5780 (+0.18%); q4_0/q4_0 7.5761 (+0.15%); K q8_0 / V q4_0 7.5738 (+0.12%). All inside the error bar. Caveats: a 2K context and one model; degradation at 32K-128K, on other models and on a task-level (coding) benchmark was NOT measured. The app's "Large" and "Maximum" window notes now quote these numbers.

### ROCm (HIP b12360) vs Vulkan (b12360) on the RX 9060 XT (2026-11-14, llama-bench, ngl 99, ub 256, 1 run; tok/s)
Flash attention ON (what the app uses), depth 0 / 16K:

| model | ROCm pp | Vulkan pp | ROCm tg | Vulkan tg |
|---|---|---|---|---|
| Qwen3.8-27B | 596 / 506 | 483 / 421 | 21.6 / 19.9 | 21.1 / 19.6 |
| granite-30B (spills) | 695 / 207 | 409 / 188 | 20.0 / 6.2 | 21.9 / 8.1 |
| Ornith-35B-A3B (MoE) | 1683 / 702 | 534 / 534 | 44.2 / 42.5 | 72.8 / 44.9 |
| flint-40B-MoE (spills) | 564 / 462 | 343 / 328 | 27.9 / 18.3 | 38.4 / 21.4 |

- **Prompt processing: ROCm wins everywhere with fa=1** (up to 3x on Ornith at depth 0). **Generation: Vulkan wins on MoE/spilling models** (Ornith 72.8 vs 44.2 at depth 0, flint 38.4 vs 27.9) and ties on fully-resident dense Qwen. Both are one run each; Ornith's fa=0 ROCm tg (43.7) matches fa=1, so the Vulkan lead is real, not noise from fa.
- Without flash attention ROCm collapses at depth (Qwen pp 106, granite tg 2.9); Vulkan degrades less. The app always uses fa=1, so this only matters if fa is off.
- Takeaway: a per-model backend choice is justified for MoE/spill models (generation), ROCm stays better for prompt-heavy/agent use. Not yet wired into the app; the Vulkan runtime lives in the scratchpad only. Caveat: llama-bench is plain decoding (no MTP/ngram speculation), which the ROCm path uses in the app.

### Prompt-prefix reuse, quantized cache on spilling models, Lemonade ROCm build (2026-11-14)
- **Prefix reuse WORKS** (llama-server defaults: `--cache-prompt` on, `--cache-ram` 8 GiB; the app changes nothing). Qwen, 17K-token prompt: first turn 31.5 s (552 tok/s); same prefix + new question 1.5 s (only ~260 tokens read). **An edit in the middle of the prompt re-reads everything (31.5 s again)** - Qwen is a hybrid model that cannot rewind to a partial prefix. So agents should keep prompts append-only (system prompt, tools, history, then the new turn).
- **Quantized KV on spilling models at 32K** (llama-bench -d 32768, fa on, single run, so noisy): Ornith tg f16 25.3 / q8_0 11.1 / q4_0 47.2 tok/s, shared-memory growth 763 / 465 / 398 MiB; flint tg 15.0 / 12.2 / 16.9, growth 9,595 / 9,105 / 8,870 MiB. q4_0 spilled least and was fastest on both, but the q8_0 dip looks like an outlier; needs repeats before changing the default.
- **Lemonade ROCm build b1375** (github.com/lemonade-sdk/llamacpp-rocm, `llama-b1375-windows-rocm-gfx120X-x64.zip`, 691 MB, sha256 5c0689eb... verified against the release checksums; extracted to its own folder, llama-bench only, 2 runs, fa on). Qwen3.8-27B prompt **746 vs 590 tok/s (+26%) at empty, 616 vs 506 (+22%) at 16K**; generation identical (21.7 vs 21.6; 20.1 vs 20.1). Ornith: within noise (2206 vs 2171 pp, 57.3 vs 56.1 tg). So the newer build helps dense-model prompt processing only. Not integrated; updating the app's HIP runtime source to this build is a candidate.

### Vulkan live in the app (2026-11-14, Ornith-1.5-35B-A3B, n-gram speculation on, 3 identical 300-token chat requests)
- **Install and load WORK in the real app.** `dependencies_install_vulkan` installed Vulkan b12374 (hash-verified, next to ROCm); Ornith set to Vulkan loaded in 21 s with the GPU verdict "verified" (backend vulkan, 41/41 layers, flash attention on), device Vulkan0 RX 9060 XT 16,304 MiB, spill warning 568 MiB (expected for this MoE).
- **Generation, Vulkan vs ROCm in the app:** Vulkan 71.4 / 70.1 / 120.2 tok/s; ROCm (HIP b12374) 51.6 / 59.8 / 83.4 tok/s, i.e. **Vulkan about 35-45% faster** (the third run is a repeated reply that n-gram speculation likes). Prompt speed was not compared here (llama-bench showed ROCm faster). Confirms the llama-bench result with speculation on.
- Still untested: MTP on Vulkan (guarded off), a dense model on Vulkan in the app, chat/API through it for long sessions.

### Review of `llama_cpp_optimization_guide_nov2026.md` (2026-11-14; second guide, same flags exist in b12374)
- **Already agrees with our measurements:** Vulkan wins generation / ROCm wins prefill; never lower `-ngl` on MoE; drop a quant tier instead of spilling dense layers; `-np 1` for single user; KV quant costs speed at long context; n-gram + MTP speculation.
- **Contradicted by our measurements:** `--cpu-moe`/`--moe-cache-mib`/`--moe-cache-cpu-share` (guide's Profile B) were 2-3x slower than oversubscription on this card (flint 10-21 vs 32.6 tok/s); the guide's numbers are an RTX 5060 Ti. Its "never go below q8_0 for K" is untested here beyond 2K perplexity. Its gfx1201 is wrong for the 9060 XT (gfx1200). iGPU display tip n/a (3800X has no iGPU).
- **NEW, untested, to try (the app passes none of these today):** `--cache-reuse 256`; `-cms 4096 -ctxcp 64` (hybrid-model checkpoints, should fix the full re-read after a mid-prompt edit that we measured on Qwen); `--no-context-shift` (avoids silently dropping tool definitions in agents); `-fit off` (stop llama.cpp shrinking our chosen context); stacked `--spec-type draft-mtp,ngram-mod` (our SpecMode is either/or); `ngram-map-k4v`; `--slot-save-path` for warm contexts across restarts; stripping old `<think>` blocks and volatile timestamps in the gateway; dual-model routing (small fast model for sub-agents).

### Second guide's flags, tested (2026-11-14, b12374, ROCm, Qwen3.8-27B unless noted)
- **`--cache-reuse 256` + `-cms 4096 -ctxcp 64`: NO EFFECT on a hybrid model.** The server logs "cache_reuse is not supported by this context, it will be disabled". A mid-prompt edit still re-read all 17K tokens (31.7 s vs 31.5 s without the flags); repeat turns stay 1.5 s. Rejected for Qwen. Not tested on a pure-transformer model (where cache-reuse may work).
- **Stacked speculation `draft-mtp,ngram-mod`: NO GAIN on a fresh reply.** Rust-code request, first reply: none 21.5, draft-mtp 31.3, ngram-mod 20.3, stacked 31.4 tok/s. Replies 2 and 3 of the *identical* request jump (ngram 44.8 / 93.7, stacked 63.8 / 79.8) only because n-gram copies the earlier reply; that is not a gain on new work. Keep MTP for MTP models, n-gram otherwise. (This code prompt gave MTP 31 tok/s vs 47.5 measured earlier on a different prompt: MTP speed depends heavily on the text.)
- **`-fit off`: NOT NEEDED.** With an explicit `-c 163840 -ngl 99` the server kept n_ctx 163,840; `--fit` only adjusts unset arguments.
- **`--no-context-shift`: NOT NEEDED.** Context shift defaults to disabled in this build.
- Still to test: `ngram-map-k4v`, `--slot-save-path`, gateway `<think>` stripping, repeat of the 32K KV-type test (running).

### Deep tune (opt-in long-prompt setting) 2026-11-14 - UNTESTED live
- Built, unit-tested only (no GPU run yet): "Also tune for long prompts" re-measures every quick-tune setup (layers x micro-batch 256/512) with llama-bench at 16,384 tokens of context, saves the fastest as a second, long-prompt setting, and loads use it for windows >= 16,384. Hypothesis to check: the micro-batch/layer ranking at 16K differs from the one at empty context (it did for the KV type, and ROCm vs Vulkan flips with depth). If it never differs on Qwen/Ornith/flint, the option is not worth its extra minutes and should be dropped. Expected cost: one extra model load plus a 16K prefill per setup (typically 2-4 setups); not yet timed.

### Gateway agent hygiene: stripping old reasoning, volatile-prompt warning (2026-11-14; unit/integration tests only, LIVE BEHAVIOUR UNTESTED)
- **Built** (from the second guide): reasoning of earlier assistant turns (`<think>` blocks, `reasoning_content`) is removed from chat requests before they reach llama-server; reasoning in the turn in progress is kept. Intended effect: 2-5K thinking tokens per turn no longer fill a 32-64K window in a few agent steps, and the stripped history is identical on every request, so the prompt-prefix cache (1.5 s repeat turn, measured above) still matches. A volatile clock time / UUID near the top of the system prompt is flagged once in the log (it makes the prefix differ from token 0, so every turn would pay the full 31 s read measured on Qwen).
- **Untested live:** no real agent client or model was run. Not measured: tokens saved on a real session, whether any client breaks when the model sees history without its own reasoning (Qwen's own chat template drops it for earlier turns, which is why this is expected to be safe). Orphan `</think>` with no opening tag (reasoning opened by the template) is NOT stripped, to avoid cutting real text that mentions the tag; llama-server normally returns that reasoning in `reasoning_content`, which is handled. The in-app chat path is not covered.

### Thinking effort per request (2026-11-14, Qwen3.8-27B UD-IQ3_S, llama-server b12374 `--jinja`, temperature 0)
- **WORKS (measured probe):** `GET /props` returns `chat_template_caps` with `supports_reasoning_effort: true` and `reasoning_effort_levels: ["low","medium","xhigh"]`. That is the detection source: no capability, no control. A per-request `reasoning_effort` field on `/v1/chat/completions` changes the thinking without reloading the model. Easy question ("Is 91 prime?"): low/medium 75 chars of thinking, 48 tokens, 2.5-2.6 s; default/high/xhigh 121 chars, 66 tokens, 3.3-3.7 s. Harder question (sheep puzzle + Python function, 4000 max tokens):

  | request | thinking chars | answer chars | tokens | seconds |
  |---|---|---|---|---|
  | default | 3,521 | 938 | 1,290 | 63.4 |
  | low | 1,084 | 832 | 681 | 32.7 |
  | medium | 1,227 | 1,312 | 871 | 41.2 |
  | xhigh | 3,521 | 938 | 1,290 | 61.1 |
  | `chat_template_kwargs {enable_thinking:false}` | 0 | 2,542 | 746 | 37.3 |

  So **xhigh is the model's default** (identical output to no field), low is about half the time, and `enable_thinking:false` also works per request. **A per-request token budget is IGNORED** (`thinking_budget_tokens` / `reasoning_budget` of 64 changed nothing); only the server flag `--reasoning-budget` caps thinking, so the app does not offer a token cap. One question each, one run each: the sizes are indicative, not a benchmark.
- **Built (unit/integration tests only, LIVE UI UNTESTED):** four levels Low / Medium (default) / High / Max, for a model with `supports_reasoning_effort` only; any other model shows no control and its requests are byte-identical. Mapping (`raptor_gateway::thinking::map_effort`): low->low, medium->medium, high->"high" else the highest level at or above medium, max->"max" else the highest level. For Qwen: Low=low, Medium=medium, **High=xhigh, Max=xhigh** (one setting; the selector's tooltip says so). The in-app chat sends the conversation's choice (else the Settings default) per request; the API gateway adds the Settings default to a chat request that sets none of `reasoning_effort`, `reasoning`, `thinking`, `thinking_budget_tokens`, `reasoning_budget`, `enable_thinking` or `chat_template_kwargs.enable_thinking/reasoning_effort` (a client's choice is never overridden). The Settings default (Medium) applies via a gateway reconfigure.
- **Not verified live:** the selector in the real window, the Settings default end to end, a real API client, and whether `high`/`max` do anything on a template that lists them (only Qwen's three levels were probed). A template that accepts an effort but lists no level names is treated as unsupported (nothing to send); never seen in practice. Thinking models other than Qwen (Ornith, flint, Nanbeige) were not probed.

### Quantized KV on spilling MoE models at 32K, repeated (b12374, llama-bench -d 32768, fa on, 3 runs per cell; REPEATS the earlier noisy single run)
| model | cache | prompt tok/s | generation tok/s | shared-memory growth |
|---|---|---|---|---|
| Ornith | f16 | 552 | 24.0 | 1,048 MiB |
| Ornith | q8_0 | 514 | **10.8** | 752 MiB |
| Ornith | q4_0 | 514 | **46.9** | 398 MiB |
| flint | f16 | 391 | 15.5 | 9,517 MiB |
| flint | q8_0 | 357 | 12.3 | 9,098 MiB |
| flint | q4_0 | 355 | 16.9 | 8,871 MiB |
- **The first run was right: q8_0 is the worst at 32K on these models and q4_0 the best** (Ornith 2x f16, 4x q8_0). Likely cause: less memory pressure (smaller spill) plus kernels that are not slower. So for a sparse MoE that spills, the app should pick q4_0, never q8_0, at long context. Quality of q4_0 at long context is still unmeasured (perplexity at 2K was within noise).

### N-gram speculation modes on code rewriting (b12374, ROCm, thinking off, fresh server per mode, temp 0, 350 tokens; tok/s)
Six "return this Rust file with a doc comment above each function" prompts (output copies the prompt), then two "write new code" prompts:

| model | mode | rewrite 1-6 | new code 1-2 |
|---|---|---|---|
| Qwen3.8-27B | none | 21 21 21 20 21 21 | 22 22 |
| Qwen | ngram-mod (app default) | 78 149 149 94 150 150 | 21 22 |
| Qwen | ngram-map-k4v | 178 21 21 21 21 21 | 22 22 |
| Qwen | **ngram-simple** | **178 181 181 179 181 180** | 22 21 |
| Ornith | none | 64 55 66 67 67 67 | 67 68 |
| Ornith | ngram-mod | 100 65 203 260 384 64 | 67 67 |
| Ornith | ngram-map-k4v | 234 66 67 66 67 67 | 67 67 |
| Ornith | ngram-simple | 140 **25** 215 425 416 66 | 65 65 |

- **`ngram-simple` is up to 8.5x on Qwen for code rewrites (consistent on all six) and no slower on new code.** `ngram-mod` (the app's current mode) is 4-7x, `ngram-map-k4v` only helped on 1 of 6 on each model. On Ornith `ngram-simple` has the highest peaks (425) but one clear loss (25 vs 55-66), so it is not strictly safe there.
- Caveat: the rewrite prompts are the best case for n-gram; real agent edits land between "rewrite" and "new code". MTP models (Qwen has a head) were not combined with ngram-simple here: next test is `draft-mtp,ngram-simple` vs `draft-mtp`.

### MTP stacked with ngram-simple (Qwen3.8-27B, ROCm b12374, same 6 rewrite + 2 new-code prompts, one run each; the GPU may have been shared with the user's other use, so absolute numbers are lower/noisier than the earlier table)
- `draft-mtp` alone: rewrite 46 43 44 41 32 28, new code 25 29 tok/s.
- **`draft-mtp,ngram-simple`: rewrite 186 206 205 213 201 197, new code 42 41 tok/s.** About 4-7x MTP alone on rewrites and +45-65% on new code. (The earlier `draft-mtp,ngram-mod` first-reply test showed no gain: ngram-simple is the better partner.)
- `ngram-simple` alone this run: 124 123 114 119 130 123 / 16 20 (it was 180 in the quiet run), so MTP + ngram-simple is the best of the three and the stack is robust to noise.
- Candidate default for MTP models: `draft-mtp,ngram-simple`; for non-MTP models `ngram-simple` has one clear loss on Ornith (25 vs 55-66), so keep `ngram-mod` there until more data.

### 4-bit cache policy for spilling MoE models (built from the repeated 32K cache test; live behaviour UNTESTED)
- `context_options`: a sparse MoE that spills (`oversubscription_applies`) used to get only the "default" window (the old "does not fit" early return) and would never have been offered anything but q8_0 "Large", its slowest cache. It now gets **"Long (4-bit cache)": min(32768, trained maximum), q4_0, 1 slot, speculation on**, note quoting the measured numbers and the unmeasured long-context quality. No q8_0 "Large" for these models. Dense and fully-fitting models are unchanged.
- Load path: a spilling sparse MoE at a window >= 32768 with no cache chosen gets q4_0 (+ flash attention) and a rationale line (`tuning::long_window_cache`).
- Unit-tested (helpers + both rules, 4 new tests). NOT verified in the app: that Ornith/flint show the "Long" option in the Library, that loading it behaves (spill report, speed vs f16 at 32K through the server rather than llama-bench), or quality of a 4-bit cache at 32K.

### Live checks in the real window (2026-11-14, Qwen3.8-27B via the chat model picker)
- **Thinking selector WORKS:** `models_thinking` reports supported with levels low/medium/xhigh and default Medium; the composer shows Low / Medium / High / Max with Medium checked, tooltips say "High sends xhigh. Max uses the same setting on this model". The same sheep question took **27 s at Low and 173 s at High** through the UI (reply text lengths similar), so the level reaches the model. Reasoning text length was not captured (collapsed block). Not yet checked: Settings default end to end, an API client without a level, a non-thinking model (selector should be absent).
- **`draft-mtp,ngram-simple` is what the app launches for Qwen now** (server command line shows `--spec-type draft-mtp,ngram-simple`); app tests 255 + 206 pass.
- The UI only tracks models it loaded itself, so a model loaded through the raw command shows as "Loaded model (not reported by the backend)" and no thinking selector; this is by design but surprising when scripting.
- Harness lesson: processes started with `Start-Process` die when the session rolls over (Vite, curl); start them with `Invoke-CimMethod Win32_Process Create` to survive.

### Faster benchmarking (2026-11-14)
- **Autotuner sweep trimmed from micro-batch 256/512/1024 to 256/512.** 1024 never won on any of the five models measured, so each sweep is a third shorter. Unit tests updated (7 pass). Not yet re-timed end to end on a model.
- Still open for speed: fewer layer candidates when the fit is obvious, shorter depth lists in the matrix, skipping `fa=0` in the Vulkan comparison harness (Ornith/flint runs are the slow ones).

### Memory and context (llama-fit-params, Qwen3.8-27B, device MiB = model 10,616 + context + compute)
| KV | ctx 8,192 | 65,536 | 131,072 | 262,144 (trained max) |
|---|---|---|---|---|
| f16, 1 slot | 11,543 | 15,155 | 19,283 | 27,539 |
| q8_0, 1 slot | 11,307 | 13,283 | 15,747 | 20,675 |
| q4_0, 1 slot | 11,179 | 12,259 | 13,699 | 16,579 |
Four slots add ~450 MiB to every cell. The compute buffer grows with context for quantized KV (266 -> 1,202 MiB), so a fixed compute estimate under-counts long contexts; use llama.cpp's own projection for "maximum context".

### Correctness and plumbing
- Embeddings on a chat model need `--embeddings --pooling mean`; chat keeps working at full speed (21 tok/s). Embeddings need `-ub` raised to the batch size (524-token chunks failed at the default 512 with HTTP 500). Real GPU: 5,120-dim vectors, 3 inputs in 0.65 s, a 1,088-chunk workspace in 359 s (~3 chunks/s, slow for a 27B model), retrieval hit `memory.rs` first for a stale-summary query (semantic similarity 0.72-0.77).
- Clear the server log buffer when a new server starts. Before this the GPU verdict summed buffers across loads (granite showed 26,510 MiB on a 16 GB card = flint + granite).
- Folder scans must recognise an already-referenced file by path; hashing it cost minutes per scan.
- LM Studio's real models folder is in its `settings.json` (`downloadsFolder`), which found `D:\AI_Storage\lmstudio`; fixed paths would not.
- llama.cpp's GitHub "latest release" is a `v1.0.0` marker with no binaries and every real build is flagged pre-release: read the release list and take the highest `b<number>`.
- HEAD `Content-Length` must be read from the header, not `reqwest::Response::content_length()` (0 for HEAD).
- Runtime update b12340 -> b12360 through the UI (280,133,027 bytes, SHA-256 from GitHub's asset digest), GPU verified afterwards.
- Stock llama.cpp HIP zip works with HIP SDK 7.2 (66/66 layers, 21 tok/s on Qwen).

---

## REJECTED / DOES NOT WORK

- **`--n-cpu-moe` (experts of N layers on the CPU)** for flint: 31.5 / 26.1 / 21.0 tok/s at 25 / 35 / 50 layers' experts on CPU vs 34.2 with none. The oversubscribed GPU run is faster.
- **Partial layer offload of a dense model** here: granite with 25 of 65 layers on the CPU generates at 1-4 tok/s vs 20.5 fully on the GPU (DDR4 bandwidth). Do not trade layers for context; shrink context or quantize KV instead.
- **`--cpu-moe` with `--moe-cache-mib` (and `--moe-cache-cpu-share`)** for flint: 10.3 / 16.8 / 20.7 tok/s at a 4,096 / 8,192 / 11,000 MiB expert cache and 14.7 with CPU share 0.5, vs 32.6 for all layers requested on the GPU with spill. Rejected (from the GPU optimization guide, hybrid items 14-16).
- **`draft-mtp` on Ornith**: the server exits at start ("failed to create MTP context"). No usable head in that quant. The load plan retries without it.
- **Micro-batch 1024** was never the best (Ornith 788 vs 2,235 at 256; flint 536 vs 695).
- **Larger thread counts with all layers on the GPU**: no effect (see above); not worth a setting there.
- **Batch threads (`-tb`)**: prompt processing speed was flat or slightly worse as threads grew (477 -> 444 pp tok/s at 2 -> 16 threads on granite partial offload). Not worth tuning.
- **GitHub `releases/latest` for llama.cpp** (see above).
- **InferSSD (wici.ai) as an engine here**: llama.cpp-based, CUDA/Metal only, built for 700B+ MoE models streamed from SSD; no ROCm/Vulkan, so it cannot use this GPU. Its headline multiples are against llama.cpp mmap thrash and the page says the comparisons differ in weights, templates and input lengths. Ideas worth keeping are already used (read only routed experts; batch prefill).

## BROKEN (found, fixed unless noted)

- **Offloadable layer count was one too low for a model with an MTP head** (`main_layer_count + 1` = 65; `block_count + 1` = 66, because `block_count` already includes the NextN layer). One layer stayed on the CPU: generation fell from 47 to ~5 tok/s, and a saved tuning of 66 layers was capped to 65. Fixed with `offload_layers_of` (uses the same arch hints as the recommender), test added.
- **A window that nearly fills VRAM spills into shared memory and generation collapses** (Qwen 167k with MTP: dedicated 15,609 of 16,304 MiB plus 2,281 MiB shared). Always check `\GPU Adapter Memory(*)\Shared Usage` after a big-window load.
- **Small test replies cannot measure generation speed**: a 1-3 token answer gave "tg 5-12 tok/s" for a server that does 47. Use 200+ token replies (and the code prompt for MTP, since acceptance is lower on prose: story ~29 vs code ~47).
- The context-options probe ran `llama-fit-params` without `-m` because the recommendation carries no model path until `with_model` (found only because errors were logged; do not `.ok()` away probe errors).

- No tracing subscriber existed, so every `warn!` went nowhere (fixed: daily files, 7 days).
- A persisted runtime selection was not restored after an app restart, so Load failed with "No runtime installed" (fixed: `restore_installed_runtime`, never downloads).
- The vector store's retry set was in memory only, so a half-embedded file looked unchanged after a restart (fixed: rebuilt on open).
- Blank chunks made llama-server answer HTTP 400 for a whole embedding batch (fixed: skipped, sentinel key).
- Importing a name already in the library hashed the whole file before refusing (fixed: checked first).
- The GPU verdict could accumulate across loads (fixed, see above).
- Leaving a model configured as the embedding model silently disabled speculation for it (the load rationale now shows the speed setup; cleared the setting during testing).
- Vite dev server dies when started as a background tool job; start it detached (`Start-Process cmd /c npm run dev`) in the same call as the app, and restart the app if its page is `about:blank`.

---

## UNTESTED or OPEN

- **Per-model backend choice, ROCm or Vulkan (built 2026-11-14, code and unit tests only; NOT run on the GPU)**. Library > per-model "Backend: Auto / ROCm / Vulkan". Auto = ROCm, except a sparse-MoE model larger than VRAM (the kind Vulkan generated faster on) goes to Vulkan when the Vulkan runtime is installed; otherwise a hint says it is not. A forced backend that is not installed is refused at load (no spawn, no silent run on the other one). Tuning and the context matrix run on the model's own backend (the tuned key already includes the runtime id). Settings > Dependencies installs the Vulkan runtime (`llama-<tag>-bin-win-vulkan-x64.zip`, checksum from the GitHub release digest, confirmed first, installed next to ROCm without changing the runtime in use). **Untested live:** (1) an actual Vulkan load in the app (device preflight for `Vulkan0`, GPU verdict from the Vulkan log, load progress stages, spill report, chat and gateway through a Vulkan server); (2) **MTP on Vulkan**: a model with an MTP head is deliberately given n-gram speculation on Vulkan and the load report says so, until `--spec-type draft-mtp` is checked on the Vulkan build; (3) n-gram speculation speed on Vulkan (the comparison above is plain decoding); (4) context windows (`context_options`) are projected with whichever runtime is active, not the model's own, so for a Vulkan model they can be slightly off until it is loaded once; (5) "Forget tuning" clears the key of the runtime in use only. Switching backend restarts the server (the previous model is unloaded first, as on every load).

- **AMD HRX Loom** (hwbusters.com article, 2026-11): AMD agents ported 51 trending LLMs to AMD client GPUs in four days on its experimental Loom compiler/runtime (`loom-serve`, `.loom` packages, tech preview). Claimed median 1.5x faster 32K-token prefill than llama.cpp's ROCm backend (secondhand: ~1.5-2.1x vs ROCm, ~1.9x vs Vulkan). Tested on Strix Halo only; discrete RDNA4 / gfx1200 not mentioned; its own model format, not GGUF; no decode numbers; llama.cpp version not stated; license not stated. Not actionable for this GPU today; watch it. The article does not mention MTP or TurboQuant.
- **TurboQuant KV cache** (user question): not in our mainline llama.cpp b12360: `-ctk/-ctv` accepts f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 only. Exists in forks (CUDA/Metal, some ROCm on gfx1030) and as an unmerged CPU-only upstream PR; HIP/Vulkan status unconfirmed. Its gain over what we already have is small: a 4-bit cache measured +0.15% perplexity (noise), so 3-bit saves ~25% more cache memory, against running an unvetted fork binary. Not pursued; re-check when a newer mainline build lists a turbo/tbq type (the runtime updater will surface it). Sources: https://github.com/ggml-org/llama.cpp/pull/21089 , https://github.com/ggml-org/llama.cpp/discussions/20969
- **Untested cache types already in our build**: `iq4_nl`, `q5_0`, `q5_1`, `q4_1` (and `f8_e4m3`) for K/V. A perplexity run like the one above would show whether any beats q4_0 at a similar size.
- **MTP ("boosters")**: already applied (Qwen3.8-27B 21.5 -> 47.5 tok/s). It only helps GGUFs that carry an MTP head (`nextn_predict_layers` > 0); Ornith, granite and flint have none. Preferring MTP-capable quants of a model family is the practical lever. The build also lists `draft-eagle3`, `draft-dflash`, `draft-dspark` and `draft-simple` (need a separate draft model): none tried.

- **llama.cpp's Vulkan backend prints `rebar: 0` for this card although Resizable BAR is on** (user-confirmed). Its flag is its own detection and does not match the firmware setting here; the cause is unknown (not investigated). Do not treat `rebar: 0` from ggml_vulkan as the real state.

- **Vulkan vs ROCm on this card (user feedback, 2026-11-14: "AMD has custom ROCm stuff as well as Vulkan (RADV) can perform very well")**. Web search (standard mode; the sources are blogs, forum posts and llama.cpp discussions that mix cards, OSes and builds, so treat the numbers as leads): decode is often faster on Vulkan for small and MoE models on RDNA4 (one table: 137 vs 101 tok/s on a 7B Q4_0 on an RX 9070 XT; another: Vulkan +20% on an 8B and +9% on a 14B); prompt processing and large dense models are reported both ways (a 27B on an R9700: ROCm 42.8 vs Vulkan 29.1 tok/s; a 7B pp512 table: Vulkan ~3,164 vs ROCm ~2,381). Flash attention on/off changes the ranking. Windows ROCm support is narrow (gfx1200/gfx1201 on the official list). **RADV is the Mesa Linux driver; on this Windows machine Vulkan runs on AMD's own driver, so Linux RADV results do not transfer directly.** "Custom ROCm" leads: AMD's TheRock / nightly ROCm builds, the Lemonade llama.cpp-ROCm builds, and a HIP build with `-DGGML_HIP_ROCWMMA_FATTN=ON`. None run yet. Needs the Vulkan runtime (`llama-b12360-bin-win-vulkan-x64.zip`, 38.1 MB, sha256 42a6f9df608b99c1c83fcaf77284a516e3f167b32a4690f4e4e364c377547055, from GitHub; the Vulkan loader `vulkan-1.dll` is present). Plan: llama-bench on the same GGUFs, flash attention on and off, prompt and decode separately, at 0 and 16K depth, for Qwen (dense), granite (dense), Ornith (MoE) and flint (MoE, spills). Sources: https://runaihome.com/blog/rdna4-vulkan-vs-rocm-local-llm-benchmark-2026/ , https://github.com/ggml-org/llama.cpp/discussions/10879 , https://github.com/ggml-org/llama.cpp/discussions/15021 , https://www.phoronix.com/review/lemonade-vulkan-rocm

- **Quality at long context with a 4-bit KV cache**: perplexity at a 2K context is unchanged within noise (see above), but accuracy at 32K-128K, on other models and on a coding task is unmeasured; a needle-in-a-haystack or coding-task check would settle it.
- **Larger windows for MoE models that only fit by spilling** (Ornith, flint): not offered; the memory of the spill plus a bigger cache is unmeasured.
- **Context windows for models whose layers do not all fit** are deliberately not offered; untested whether a modest CPU share could pay off for MoE.
- **Slot count 2 vs 1 vs 4 under real concurrent agent load** (prompt-cache hit rate, queueing): memory known, behaviour with parallel Claude Code subagents unmeasured.
- **Summariser** (text generation through the loaded model) never run on real hardware.
- **A small dedicated embedding model** instead of the 27B chat model (~3 chunks/s); needs a second server or a swap, a design decision.
- **Pre-reading the GGUF into the OS cache / `--no-mmap` / `--mlock`** for dense models that keep layers in RAM; flint's cold load is 130-140 s of disk reads, a warm one is much faster. Unmeasured.
- **Oversubscribed MoE on Linux** (shared-memory spill may not exist; the fallback covers it). Untested.
- **Installer on a clean machine**: built, uninstall script read, never run.
- **Linux build and packaging**, **live Hugging Face search/download**, **publish the first signed app update** (needs the GitHub Actions signing secret), **Windows key-file ACLs**.
- **Quality of the three extra models** and long-context behaviour (only speed and GPU use were checked).
- **Sustained (30-minute) performance, GPU clocks and temperature**: never measured; no reader exists.
- **Matrix for granite and flint** (done for Qwen to 128K and Ornith to 32K); **matrix with speculation on** (llama-bench is plain decoding); **Ornith/flint above 32K** (their spill makes the numbers unstable).
- **Vulkan vs ROCm** on this GPU: not run (needs a download).
- **Gateway request with no `model` field on a non-Qwen model** returned an empty result once (named model works); not investigated.
- **`draft-eagle3` / `dflash` speculative modes** and a separate draft model: not tried.
- **Autotuner**: tunes layers and micro-batch only (KV type and slots do not change speed; threads only matter with CPU layers and use a rule). Does not re-run after a runtime update; no "tune all models" action.
- **Speculation setting for the user**: no switch to turn MTP/n-gram off globally, and no way to pick Large/Maximum from the chat window (Library only).
- **ContextChoice error display** (the rejected-probe message): a unit test for it kept failing as an unhandled rejection in the test harness although the component catches it; verified only by reading the code and the live error path.

---

## Review of `local_llm_gpu_optimization_guide.md` (2026-11-14)

A third-party guide of 60 + 20 techniques for NVIDIA, AMD and Intel GPUs. Every item was checked against this project: already applied, measured here, not applicable to this hardware, or untested. Its own warning holds: each setting is a benchmark candidate, not a promise.

**Already applied or confirmed by our own measurements**
- Keep every layer on the GPU (guide NVIDIA 8, AMD 9/17/19, hybrid 1): confirmed hard. granite with 25 of 65 layers on the CPU generates at 1-4 tok/s vs 20.5 on the GPU. The load plan requests all layers whenever it can.
- FlashAttention on (NVIDIA 4/12, AMD 8): always on for GPU backends; confirmed active in the server log.
- KV-cache quantization (NVIDIA 5, AMD 7, hybrid 12): done, and measured not to change speed here (within 1%). Used for the Large and Maximum windows.
- Right-size the context (NVIDIA 7, AMD 10, hybrid 13): the default is 8,192 (Settings); larger windows are opt-in and computed from llama.cpp's own projection.
- Speculative decoding (NVIDIA 16-18, cross-vendor 6): applied (MTP head and n-gram). 2.2x on Qwen. The guide's "a poor draft can make it worse" held for MTP only in the memory sense (see the overhead measurement).
- Batch / micro-batch tuning (NVIDIA 20, AMD 21): the per-model autotuner. The guide's point that prefill and decode must be measured separately is exactly why the score weighs both.
- CPU threads (AMD 22): measured; only matters with CPU layers, then logical cores win by ~13%.
- Benchmark prefill and decode separately, record peak VRAM, report cold vs warm: partly done (llama-bench pp/tg; cold load of flint 130-140 s vs warm); see UNTESTED for the missing parts.
- MoE with few active experts (hybrid 14-16): tested below. The best result for flint remains "all layers on the GPU, let the driver spill the cold experts".
- Architecture-appropriate build (AMD 2/25): gfx1200 is detected and the HIP build matches; the runtime updater keeps it current.
- Keep drivers/ROCm compatible (AMD 3): the HIP SDK 7.1 vs 7.2 mismatch we hit is the guide's warning in practice; the app now reports the SDK state and refuses a silent CPU run.

**Measured now, from the guide**
- **`--cpu-moe` + `--moe-cache-mib` (hybrid 14/16): REJECTED on this GPU.** flint (24 GB, 256 experts, 6 used), generation tok/s: 32.6 baseline (all layers requested on the GPU, spill allowed) vs 10.3 / 16.8 / 20.7 with a 4,096 / 8,192 / 11,000 MiB expert cache, and 14.7 with an 8,192 MiB cache plus `--moe-cache-cpu-share 0.5`. A bigger cache helps but never reaches the baseline; letting the CPU compute misses is worse. Earlier `--n-cpu-moe` was also worse.
- **PCIe link and memory** (AMD 23/24, hybrid 4): the GPU runs at the maximum link speed the device reports, x16; RAM is two 32 GB DDR4-3200 modules in dual channel (both configured at 3200). Nothing to fix. Resizable BAR is ON (user-confirmed 2026-11-14, firmware setting); the app cannot read it independently yet.
- **Flags present in b12360** (so the guide's examples are valid here): `--moe-cache-mib`, `--moe-cache-cpu-share`, `--n-cpu-moe`, `--n-cpu-ffn`, `--load-mode` (auto/none/mmap/mlock/mmap+mlock), `--no-kv-offload`, `--numa`, `--tensor-split`, `--split-mode`, `--cache-ram`, `--fit`, `--spec-type`, RPC. Not present: `--mlock`/`--no-mmap` as separate flags (replaced by `--load-mode`).
- **Spill detection was missing and the guide is right to stress it** (AMD 17, Intel 8, long-context section): built `gpumem.rs`. It reads the Windows `GPU Adapter Memory` counters before and after a load and reports growth in shared memory. Live: Qwen default 14,023 MiB dedicated, +123 MiB shared (fits); Qwen at 167k with MTP +2,307 MiB shared (warned: "works but generation can be several times slower"); flint +8,991 MiB shared (reported as expected, sparse MoE larger than VRAM). Shown in the load rationale and the load report. Not available on Linux (reported as unknown, never as fine).

**Not applicable to this machine (cannot be tested here; keep in mind for other users)**
- NVIDIA: CUDA Graphs, NVFP4, TensorRT-LLM, GPU-side sampling, pinned memory, AWQ/GPTQ engines, power/thermal tuning of a GeForce card.
- Intel Arc: SYCL, SYCL Graphs, `GGML_SYCL_*`, Level Zero options, relaxed allocation limits, USM.
- Multi-GPU layer/tensor split, NUMA tuning, RPC offload: one GPU, one NUMA node, no second machine. The app has no tensor-split option; adding one without hardware to verify it would be guesswork.

**Worth doing next (ordered by value, none needs a download unless stated)**
1. Context-length benchmark matrix (guide section 8): prefill, decode, time to first token and peak VRAM at 4K / 16K / 32K / 64K / 128K, stored per model, so a user sees what a window really costs. We have only spot checks (16k and 82k for Qwen).
2. Sustained run (30 min) with clocks and temperature, to catch thermal or power throttling that a short benchmark hides. No clock/temperature reader exists yet on Windows here.
3. Vulkan vs ROCm on this GPU (AMD 1, 12-15): needs the Vulkan runtime zip from llama.cpp (a download, ask first). The manifest already lists `llama-vulkan-win`.
4. A separate small draft model for models without an MTP head (NVIDIA 16): needs a download.
5. `--no-kv-offload` and `--load-mode mlock` as measured options (hybrid 11, 6): cheap to test with llama-server; expected small.
6. Read ReBAR/SAM and show PCIe link width/speed in the Hardware panel: cheap, informative when a spill makes the link matter.

## Nanbeige4.3-33B Q4_K_M (dense, 19.36 GiB, does not fit 16 GB) - ROCm vs Vulkan (2026-11-14)
File verified: size 20,788,825,216, SHA-256 matches HF. llama-bench b12374, -ub 256 -fa 1.

| config | ROCm pp512 / tg64 | Vulkan pp512 / tg64 |
|---|---|---|
| -ngl 99 | 304.5 / 4.78 | 83.4 / 4.69 |
| -ngl 99 @ d16384 | 153.0 / 2.23 | 72.2 / 2.88 |
| -ngl 48 | 368.4 / 4.96 | 88.3 / 5.30 |
| -ngl 40 | 287.8 / 5.13 | 217.9 / 5.16 |
| -ngl 32 | 238.4 / 3.81 | 173.7 / 3.83 |

Findings:
- A dense model that spills is ~5 tok/s on either backend: generation is bound by system RAM / PCIe bandwidth, so the backend barely matters (unlike sparse MoE, where Vulkan won generation).
- ROCm wins prompt processing 1.4-3.6x; keep Auto = ROCm for dense spilling models (matches existing rule).
- Best generation is at ngl ~40-48 (~5.1-5.3 tok/s), only ~5% above -ngl 99; no layer split makes it interactive. Dense models over VRAM should be avoided; a smaller quant or MoE is the real fix.
- Vulkan pp512 is erratic at ngl 48 (88 vs 218 at ngl 40) - not trusted.
- Not yet tested: loading Nanbeige through the app (import by reference, spill reporting).

## Nanbeige4.3-33B: push to 30 tok/s (2026-11-14, HIP b12374, llama-server -np 1 -c 4096 -ub 256 -fa on, temp 0, thinking off, 256-300 token replies)
Note: under Git Bash the HIP runtime needs `C:\Program Files\AMD\ROCm\7.2\bin` on PATH, otherwise llama-bench silently falls back to CPU (1.9 tok/s); the app's `launch_env` already does this.
- **Verify cost is flat with a spilled dense model (WORKS, the key fact).** llama-bench -ngl 99 (4.2 GiB spills to shared memory): pp1 4.95, pp2 9.8, pp4 18.5, pp8 30.6, pp16 71.0, pp32 136.8 tok/s, so one pass costs ~0.2 s for 1 to 32 tokens (PCIe-bound, not compute). With layers on the CPU instead (-ngl 44) it does not scale (pp16 35.5, pp32 52.9): for speculation keep -ngl 99 and spill.
- Baseline llama-server: code 4.90, story 4.91, rewrite 4.74 tok/s.
- **`--spec-type ngram-simple`: rewrite 73.4 tok/s (284/284 drafts accepted), code 4.92, story 4.91.** Above 30 on copy-heavy work only; neutral on fresh text.
- **Parallel streams (llama-batched-bench, pp128 tg64): aggregate generation 4.92 / 17.6 / 28.2 / 59.6 tok/s at 1 / 4 / 8 / 16 streams.** 30 tok/s total across requests is reached at ~9 streams; each stream stays ~3.7-4.9.
- **DFlash2 drafter (ProCreations/Nanbeige4.3-33B-DFlash2, converted with b12374 `convert_hf_to_gguf.py --target-model-dir`, Q8_0 1.0 GiB, `--spec-type draft-dflash`): REJECTED.** code 6.21 tok/s (127/879 accepted, 14%), story 4.41 (74/1246, 6%); the server then died silently on the third prompt. Matches the model card's AL 1.9.
- **Nanbeige4.2-3B as a classic draft (bartowski Q4_K_M 2.5 GiB): REJECTED.** llama.cpp refuses it as-is (vocab 166,144 vs 166,912, > 128 difference); tokens 0-166,143 are identical and the extra are `[PAD]`, so the vocab and the embd/output rows were zero-padded (scratchpad `padvocab.py`). Then code 2.49 tok/s (193/484 accepted, 40%), story 1.23 (130/983, 13%): the 3B (a looped model) predicts the 33B badly, and its VRAM pushes more of the 33B over PCIe.
- **Pure 4-bit IQ4_XS, everything on the GPU (WORKS, biggest single gain).** bartowski IQ4_XS still has 2.5 GiB of Q6_K/Q8_0/Q5_K (output head, attn_v/k, some attn_output/ffn), 16.76 GiB on the GPU; a `llama-quantize --pure --imatrix` IQ4_XS is 15.80 GiB on the GPU (proxy built from the local Q4_K_M with `--allow-requantize`; quality of a requant-from-Q4_K_M is a little worse, speed is what was measured here). llama-bench pp1 4.95 -> 11.34 tok/s (official b12374 DLL). Context 2,048 with q4_0 KV and -ub/-b 64 keeps KV + compute at ~170 MiB.
- **Own gfx1200-only build of `ggml-hip.dll` (b12374 source, ROCm 7.2 clang, `GGML_BACKEND_DL=ON GGML_CPU=OFF`, dropped into a copy of the official runtime): pp1 11.34 -> 13.26 tok/s, tg128 14.56 (WORKS).** Built without `GGML_BACKEND_DL` the DLL silently does not load and everything runs on the CPU (2.2 tok/s). VS 2019 Build Tools cannot build llama-server itself with this clang (jinja `reinterpret_cast` error), so only the backend DLL is rebuilt.
- **MMVQ/MMQ crossover on RDNA4 (WORKS).** Batch 1-8 uses MMVQ, whose cost grows per token (0.075 s at 1 -> 0.122 s at 8), MMQ from 9 is flat (~0.100 s at 9-16), so an 8-token DFlash verify was the most expensive point. Patched `ggml_cuda_should_use_mmvq` with an env override `GGML_MMVQ_MAX`; at 4: pass 0.085 s at 6 and 8 tokens (was 0.090 / 0.122), batch 1-4 unchanged.
- **Server, pure IQ4_XS + own DLL + MMVQ_MAX=4, -c 2048 q4_0 KV:** plain code 14.19, story 14.21 tok/s (2.9x the original 4.9). With DFlash2 Q4_K_M + ngram-simple: code 16.57, story 12.44, explain 13.78; with `--spec-draft-p-min 0.3`: code 16.69, story 13.07, rewrite 150.3. DFlash only helps code: its 0.55 GiB pushes the target into shared memory (llama-server 16,119 MiB dedicated + 1,362 MiB shared).
- **Run-to-run VRAM contention:** the same plain config gave 11.78 tok/s in one run and 14.19 in another; dwm.exe (5120x1440 desktop) held 2,040 MiB dedicated + 648 MiB shared. Close or minimise windows before benchmarking near the VRAM limit.
- REJECTED: output head on the CPU (`-ot output.weight=CPU`): pp1 7.54 (was 11.34). Vulkan on the pure IQ4_XS: pp1 12.0 but pp8 5.65 (a slow path at batch 8). Server sampling is not a cost: `-bs` and `--samplers top_k --top-k 1` both 14.3 vs 14.2 default. The earlier `--spec-draft-p-min` sweep ran during a compile and is void.
- **Realistic coding-agent edit (WORKS, >30 tok/s single stream).** Prompt: a 3 KB Rust file from this repo + "add `fn total_mib(rows: &[Row]) -> u64`, a unit test, fix typos, output the whole file"; the model copied the file, wrote the function, a test module and a change summary (1.6-1.7K tokens, checked by hand). Pure IQ4_XS proxy, own DLL, MMVQ_MAX=4, -c 4096 q4_0: plain 13.42 tok/s (700-token cap); `ngram-simple` 25.63 (1,724 tokens, 938/1,392 accepted); **DFlash2 + ngram-simple, p_min 0.3: 31.73 tok/s** (1,604 tokens, 1,231/1,872 accepted). At a 700-token cap (copy part only) 97-104 tok/s, so a short cap overstates it.
- **Fresh text stays ~14 tok/s.** DFlash n-max 4 / p_min 0.3: code 16.54, story 13.36; p_min 0.5: code 15.16, story 11.94; plain story 14.2. Plain server still spills 546 MiB (16,118 dedicated + 546 shared). Ceiling: 16.2 GB of 4-bit weights / 320 GB/s peak = ~20 tok/s per pass, so prose needs AL >= 1.5 from a drafter that costs no VRAM; DFlash2 gives ~1.3 on prose.
- **Aggregate with parallel requests on the pure IQ4_XS (llama-batched-bench, own DLL): 14.3 / 27.9 / 46.2 / 84.2 tok/s at 1 / 2 / 4 / 8 streams** (was 4.9 / - / 17.6 / 28.2 on Q4_K_M).
- **Quality (perplexity, ~100 KB of repo docs + fit.rs, ctx 2048, 15 chunks): Q4_K_M 17.44 +/- 0.47; pure IQ4_XS proxy (requantized from Q4_K_M) 17.95 +/- 0.48 (+2.9%, inside the error bar).**
- **Final 4-bit builds from bartowski IQ4_XS** (sha256 510cb2e0... verified; `llama-quantize --allow-requantize --pure --imatrix <bartowski imatrix>`; tensors already IQ4_XS are copied unchanged, `src/llama-quant.cpp:1307`). Perplexity (same docs corpus, ctx 2048): Q4_K_M 17.44, stock IQ4_XS 17.43, **pure IQ4_XS 17.99 (+3.2%, 15.80 GiB on GPU)**, pure + attn_k/attn_v Q8_0 18.16 (no help, 16.17 GiB), **pure + output head Q6_K 17.29 (16.12 GiB on GPU, quality equal to stock)**. The output head carries the quality; K/V projections do not.
- **Benchmark noise from the desktop:** dwm.exe grew from 2,040 to 2,576 MiB dedicated while the user worked (plus the Claude app, Explorer); pure IQ4_XS tg128 fell 14.56 -> 11.41 with no config change. `--list-devices` still reported 16,158 MiB free, so llama.cpp's free figure does not see this. Compare configs only back to back.
- **Final back-to-back suite (desktop busy, dwm ~2.5 GB; own DLL, MMVQ_MAX=4, -c 4096 q4_0 KV, DFlash2 Q4_K_M + ngram-simple, p_min 0.3; tok/s):**

  | task | pure IQ4_XS (PPL 17.99) | IQ4_XS + Q6_K head (PPL 17.29) |
  |---|---|---|
  | Rust file edit (add fn + test, whole file) | **38.82** (1,274 tok) | 20.39 (1,807 tok, a different, longer answer) |
  | React component edit (prop + Escape key) | 26.31 | 28.60 |
  | JSDoc on every function (mostly new text) | 16.75 | 17.43 |
  | fresh code, 2,000 tok | 14.83 | 13.35 |
  | story, 2,000 tok | 10.31 | 9.13 |

  Verdict: **30+ tok/s single stream is reached on coding-agent edits that keep most of a file (31.7-38.8 on two tasks, 150 on a pure rewrite), not on fresh text (10-17 tok/s).** Fresh text is capped by memory bandwidth (~20 tok/s for 16 GB of weights at 320 GB/s) plus the drafter's acceptance (DFlash2 AL ~1.3 on prose, ~1.9 on code). Aggregate over parallel requests passes 30 at 2-3 streams.
- **UNTESTED, ordered by expected value for fresh text:** (1) a quiet desktop / fewer GPU-composited windows (dwm took 2.0-2.6 GB; with it gone, target + DFlash fit with no spill: estimated plain ~17-19, code ~22); (2) tree verification of the DFlash2 lattice (32 candidates per position; verify cost is flat at 9-16 tokens, ~0.10 s): estimated AL 1.9 -> 2.4 on code; needs a C++ change in `common/speculative.cpp`; (3) a better-trained drafter (DFlash2 "v2 planned" upstream). None of these makes prose reach 30.
- **How to reproduce the fast setup:** runtime = official b12374 HIP zip with `ggml-hip.dll` replaced by a gfx1200 build of the same tag (+ the `GGML_MMVQ_MAX` patch in `ggml/src/ggml-cuda/mmvq.cu`), env `GGML_MMVQ_MAX=4`; model `Nanbeige4.3-33B-IQ4_XS-pure.gguf`; `llama-server -ngl 99 -fa on -c 4096 -ub 64 -b 64 -ctk q4_0 -ctv q4_0 -np 1 -md dflash2-q4_k_m.gguf -ngld 99 --spec-type draft-dflash,ngram-simple --spec-draft-p-min 0.3`. Files: `D:\AI_Storage\downloads\nanbeige-33b-pure\`, `...\nanbeige-33b-dflash2\`, `D:\AI_Storage\build\llama-hip-b12374-patched\`. Not integrated into the app.

## Training an EAGLE-3 drafter for Nanbeige4.3-33B (started 2026-11-14, user approved the downloads)
Why: fresh text is bandwidth-bound (~14 tok/s plain), so 30 tok/s needs ~2.2+ accepted tokens per pass on prose; the only public drafter (DFlash2) gives ~1.3 and its training method is not published. EAGLE-3 has public training code and llama.cpp runs it (`--spec-type draft-eagle3`).
- **PyTorch on this GPU (WORKS):** AMD's Windows wheels `torch-2.9.1+rocm7.2.1` (repo.radeon.com rocm-rel-7.2.1, Python 3.12 venv at `D:\AI_Storage\drafter-train\venv312`) report `cuda.is_available() True`, device "AMD Radeon RX 9060 XT". Adrenalin 26.10 is newer than the required 26.2.2.
- **Feature capture (WORKS):** `D:\AI_Storage\drafter-train\featdump\featdump.cpp`, links only `llama.h` (built against b12374 import libs, run from the patched runtime folder): saves target `l_out` of layers 1/25/49 (= llama.cpp eagle3 layer inputs 2/26/50, `common/speculative.cpp` uses `llama_set_embeddings_layer_inp(id)`) as fp16 plus the target's top-32 next-token log-probs. 224 tok/s, 43 KB per token. Sanity: target top-1 = next token 90.3% on its own temp-0.6 outputs, no NaNs.
- **Trainer (WORKS, pilot):** `D:\AI_Storage\drafter-train\train_eagle3.py`, one decoder layer matching `src/models/eagle3.cpp` (fc 21504->7168, attention over [norm(embd), norm(g)], FFN 8192, 32k draft vocab with d2t, 743M params), training-time-test unrolling over 5 steps, soft targets from the stored top-32. Exports SafeAILab layout; `convert_hf_to_gguf.py --target-model-dir` converts it, Q4_K_M 0.50 GB, and llama-server loads and uses it.
- **Pilot on 65 on-policy records (70.7K response tokens), 4 epochs, 3.5 min:** held-out step-1 accuracy 0 -> 23%; initialising lm_head from the target's output head made no difference (23.2% both). In llama-server: story 6.20 / explain 6.34 tok/s, 40/852 and 45/832 drafts accepted (~0.19-0.22 per round, consistent with the held-out numbers, so the feature/token alignment is right). Far too little data; the point was to prove the pipeline.
- In progress: 1.0M-token UltraChat run (2,768 conversations from `HuggingFaceH4/ultrachat_200k` train_sft shard 0; target soft labels make it distillation from the 33B), to measure how acceptance scales with data before a multi-day run.
- **1M-token run (2,768 UltraChat conversations, 68.7 GB of fp16 features, capture ~2 h at ~230 tok/s; 2 epochs = 5,536 steps, 65 min at 0.7 s/step):** held-out AL (validated on the 33B's own temp-0.6 answers, leading run of correct greedy drafts) 1.18 / 1.21 / 1.25 / 1.27 / 1.32 / 1.34 / 1.37 / 1.38 / 1.39 / 1.40 / 1.40 at steps 500-5,500. **Live in llama-server (pure IQ4_XS, own DLL, desktop busy): plain 10.0 / 10.1 / 10.1 tok/s (story / explain / code); EAGLE-3 n-max 2: 8.30 / 8.40 / 8.70 (57/393, 59/389, 65/379 accepted); n-max 4: 6.87 / 7.02 / 6.33. REJECTED as it stands: a net loss.**
- **Scaling verdict:** ~10x more tokens seen moved held-out AL by +0.22 (about +0.1 per doubling). Reaching ~2.2+ (needed for 30 tok/s on prose) would take ~8-9 more doublings, hundreds of times this data; at ~230 tok/s capture and 43 KB/token (21.5 KB with the int8 option, `featdump` writes it for `.q8.bin` outputs) that is months of capture and tens of TB on this machine. Published EAGLE-3 drafters reach AL 3+ with datacentre-scale data and compute. A drafter good enough for 30 tok/s here would have to be trained elsewhere; this pipeline (`featdump.cpp`, `train_eagle3.py`, verified to load in llama.cpp) is reusable for that.
- Files: `D:\AI_Storage\drafter-train\` (pipeline, `feats_uc1m.bin` 68.7 GB, `feats_part1.bin` 3.2 GB, `eagle3_uc1m-q4km.gguf`), ROCm PyTorch venv `venv312`, UltraChat shard `data\ultrachat_sft0.parquet`. The feature files can be deleted to free ~72 GB.

## Parallel expansion (skeleton-of-thought) for one answer (2026-11-14, pure IQ4_XS, own DLL, llama-server -np 8 -c 12288 q4_0)
Method (scratchpad `sot.py`): one request writes a plan (a "Facts:" line with names/setting/terms + a 6-section outline, ~10-19 s), then all sections are written at once in parallel slots and joined. Wall-clock tok/s = all tokens / total time. Output structure differs from a single pass, so quality was read by hand.
- **Run 1 (dwm ~2.5-2.9 GB): TCP explanation 34.1 tok/s (2,002 tok), story 51.2 (2,760), gardening guide 40.7 (2,088).** Explanation and guide read well (coherent, consistent headings). **Story was broken**: heroine "Mara" in one section, "Elena" in another; the bottle discovery retold in several sections; sections cut mid-sentence at the 330-token cap.
- **Run 3 with a shared Facts line, no-overlap outline lines, '## n.' headings, 450-token cap: story consistent (one name, 21 mentions, six ordered sections), but story 37.1 / explanation 22.2 / guide 19.6 tok/s**, because dwm had grown to 3,246 MiB and each of the parallel streams ran at 3.86 tok/s (~23 combined, vs 84 combined at 8 streams on a quieter desktop). Wall time is set by the slowest section.
- Verdict: **WORKS for long explanatory answers (34-41 tok/s wall with good quality when VRAM is free); needs the Facts line for narratives; very sensitive to desktop VRAM.** Not exact-output-preserving.
- **Desktop VRAM:** dwm.exe reached 3,291 MiB. Closing Riot Client, Razer Cortex and the Settings window (graceful close) changed nothing (3,247 -> 3,248). An active RustDesk session and the 5120x1440 display remain (not touched).
- **Lookahead (Jacobi) decoding, exact output (REJECTED, at most +8%).** Built `llama-lookahead` from b12374 source (W/N/G made env-configurable). The build needed one source patch: `common/jinja/value.h` `val_func.target<func_hptr>()` -> `target<func_hptr*>()` (VS 2019's `<functional>` + ROCm clang reject `target<>` of a plain function type; the code notes these paths are unused). With that patch the whole common library builds, so a custom llama-server is now buildable here. Pure IQ4_XS, own DLL, MMVQ_MAX=4, desktop busy (llama-bench tg128 baseline 11.43 tok/s, same session), 256 tokens greedy: W3/N3/G3 story 10.94 / code 12.30 tok/s (44 / 69 tokens accepted); W4/N3/G4 10.80 / 12.02; W5/N4/G5 9.14 / 11.00; W8/N4/G8 7.77 / 9.93. Bigger windows cost more than they win on this GPU.

## After the reboot (2026-11-14 evening; dwm 478-492 MiB, RustDesk 1.5.3)
- **RustDesk 1.4.9 -> 1.5.3 did not free dwm's VRAM (3,411 MiB after the update, paged pool unchanged ~1 GB); a reboot did (478 MiB).** So the RustDesk capture leak (#15945) was not the holder; dwm keeps what it grew to until it restarts.
- **System-wide single-token slowdown since ~14:00, survives the reboot, not explained:** tg128 14.56 (13:40) -> 11.33-11.41 now with both the official and the own DLL, on both the proxy and the final pure file; pp1 13.26 -> 11.35, pp8 93.8 -> 80.4, but pp16 155 -> 156.8 (unchanged). So the latency-bound MMVQ path got slower, the compute-bound MMQ path did not. Ruled out: VRAM (dwm 0.5 GB), GPU memory bandwidth (PyTorch copy 284 GB/s = 89% of 320), CPU clock (boosting, plan "LLM + Gaming"), PCIe ASPM (off), HAGS (on, unchanged), file differences. UNTESTED: Adrenalin Performance > Tuning preset / VRAM tuning, GPU clocks under load (needs Adrenalin metrics or GPU-Z).
- Clean-desktop rerun (plain / DFlash2 + ngram-simple p_min 0.3, 256 tok): code 11.09 / 12.84, story 11.14 / 11.35, explain 11.12 / 11.57 tok/s.
- **Parallel sections, corrected diagnosis:** the slow 20-23 tok/s runs were not dwm but my server settings: `-c 12288 -ub 256` with 6-8 slots spilled the KV/compute buffers (each stream 4.3-4.9 tok/s, 0.23 s per step vs 0.10 s for an 8-token batch in llama-bench). With `-c 6144 -ub 64 -b 64`, 6 slots: story 37.3, explanation 27.8, guide 37.8 tok/s wall (streams ~8 tok/s). **With a tighter plan (Facts line <= 25 words, 8 outline lines of 6-12 words, 170-token cap) and `-np 8 -c 8192 -ub 64 -b 64`: story 44.7 tok/s (3,770 tokens, 84 s), explanation 33.7 (2,662, 79 s), guide 31.6 (2,638, 84 s) - all three above 30.** Quality read by hand: story consistent (one heroine, 8 ordered sections); explanation and guide coherent and ordered with complete endings; artifacts: two sections echo the instructions ("without extending into later stages..."), one ends on a dangling term list, every section ends with a similar wrap-up sentence, heading style varies. Planning step 15-19 s is now the largest fixed cost.
- **Hunting the ~22% single-token slowdown (all REJECTED, cause still unknown; tg128 now 10.7-11.4 vs 14.56 at 13:40):**
  - Correction: `HwSchMode = 1` means hardware-accelerated GPU scheduling is OFF (2 = on); the GraphicsDrivers key was last written 2026-08-13, so it did not change today.
  - Adrenalin tuning: none active in the driver key (`OD8Settings 0`, `PP_AutoODActive 0`, Chill off, auto-overclock offsets 0); driver unchanged (32.0.31044.16, 2026-09-08); last driver-key write = the 21:42 reboot.
  - Process priority (`--prio 0/2/3`): 11.34 / 11.34 / 11.35. `--poll 0`: 11.26.
  - Background GPU keep-alive (tiny matmuls, spinning / 1 ms sleeps): 9.56 / 10.17 (worse).
  - RustDesk hardware video encoder (6.4% "video codec" engine while streaming): user switched it off, encoder idle, tg128 10.68 (no gain; software encoding loads the CPU).
  - Physical display asleep (turns off after 15 min; user is remote): kept awake with `SetThreadExecutionState(ES_DISPLAY_REQUIRED)` during the run: tg128 10.69 (no gain; whether the panel really woke was not verified).
  - CPU idle states (the 19 GB download was running during every fast measurement and ended ~14:15): 1 / 4 spinning cores: tg128 10.87 / 11.02 (no gain).
  - Remaining UNTESTED: GPU core/memory clocks under this load (Adrenalin metrics overlay or GPU-Z), temperature/power limit.
- **Root cause of the "slowdown" found: VRAM spill, not kernels or clocks (WORKS as a diagnosis).** Half-model test (layers 26-51 pruned with `llama-quantize --prune-layers`, 8.2-8.7 GiB, fully resident): tg128 IQ4_XS 32.99 tok/s = **290 GB/s effective (91% of the 320 GB/s peak)**, Q4_0 31.95 = 298 GB/s, Q4_K 31.99 = 298 GB/s; MMVQ_MAX made no difference. So the IQ4_XS kernel is efficient. The full model gets only ~197 GB/s because it spills: llama-server now holds 15,521 MiB dedicated + **1,142 MiB shared** (546 MiB shared at 13:58, when it ran 14.2). The difference is how much VRAM other processes (dwm, Claude app, helpers) hold at the moment, not a setting. With zero spill the ceiling is ~16.5 tok/s single-stream (16.2 GiB x 290 GB/s).
- **Layers on the CPU instead of spill: no gain.** llama-bench tg128 at -ngl 53 / 51 / 50 / 49 / 48: 11.23 / 11.33 / 11.16 / 10.71 / 9.78. A layer on the CPU (DDR4) costs about what the spilled bytes cost over PCIe.
- **Reviewed a user-supplied list of "exploits"** (tiny draft model in the 32 MB Infinity Cache, async CPU draft, native INT4 repacking, 2:4 sparsity, MoE offload, KV fixes): the useful one was "check kernel efficiency" (answered above: 91%, not the bottleneck). The others do not apply here: drafter acceptance is the limit (a 30 MB drafter cannot even hold Nanbeige's 166,912-token embeddings), the CPU is a 3800X with DDR4, the board is PCIe 4.0, the model is dense, llama.cpp has no 2:4 kernels and pruning would cost quality, KV is already q4_0 with flash attention.

## In-app verification of DFlash draft + Long answer (2026-11-15, real window, desktop busy: dwm ~3 GB, 1.2-4.3 GB of the server spilled)
- **DFlash2 draft chosen automatically, 1 slot, q4_0, ctx 4096, back to back vs speculation off (tok/s):** fresh code 9.39 vs 7.20, story 8.18 vs 7.21, explanation 8.85 vs 7.18, **whole-file edit 32.66 vs 7.00** (1,236 tokens, 1,052/1,375 drafts accepted). WORKS.
- **Long answer at 8 slots, gardening guide, wall clock in the window:** with the draft 11.0 tok/s (335 s, shared-memory spill 4,310 MiB) vs **30.7 without** (102 s, 1,243 MiB). The draft is now skipped above 2 slots. After that change: 35.4, 30.9, 32.9, **32.4 tok/s** on four runs while the section prompt was tuned (final run: 8 complete sections of 209-415 words, ~22 s to the first text). REJECTED: draft with many slots; asking for "about 200-300 words" (instructions echoed into the answer, one section repeated). WORKS: Long answer at 8 slots without the draft.

- **App planner after the fix (2026-11-15, default load, no hand settings; desktop busy):** Nanbeige4.3-33B IQ4_XS-pure at a 6,144-token window: 53/53 layers, q4_0 cache, DFlash draft, 2,725 MiB spilled; built-in benchmark 12.3 tok/s (old plan 8.9). Prompts through the server: code 12.66, story 11.03, **whole-file edit 41.57 tok/s** (1,236 tokens, 1,052/1,375 drafts accepted). WORKS.

## Dense Q4/Q6 parameter sweep 2026-11-23 (RX 9060 XT 16 GB, ROCm build b12374-patched, Nanbeige4.3-33B)
Framed as a hyperparameter search: every launch knob is a feature, wall time and tokens/s are the loss. Server-level measurements (llama-server, one slot, fa on, t16, b4096) except where noted.

### Same 33B dense model, three files / placements (8,192-token context)
| Setup | prompt tok/s (4k or 2k fresh) | generation tok/s |
|---|---|---|
| Q4_K_M 19.4 GiB, 40/53 GPU layers, f16 KV, ub 512 | 328 | 4.5-4.6 |
| Q4_K_M, 40 GPU layers, **q8_0 KV** | 327 | 4.5 |
| Q4_K_M, 40–43 GPU layers, **q4_0 KV** | 196-214 | 4.0-4.9 |
| Q4_K_M, every layer on GPU (llama.cpp reports 53/53; ~6.5 GiB paged by WDDM) | 217-237 (q4 KV) / 230-252 (f16 KV) | 4.0-4.4 |
| Q4_K_M, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, all layers | 145 | **1.3 — REJECT** |
| Q4_K_M, split + classic same-family 3B Q4 draft (-md, draft-simple) | ≈ plain | 3.5 (worse than plain) |
| Q4_K_M + DFlash2 q4 draft (0.58 GiB GPU-resident) | ≈ plain | **9.6 continuation / 4.9 code-edit** |
| **IQ4_XS-pure 16.4 GiB, every layer on GPU (5-6 GiB over VRAM? no: ~0.6 GiB paged)** | 329-346 | **8.7-9.0** |
| IQ4_XS-pure, all layers + DFlash2 q4 draft | 265-346 | **20.3 continuation / 12.4 code-edit** |

- Quant-file choice is the largest single lever on this card: the same weights at IQ4_XS-pure run **2× the decode** of Q4_K_M because the model fits (0.6 GiB of mild spill) instead of needing 13 CPU layers. With the DFlash draft stacked: **20.3 tok/s = 4.4-4.8× the worst config**, ~4.5× its split generation speed, at the same prompt-processing speed (~330).
- The 0.58 GiB DFlash draft is the small-weights lever that works on a split/spilling dense model (the classic 3B draft loses: its verify batches raise CPU-layer work). DFlash gain shrinks on text it has not seen a pattern for (12.4 tok/s on a fresh 1k code prompt vs 8.4 plain ≈ +48%); on continuation text it is 2.3×. Raptor already applies DFlash automatically (`SpecMode::Dflash`); EAGLE-3-style drafts (`--spec-type draft-eagle3`, supported by b12374 server) are the peer-reviewed direction this points at (arXiv 2503.01840: 3-6× with ~0.5-1.5B feature-level drafters).
- **q4_0 KV costs ~40% of prompt speed on a dense model at the same layer count** on the HIP flash-attention path (FA dequantizes K/V per read). q8_0 KV is free at 8k (±1% vs f16) and doubles the cache capacity. Raptor's tuner now skips q4_0-KV candidate trials for dense models and keeps them for sparse MoE (where the cache is the spill and the whole KV is read per token: Ornith 46.9 vs 10.8 tok/s q8).
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` on HIP/RDNA4 is a trap: 3× slower generation (1.3-1.4 tok/s), and it left the driver in a degraded state that slowed other large-VRAM processes (the 3B probe stayed at full speed; the 33B prefill dropped 328→124 until the state decayed). Never pass it on this backend.
- All-layer spill on a +37%-over-VRAM dense model loses to the layer split on both prefill (217-252 vs 327) and generation (4.0-4.4 vs 4.5-4.6): the earlier `dense_spill_applies` ±10% VRAM gate is right.
- ubatch on a fully-offloaded model: ±5% (4.17B Q4, peak at 1024: 2045 vs 1966), so a ubatch candidate is only worth a trial when layers or caches move with it.
- Threads 16 vs 8 on a split model: generation 4.6-4.9 vs 4.4-4.8 (within noise); prefill rows were too noisy (±60% re-runs) to judge; the 3B llama-bench rows reproduce within ±2%.
- At the 65,536-token floor this 16 GB card must choose: q4_0 KV (3.4 GiB cache) at ~33 GPU layers on the 19.4 GiB model ran **89 tok/s to build 57k fresh, then 161 tok/s prefill and 2.9 tok/s generation at depth** — usable for the qualification but slow. The better answer for the floor is a model that fits with the cache: a 12-14 GiB dense Q4/Q6 (or the 12.0 GiB Qwen3.8-27B IQ3_S, which Raptor already runs 66/66 layers at 8,192 and loads the 82,640-token window on the maximum option). Pushing users to Q4-Q6 *of a file that fits with the target window* is worth 2× over pushing them to the largest Q4 file.

## Nanbeige4.3-33B on the RX 9060 XT: the file that fits wins (2026-11-24, ROCm 7.2, b12374-patched and official b12528)

The 40 tok/s goal for this 33B dense model on this 16 GiB card was chased from every side
(draft models, runtimes, batch sizes, parallel slots, quantization levels). The lever that moved
the needle was the size of the file: a model that fits in VRAM with room for the KV cache
decodes 2.4x faster than one that spills, and 4.7x faster than the 19.4 GiB Q4_K_M that needs a
layer split.

### The quant ladder (llama-bench, `-ngl 99 -fa on`, q4_0 KV, `-ub 512 -b 2048 -t 16`, 128 generated tokens)

| File | bpw | VRAM | tg128 d0 | tg128 d4096 | ppl (15 chunks) | vs Q4_K_M |
|---|---|---|---|---|---|---|
| Q4_K_M (19.4 GiB, does not fit) | 4.85 | 19.36 GiB | 4.7-5.3 (server, layer split) | - | 17.7119 ± 0.9161 | - |
| IQ4_XS-pure (16.4 GiB, 3% over) | 4.25 | 16.39 GiB | 9.11 ± 0.09 | 8.9 | 17.8926 ± 0.8682 | +1.02% |
| IQ3_XXS | 3.06 | 13.49 GiB | 19.04 ± 0.02 | - | 18.2695 ± 0.9649 | +3.15% |
| IQ2_S | 2.50 | 10.38 GiB | 21.58 ± 0.02 | - | 19.6351 ± 1.0981 | +10.86% |
| IQ2_XS | 2.31 | 9.90 GiB | 21.99 ± 0.02 | 19.26 ± 0.03 | 19.3616 ± 1.0651 | +9.31% |
| IQ2_M | ~2.5 | 11.35 GiB | 20.71 ± 0.04 | - | 19.1939 ± 1.0233 | +8.37% |
| **IQ2_XXS (pure, imatrix)** | **2.06** | **9.38 GiB** | **22.17 ± 0.03** (patched) / 22.53 ± 0.36 (b12528) | 19.36 ± 0.05 (patched) / 21.11 ± 0.05 (b12528) | 18.0181 ± 0.6941 | **+1.73%** |

- **WORKS: IQ2_XXS is the file to run on this card.** 22.2-22.7 tok/s against 9.1 for the
  IQ4_XS-pure file (2.4x) and 4.7-5.3 for Q4_K_M (4.7x), with a perplexity 1.73% above Q4_K_M.
  The 2.06 bpw "pure" file beats the *higher* bit-rate mixed files (IQ3_XXS 3.06 bpw, IQ2_S
  2.50 bpw, IQ2_XS 2.31 bpw, IQ2_M ~2.5 bpw) on both quality and speed: it is an imatrix file
  with every tensor at 2 bit, the mixed files keep some tensors at 4-6 bit and score worse. Bit
  rate alone does not rank these files; which quantizer made them does.
- Perplexity method: `llama-perplexity -f results/ppl_corpus.txt -c 2048 --chunks 15` over a
  177,318-byte corpus built from this repository's own `docs/*.md` and
  `crates/raptor-runtime/src/fit.rs` (English prose plus Rust code). The chunk-to-chunk spread
  is large (SE 0.69-1.02), so the +1.7% difference sits inside the noise: the ladder says "no
  measurable loss", not "no loss". Task-level coding checks are still open.
- **REJECTED: IQ2_M as the default.** Slower than IQ2_XXS (20.71 vs 22.17 tok/s) *and* 4.6 points
  worse in perplexity (+8.37% vs +1.73%).
- **REJECTED: the official b12528 ROCm build as a speed fix.** 9.18 ± 0.09 vs 9.11 ± 0.09 tok/s
  on IQ4_XS (+0.7%) and 22.53 ± 0.36 vs 22.17 ± 0.03 on IQ2_XXS (+1.6%, inside the run-to-run
  spread). The patched b12374 build stays the default; b12528 is a prerelease anyway.
- The card itself is healthy and not the limit: device-copy bandwidth 264.9 GiB/s, FP16
  4096^3 matmul 6.7 TFLOP/s, and the 3B draft model decodes at 57.57 ± 0.57 tok/s.

### DFlash2 speculative decoding on a target that fits: always slower

DFlash2 (Q4_K_M draft, 0.58 GiB) was re-measured against the plain IQ2_XXS target, both on the
official b12528 server and on the patched b12374 one (draft block sizes 1/3/5/7):

| Target | Draft | Code tok/s | Prose tok/s | Acceptance (code/prose) |
|---|---|---|---|---|
| IQ2_XXS | none | 21.98 | 21.99 | - |
| IQ2_XXS | DFlash2, n_max 1 | 21.72 | 20.25 | 39.6% / 29.6% |
| IQ2_XXS | DFlash2, n_max 3 | 18.11 | 15.65 | - |
| IQ2_XXS | DFlash2, n_max 5 | 19.46 | 15.13 | - |
| IQ2_XXS | DFlash2, n_max 7 | 19.25 | 15.26 | 10.8% / 5.0% |
| IQ2_XXS | DFlash2 (b12528, n_max 8) | 13.98 | 11.76 | 9.5% / 5.5% |

- **REJECTED: DFlash2 on a model that fits in VRAM.** Every configuration is at or below plain
  decoding. The reason is visible in the acceptance column: with the weights resident, the
  target verifies a draft in microseconds, so a 4-10% acceptance rate cannot pay for the draft
  tokens, and the extra KV work of the draft costs more than the saves. The earlier DFlash wins
  in this file (20.3 tok/s on IQ4_XS) were wins against a *spilling* target, where every token
  costs a PCIe transfer.
- Consequence for the app: `src-tauri/src/tuning.rs` now only reaches for a DFlash draft when the
  target spills (`target_spills`). A target that fits gets n-gram drafting instead, and the
  measured numbers for both are in the note the plan writes into the load rationale.

### Parallel slots: 40 tok/s is reached in aggregate, not per request

`llama-server -np N` with the IQ2_XXS target, every slot decoding the same prompt at the same
time (256 tokens each, 2048-token window per slot). One measurement per row; the spread over
three repeats of the four-slot case is 0.03 tok/s:

| Slots | Aggregate tok/s | Per request tok/s | Total tokens | Wall time |
|---|---|---|---|---|
| 1 | 21.60 | 21.99 | 256 | 11.85 s |
| 2 | 33.72 | 17.14-17.34 | 512 | 15.19 s |
| 3 | **41.44** | 14.11 | 768 | 18.53 s |
| 4 | **47.17-47.21** | 11.99-12.16 | 1,024 | 21.7 s |
| 6 | 68.99 | 11.70-11.95 | 1,536 | 22.26 s |
| 8 | 89.16 | 11.27-11.64 | 2,048 | 22.97 s |

- **WORKS: the goal of 40 tok/s is met in aggregate from three slots up**, and the app's default
  of four slots (`SERVER_DEFAULT_SLOTS = 4` in `crates/raptor-runtime/src/fit.rs`) sits at
  47.2 tok/s. This is the app's "Long answer" mode (one slot per parallel request). It is not
  40 tok/s for one request: per-request speed falls as the slots share the card (22 tok/s alone,
  about 12 tok/s each at four slots).
- A first sweep with six *different* prompts per slot gave 33.7 / 37.3 / 43.1 / 47.6 / 61.5 tok/s
  for 2 / 3 / 4 / 6 / 8 slots, but the spread over repeats of the same configuration was 20%
  (37.0-43.8 at four slots). The cause is the content, not the setup: with one prompt per slot
  every slot runs at the same speed, with six prompts some slots finish early and the last slot
  sets the wall time. Parallel numbers need one prompt per slot or a median over repeats.
- The app's long-window mode (`--kv-unified`, added above a 32,768-token window) measured
  44.31 tok/s at four slots with a 32,768-token unified cache, so the shared cache costs
  nothing measurable.

### N-gram drafting: transformative for copied text, neutral for everything else

A full-file rewrite workload (a 3 KB Rust source file, the model asked to rewrite the whole
file) and ordinary prose/code prompts, IQ2_XXS target unless stated:

| Workload | Setup | tok/s | Acceptance |
|---|---|---|---|
| Whole-file rewrite | plain | 20.83 | - |
| Whole-file rewrite | `ngram-simple` | **187.39** | 87.6% (mean accepted length 43) |
| Whole-file rewrite | `ngram-simple` on the IQ4_XS-pure target | 81.82 | - |
| Whole-file rewrite | DFlash2 + `ngram-simple` | 173.65 | 83.8% |
| New code, no copy | plain / `ngram-simple` | 21.98 / 22.00 | no matches |
| Prose, no copy | plain / `ngram-simple` | 21.99 / 22.03 | no matches |

- **WORKS: n-gram drafting for copy-heavy work.** When the answer repeats text the prompt
  already contains (file edits, refactors, templated text), the draft matches almost every
  token and the speedup is 9x. On ordinary prompts there are no matches, so it is free.
- **REJECTED: DFlash2 together with n-grams.** The mixed run is slower than n-grams alone
  (173.65 vs 187.39 tok/s): the draft model spends time proposing tokens the n-gram would have
  supplied for free.
- **WORKS: `ngram-mod` stays the app's n-gram mode.** The app maps `SpecMode::Ngram` to
  `ngram-mod`; the benchmark winner was `ngram-simple`. Compared head to head (IQ2_XXS):

| Workload | `ngram-mod` | `ngram-simple` | plain |
|---|---|---|---|
| Whole-file rewrite (1 slot) | **187.38 tok/s**, acceptance 95.8% (920/960) | 184.68 tok/s, acceptance 87.6% (925/1056) | 20.83 tok/s |
| New code, prose, memo (1 slot) | 22.02 / 22.06 / 22.02 | 22.00 / 22.03 / 22.00 | 21.98 / 21.99 / 21.99 |
| Four slots, same prompt | 47.21 / 47.15 / 47.30 | 46.65 / 46.84 / 46.63 | 47.17 / 47.21 / 47.15 |

  `ngram-mod` accepts longer drafts (mean accepted length 62 against 43) and is free in the
  parallel case, so the app's mapping stays. `ngram-simple` is 1% behind at four slots.

### Tooling notes (the quiet half of the work)

- **REJECTED: the single-shot Hugging Face downloader for large GGUFs.** It silently produced
  three truncated files (IQ3_XXS 11.95 of 14.49 GB, IQ2_XS 6.99 of 10.63 GB, IQ2_S 2.86 of
  11.15 GB) while printing "done". `scripts/hf_fetch_mt.py` (24 parallel range requests) lifted
  the transfer rate from ~2.94 MB/s to 50-92 MB/s, and `scripts/hf_verify.py` checks the file
  size and the Hub's LFS SHA-256 before anything is benchmarked. Verified hashes:
  IQ2_XXS `3d5bba3c9c7567b5709243308c741cb9938f44eaa7905ab966f0029e82f08bf4`,
  IQ2_M `efb45fc475fac828741ba993ff2c4da7dd26bc33262f0b5b994164e394554930`,
  IQ3_XXS `5d959f2812f07632fc6e8de21373f90d12bda81d0178a2a8868fc3e6812a25da`,
  IQ2_XS `69db5647e45ea37de7a533c4452598c124c65f09204d8aac7adf7148def5a28e`,
  IQ2_S `c84bc91903f913da29d635876fa504275964efc487a49950edb49dca4a9a0241`.
- **UNTESTED: AMD clock and power telemetry.** `amdsmi` is not installed in ROCm 7.1/7.2
  (`Could not find module 'libamd_smi.so'`), so the GPU clocks during these runs are unknown.
- **UNTESTED: IQ1_S / IQ1_M** (below 1.5 bpw, not in the Bartowski repository for this model)
  and tree-based verification (`--spec-type` tree drafts). Both would be needed for a genuine
  single-request 40 tok/s: IQ2_XXS at 2.06 bpw is the smallest file that still fits the card
  with the KV cache, and every further bit of compression costs quality.

### What changed in Raptor because of this

- `src-tauri/src/tuning.rs`: a DFlash draft is now only used when the target model spills out of
  VRAM (`target_spills`). A target that fits gets n-gram drafting instead, because the measured
  DFlash2 runs against a fitting target were 19.3 tok/s with the draft against 22.0 without, and
  174 against 187 tok/s on a whole-file edit. The plan also writes a note when the file is larger
  than the card, naming the measured numbers above so the user can pick a smaller quant.
- `src/lib/compat.ts`: the model browser now recommends a file that fits *with headroom* before
  one that only just fits, because the KV cache and the compute buffers need that room. Before
  this change the browser offered the 12.2 GiB IQ2_M file on this card (8.4% worse in perplexity
  and slower than IQ2_XXS) and would have offered the 18.6 GiB IQ4_XS file to anyone who asked
  for it.
- `src/components/models/DiscoverTab.tsx`: the recommendation notice says what the rule is now.
- The runtime manifest is unchanged: b12528 is a prerelease and at most 1.6% faster, so the
  patched b12374 build stays the reference runtime.

## N-gram mode head-to-head on the models the app ships (2026-11-19, RX 9060 XT 16 GB, ROCm build b12374-patched, temperature 0.6, seed 470, 256-token replies, `-np 1 -c 8192 -ub 256`, IQ3 quants, first pass each)

The IQ2_XXS head-to-head above was run on one dense 33B file. This reruns it on the two
models the app can actually serve on this card today: Granite 4.2 30B dense (IQ3_M) and
Ornith 1.5 35B-A3B sparse MoE (IQ3_S/IQ3_XXS), with the same five prompts used for the
Nanbeige sweep below (`code`, `prose`, `chat`, `report`, plus a new `rewrite` prompt: add doc
comments to a Rust module, answer repeats the prompt).

| Workload | Granite ngram-mod | Granite ngram-simple | Ornith ngram-mod | Ornith ngram-simple |
|---|---|---|---|---|
| Whole-file `rewrite` (Rust) | 19.92 tok/s | 19.89 tok/s (2/48 drafts accepted) | 76.59 tok/s (40% drafts accepted) | **92.50 tok/s (64% accepted)** |
| New `code` | 20.19 | 20.15 | **63.61** | 61.78 |
| `prose` | 20.18 | 20.17 | **63.86** | 46.53 |
| `chat` | 20.20 | 20.20 | **63.13** | 40.16 |
| `report` | **20.16** | 12.89 (rerun: 13.68) | **63.65** | 34.19 |

Granite is prompt-eval bound at ~20 tok/s and neither mode moves it; the `report` prompt is
~2-3.5x slower to read with ngram-simple in two runs (415→109/108 prompt tok/s on the rerun),
which is the one anomaly of the day and not reproduced on the other four prompts. On Ornith,
`ngram-simple` is 21% faster on the rewrite-style prompt and 26-46% slower on everything else,
matching the earlier observation that it only pays off when the answer echoes the prompt.

- **WORKS: `ngram-mod` stays the app's `ngram` mode.** No change. The one regime where
  `ngram-simple` wins (verbatim copying) is what the DFlash2/MTP stacks already exploit.
- **REJECTED: switching the default to `ngram-simple`.** It is a REGRESSION on three of five
  Ornith workloads (prose -27%, chat -36%, report -46%) and never wins on Granite.
- **UNTESTED: repetition-heavy MoE at very long context.** All five prompts are ≤ 2K tokens;
  whether `ngram-simple` acceptance grows enough at ≥ 32K to flip the table is unknown.
- The `bench_spec.py` prompt set gained the `rewrite` task; `results/*.log` holds the raw
  per-task lines. Run-to-run variance of ±40% on prompt-eval throughput was observed on a
  busy machine; treat single-digit decode differences as noise.

## What changed in Raptor because of this (2026-11-19)

- Benchmark visibility: benchmark records now carry prompt/generated token counts, the
  history table shows a run-vs-best column and a token summary, `BenchmarkResult` prints the
  prompt/completion tokens, and the context-length matrix gained per-row "% of empty-context"
  generation, a plain-English comfort summary, and a Copy-as-Markdown export.
- Default chat context target raised from 65,536 to 122,880 tokens ("120K ideal"). The 65,536
  floor is unchanged: the recommender shrinks the window, never below the floor, when the
  model or the hardware cannot hold the new default. No measured speed data yet — the change
  is a default-policy change, not a measurement.

## KV cache in system RAM (`--no-kv-offload`) measured on hardware (2026-11-20, RX 9060 XT 16 GB, ROCm b12374-patched, Nanbeige4.3-33B-IQ2_XXS 9.38 GiB, 52/52 layers on the GPU, flash attention on, warm page cache, `-lv 4`, `svrbench.py`, prompt 4,096 tokens, 64-128 generated tokens, dwm holding ~5.6 GiB desktop VRAM as usual)

Driver is AMD's; "free" below is what `ggml-cuda`/`ROCm0` reports (16,146 MiB of 16,304).

| Context | KV placement | pp tok/s | gen tok/s | Notes |
|---|---|---|---|---|
| 16,384 | f16 in VRAM (default) | 682 (2K prompt) | 21.5 / 19.7 | fits easily |
| 16,384 | f16 in RAM (`-nkvo`) | 604 / 538 | **7.4 / 4.4** | works, ~3-4.5x slower |
| 32,768 | f16 in VRAM | 314 | 15.4 | projected 15,983 MiB vs 16,146 free: fits, but below llama.cpp's own 1,024 MiB free target |
| 32,768 | f16 in RAM (`-nkvo`) | 578 | 6.1 | KV 13.3 GiB on the host bus |
| 65,536 | f16 in VRAM (oversubscribed, untouched flags) | 313 | 13.8 | projected 22,671 MiB: runs by spilling ~6.5 GiB into Windows shared memory |
| 65,536 | f16 in RAM (`-nkvo`) | 578 | 6.1 | |
| 65,536 | q4_0 KV in RAM (`-nkvo`) | 563 | 7.0 | 4-bit does not help: the bottleneck is the per-token host round-trip, not cache size |
| **65,536** | **q4_0 KV in VRAM** | **647** | **19.3** | projected 13,248 MiB, 2,897 MiB headroom — the right tool |
| 131,072 | q4_0 in VRAM (spills ~1.2 GiB) | 278 | 18.0 | projected 17,312 vs 16,146 free; driver serves the overflow |

- **Works: `-nkvo` loads and generates on this card.** No flag error, no crash at 16K/32K/64K. Generation is 3-4.5x slower than a VRAM-resident cache at the same context. Prompt processing is not hurt (538-578 tok/s vs 313-682): the penalty is per generated token.
- **The right large-window configuration on this card is q4_0 KV resident in VRAM**, not RAM-KV: 19.3 tok/s at 64K (and 128K still runs at 18.0 via a small driver spill). The existing planner already prefers q4_0 when f16 does not fit; this measurement confirms that ordering is also the fast one.
- **Windows' shared-memory spill beats `-nkvo`** (13.8 vs 6.1 tok/s at 64K) but is uncontrolled: the driver decides what and when to move, and the earlier BROKEN entry shows it can collapse generation on much larger spills. The app does not model spill; it is not something we can recommend deliberately.
- **RAM-KV is an opt-in escape hatch, nothing more.** It only earns its keep when the requested context cannot fit VRAM at any KV quantization *and* the user accepts 3-4.5x slower generation in exchange for a deterministic, driver-independent memory layout. The app keeps it behind a setting that defaults off.
- UNTESTED: RAM-KV end-to-end through the app (toggle -> recommend -> spawn). The numbers above are raw flag-level benchmarks, not an app run.

## Unresolved from the 2026-11-19 session (logged so it is not lost)

- **RESOLVED (2026-11-20): quantized-KV generation at depth looked nondeterministic; it is not a KV-quant kernel defect.** Fresh runs on an idle GPU (llama-bench, IQ2_XXS, FA on, `-ngl 99`): d4096 q4_0 tg128 = 19.42/19.39/19.38/19.38 across four separate processes (±0.05), d8192 f16 18.8 / q4_0 17.4 / q8_0 17.4-18.2, d2048 f16 21.4 / q4_0 20.7. See the late-2026-11-20 section below for the root-cause analysis. One residual anomaly (kv-depth.log task 129) remains unexplained; see below.

## Research notes (web, 2026-11-20; leads, not measurements)

- **llama.cpp b12430 (2026-11-16) fixes a MoE prompt-processing regression on HIP/AMD** (commit 05db09a, `mmq.cuh` grid order back to row-tiles-innermost on HIP; root cause issue #31603: #29948's MMQ change cost ~20% MoE prefill, measured on gfx1010/RDNA1 with experts on CPU; gfx1200 not measured). Our pinned b12374 post-dates #29948, so MoE prompt processing at Ornith/flint rates may be affected. Not adopted: MoE tg was unchanged in the report, our models run fully on the GPU (the report used `-ot exps=CPU`), and the pinned build is measured. Candidate for the next deliberate runtime bump; verify with `llama-bench -p 512 -fa 1` on Ornith before switching. b12430 also ships Windows x64 ROCm builds. Sources: commit 05db09a4, issue #31603, release b12430.
- **Vulkan vs ROCm on gfx1200/gfx1201 is build- and driver-dependent; no blanket winner.** Same-GPU data point (RX 9060 XT, Arch, ROCm 7.2.4 vs RADV, llama-2-7b Q4_0, fa=1, oct 2026, llama.cpp f1cee99, Discussion #15021): pp512 3,015 (ROCm) vs 3,228 (Vulkan), tg128 70.2 vs 74.0, and ROCm tg is the same with or without FA. On R9700/gfx1201 (Fedora 44, ROCm 7.1.1 vs RADV): pp512 equal (~4,300), ROCm loses at long prompts (2,755 vs 4,236 at pp8192), Vulkan tg +8%. On RX 9070 XT (gfx1201, llama.cpp b10034): Vulkan tg 132 vs ROCm 122. On RX 9070 XT (earlier commit): ROCm pp 2.4x Vulkan on dense qwen3-32B. Our own Windows numbers (2026-11-14, this card): ROCm pp much faster, Vulkan tg faster on MoE (Granite/flint), Vulkan ~35-45% faster tg in-app for Ornith. FA on ROCm matters: pp512 +14% on gfx1200 (2,641 → 3,015). Net: keep both backends; the app's per-model Vulkan backend remains the MoE-decode lever; not re-measured since. Sources: github.com/ggml-org/llama.cpp discussion #15021.
- **Flash-attention rocWMMA build flag — tested, dead end.** b12374's source contains no reference to `GGML_HIP_ROCWMMA_FATTN` at all (the CMake option is accepted but gates nothing); a `build-rocwmma` with `-DGGML_HIP_ROCWMMA_FATTN=ON` produced binaries perf-identical to the stock build on gfx1200 (d2048 f16: pp 635.9 vs 636.4, tg 21.41 vs 21.39; d8192 q4_0 17.35 vs 17.40; d4096 f16 20.49 vs 20.53). The HIP FA path in this build is the generic tile implementation; rocWMMA FA is not wired up in b12374. Do not chase this flag on this source drop.
- **MTP (NextN) is mainstream mainline now** (e.g. GLM-DSA/GLM-5.2 via `--spec-type draft-mtp`). The app already exploits MTP for Qwen (2.2x). **Nanbeige4.3-33B has no MTP/NextN head — verified 2026-11-20** via `ggufmeta.py` on IQ2_XXS, IQ3_XXS and IQ4_XS-pure: `general.architecture = llama`, `block_count = 52`, no `nextn_predict_layers` / NextN tensor keys anywhere. Do not attempt `--spec-type draft-mtp` on this model. The only Nanbeige draft is DFlash2 (`dflash` arch, SGLang-only, measured on Modal RTX PRO 6000 in the vendor harness at 1.1-2.7x; requires SGLang + custom draft runtime, not runnable under llama.cpp/ROCm) — not adoptable here.
- **Research KV compression not in llama.cpp**: KIVI (2-bit, per-channel K / per-token V, ~2.6x peak-memory reduction at iso-quality on Llama-2/Falcon/Mistral) and the SnapKV / DuoAttention / PyramidKV line of work are fork/research territory. llama.cpp only offers symmetric per-tensor K/V types (f16..q4_0). Not adoptable without a custom runtime; revisit only if a maintained mainline path lands.
- **KV quality at long depth remains unmeasured** (our perplexity evidence is 2K ctx only). q4_0 at 32K-128K on a coding task is the open question that still gates recommending q4_0 KV for maximum-context loads.

## What changed in Raptor because of this (2026-11-20)

- `raptor-runtime`: the recommender can now plan `--no-kv-offload` (KV cache in system RAM) when VRAM cannot fit the requested window at any quantization: `recommend_with_arch_and_floor(..., allow_ram_kv)` calls `plan_ram_kv`, which keeps as many layers on the GPU as fit (weights stay on the GPU whenever possible) and moves only the KV cache (or the KV cache plus the layers that do not fit) to RAM. It is chosen only when it reaches a strictly longer context than any VRAM plan (or no VRAM plan exists); otherwise the VRAM plan wins, unchanged. CPU backend never uses it.
- New `allow_ram_kv` setting (default off, `Settings > Runtime`, checkbox + explicit warning). Rationale lines say exactly what was chosen and why ("VRAM-resident cache reaches only N tokens; keeping the KV cache in system RAM instead (--no-kv-offload) reaches M …").
- 264 raptor-runtime + 241 raptor-llm unit tests pass; `tsc` clean; settings UI tests 38 pass. Hardware numbers in this entry are flag-level, not yet through the app toggle.

## 2026-11-20 (late): quantized-KV "nondeterminism" root-caused; VRAM cliff; rocWMMA probe

Measurement hygiene note: 2026-11-19's "nondeterministic q4_0 KV" was measured through llama-server on a GPU that was also serving the QAT training job and desktop compositor. All numbers below were taken on an idle GPU. **Rule going forward: no kernel-level conclusion may be drawn from a server log taken while another GPU consumer is active; use `llama-bench` with the GPU idle.**

### Quantized-KV at depth is stable (IQ2_XXS, -ngl 99, FA on, idle GPU)

| depth | KV (K/V) | pp512 (tok/s) | tg128 (tok/s) | notes |
|---|---|---|---|---|
| 2048 | f16/f16 | 636.4 | 21.4 | |
| 2048 | q4_0/q4_0 | 635.5 | 20.7 | |
| 4096 | f16/f16 | 577.3 ± 2.3 | 20.53 ± 0.01 | 3 reps |
| 4096 | q4_0/q4_0 | 565–573 | 19.38–19.42 | 5 reps in 4 processes, ±0.05 |
| 8192 | f16/f16 | 485.7 | 18.8 | |
| 8192 | q4_0/q4_0 | 477.6 | 17.40 ± 0.00 | |
| 8192 | q8_0/q8_0 | 479.6 | 18.58 | |
| 8192 | f16/q4_0 | 483.3 | 18.2 | |
| 8192 | q4_0/f16 | 481.8 | 18.3 | |
| 8192 | q8_0/f16 | 481.0 | 18.15 | |

q4_0/q8_0 KV costs ~5–8% decode vs f16 at every depth. No run showed the 1.15/1.26 collapse. f16 and q4_0 differ by ~1 t/s consistently.

### Where the 2026-11-19 collapses actually came from

Every 2026-11-19 log that shows ~1–9 t/s needed ≥14.3 GiB of VRAM (vs 16,158 MiB driver-reported free):

| log | model bytes + ctx (MiB) | needed | tg/s |
|---|---|---|---|
| plain-baseline | 13,970 + 354 + 315 | 14,639 | 4.18 |
| isol-f16-40 / isol-q8-40 | 13,970 + 1,248 + 288 | 15,506 | 4.55 / 4.55 |
| maxlayers-43-q4 | 15,188 + 1,344 + 288 | 16,820 | 1.40 |
| at65k-q4-33 (IQ3_XXS @65K) | 11,613 + 8,192 + 761 | 20,566 | 1.33 |
| iq4xs-pure-plain | 16,176 + 1,664 + 97 | 17,936 | 8.78 |

All of these ran while dwm was holding ~5.7 GiB dedicated (Task Manager). This is the Windows shared-memory cliff: once commit exceeds real VRAM, the driver pages KV/weights/activations to system memory and decode falls off a cliff. Quantization of KV is irrelevant to the mechanism — f16 and q8_0 collapsed identically. **The 4.18 t/s plain-baseline case matters: 14.6 GiB is only ~100 MiB over this app's 14.5 GiB budget (free − max(10%, 512 MiB)), so the margin between "recommended" and "cliff" is thin on this machine.** No code change made; the app's `probe_fit` second-guess is the existing safety net. Flagged for the maintainer: consider whether the desktop-reluctant reserve should be a user setting (e.g. "conservative fit for desktop GPUs") rather than silently trusting driver-reported free VRAM.

Unresolved exception: kv-depth.log task 129 (IQ2_XXS, 16K ctx, only 10.25 GiB needed) ran task 0 at 19.8 t/s, then task 129 collapsed to 1.03 t/s and *recovered* to 6.3 t/s by the final tokens (tg_3s). Not reproducible on an idle GPU; consistent with external transient (compositor/driver/power), not with the model fit. Logged as unexplained, not as a product defect.

### llama-bench state-restore artifact

`-ctk f16 -ctv q4_0` inside one process printed `state_read_data: incompatible value rotation (0 instead of 64)` / `failed to restore kv cache` once each when mixed cache types ran in sequence. 0/10 recurrence across fresh processes and across f16/q4_0/q8_0 combos at d8192. Cause: llama-bench reuses a state slot across test cases; the rotation tag differs by V-cache type. Not a llama-server path (single cache config per server), not a correctness issue for serving.

### rocWMMA flash attention: not present in b12374

Built a throwaway `build-rocwmma` from the b12374 source with `-DGGML_HIP_ROCWMMA_FATTN=ON` (253 MiB; removed after verification). Findings: (a) the source tree contains **zero** occurrences of `ROCWMMA` outside the CMake cache line — the option gates nothing in this source drop; (b) measured d2048/d4096/d8192 × f16/q4_0 × pp/tg are all within noise of the stock patched build. The ggml FA tile kernels are what ships. Do not spend time on this flag; the experiment cost ~15 min. If rocWMMA FA ever lands upstream, revisit.

### Nanbeige MTP/NextN verdict

All local 33B GGUFs (IQ2_XXS, IQ3_XXS, IQ4_XS-pure) carry `general.architecture = llama`, `block_count = 52`, and no `nextn_*` keys. `--spec-type draft-mtp` cannot work; only the external SGLang DFlash2 draft exists (not llama.cpp-compatible). Decode speed for this model family on this machine remains bounded by memory bandwidth; speculation via llama.cpp would need an n-gram/self-speculative path (already used successfully for whole-file rewrite tasks) or the 3B Nanbeige-3B draft via `--model-draft` with `draft-simple`/`draft-eagle`-style flow — untested, and Nanbeige-3B as draft for a 4.3 33B target is plausible but must be measured before any recommendation changes.

### Status vs 2026-11-19

The "quantized-KV nondeterminism" entry is closed as **not a defect**: stable ±0.05 t/s across processes, consistent across f16/q4_0/q8_0 and depths to 8K. RAM-KV remains opt-in for the does-not-fit-in-VRAM case; q4_0-KV-in-VRAM remains the default large-window answer on this card.

## Open leads (logged, not yet tested — do not claim without measurement)

- `--cache-reuse <N>` (KV-prefix reuse across requests) and `--cache-ram` (PR #16391, in b12374) for repeated long-prefix chats; needs a scripted two-turn replay harness against llama-server to quantify prefill savings. Would directly serve the 122,880-token default chat target.
- `-ctxcp` / checkpoint-style KV offload for mid-conversation forks; behavior and interaction with `-nkvo` unknown on this build.
- `--kv-unified` : reduces per-slot KV waste with `-np > 1`; decode perf impact unmeasured on gfx1200; we default to 2 slots.
- Prompt-lookup / n-gram speculative decoding beyond simple cases — measured 187 t/s on whole-file rewrite tasks already (see DFlash2 notes); coverage is task-shaped, not general.
- Raising the app VRAM reserve for desktop GPUs (see the oversubscription cliff above) — needs a product decision, not just a number; affects every recommendation the app emits.
- b12430 runtime (MoE HIP prefill fix): needs `llama-bench -p 512 -fa 1` on Ornith before any adoption; our models are dense/MoE-off-GPU in the common case, so expected gain is nil for this machine's main model.
- Per-token cost of `-ctk q8_0` vs `q4_0` at 32K+ depth on Ornith (measured only at 8K on this dense 33B; KV-heavy models may differ).
- Windows-only: whether `--mlock` for the KV/weights changes spill behavior near the CLI reports; we have no positive evidence either way.

## Research notes (web, Sept-Nov 2026, reviewed 2026-11-20; leads, not measurements)

Goal behind this pass: tune faster, and reach >= 1.5x tok/s over an untuned llama-server on this card. Nothing below was run here.

- **Speculative speed on AMD is limited by the small-batch verify kernels, not by draft quality.** llama.cpp v1.0.0 (= b12359, 2026-11-13; our b12374 already includes it) moved MTP on a 27B from +11-17% to +52-55% (RTX 3090, homebrewserver.club, acceptance unchanged at 32-33%), and the gain came from a new CUDA kernel for 2-24-token batches (PR #31533) that is **NVIDIA-only**. Our own `GGML_MMVQ_MAX` patch hits the same bottleneck on HIP. Lead: measure verify cost directly with `llama-bench -ub 2..8` (or `--spec-synth-rates`) per backend before and after any runtime bump.
- **Vulkan PR #31804 (RDNA4 + AMD proprietary driver, MMVQ rows for Q3_K / Q5_K / IQ4_XS at batch 5-8)**: open, "merge ready" 2026-11-20. R9700 / Windows / Adrenalin 26.9.2: op-level IQ4_XS 1.28x (n=5) / 1.41x (n=8); Qwen3.8-27B UD-Q4_K_XL pp128 at ub 6/7/8 1.45x / 1.38x / 1.58x, tg128 unchanged. Batch 5-8 is the speculative-verify shape, and Nanbeige IQ4_XS-pure is an IQ4_XS file. UNTESTED: Vulkan + DFlash/n-gram/MTP on a build that has it.
- **`--spec-draft-n-max` was never set by the app** (default 3; now 2 for MTP, see the draft-length section below). The v1.0 MTP numbers above used 6; DFlash docs use 15, DFlash2/DSpark 7. UNTESTED: draft length 2/3/4/6 as an autotune dimension for MTP models (Qwen3.8-27B). Cheapest lead on this list.
- **stew675/llama-cpp-rdna-boosts** (MIT, active, last commit 2026-11-20, fork point 4f9dfa0e3, Windows supported): opt-in adaptive MTP draft depth (`--spec-type draft-mtp-adaptive`), MTP draft-vocab subset (`--spec-draft-mtp-vocab N`), K-quant/IQ4_XS WMMA verify matvec (`--wmma-verify`), q4_0 K/V fold (+10.8% tg128 on a q4_0-KV 30B-A3B, GPU not named), BF16 KV + native-BF16 FA. Most headline numbers are 2-3 GPU tensor-split and self-reported. Its Windows note matters to us: `amdhip64_7.dll` is usually loaded from `C:\Windows\System32` (the Adrenalin driver's copy) rather than the copy beside the exe, and that changed pinned-memory limits and prefill (370.7 vs 39.8 t/s on the author's box). UNTESTED here: whether the unexplained ~22% single-token slowdown since 2026-11-14 matches a driver-side `amdhip64_7.dll` change; check which copy llama-server loads (Process Explorer / `listdlls`) and its version.
- **Lemonade llamacpp-rocm b1383** (2026-11-21, ROCm 10.3.0a20261120 TheRock nightly, gfx120X Windows zip ~694 MB). b1375 already measured +22-26% dense prefill, tg identical. Re-test candidate, prefill only.
- **Mainline drafter family grew** (docs/speculative.md): `draft-dflash2`, `draft-dspark` (Qwen3 backbone only), `draft-lilicorr`, `--spec-draft-p-min` confidence truncation, `--spec-draft-sampling probabilistic`. No tree verification. Nothing new for Nanbeige (DFlash2 already rejected at 6-14% acceptance).
- **Tuning faster**: `--spec-synth-rates P0,P1,...` / `--spec-synth-len L` give a synthetic acceptance profile, so the cost of speculation can be benchmarked without prompts or drafts. llama-optimize (bigattichouse) screens many knobs with Morris/Taguchi designs instead of a grid. Both are leads for shortening the autotuner.
- **Looked at, not useful here**: FreeToken on RX 9060 XT / Windows (issue #122, Aug 2026): dense decode 93.6 vs llama.cpp ROCm 89.9 but Vulkan 113.0; MoE 61 vs 83.5. Behind llama.cpp. Papers: D3D (arXiv 2611.06425, async diffusion drafting), layer-wise CPU-GPU scheduling (2611.01393, OPT-30B vs Accelerate/DeepSpeed baselines), KnapSpec (2602.20217, training-free layer-skip self-speculation, not in llama.cpp). llama.cpp v1.0 breaking change: `--defrag-thold`/`-dt` removed (the app does not pass it).

## MTP draft length (`--spec-draft-n-max`) on Qwen3.8-27B (2026-11-20, RX 9060 XT, ROCm b12374-patched, `scripts/bench_spec.py`: `-np 1 -c 8192 -fa on -ctk/-ctv f16 -ub 512`, thinking off, temperature 0.6, seed 470, 256-token replies, 5 prompts x 2 reps, fresh server per arm; GPU idle, a CPU-only Python job was running alongside)

| arm | code | rewrite | prose | chat | report | mean | vs none | accepted |
|---|---|---|---|---|---|---|---|---|
| none | 21.1/21.4 | 21.4/21.4 | 21.5/21.4 | 21.4/21.4 | 21.5/21.4 | 21.4 | 1.00x | - |
| draft-mtp n2 | 33.5/33.0 | 39.9/39.6 | 29.0/28.9 | 33.0/33.3 | 31.9/32.0 | **33.4** | **1.56x** | 70% |
| draft-mtp n3 (llama.cpp default, what the app runs) | 32.8/32.5 | 43.3/43.7 | 26.2/26.0 | 30.7/30.7 | 30.6/30.5 | 32.7 | 1.53x | 60% |
| draft-mtp n4 | 28.9/31.2 | 46.5/46.5 | 21.4/21.2 | 26.0/25.7 | 28.3/28.0 | 30.4 | 1.42x | 49% |
| draft-mtp n6 | 31.8/31.1 | 57.7/57.5 | 19.7/19.6 | 25.1/25.7 | 27.1/27.1 | 32.3 | 1.51x | 38% |
| draft-mtp n8 | 25.9/30.1 | 58.4/58.4 | 16.3/17.8 | 20.1/23.4 | 20.4/26.5 | 29.7 | 1.39x | 31% |

- **Longer MTP drafts only pay on copy-heavy text.** Rewrite climbs 39.8 -> 58.4 tok/s from n2 to n8, while prose falls 29.0 -> 16-18 (below no speculation at n8). Acceptance drops 70% -> 31% as the draft grows, and on this card each extra verified token is not cheap enough to carry the misses. Fits the research note above: the RTX 3090's +52-55% at n6 came from an NVIDIA-only small-batch kernel.
- **n2 vs the default n3: +2% on the mean**, +11% prose, +8% chat, +4% report, +1.5% code, -8% rewrite. Small. Since the app stacks `ngram-simple` (which covers copied text), the decision depends on the stacked run below.
- Repeats agree within ~1% except n8 (code 25.9/30.1, report 20.4/26.5), which is noisy.

**Stacked with `ngram-simple` (what the app launches for MTP models), same harness, same day:**

| arm | code | rewrite | prose | chat | report | mean | vs none | accepted |
|---|---|---|---|---|---|---|---|---|
| draft-mtp,ngram-simple n3 (current app) | 32.4/32.3 | 42.1/44.9 | 25.7/25.8 | 30.6/30.5 | 30.4/30.1 | 32.5 | 1.52x | 52% |
| draft-mtp,ngram-simple **n2** | 33.0/33.2 | 52.7/53.2 | 28.7/28.8 | 33.0/33.0 | 32.0/31.9 | **36.0** | **1.68x** | 61% |

- **WORKS (measured, applied 2026-11-20 as `MTP_DRAFT_N_MAX` in `raptor-runtime` config.rs; unit-tested, not yet re-measured through the app): `--spec-draft-n-max 2` with the stacked mode is +11% on the mean over the app's current launch and wins every prompt** (prose +12%, chat +8%, rewrite +21%, report +5%, code +2%). With n2 the stack also beats MTP alone at n2 (36.0 vs 33.4), mostly on rewrite (53 vs 40), so n-gram drafts complement short MTP drafts. At n3 the stack is no better than MTP alone (32.5 vs 32.7).
- Caveats: one model (Qwen3.8-27B UD-IQ3_S), one card, ROCm only, 8K context, 256-token replies, thinking off. Not measured: Vulkan (its small-batch verify cost differs, see PR #31804), deep context, thinking on, other MTP models. The best length is likely backend- and model-specific, so it belongs in the autotuner rather than as a fixed constant.
