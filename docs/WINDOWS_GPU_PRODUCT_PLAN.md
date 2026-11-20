# Windows GPU support and workflow plan

**Scope:** Windows desktop first. Keep current runtime support, add capabilities through explicit
contracts, and promote GPU families only after repeatable device and inference gates pass.

## Product outcomes

1. A new user can see which GPU Windows detected, what runtime can use it, whether the backend
   actually sees it, and the next useful action.
2. Auto chooses only among installed and probed backends, fits the selected model to available
   memory, and explains the reason. A hardware family match alone is never proof of support.
3. Chat and Models show the active runtime, verified offload state, and recent measured performance.
4. Optimization can be started from the normal load path, reuses compatible profiles, and reports
   comparable measurements with their limits.
5. Runtime, model, benchmark and presentation concerns remain independently changeable.

## Recommended architecture

Keep the current Rust composition root (`src-tauri`) and subsystem boundaries. Add new behavior
through the following layers rather than putting GPU-specific decisions in React components:

| Layer | Owns | Contract |
|---|---|---|
| Hardware inventory (`raptor-runtime`) | Adapter identity, driver, memory and OS facts | Detected hardware facts; no support inference from marketing names alone |
| Runtime catalog (`raptor-runtime`) | Signed/hash-checked packages and declared backend capabilities | Runtime id/version, OS/architecture, backend and requirements |
| Runtime probe (`raptor-runtime`) | Executable health and `--list-devices` evidence | Installed, executable, backend initialized, device visible, structured diagnostics |
| Model compatibility (`raptor-runtime`, using `raptor-models` facts at composition root) | Architecture, quantization, memory fit, context and supported acceleration | A recommendation with confidence, evidence and caveats |
| Load and verification (`src-tauri`) | Apply preference, start server, inspect logs and model readiness | Actual loaded backend, GPU layers, fallback and typed failure |
| Benchmark/profile (`raptor-runtime`, persisted by host) | Repeated samples, stability, environment key and invalidation | Reproducible profile keyed by GPU/driver, model hash, runtime/backend and config/context |
| UI (`src/`) | Read-only views, explanation and user actions | Read DTOs from Tauri; never independently decide support or launch a runtime |

Use typed evidence states consistently: **detected → runtime installed → backend initialized →
device visible → model loaded → GPU offload verified → speed measured**. Persist benchmark metadata
with results; don't collapse these states into a single “supported” boolean. Auto can prefer a native
runtime initially, but should use successful preflight and same-machine measurements to rank choices
once more than one backend is available. It must not silently switch to CPU.

## Ranked implementation plan

### Phase 1 — trust and first-run readiness

1. **GPU-ready checklist:** surface adapter, VRAM, driver, installed/available runtime, device
   preflight, and one next action (install, retry, choose Vulkan, or load a model). Name unsupported
   and unverified combinations separately.
2. **Shared status summary:** show active backend, GPU/offload verdict and last stable generation,
   prompt and TTFT measurements in Chat and Models. Keep detailed evidence in the load report.
3. **Evidence-based compatibility matrix:** run the hardware gate for RX 6000+, RTX 40/50,
   Arc A770/B580 and Arc Pro B-series. Promote each exact tested driver/runtime/device tuple by
   gate stage, not by vendor name.

### Phase 2 — predictable Auto and optimization

4. **Evidence-driven Auto:** probe backend executables/devices first; estimate model architecture,
   quantization, VRAM fit and context; use compatible stable local measurements where available;
   explain a heuristic choice and offer a short same-model comparison when evidence is insufficient.
5. **Quick optimization in load flow:** offer a fast tune before first load or after import, estimate
   elapsed time and peak resources, show progress/cancel, and retain deep-context tuning as optional.
6. **Fair default comparison:** compare stock defaults and optimized settings with identical model,
   runtime, backend, context, prompt and output. Report generation tokens/s, prompt tokens/s, TTFT and
   percent change. TTFT requires a real streamed request per candidate; `llama-bench` alone does not
   measure it. Warm up, repeat, preserve samples and omit unstable runs from headline results.
7. **Reusable measured profiles:** key by hardware identity and driver, model content hash and
   quantization, runtime/backend version, acceleration flags, and context range. Reuse exact matches;
   invalidate only on relevant environment changes. Retain history so profile changes are explainable.

### Phase 3 — workflow and sustained performance

8. **Contextual Models actions:** make install runtime, import/download model, load and optimize
   available at the point a user needs them; keep backend and tuning detail in progressive disclosure.
9. **Performance options behind evidence:** evaluate Flash Attention, speculation and cache modes
   per runtime/model. Enable automatic recommendations only after repeatable tests; state hardware
   and model limits next to each option.
10. **Operational quality:** keep Windows driver/runtime troubleshooting actionable, collect opt-in
    diagnostics without model content, and add CI checks for manifest hashes, package extraction,
    device-log parsing and DTO/UI contracts. Hardware gates remain a release qualification step.

### Coding-agent context profile

Treat agent context as a workload budget, separate from the model's advertised maximum and from the
VRAM-fit context recommendation used for ordinary chat. Set the coding-agent target minimum
to **65,536 tokens** and the default target to **122,880 tokens** ("120K ideal"), with a hard
setting ceiling of **200,000 tokens**. The effective
context remains capped by the model's trained length and measured hardware fit; report explicitly
when it falls below 65,536. Offer larger windows through 200K when supported. Keep lightweight
chat/code questions available below that floor, but explain that multi-step repository work may
compact more often or lose tool history. This is a design target informed by Hermes' 65,536-token
local tool-use guidance, not a cross-harness quality result; Raptor should validate the floor with
its own eval suite before calling it empirically validated.

The 65,536 floor is evidence-based as a conservative **runtime readiness floor**: Hermes currently
checks that local tool-using models meet this minimum and calls 65,536 a known-good starting point.
OpenCode's documented compactor retains about 15,000 recent tokens and reserves at least 16,000
tokens or 15% of the model window, whichever is larger. These establish practical harness patterns,
not that every model completes coding tasks equally well at 65K. That claim still needs the task
success evaluation below.

Use a cache-friendly, retrieval-first compactor:

1. Keep stable instructions and tool schemas at the beginning of the prompt. Keep the original task,
   constraints, current plan, and any in-flight tool call/result group intact.
2. Before summarizing, replace old completed tool output with a small record: command, exit status,
   key lines, and artifact/file path. Keep full logs and large diffs on disk for exact retrieval.
3. Summarize completed middle turns into structured state: goal, constraints, decisions, touched
   files, checks and outcomes, unresolved questions, and next actions. Do not rewrite the stable
   prefix on every turn; background compaction can otherwise defeat prompt-cache reuse.
4. Retain a budgeted recent tail, aligned to whole assistant/tool rounds. Retrieve older evidence
   from the transcript or repository when needed instead of relying on a lossy summary.
5. Trigger compaction from the effective prompt token count, reserving output space and the larger
   of a 16K safety margin or 20% of the model window. Keep roughly the latest 15K tokens verbatim;
   summarize older completed turns. Use provider-reported token counts when available, with a safe
   tokenizer-based estimate otherwise. Back off when compaction makes no meaningful reduction.
   Report the remaining budget and let users compact or start a fresh task explicitly.

Validate 16K, 32K, 64K, and 128K budgets using the same local model, harness prompt, tools, and
repository tasks. Score completed task and test success, constraint retention, wrong-file edits,
tool recovery, compaction count, latency, and memory use. Adopt the smallest budget whose success
rate is within a predeclared margin (for example, 3 percentage points) of 128K and has no increase
in constraint or tool failures. Include bug fixes, cross-file changes, test-driven edits, and long
tool-output sessions; speed-only llama-bench numbers cannot establish agent quality.

## Boundaries and references

Atomic Chat is a useful reference for one-click provider setup, a common inference API across engines,
and clearly qualified performance features. Jan is a useful reference for explicit engine variants and
concrete Windows build/troubleshooting guidance. Raptor should reuse these product patterns while
keeping its own evidence-first runtime qualification and safe explicit install confirmation. Do not
copy their multi-engine breadth until Windows qualification and user demand support the maintenance
cost.

Relevant upstreams to track:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): Windows backend packages and inference behavior.
- [Atomic Chat](https://github.com/atomicbot-ai/atomic-chat): engine/provider orchestration and setup.
- [Jan](https://github.com/janhq/jan): Windows engine configuration and troubleshooting.
- [Hermes context sizing and compression](https://github.com/NousResearch/hermes-agent/blob/main/agent/conversation_loop.py): local tool-use guidance starts at 65,536 tokens and uses automatic context growth/compaction.
- [OpenCode compaction](https://opencode.ai/docs/compaction/): context-aware compaction for continuing long sessions.
- [Claude Code context guidance](https://support.claude.com/en/articles/14552983-models-usage-and-limits-in-claude-code): selective file references, `/compact`, and task resets.
- [Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/): automatic compaction and preserving a stable prompt structure for cache reuse.

## Implementation tracking

- [x] CUDA and SYCL runtime packages and explicit backend choices; Vulkan remains available.
- [x] First-run checklist and device visibility preflight; checklist now includes the detected
  driver's reported version when Windows provides one.
- [x] Chat and Models sidebar summary for active runtime, GPU verdict and latest stable speed/TTFT.
- [x] Context setting targets 65,536 tokens by default, accepts targets through 200,000, migrates
  older saved settings, and reports when model/hardware fit produces a smaller effective window.
- [x] Context fit probing and the context benchmark matrix can qualify windows up to 200K (192K
  benchmark depth plus turn reserve); actual support remains model and device specific.
- [x] Quick-tune llama-bench sweep uses the effective context depth, and its streamed baseline uses
  the same first launch variant, cache, flash attention, speculation and slots as a normal load.
- [ ] Validate the 65,536-token coding-agent floor with a cross-harness task-quality evaluation;
  performance benchmarks alone do not validate agent context quality.
- [x] Auto retries Vulkan when a native GPU runtime cannot initialize a visible device, and records
  the evidence in the load rationale. Explicit backend preferences remain strict.
- [x] Driver-aware hardware fingerprints and profile compatibility for KV-cache and Flash Attention
  settings; incompatible or legacy profiles are skipped with a user-facing explanation.
- [x] Three-repeat llama-bench sampling plus a matched live comparison: one warm-up and three
  streamed samples for default and winner, with prompt/decode throughput, TTFT, exactness, context,
  runtime/backend, cache, Flash Attention and speculation metadata. Unstable live samples are
  excluded and reported as comparison errors.
- [x] Quick and deep profile settings retain their own cache/attention compatibility and context
  range. Machine fingerprint includes GPU and driver; model SHA and runtime id/version are in the
  profile key. Mismatched and legacy profiles are skipped with an explanation.
- [ ] Rank alternate backends using successful model-specific benchmarks where multiple runtimes
  are installed; Auto fallback currently uses device preflight evidence, not a backend speed sweep.
- [ ] Run and record physical validation for each requested GPU family and relevant driver/runtime
  combination. Package availability is not hardware qualification.
- [ ] Add cancelable short tuning and an explicit estimate of peak memory; deep-context tuning
  remains optional. Progress now includes the live comparison steps.
- [ ] Complete runtime/model actions at point of need and add CI checks for package and UI contracts.
- [ ] Evaluate coding-agent quality across 16K, 32K, 64K and 128K effective contexts before
  advertising a supported agent-mode minimum; throughput-only benchmarks do not qualify context.

Hardware is the limiting external dependency for the uncompleted qualification rows. The software
work that doesn't require those GPUs can continue independently; unsupported combinations must stay
marked unverified until the release gate passes.
