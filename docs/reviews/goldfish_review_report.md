Status: resolved (round 1: 17 fixes applied by replacing docs/ARCHITECTURE.md; goals and non-goals escalated and left out pending the owner)

# Goldfish Annotated Review Report

**Target:** `docs/ARCHITECTURE.md` (full file, lines 1-50) · **Round 1** · reviewers: zoom_out
(zoom-out-standard), design_critic (zoom-out-standard), codebase_details (re-implementation-standard),
all completed. Raw merged findings: [round1_findings.md](round1_findings.md).

**Overall:** all 17 findings are valid. Five key claims were re-checked against the code before this
report (gateway on 127.0.0.1:8790, `server.rs:93-111`; LAN refused without a key and without TLS
unless `allow_insecure_lan`, `server.rs:121-130`; manifest signatures "Not verified yet",
`manifest.rs:70`; no GPU telemetry referenced in `src-tauri/src`; each subsystem crate depends only
on `raptor-core`, `crates/*/Cargo.toml`). One correction to the reviewers: there is no `ModelCatalog`
trait; the gateway's seams are `Upstream` and `ContextService`.

**Proposed fix for all findings:** replace `docs/ARCHITECTURE.md` with
[ARCHITECTURE.proposed.md](ARCHITECTURE.proposed.md), one consolidated rewrite. Patching 17 gaps
line by line would keep a 50-line document built around the wrong topology. The rewrite keeps the
principles that were right (positive-evidence GPU verification, no silent CPU fallback, protected
context items) and points to `docs/specs/` for detail instead of repeating it (repetition is what
drifted).

## Needs your decision first

### [Escalate] Goals and non-goals (added for zoom_out-1-1)
*   **Affected Scope:** proposed doc, "What it is"
*   **Why escalate:** the code cannot confirm product intent. The draft says the goals are verified GPU
    acceleration, safe defaults, measured speed rules and an agent-facing API with repository context,
    and the non-goals are multi-user serving, cloud inference, training and running more than one model
    at a time. Please confirm or correct.

## Prioritized Findings & AI Suggestions

### [Issue 1] [`zoom_out` + design_critic + codebase_details] - Severity: Blocker
*   **Affected Scope:** [Lines 1-8](../ARCHITECTURE.md#L1-L8) (whole document)
*   **Discrepancy Description:** No purpose, goals, non-goals, status or verified date; no links to the specs or RULES.md; README.md calls implemented what this document calls future.
*   **Rationale:** a reader cannot tell what the system is for or whether the document is current.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid; README.md:3-19 confirms the contradiction.
*   ***Proposed Edit:*** proposed doc header (status line with commit and date, related-doc links) and "What it is" (goals and non-goals; see the escalation above).

### [Issue 2] [`zoom_out` + design_critic + codebase_details] - Severity: Blocker
*   **Affected Scope:** [Lines 10-15](../ARCHITECTURE.md#L10-L15) (Process boundaries)
*   **Discrepancy Description:** No crate map and no dependency rule, though the code follows one strictly (subsystems depend only on raptor-core; src-tauri is the composition root; gateway trait seams). CI does not enforce it; RULES.md:7 misstates the gateway's role.
*   **Rationale:** this is the rule a cleanup must keep.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** verified in `crates/*/Cargo.toml` and `src-tauri/Cargo.toml`.
*   ***Proposed Edit:*** new section "Crates and the dependency rule" (table plus rule plus trait seams); the CI gap is listed under "Testing". RULES.md:7 is outside this target (cleanup).

### [Issue 3] [`zoom_out` + design_critic + codebase_details] - Severity: Blocker
*   **Affected Scope:** [Lines 15, 28-40](../ARCHITECTURE.md#L28-L40) (Context Intelligence)
*   **Discrepancy Description:** Labelled future, but the whole pipeline, its local API and MCP tools ship; it is an in-process library, not a service; recovery, memory and git modules are missing; lines 31-38 repeat context.md.
*   **Rationale:** a cleanup guided by this text could delete working code.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid (`raptor-context/src/lib.rs`, gateway `context_api.rs`, `mcp.rs`).
*   ***Proposed Edit:*** section "Context engine": implemented, in-process facade, exposure via routes and MCP, the three rules the architecture depends on; pipeline detail deferred to context.md.

### [Issue 4] [`design_critic` + zoom_out + codebase_details] - Severity: Blocker
*   **Affected Scope:** [Lines 10-15](../ARCHITECTURE.md#L10-L15)
*   **Discrepancy Description:** raptor-gateway is missing entirely; line 14 implies clients reach llama-server directly.
*   **Rationale:** the gateway is the product's external interface and its main security boundary.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid (`raptor-gateway/src/server.rs`, `adapters.rs` `RuntimeUpstream`).
*   ***Proposed Edit:*** "Processes and how they talk" (diagram, shared upstream, llama-server port never offered), gateway row in the crate table, "Main flows" (agent request).

### [Issue 5] [`codebase_details` + zoom_out + design_critic] - Severity: Blocker
*   **Affected Scope:** [Lines 44-47](../ARCHITECTURE.md#L44-L47) (Security)
*   **Discrepancy Description:** LAN API keys and the 0.0.0.0 rule are listed as to-do but are done, and stricter (LAN needs a key and TLS unless `allow_insecure_lan`; 0.0.0.0 refused outright).
*   **Rationale:** a rebuild from this text would be weaker than the code.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** verified, `server.rs:121-130`.
*   ***Proposed Edit:*** "Security > Controls in place", LAN and bind bullets.

### [Issue 6] [`design_critic` + zoom_out + codebase_details] - Severity: Blocker
*   **Affected Scope:** [Lines 48-50](../ARCHITECTURE.md#L48-L50)
*   **Discrepancy Description:** "Signature/hash verification" overclaims: manifest signatures are not verified, the update digest is not an independent signature, `use_local_runtime` runs an unverified binary; path validators are not named. README.md:15 repeats the false "signed" claim.
*   **Rationale:** a security document must not state controls that do not exist.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** verified, `manifest.rs:70`.
*   ***Proposed Edit:*** "Security > Known gaps" (three items, "do not describe these as done anywhere"). README.md:15 is outside this target (cleanup).

### [Issue 7] [`codebase_details` + zoom_out + design_critic] - Severity: Blocker
*   **Affected Scope:** [Lines 19-24](../ARCHITECTURE.md#L19-L24) (GPU strategy)
*   **Discrepancy Description:** Verification is the device preflight plus the server log only; telemetry is never called; no smoke test; HIP-only wording although CUDA and Vulkan are verified; verdict types undocumented.
*   **Rationale:** the stated evidence chain does not exist; the principle itself is right.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** verified (no telemetry use in `src-tauri/src`; `verify.rs`).
*   ***Proposed Edit:*** "GPU verification and runtime selection": the two evidence sources, the three verdicts, backend-neutral wording, CPU-fallback policy, and a "Not done" line for telemetry and a smoke test.

### [Issue 8] [`codebase_details`; zoom_out and design_critic rated Nit] - Severity: Blocker
*   **Affected Scope:** [Lines 5-8](../ARCHITECTURE.md#L5-L8) (Product structure)
*   **Discrepancy Description:** Settings lists "appearance" (absent) and "performance" (on Models); Models omits Performance and Compare tabs, presets and load-failure handling; Chats is undersold.
*   **Rationale:** a rebuilt UI from this text would be wrong.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** verified, `Settings.tsx:20-29`, `Models.tsx:15-23`.
*   ***Proposed Edit:*** section "User interface" with the real pages, tabs and sections.

### [Issue 9] [`codebase_details`] - Severity: Major
*   **Affected Scope:** [Lines 42-50](../ARCHITECTURE.md#L42-L50)
*   **Discrepancy Description:** Existing controls undocumented (keyless loopback, CORS allowlist, body limit, rate limit and queue, TLS, hashed keys, workspace confinement, llama-server loopback bind); should be a threat model with controls, not a to-do list.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid.
*   ***Proposed Edit:*** "Security": threat model, controls in place, known gaps.

### [Issue 10] [`codebase_details`] - Severity: Major
*   **Affected Scope:** [Line 26](../ARCHITECTURE.md#L26)
*   **Discrepancy Description:** "Tested runtime profiles" have no mechanism; the real one (seeded manifest, `select_runtime`, minimum HIP SDK, rollback rules), the CPU-fallback policy, the per-model backend choice and the speed plan are undocumented.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid.
*   ***Proposed Edit:*** "GPU verification and runtime selection" (manifest, selection, per-model backend, rollback rule) and "Main flows" (speed plan).

### [Issue 11] [`design_critic`] - Severity: Major
*   **Affected Scope:** [Lines 12-15](../ARCHITECTURE.md#L12-L15)
*   **Discrepancy Description:** "Rust application core" matches no component; the real topology is three processes joined by Tauri IPC and loopback HTTP, with a kill-on-close Job Object; state ownership and "isolated" undefined.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid (`job.rs:1-11`).
*   ***Proposed Edit:*** "Processes and how they talk". Start-up and shutdown ordering is left to runtime.md.

### [Issue 12] [`design_critic`] - Severity: Major
*   **Affected Scope:** whole document
*   **Discrepancy Description:** No state, storage, configuration, error or UI-contract architecture.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid.
*   ***Proposed Edit:*** section "State and storage" (DataDirs, JSON stores with atomic writes and versioning, logs, `RaptorError`, link to ui-contract.md).

### [Issue 13] [`design_critic`] - Severity: Major
*   **Affected Scope:** whole document
*   **Discrepancy Description:** RULES.md's cross-cutting principles (traits with fakes, no network/GPU in unit tests, real-hardware tier, no silent fallbacks) are not carried into the architecture.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid.
*   ***Proposed Edit:*** trait-seam sentence in the crate section, section "Testing", and the no-silent-fallback rule in the GPU section.

### [Issue 14] [`codebase_details`] - Severity: Major
*   **Affected Scope:** [Lines 34-35](../ARCHITECTURE.md#L34-L35)
*   **Discrepancy Description:** Scoring factors are wrong (weighted rank fusion over ten signals by task kind; no "test coverage" signal; embeddings optional).
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid (`retrieve.rs:18-80`).
*   ***Proposed Edit:*** the factor list is removed; "weighted rank fusion over several retrieval signals" with detail deferred to context.md (avoids a second copy that drifts). The stale `retrieve.rs:3` comment is a cleanup item.

### [Issue 15] [`codebase_details`] - Severity: Major
*   **Affected Scope:** [Lines 36, 40](../ARCHITECTURE.md#L36-L40)
*   **Discrepancy Description:** Budget contract unspecified; protected categories differ from `budget.rs:14-24`; overflow reported as `over_budget_by`.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** valid.
*   ***Proposed Edit:*** "Context engine" rules point to `budget.rs` as the source of the protected categories and name `over_budget_by`.

### [Issue 16] [`design_critic`] - Severity: Nit
*   **Affected Scope:** [Lines 21, 26](../ARCHITECTURE.md#L21-L26)
*   **Discrepancy Description:** RDNA 4 detail in a general section; HIP-specific wording.
*   ***AI Suggestion:*** PROPOSE FIX (covered by the rewrite)
*   ***Proposed Edit:*** backend-neutral GPU section; card-specific facts stay in EXPERIMENTS.md and runtime.md.

### [Issue 17] [`design_critic`] - Severity: Nit
*   **Affected Scope:** [Lines 10-15, 30-38](../ARCHITECTURE.md#L10-L38)
*   **Discrepancy Description:** Process and module concepts mixed; no diagram.
*   ***AI Suggestion:*** PROPOSE FIX (covered by the rewrite)
*   ***Proposed Edit:*** separate "Processes" and "Crates" sections, with a process diagram.

## Outside the target (for the cleanup, not this review)

- README.md:15 claims "signed" runtime manifests (false); README.md:18 lists 4 MCP tools (the code has 6).
- RULES.md:7 says the gateway does wiring (src-tauri does).
- `raptor-context/src/retrieve.rs:3` comment says six signals (there are ten).
- CI does not check the crate dependency rule.
