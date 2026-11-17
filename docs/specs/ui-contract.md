# UI contract: Tauri commands, events and types

Source of truth: `src-tauri/src/commands/*.rs` (commands), `src-tauri/src/lib.rs` (registration),
and the crates' serde derives. The TypeScript mirror is `src/api.ts`. When a Rust type changes,
change `src/api.ts` in the same commit.

## Conventions

- Every command returns `Result<T, RaptorError>`. A rejected call is the serialized error
  `{ "kind": "<RaptorErrorKind>", "message": "<text>" }`. `src/api.ts` wraps it as `RaptorApiError`.
  Kinds: `InvalidInput`, `NotFound`, `Conflict`, `Io`, `Network`, `Integrity`, `Runtime`,
  `Unsupported`, `Unauthorized`, `Internal`.
- Arguments are passed by JavaScript name (camelCase); Tauri maps them to the Rust parameters.
  Struct arguments (for example `request`) keep their serde field names (snake_case).
- `Option<T>` is `T | null` on the wire. Omitted optional arguments are `None`.
- Nothing returned to the UI contains an API key secret. The secret is returned once, by `api_key_rotate`.
- `SettingsView` has a new field `pinned_runtime` (`{id, version}` or `null`). `PinnedRuntime` is the shell type for it.
- The API key is persisted only as the gateway's digest record (`key_id`, `salt`, `digest`, `created_at`).
  Settings files from before this change held the plaintext secret; it is converted on load, the file is
  rewritten at once, and the existing client key keeps working.

## Commands

### Hardware and runtime

| Command | Args | Returns | Notes |
|---|---|---|---|
| `hardware_info` | - | `HardwareInfo` | Returns the hardware detected at startup (cached). Probes once only when nothing is cached yet. |
| `setup_status` | - | `SetupStatus` | First-run checklist: `checks` (`hardware`, `runtime`, `gpu`, `model`, `storage`, `performance_memory`, each `ok` + `detail`), `available_runtime` (compatible, not installed), `runtime_needs_confirmation`, `degraded`, `storage_free_bytes`, `notes`. Nothing is downloaded or installed at startup; installing is `runtime_ensure`, after the user confirms. If a runtime is selected, it runs `llama-server --list-devices` (no model is loaded) and returns `device_preflight`. When that finds no device for a GPU backend, the `gpu` check is `ok: false` and its `detail` starts with `GPU unavailable:` and the typed message. |
| `runtime_status` | - | `EngineStatus` | Includes the `gpu_verdict`. |
| `runtime_ensure` | - | `RuntimeSelectionView` | Installs the best runtime. Needs a runtime index file (see below). Refused while a model is loaded. While a runtime is pinned, only the pinned version is considered. `rejected` also lists index entries refused for lacking a verified sha256. |
| `runtime_rollback` | `id` (must be the runtime in use) | `RuntimeRollbackView` | Makes the previous installed version current and pins it. Refused while a model is loaded. Needs a prior `runtime_ensure`. |
| `runtime_unpin` | - | `void` | Clears the pin, so `runtime_ensure` follows the newest compatible version again. |
| `runtime_list` | - | `RuntimeListView` | Lists installed runtimes on disk and available runtimes from the manifest index. |
| `runtime_recommend` | `modelId`, `context?` | `Recommendation` | Launch settings and rationale for this machine. Nothing is loaded. Architecture-aware (same as `models_load`): the rationale starts with the architecture facts, for example `hybrid model: 16 of 64 layers hold KV, 48 hold recurrent state (4 slots)` or `architecture 'x': not in Raptor's tested list; llama-server will decide when you load it`. On a GPU runtime, llama.cpp's projection (`llama-fit-params`, no weights loaded, about a second) calibrates the estimate; if it fails the rationale says so and the GGUF estimate is used. |
| `runtime_install_preview` | - | `RuntimeInstallPreview` | What `runtime_ensure` would install, before anything is downloaded: `url` and `sha256` from the index, `download_bytes` from a HEAD request (`Content-Length`, null when the server does not report it or cannot be reached; `size_source` says which), `already_installed`, `rejected`, `notes`. Confirm with the user, then call `runtime_ensure`. |

### System and native dialogs

| Command | Args | Returns | Notes |
|---|---|---|---|
| `dialog_pick_file` | `filterExt?` (`string[]`) | `string | null` | Native file picker dialog (e.g. for .gguf model imports). |
| `dialog_pick_folder` | - | `string | null` | Native directory picker dialog (for storage or workspace selection). |
| `dialog_save_file` | `defaultName`, `filterExt?` | `string | null` | Native save dialog (Windows; null elsewhere or on cancel). Writes nothing: pass the chosen path to `conversations_export_to`. `defaultName` must be a plain file name. |

### Gateway

| Command | Args | Returns | Notes |
|---|---|---|---|
| `gateway_status` | - | `GatewayStatus` | `lan`, `tls`, `tls_fingerprint_sha256` (pin this), `insecure` (always false in the shell). |
| `gateway_rotate_tls` | - | `GatewayStatus` | Replaces the LAN self-signed certificate on the running listener and returns the new fingerprint. Clients must re-pin. Refused with `Conflict` when the gateway is not running. Emits `gateway://state`. |
| `diagnostics_log_tail` | `lines?` (1-2000, default 200) | `string[]` | Newest llama-server output lines held in memory, oldest first, each a JSON object `{"stream","text"}`. The file `logs/llama-server.log` is still written. |

### Models

| Command | Args | Returns | Notes |
|---|---|---|---|
| `models_search` | `query`, `sort?`, `limit?` (1-100) | `HfModelSummary[]` | GGUF repositories on Hugging Face. |
| `models_repo_files` | `repo` | `RepoFile[]` | Top-level GGUF files with sha256 and quant label. |
| `models_repo_info` | `repo` | `HfRepoInfo` | |
| `models_list` | - | `ModelEntry[]` | The library. |
| `models_storage` | - | `StorageUsage` | |
| `models_delete` | `id` | `void` | Removes the entry and its presets. Deletes the file only when it is managed. A split model is removed with all its shards. |
| `models_import` | `path` (absolute), `mode` (`copy` or `reference`) | `ModelEntry` | A path to the first shard of a complete split set imports the whole set as one entry. |
| `models_details` | `id` | `ModelDetails` | Adds the pinned Hub `revision` (commit sha, null for models installed before revisions were recorded), `hub_tags`, every file (`files`, all shards) and `architecture` (`ArchInfo`: `name`, `support` = `known` / `known_unsupported` / `unknown`, `note`). `unknown` shows the note "not in Raptor's tested list; llama-server will decide when you load it" and never blocks the model. Only `known_unsupported` (today: `clip`, a vision projector) blocks a download or an import, with the reason as the error. |
| `models_compare` | `ids` (1-8 installed), `context?` | `Comparison` | Side-by-side rows (quant, bits per weight, estimated size at the context, fit `compat`, heuristic `tags`), rated against this machine's RAM and summed VRAM. Heuristic parts: see the models README. |
| `models_rank_for` | `task` (`coding`, `reasoning`, `general`, `agentic`), `context?` | `RankedCandidate[]` | Installed models ranked for a task, with reasons. Heuristic. |
| `models_check_updates` | - | `UpdateCheck[]` | Compares each installed Hub model's pinned commit with the repo head. Makes no download and changes nothing. `Network` when offline. `changed` means the head moved, not that the bytes changed. |
| `models_scan_folder` | `dir` | `ScanItem[]` | Finds `.gguf` files (header read only). An unfinished `*.gguf.part` download is returned as `invalid_gguf` with a reason that starts `incomplete download`; it is never a model and never in `models_list`. |
| `models_download_start` | `repo`, `fileName`, `overwrite?` | `string` (job key) | Verified download. Progress and completion arrive as events. |
| `models_download_pause` | `key` | `void` | Keeps the partial file. |
| `models_download_resume` | `key` | `void` | Continues from the partial file. |
| `models_download_cancel` | `key` | `void` | Deletes the partial file. |
| `models_download_queue` | - | `QueueEntry[]` | Downloads started in this session. |
| `models_recover_downloads` | - | `RecoveryReport` | Partials left by earlier sessions. Start the same file again to resume. |
| `models_discard_partial` | `key` | `void` | |
| `models_load` | `modelId`, `context?`, `preset?` | `LoadModelOutcome` (`LoadModelResult` plus `preset`, `cpu_fallback`) | `preset` (a preset of this model, else `Conflict`/`NotFound`) supplies the context length when `context` is not given; the result names it. A failure is `{kind, message, backend_failure}`: `backend_failure` is the typed GPU diagnosis (`cause`, `message`, `suggestions`) for a GPU runtime with no device, and null otherwise. `cpu_fallback` is true only when `allow_cpu_fallback` is on and a GPU backend had no device; the run is then CPU-only and the UI must say so. Requires an installed runtime. Replaces any loaded model. `status.gpu_verdict` is the honest GPU state. For a GPU runtime, the device check runs first: if llama-server initializes no device of that backend, the load is refused with a `Runtime` error that names the cause (for example `hipblas.dll` needed but the HIP SDK ships `libhipblas.dll`), and no process starts. The same diagnosis is in `status.device_preflight.failure`. There is no silent CPU run. `status.device_preflight` lists the devices llama-server reported. `model_id` confirms what is running. `config_source` is `known_good` (the fastest stable config measured on this machine for this model and runtime, when its context matches) or `recommended`. `rationale` explains the choice and starts with the architecture facts (`hybrid model: N of M layers hold KV`, dense, MoE, or the unknown-architecture note). `benchmark` is the run recorded right after Ready: `stable`, `recorded` (`inserted`, `merged`, `kept_stable`, `not_saved`), `reasons` (why unstable, or the error). A failed run is stored as unstable and never hides. A crashed or failed load leaves no model loaded; calling `models_load` again retries. |
| `perf_history` | `model?` | `PerfEntry[]` | Stored benchmark records, newest first. `model` (library id) limits them to one model. Each entry has a stable `id`. |
| `perf_compare` | `a`, `b` (entry ids) | `PerfComparison` | Deltas of `b` against baseline `a`, in percent. A negative `ttft_delta_pct` means `b` answers faster. |
| `perf_run_benchmark` | - | `BenchmarkOutcome` | Runs the fixed benchmark again on the ready model and records it. `Conflict` when no model is ready. |
| `perf_best_known` | `model` | `PerfEntry | null` | The fastest stable record for this model on this machine with the installed runtime. |
| `models_unload` | - | `EngineStatus` | |
| `models_current` | - | `ModelEntry | null` | The library entry of the currently loaded model, or null if stopped. |

### Presets

Per model. A preset holds the system prompt, `n_ctx`, `temperature`, `top_p`, `top_k`,
`repeat_penalty` and `max_tokens`. Ranges are checked on every save. A model has at most one
default preset. The model must be installed.

| Command | Args | Returns | Notes |
|---|---|---|---|
| `presets_list` | `modelId` | `PresetList` (`presets`, `default`) | |
| `presets_get` | `modelId`, `name` | `Preset` | `NotFound` for an unknown name. |
| `presets_save` | `preset` | `Preset` | Creates or replaces. |
| `presets_delete` | `modelId`, `name` | `PresetList` | Deleting the default clears the default. |
| `presets_set_default` | `modelId`, `name?` | `PresetList` | `name` null clears the default. |

A preset is applied by `models_load` (`preset`) and by `chat_start` (`options.preset`, which must
name a preset of the loaded model). A chat preset adds its system prompt unless the messages already
start with one, and sets `top_p`, `top_k`, `repeat_penalty`, `temperature` and `max_tokens` unless
the same options are given explicitly.

### Settings, gateway exposure and workspaces

| Command | Args | Returns | Notes |
|---|---|---|---|
| `settings_get` | - | `SettingsView` | `notes` lists startup recovery problems. |
| `settings_set_storage_dir` | `path` (or `null`) | `SettingsView` | `restart_required` becomes true. |
| `settings_set_gateway_port` | `port` | `SettingsView` | Applied live. Reverted if the gateway refuses it. |
| `settings_set_gateway_enabled` | `enabled` | `GatewayStatus` | |
| `settings_set_lan` | `enabled` | `SettingsView` | `true` is refused with `Unauthorized` unless an API key exists. |
| `settings_set_context_target` | `tokens` (65536-200000) | `SettingsView` | Target; effective load context can be lower when model or hardware limits apply. |
| `settings_set_allow_cpu_fallback` | `enabled` | `SettingsViewV2` | Off by default. On: a load whose GPU backend has no device runs on CPU, and the result has `cpu_fallback: true`. `SettingsViewV2` adds `embedding_model` and `allow_cpu_fallback` to `SettingsView`. |
| `api_key_rotate` | - | `ApiKeyReveal` | Creates or replaces the key. Show the secret once. |
| `api_key_revoke` | - | `SettingsView` | Refused with `Unauthorized` while LAN is enabled. |
| `workspace_register` | `path` | `WorkspaceInfo` | The gateway serves only registered workspaces. |
| `workspace_remove` | `id` | `SettingsView` | |

### Chat

| Command | Args | Returns | Notes |
|---|---|---|---|
| `chat_start` | `requestId`, `messages`, `options?` (`temperature`, `max_tokens`, `preset`, `top_p`, `top_k`, `repeat_penalty`; see Presets) | `void` | Subscribe to `chat://<requestId>` first. The same stream runs for the command and for the gate test. Refused with `Conflict` when no model is loaded. |
| `chat_cancel` | `requestId` | `boolean` | `false` when the request is not running. |

`requestId`: 1-64 characters, letters, digits, `-` or `_`.

### Conversations

| Command | Args | Returns |
|---|---|---|
| `conversations_list` | - | `ConversationList` |
| `conversations_create` | `title` | `Conversation` |
| `conversations_get` | `id` | `Conversation` |
| `conversations_rename` | `id`, `title` | `Conversation` |
| `conversations_delete` | `id` | `void` |
| `conversations_append` | `id`, `messages` | `Conversation` |
| `conversations_truncate` | `id`, `count` | `Conversation` | Truncates to the first `count` messages (used by regenerate/rollback). |
| `conversations_replace_message` | `id`, `messageId`, `content` | `Conversation` | Replaces a single message's text in place. |
| `conversations_export` | `id` | `string` (JSON text; the UI saves the file) |
| `conversations_import` | `json` | `Conversation` (always stored under a new id) |
| `conversations_search` | `query`, `limit?` (1-200, default 50) | `ConversationSearch` | Case-insensitive, across all conversations, newest first. Each hit is one message: `conversation_id`, `title`, `message_id`, `role`, `snippet`. `truncated` is true when the limit or the 256 MiB read budget stopped the search. Files that cannot be read are listed in `problems`. |
| `conversations_export_to` | `id`, `path` (absolute, `.json`) | `ExportedFile` (`path`, `bytes`) | Writes the export atomically to a path the user picked with `dialog_save_file`. |

### Context Intelligence

Paths must be directories. The engine for a path is opened on first use and kept for the session.

| Command | Args | Returns |
|---|---|---|
| `context_index` | `path` | `UpdateReport` (re-scan; only changed files are parsed) |
| `context_index_status` | `path` | `IndexStatus` |
| `context_search` | `path`, `request: SearchRequest` | `Hit[]` (re-indexes first) |
| `context_build` | `path`, `request: ContextRequest` | `ContextResponse` (`manifest` explains every item) |
| `context_recover` | `path`, `request: RecoverRequest` | `RecoverResponse` |
| `context_symbols` | `path`, `query: SymbolQuery` | `SymbolHit[]` |
| `context_impact` | `path`, `file` | `ImpactReport` |
| `context_set_embedding_model` | `modelId` | `string` | The installed model that serves embeddings (settings `embedding_model`). Semantic retrieval uses it only while it is the loaded model: the backend calls its `/v1/embeddings`, and the server must support that endpoint. |
| `context_clear_embedding_model` | - | `void` | Turns semantic retrieval off. |
| `context_semantic_status` | `path` | `SemanticView` | `configured_model`, `reason` (why semantic retrieval is off for this workspace: no model loaded, another model loaded, or the endpoint did not answer; null when active or unconfigured) and the engine's `status` (vectors, chunks without a vector, last error). Lexical and graph retrieval always run. Summaries (`summarize_session`) are written by the loaded model. |
| `context_remember_project` | `path`, `kind` (`decision`, `convention`, `summary`, `completed_work`, `failed_attempt`), `text`, `paths` | `ProjectMemoryAdded` | Long-term memory for the workspace. Entries that cite files go stale when those files change. |
| `context_forget_project_memory` | `path`, `id` | `void` | Removes an entry; the way to retire a decision or convention. |

## Events

Subscribe with the helpers in `src/api.ts` (`subscribe.*`). Each returns an unlisten function.

| Event | Payload | When |
|---|---|---|
| `server://state` | `EngineStatus` | Read about once a second; sent only when the status changes (server state, GPU verdict, runtime, device preflight). A `crashed` state means the model is no longer loaded: call `models_current` to refresh. |
| `download://progress` | `DownloadProgress` | Every change of a download job. |
| `download://finished` | `DownloadFinished` | Completed, cancelled or failed. The library entry exists only on `completed`. |
| `gateway://state` | `GatewayStatus` | After the gateway starts, stops or is reconfigured. |
| `chat://<requestId>` | `ChatEvent` | `chunk` (text, in order), then exactly one `done` (`cancelled` true when stopped) or `error` (the partial text stays). |

## Types

Defined in `src/api.ts`, with the Rust source named here.

- Core: `Role`, `ChatMessage` (`raptor-core`).
- Hardware and runtime: `HardwareInfo`, `CpuInfo`, `GpuInfo`, `Os`, `GpuVendor`, `Backend`,
  `EngineStatus`, `RuntimeSummary`, `ServerStatus`, `ServerState`, `GpuVerdict` (with `offloaded_layers` and `total_layers`),
  `LaunchConfig`, `KvCacheType`, `Recommendation`, `MemoryEstimate` (`raptor-runtime`).
- Device preflight (`raptor-runtime::devices`): `DevicePreflight` (`device_preflight` on `EngineStatus` and `SetupStatus`),
  `RuntimeDevice`, `BackendLoadFailure` (`cause`: `hipblas_name_mismatch`, `hipblas_missing`, `hip_sdk_not_found`,
  `no_device_visible`; `message`; `suggestions`), `HipDependencies`. Defined at the end of `src/api.ts`.
- Models: `HfModelSummary`, `RepoFile`, `HfRepoInfo`, `SearchSort`, `ModelEntry`, `ModelSource`,
  `GgufMeta`, `StorageUsage`, `ScanItem`, `ScanStatus`, `DownloadProgress`, `JobState`,
  `RecoveryReport`, `PartialDownload` (`raptor-models`); shell: `DownloadFinished`, `QueueEntry`,
  `LoadModelResult` (with `model_id`), `ImportMode`.
- Context: `TaskKind`, `Category`, `SymbolKind`, `ContextRequest`, `ContextResponse`,
  `ContextManifest`, `ManifestItem`, `ExcludedItem`, `CategoryUsage`, `LineRange`, `SearchRequest`,
  `Hit`, `SymbolQuery`, `SymbolHit`, `ImpactReport`, `IndexStatus`, `UpdateReport`,
  `RecoverRequest`, `RecoverResponse`, `RecoveredItem`, `DiffInput` (`raptor-context`).
- Wave 6 (appended at the end of `src/api.ts`): `LoadModelOutcome`, `LoadFailure`
  (with `loadFailureOf`), `ChatStartOptions`, `Preset`, `PresetList`, `SearchHit`, `ConversationSearch`,
  `ExportedFile`, `RuntimeInstallPreview`, `ModelFile`, `ModelDetails`, `CompatRating`, `ModelTask`,
  `TaskTag`, `HardwareSummary`, `ComparisonRow`, `Comparison`, `RankedCandidate`, `UpdateStatus`,
  `UpdateCheck`, `SemanticStatus`, `SemanticView`, `MemoryKind`, `ProjectMemoryAdded`, `SettingsViewV2`.
- Architecture support (appended at the end of `src/api.ts`): `ArchSupport`, `ArchInfo`, and the
  `architecture` field merged into `ModelDetails` (`raptor-models` `gguf::ArchSupport`, `ArchInfo`).
- Shell: `SettingsView`, `ApiKeyInfo`, `ApiKeyReveal`, `WorkspaceInfo`, `GatewayStatus`,
  `MetricsView`, `RuntimeSelectionView`, `RuntimeRollbackView`, `RuntimeEntryView`, `RuntimeListView`,
  `ChatOptions`, `ChatEvent`, `StoredMessage`, `Conversation`, `ConversationSummary`,
  `ConversationList`, `NewMessage`.

## Files and external state

- Settings: `<platform data>/RaptorLLM/config/settings.json`. Always in the default location. Corrupt
  files are moved to `settings.json.corrupt-<unix>` and defaults are used; `settings_get.notes` says so.
- Storage root (default `<platform data>/RaptorLLM`, or `storage_dir`): `models/`, `runtimes/`,
  `downloads/`, `logs/llama-server.log`, `db/conversations/<id>.json`, `context/`, and
  `config/runtime-manifest.json`.
- Performance memory: `config/perf-memory.json` (schema 2). A corrupt file is reported by `setup_status` and `perf_history`, and is never overwritten.
- Gateway TLS (LAN mode): `config/gateway-tls/` holds the self-signed certificate and key, reused across restarts so the pinned fingerprint stays valid until `gateway_rotate_tls`.
- Runtime index: `config/runtime-manifest.json`, a `parse_manifest_index` document (schema 1).
  Without it, `runtime_ensure` fails with `NotFound`. At startup the bundled copy
  (`src-tauri/resources/runtime-manifest.json`) is copied here when no file exists; an existing file is never overwritten.
  Entries whose `sha256` is missing or `null` are refused (never installed) and reported in `rejected`.
- Workspace instructions: the first of `RAPTOR.md`, `AGENTS.md`, `CLAUDE.md` at the workspace root
  is included as protected context (64 KiB maximum).

## Gateway (for reference)

Started at launch when `gateway_enabled` is set, on `127.0.0.1:<gateway_port>` (default 8790).
LAN mode (`0.0.0.0`) requires an API key. Clients authenticate with `Authorization: Bearer <key>`
or `x-api-key`. Routes are described in `docs/specs/gateway.md` and `crates/raptor-gateway/README.md`.
The gateway forwards to the loaded llama-server (loopback only). Context routes serve registered
workspaces by id (`ws-` plus 16 hex digits).
