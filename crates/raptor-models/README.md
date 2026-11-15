# raptor-models

Model Manager for Raptor LLM: Hugging Face discovery, GGUF header metadata, quant guidance,
pinned and verified downloads (single files and split shard sets), the local model library,
model comparison and per-model presets. Spec: `docs/specs/models.md`. Depends only on `raptor-core`.

## Public API

Facade (re-exported from the crate root):

- `ModelManager::with_hugging_face(dirs, token)`: production wiring (HTTPS, real disk space).
- `ModelManager::new(dirs, hf, fetch, disk)`: explicit wiring for tests or alternative transports.
- `search(query, sort, limit)`, `repo_files(repo)`, `repo_info(repo)`: Hub queries (`repo_files` lists `main`, subfolders included).
- `start_download(repo, path, overwrite) -> DownloadHandle`: pins the download to the repo's current commit sha, verifies sha256 against the Hub's LFS hash, and promotes atomically. If `path` is a shard of a split model, the whole set is one job and one library entry; an incomplete set is refused before any transfer.
- `check_updates() -> Vec<UpdateCheck>`: compares each installed Hub model's pinned commit with the repo head. Makes no download and never replaces anything. Returns a clean `Network` error when offline. `UpdateStatus::{UpToDate, Changed, NotRecorded}`.
- `compare(ids, ctx, hardware) -> Comparison`: side-by-side view of installed models.
- `rank_installed(task, hardware, ctx) -> Vec<RankedCandidate>`: ranking of installed models.
- `model_details(id) -> ModelDetails`: files (every shard), pinned revision, Hub tags.
- `hub_rate_limit() -> Option<HubRateLimit>`: rate-limit state from the last Hub response.
- `delete_model(id)`: removes a library entry (all shards), its presets and its default. Referenced files are kept.
- `library()`, `presets()`, `downloads()`: access to the subsystems below.

`hf`:
- `trait HfApi` (`search_gguf`, `list_repo_files`, `list_repo_files_at`, `repo_info`, `repo_snapshot`, `file_url`, `file_url_at`, `last_rate_limit`); `HttpHfApi` implements it.
- `RepoFile { file_name (repo-relative path), size, sha256, quant_label, split_shard }`, `RepoSnapshot { info, revision }`, `HubRateLimit`, `RetryPolicy` (bounded, default 3 attempts, 0.75 s doubling to 8 s).
- `parse_search`, `parse_tree`, `parse_repo_info`, `parse_repo_snapshot`, `parse_rate_limit`, `next_link`.
- `validate_repo_id`, `validate_commit_sha`, `safe_repo_path` (1 to 4 segments, no traversal), `is_split_shard`, `map_http_status`, `is_transient_status`.

`shard`:
- `parse_shard_name`, `detect_shard_sets(&[RepoFile]) -> Vec<ShardSet>`, `set_containing`; `ShardSet { name, total, shards, missing, size_bytes }` with `is_complete()`.

`gguf`:
- `parse(bytes) -> Result<GgufMeta>`, `probe(bytes) -> Result<HeaderProbe>` (works on a prefix), `parse_file(path)`, `parse_file_with_split(path) -> (GgufMeta, SplitInfo)`.
- `GgufMeta::kv_bytes_per_token()`.
- Architecture support, three outcomes: `classify_architecture(arch) -> ArchSupport` (`Known`, `KnownUnsupported`, `Unknown`),
  `arch_info(Option<&str>) -> ArchInfo { name, support, note }` (the note is what the UI shows),
  `ensure_supported(Option<&str>)` (download gate: refuses a missing name or `KnownUnsupported` only),
  `refuse_known_unsupported(arch)` (import gate). `is_architecture_supported(arch)` means `Known`.
  `ModelDetails.architecture` carries the `ArchInfo`.
- `GgufMeta` also carries hybrid, SSM and MoE keys (`full_attention_interval`, `nextn_predict_layers`,
  `ssm_*`, `value_length`, `head_count_kv_layers`, `expert_count`, `expert_used_count`). Adding fields
  breaks struct literals without `..Default::default()`.

`arch`:
- `GgufMeta::kv_layer_count()`, `recurrent_layer_count()`, `recurrent_state_bytes(n_slots)`,
  `kv_cache_bytes_per_token_f16()`, `is_moe()`, `main_layer_count()`.
- `runtime_hints(&GgufMeta) -> ArchHints { kv_layers, recurrent_bytes_per_slot, offload_layers }`: plain
  numbers for the runtime's `ModelArch`. Zero means the header lacked the input.

`quant`:
- `parse_quant_label(file_name)`, `lookup(label)` (approximate bits per weight).
- `compat_rating(footprint, ctx, hw) -> CompatRating` (`Great`, `Ok`, `Tight`, `WontFit`).
- `recommend_quant(options, kv_bytes_per_token, ctx, hw) -> Vec<QuantRecommendation>`.

`compare` (see the heuristics note below):
- `compare(&[ModelProfile], ctx, Option<HardwareSummary>) -> Comparison`; `ModelProfile::from_details`.
- `rank_for(task, &[ModelProfile], hw)`, `rank_for_at(task, candidates, hw, ctx)`.
- `task_tags(&ModelProfile) -> Vec<TaskTag>`, `params_estimate`, `params_from_name`.

`download`:
- `DownloadManager::{start, start_with, start_set, pause, resume, cancel, recover, discard_partial, list_jobs}`. `start_with` and `start_set` take a `Provenance { revision, hub_tags }`.
- `DownloadHandle::{progress, pause, resume, cancel, finish}`; progress is a `tokio::sync::watch` stream. For a set, `total` is the sum and `downloaded` spans all shards.
- Traits: `HttpFetch`, `DiskSpace` (`available_bytes`, `same_volume`), `Registrar` (`register`, `register_with`, `register_set`). `HttpFetcher`, `SystemDiskSpace` and `same_volume` are the real ones.

`library`:
- `Library::{open, list, get, details, file_path, import_copy, import_reference, register_completed, register_completed_with, register_shards, scan_folder, verify, delete, storage_usage, managed_path}`.
- `ModelEntry` (a split model is one entry: summed size, first shard as path) and `ModelDetails { entry, revision, hub_tags, files }`.

`presets`:
- `PresetStore::{open, list_for_model, get, default_for_model, create, update, save, delete, set_default, clear_default, delete_for_model}`; `validate(&Preset)`.
- Ranges: `n_ctx` 256 to 1,048,576; `temperature` 0 to 2; `top_p` in (0, 1]; `top_k` 0 to 1000; `repeat_penalty` 0.5 to 2; `max_tokens` 1 to 1,000,000 or unset; names 1 to 64 printable characters; system prompts at most 32 KiB.
- `create` refuses an existing key, `update` refuses a missing one, `save` creates or replaces. Each model has at most one default preset; deleting the default clears it.

All fallible calls return `raptor_core::Result`. Methods that touch the network or the download
queue must run inside a Tokio runtime.

### Heuristics (compare and tags)

Tags for coding, reasoning, general and agentic use are **heuristic**: they come from Hub tags,
name tokens and header metadata, with fixed weights documented on `compare::task_tags`. A task
needs a score of at least 2, so metadata alone never tags coding, reasoning or agentic use.
Parameter counts are estimates from file size and the quant table. They are for ordering and
hints, not facts.

## Status

Implemented and covered by offline unit tests (209 tests; no network, no real GPU, no real
llama-server). Nothing here has been run against live huggingface.co. The real-header tests read
local samples from `RAPTOR_GGUF_SAMPLES` (default `D:\AI_Storage\lmstudio\lmstudio-community`) and
skip when they are absent.

- **hf**: search, tree listing (subfolders, pagination with a same-host check), repo info and
  head commit parsing from fixtures. Repo id, path and revision validation (traversal, backslash,
  depth limit). Bearer token sent only as a header, redacted in `Debug`. Hub API calls retry
  transient failures (429, 5xx, transport errors) a bounded number of times and honour
  `Retry-After` up to the policy cap; permanent failures are never retried. Dropping the future
  cancels the backoff. Rate-limit headers (`RateLimit`, `RateLimit-Policy`) are parsed into
  `HubRateLimit`; the format follows the documented structured-header style and is not yet
  checked against live responses.
- **arch**: KV layer count (per-layer `head_count_kv` array, else `full_attention_interval`, else
  all main layers, 0 for pure SSM), recurrent state bytes (f32 conv plus SSM state per recurrent layer
  and slot), and f16 KV bytes per token over KV layers. Verified against a real `qwen35` header:
  16 of 64 attention layers, 256 MiB KV at ctx 4096, 598.5 MiB recurrent state for 4 slots, 66
  offloadable layers. Hand-built fixtures cover dense, hybrid, granite-style array, pure SSM and MoE.
  The formulas assume llama.cpp's layout and are not checked for Mamba-2 or delta-net layouts other
  than `qwen35`. NextN layers are excluded from KV and recurrent counts. Sliding-window layers
  (for example `flint`) are counted as full KV layers, so the KV estimate is an upper bound there.
- **shard**: `name-00001-of-000NN.gguf` parsing, grouping by directory and prefix, missing-shard
  reporting, duplicate-index refusal.
- **gguf**: hardened header parser for GGUF v2/v3 with length caps, bounds checks and prefix
  probing. It also reads `split.no` and `split.count`. The assumption that `split.no` is 0-based
  and that every shard carries both keys follows llama.cpp's writer; it is not yet verified against
  real split files. The `general.file_type` table and the architecture lists are maintained by hand.
  Architecture policy: `SUPPORTED_ARCHITECTURES` (`Known`) holds names Raptor tested or that llama.cpp b12340
  loaded from a local file (`qwen35`, `qwen35moe`, `qwen4exp`, `flint`, `granite`; confirmed through
  `llama-fit-params`, which reads the same header). `KNOWN_UNSUPPORTED_ARCHITECTURES` holds `clip`, which
  llama-server refuses as a main model. Any other name is `Unknown`: it never blocks a download or an
  import, and the UI shows "llama-server will decide when you load it". Only a local header read was
  used for the local-model checks; no network.
- **download**: queue with a concurrency limit, HTTP Range resume, pause/resume/cancel, sha256
  verification before promotion, atomic rename, no silent overwrite, cleanup on mismatch and on
  registrar failure, restart recovery through sidecar records, and architecture check from the
  first bytes. Shard sets run as one job: all shards are verified, then promoted and registered
  together. If promotion or registration fails, promoted shards go back to the downloads
  directory so the set can resume. Known gap: a crash in the short window after shards are moved
  into the models folder and before the index is saved leaves unindexed files there. A later
  start reports them as existing destinations until they are overwritten or removed.
- **Disk space**: preflight checks the downloads volume and, when the destination is on another
  volume, that volume too (promotion copies whole files). Free space comes from `fs2::available_space`,
  which is used on all platforms (`statvfs` on Unix, `GetDiskFreeSpaceEx` on Windows). Only the
  Windows build was run here. Volume identity is the device id on Unix and the drive or UNC prefix
  on Windows. When identity is unknown, the answer is "different volume", which is the stricter check.
- **revisions**: every download is pinned to the commit `main` pointed to when the download was
  queued, fetched from that commit's URLs, and recorded with the entry. `check_updates` compares
  the stored commit with the current head with one call per repo and no download. `Changed` means
  the head moved, not that the model's bytes changed. Entries installed before this change report
  `NotRecorded`.
- **library**: JSON index (atomic writes; a corrupt index is an error, never reset). Revision and
  Hub tags are kept in a side table of the same file, so `ModelEntry` and `GgufMeta` keep their
  shape. Copy and in-place import of single files and complete shard sets, duplicate detection by
  content identity, name-collision refusal, incomplete-set and header-consistency refusal, scan of
  existing model folders (one item per split set), verify, delete (managed files only), storage
  usage. Partial files and sets with a missing shard are never listed as installed.
- **compare**: side-by-side rows (quant, bits per weight, estimated parameters, context, template,
  estimated bytes at a context from KV math, fit rating) and task ranking. Heuristic parts are
  marked in the code and in this README.
- **presets**: full CRUD (`get`, `save`, `create`, `update`, `delete`), per-model default with
  pointer maintenance, range validation, failed writes leave memory unchanged. The default is
  stored in the same JSON file (format version still 1; the field is optional on read).

Stubbed or not implemented:

- Linking (symlinks or hard links) on import: in-place reference is provided instead.
- Repo card (README) content: only tags, license and gating are parsed.
- Hub candidates are not profiled from their headers: `compare` and `rank_for` use names and tags
  for candidates, and full metadata only for installed models.
- Restart does not auto-resume: the app must call `start` again. `recover` lists the interrupted transfers.
- Downloads require a sha256 published by the Hub. Files without one are refused.

Open issues for the manager:

- `start_download` for a large repo still lists the full tree; a shard set is refused if any shard is missing.
- The library holds its index lock during an import copy, so `list()` waits for large copies.
