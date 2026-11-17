// Typed client for the Raptor LLM Tauri shell. Types mirror the Rust serialization exactly
// (serde field names, snake_case variants). See docs/specs/ui-contract.md for the full list.
//
// Rust `Option<T>` is `T | null` on the wire. Command arguments use camelCase keys because
// Tauri maps them to the snake_case Rust parameter names.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

// ---------------------------------------------------------------------------
// Errors

/** Every command rejects with a serialized `RaptorError`. */
export type RaptorErrorKind =
  | "InvalidInput"
  | "NotFound"
  | "Conflict"
  | "Io"
  | "Network"
  | "Integrity"
  | "Runtime"
  | "Unsupported"
  | "Unauthorized"
  | "Internal";

export interface RaptorError {
  kind: RaptorErrorKind;
  message: string;
}

/** Thrown by every wrapper in `api`. `error.kind` tells the UI what to show. */
export class RaptorApiError extends Error {
  readonly kind: RaptorErrorKind;
  constructor(readonly error: RaptorError) {
    super(error.message);
    this.name = "RaptorApiError";
    this.kind = error.kind;
  }
}

function toApiError(e: unknown): RaptorApiError {
  if (e && typeof e === "object" && "kind" in e && "message" in e) {
    return new RaptorApiError(e as RaptorError);
  }
  return new RaptorApiError({ kind: "Internal", message: String(e) });
}

async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(cmd, args);
  } catch (e) {
    throw toApiError(e);
  }
}

// ---------------------------------------------------------------------------
// Shared vocabulary (raptor-core, raptor-runtime, raptor-models)

export type Role = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
  role: Role;
  content: string;
}

export type Os = "windows" | "linux" | "macos" | "other";
export type GpuVendor = "amd" | "nvidia" | "intel" | "other";
export type Backend = "hip" | "cuda" | "sycl" | "vulkan" | "cpu";
export type KvCacheType = "f16" | "q8_0" | "q4_0";

export interface CpuInfo {
  brand: string;
  cores: number;
  threads: number;
  features: string[];
}

export interface GpuInfo {
  vendor: GpuVendor;
  name: string;
  vram_total: number;
  /** `null` when the platform cannot report free VRAM. */
  vram_free: number | null;
  arch: string | null;
  driver: string | null;
}

export interface HardwareInfo {
  os: Os;
  cpu: CpuInfo;
  ram_total: number;
  ram_avail: number;
  gpus: GpuInfo[];
}

export type ServerState =
  | { state: "stopped" }
  | { state: "starting" }
  | { state: "ready" }
  | { state: "degraded"; reason: string }
  | { state: "crashed"; reason: string };

export interface ServerStatus {
  state: ServerState;
  endpoint: string | null;
  pid: number | null;
  restarts: number;
}

/**
 * GPU verdict from the llama-server log. Only `verified` means the GPU ran the model.
 * `cpu_only` and `unverified` must never be shown as GPU acceleration.
 */
export type GpuVerdict =
  | {
      verdict: "verified";
      evidence: string[];
      offloaded_layers?: number | null;
      total_layers?: number | null;
    }
  | { verdict: "cpu_only"; reason: string }
  | { verdict: "unverified"; reason: string };

export interface RuntimeSummary {
  id: string;
  version: string;
  backend: Backend;
  /** True only when a CPU build was chosen because no GPU build is compatible. */
  degraded: boolean;
  reason: string;
  binary: string;
}

export interface EngineStatus {
  hardware_detected: boolean;
  runtime: RuntimeSummary | null;
  server: ServerStatus;
  gpu_verdict: GpuVerdict | null;
  /** Latest device preflight; see `DevicePreflight` at the end of this file. Optional for older shells. */
  device_preflight?: DevicePreflight | null;
}

export interface LaunchConfig {
  model_path: string;
  backend: Backend;
  gpu_layers: number;
  ctx_size: number;
  batch_size: number;
  ubatch_size: number;
  threads: number;
  flash_attn: boolean;
  kv_cache: KvCacheType;
  host: string;
  port: number;
}

export interface MemoryEstimate {
  vram_bytes: number;
  ram_bytes: number;
  kv_bytes: number;
  vram_budget: number;
  ram_budget: number;
}

export interface Recommendation {
  config: LaunchConfig;
  rationale: string[];
  estimate: MemoryEstimate;
}

// ---------------------------------------------------------------------------
// Models (raptor-models)

export type SearchSort = "downloads" | "likes" | "last_modified";

export interface HfModelSummary {
  id: string;
  author: string | null;
  downloads: number;
  likes: number;
  tags: string[];
  last_modified: string | null;
  pipeline_tag: string | null;
}

export interface RepoFile {
  file_name: string;
  size: number;
  /** `null` means the file cannot be verified and will not be downloaded. */
  sha256: string | null;
  quant_label: string | null;
  /** Split shards are listed but not downloadable yet. */
  split_shard: boolean;
}

export interface HfRepoInfo {
  id: string;
  pipeline_tag: string | null;
  library_name: string | null;
  license: string | null;
  tags: string[];
  gated: boolean;
}

export type ModelSource =
  | { kind: "hugging_face"; repo: string; file_name: string }
  | { kind: "local_copy"; original_path: string }
  | { kind: "local_reference"; original_path: string };

export interface GgufMeta {
  version: number;
  kv_count: number;
  architecture: string | null;
  name: string | null;
  file_type: number | null;
  quant_label: string | null;
  context_length: number | null;
  block_count: number | null;
  embedding_length: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  key_length: number | null;
  tokenizer_model: string | null;
  has_chat_template: boolean;
}

export interface ModelEntry {
  /** `owner/repo/file.gguf` or `local/file.gguf`. Pass this id to load, delete and file lookups. */
  id: string;
  file_name: string;
  /** Managed: relative to the models directory. Referenced: absolute. */
  location: string;
  /** True when the file lives in the library and is deleted with the entry. */
  managed: boolean;
  size_bytes: number;
  sha256: string;
  added_at: number;
  source: ModelSource;
  gguf: GgufMeta;
}

export interface StorageUsage {
  models: number;
  managed_bytes: number;
  referenced_bytes: number;
}

/** One measured combination from tuning. */
export interface TuneMeasured {
  gpu_layers: number;
  ubatch: number;
  prompt_tps: number;
  gen_tps: number;
  /** Time of a 4096-token prompt plus a 256-token reply, seconds. */
  seconds: number;
}

export interface TuneReport {
  model_id: string;
  runtime_id?: string;
  runtime_version?: string;
  backend?: Backend;
  context_size?: number;
  gpu_name?: string | null;
  gpu_driver?: string | null;
  results: TuneMeasured[];
  best: TuneMeasured;
  baseline: TuneMeasured | null;
  baseline_live?: TuneMetrics | null;
  best_live?: TuneMetrics | null;
  /** How much less time the best setup takes than the untuned one; 0 when the default was best. */
  improvement_percent: number;
  errors: string[];
  /** Deep tune only: the same setups measured with `long_depth` tokens in the context. */
  long_results: TuneMeasured[];
  long_best: TuneMeasured | null;
  /** How much less time the long-prompt setting takes than the quick-tune one at that depth. */
  long_improvement_percent: number | null;
  /** Depth of `long_results`; 0 when no deep tune ran. */
  long_depth: number;
}

export interface TuneMetrics {
  gpu_layers: number;
  ubatch_size: number;
  parallel_slots?: number | null;
  kv_cache: KvCacheType;
  flash_attn: boolean;
  speculative: boolean;
  prompt_tokens: number;
  output_tokens: number;
  ttft_ms: number;
  prompt_tps: number;
  gen_tps: number;
  samples: number;
  prompt_tokens_exact: boolean;
}

/** One context length of the benchmark matrix. Numbers are null when it could not be measured. */
export interface MatrixRow {
  depth: number;
  kv_cache: KvCacheType;
  prompt_tps?: number | null;
  gen_tps?: number | null;
  /** Milliseconds until the first token of a 512-token turn at this depth. */
  turn_ms?: number | null;
  peak_dedicated_mib?: number | null;
  peak_shared_growth_mib?: number | null;
  spilled: boolean;
  note?: string | null;
}

export interface MatrixReport {
  model_id: string;
  measured_at: number;
  gpu_layers: number;
  ubatch: number;
  max_depth: number;
  rows: MatrixRow[];
  /** Seconds to read a whole prompt of each depth from empty, as `[depth, seconds]`. */
  prompt_seconds: [number, number][];
}

export interface DiscoveredFolder {
  path: string;
  source: string;
  items: ScanItem[];
}

export type ScanStatus =
  | { status: "new" }
  | { status: "already_in_library"; id: string }
  | { status: "invalid_gguf"; reason: string };

export interface ScanItem {
  path: string;
  size_bytes: number;
  status: ScanStatus;
}

export type JobState =
  | "queued"
  | "running"
  | "paused"
  | "verifying"
  | "completed"
  | "failed"
  | "cancelled";

export interface DownloadProgress {
  key: string;
  state: JobState;
  downloaded: number;
  total: number;
  attempts: number;
}

export interface PartialDownload {
  key: string;
  repo: string;
  file_name: string;
  expected_size: number;
  downloaded_bytes: number;
}

export interface RecoveryReport {
  partials: PartialDownload[];
  problems: string[];
}

/** Emitted on `download://finished`, and kept in each queue entry. */
export interface DownloadFinished {
  key: string;
  status: "completed" | "cancelled" | "failed";
  path: string | null;
  sha256: string | null;
  error: string | null;
}

export interface QueueEntry {
  key: string;
  repo: string;
  file_name: string;
  progress: DownloadProgress;
  finished: DownloadFinished | null;
}

export type ImportMode = "copy" | "reference";

export interface LoadModelResult {
  status: EngineStatus;
  rationale: string[];
  estimate: MemoryEstimate;
  model_id: string;
  /** Which configuration was used. `recommended` when nothing is known yet. */
  config_source: ConfigSource;
  /** The benchmark run after the model became Ready. Always present. */
  benchmark: BenchmarkOutcome;
  /** Whether the loaded model has a thinking-effort control. Absent in older shells. */
  thinking?: ThinkingView;
}

/** How hard a thinking model thinks. Medium is the default. */
export type Effort = "low" | "medium" | "high" | "max";

/** One app level and the loaded model's own level it maps to (Qwen3.8: high and max are both "xhigh"). */
export interface EffortChoice {
  effort: Effort;
  model_level: string;
}

/** What the loaded model offers for thinking effort. `supported: false` means show no control. */
export interface ThinkingView {
  supported: boolean;
  /** The model template's own level names. */
  model_levels: string[];
  /** The four app levels and what each means for this model; empty when unsupported. */
  choices: EffortChoice[];
  /** The level used when a conversation has not chosen (the setting). */
  default_effort: Effort;
}

// ---------------------------------------------------------------------------
// Runtime and gateway (shell DTOs)

export interface RuntimeEntryView {
  id: string;
  version: string;
  backend: Backend;
  installed: boolean;
  current: boolean;
  dir: string | null;
}

export interface RuntimeListView {
  active: RuntimeSummary | null;
  runtimes: RuntimeEntryView[];
}

export interface RuntimeSelectionView {
  id: string;
  version: string;
  backend: Backend;
  degraded: boolean;
  reason: string;
  gpu_arch: string | null;
  rejected: string[];
}

export interface RuntimeRollbackView {
  id: string;
  version: string;
  backend: Backend;
  dir: string;
  binary: string;
}

export interface MetricsView {
  queued: number;
  running: number;
  completed: number;
  rejected: number;
}

export interface GatewayStatus {
  running: boolean;
  /** Bound address, e.g. `127.0.0.1:8790`. */
  addr: string | null;
  metrics: MetricsView | null;
  last_error: string | null;
  /** True when the listener binds all interfaces (LAN mode). */
  lan: boolean;
  /** True when the listener serves TLS. LAN mode always does. */
  tls: boolean;
  /** Lowercase hex SHA-256 of the serving certificate. Clients pin this value. */
  tls_fingerprint_sha256: string | null;
  /** Always false in the shell. */
  insecure: boolean;
}

// ---------------------------------------------------------------------------
// Settings (shell DTOs). The API key secret is never part of these types.

export interface ApiKeyInfo {
  id: string;
  created_at: number;
}

/** Returned once by `api_key_rotate`. Show it to the user and do not store it in UI state. */
export interface ApiKeyReveal {
  id: string;
  secret: string;
}

export interface WorkspaceInfo {
  id: string;
  path: string;
}

export interface SettingsView {
  storage_dir: string | null;
  default_storage_dir: string;
  active_storage_dir: string;
  /** True when the configured storage directory differs from the one in use. */
  restart_required: boolean;
  gateway_enabled: boolean;
  gateway_port: number;
  lan_enabled: boolean;
  api_key: ApiKeyInfo | null;
  context_target: number;
  workspaces: WorkspaceInfo[];
  min_context_target: number;
  max_context_target: number;
  /** Startup problems, such as a recovered settings file. Show them to the user. */
  notes: string[];
}

// ---------------------------------------------------------------------------
// Conversations (shell DTOs)

/** Speed of one reply, as the gateway measured it. Any field may be missing. */
export interface ReplyStats {
  tokens_per_second?: number | null;
  prompt_tokens_per_second?: number | null;
  time_to_first_token_ms?: number | null;
  generated_tokens?: number | null;
}

export interface StoredMessage {
  id: string;
  role: Role;
  content: string;
  created_at: number;
  /** Model reasoning for an assistant reply. Never part of `content`. Absent in older files. */
  reasoning?: string | null;
  /** Speed of an assistant reply. Absent when the server reported no timings and in older files. */
  stats?: ReplyStats | null;
}

export interface Conversation {
  version: number;
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: StoredMessage[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface ConversationList {
  conversations: ConversationSummary[];
  /** Files that could not be read. They are reported, never hidden. */
  problems: string[];
}

export interface NewMessage {
  role: Role;
  content: string;
  /** Optional reasoning to keep beside the answer. Empty values are not stored. */
  reasoning?: string | null;
  stats?: ReplyStats | null;
}

// ---------------------------------------------------------------------------
// Chat (shell DTOs)

export interface ChatOptions {
  temperature?: number | null;
  max_tokens?: number | null;
}

/** Payload on `chat://<requestId>`. */
export type ChatEvent =
  | { type: "chunk"; text: string }
  /** Reasoning from a thinking model. It is not the answer. */
  | { type: "reasoning"; text: string }
  | { type: "done"; cancelled: boolean; stats?: ReplyStats | null }
  | { type: "error"; error: RaptorError };

// ---------------------------------------------------------------------------
// Context Intelligence (raptor-context)

export type TaskKind =
  | "debug"
  | "architecture"
  | "refactor"
  | "feature"
  | "ui"
  | "security"
  | "performance";

export type Category =
  | "instructions"
  | "task"
  | "active_files"
  | "diffs"
  | "errors_tests"
  | "memory"
  | "retrieved_code"
  | "graph_neighbours"
  | "history";

export type SymbolKind =
  | "function"
  | "method"
  | "struct"
  | "class"
  | "enum"
  | "trait"
  | "interface"
  | "impl"
  | "module"
  | "type"
  | "const"
  | "test";

export interface DiffInput {
  path: string;
  patch: string;
}

/** Input to `context_build`. Every field is optional on the wire (serde defaults). */
export interface ContextRequest {
  task?: string;
  task_kind?: TaskKind;
  instructions?: string | null;
  messages?: ChatMessage[];
  active_files?: string[];
  diffs?: DiffInput[];
  errors?: string[];
  pinned_files?: string[];
  session_id?: string | null;
  /** Model context window in tokens. */
  model_ctx?: number;
  reserved_output?: number;
  max_retrieved?: number;
}

export interface LineRange {
  start: number;
  end: number;
}

export interface ManifestItem {
  id: string;
  source: string;
  category: Category;
  hash: string | null;
  range: LineRange | null;
  tokens: number;
  score: number;
  why: string;
  truncated: boolean;
}

export interface ExcludedItem {
  id: string;
  source: string;
  category: Category;
  tokens: number;
  why: string;
  recover_handle: string | null;
}

export interface CategoryUsage {
  used: number;
  budget: number | null;
}

export interface ContextManifest {
  budget_tokens: number;
  used_tokens: number;
  /** Protected tokens beyond the budget. Non-zero means over budget; nothing protected was dropped. */
  over_budget_by: number;
  /** False when the counts are estimates (no model tokenizer). */
  tokens_exact: boolean;
  included: ManifestItem[];
  excluded: ExcludedItem[];
  utilization_by_category: Partial<Record<Category, CategoryUsage>>;
  warnings: string[];
}

export interface ContextResponse {
  messages: ChatMessage[];
  manifest: ContextManifest;
}

/** Input to `context_search`. */
export interface SearchRequest {
  query: string;
  task_kind?: TaskKind;
  limit?: number;
  active_files?: string[];
  pinned_files?: string[];
}

export interface Hit {
  path: string;
  start_line: number;
  end_line: number;
  symbol: string | null;
  hash: string;
  score: number;
  why: string[];
}

export interface SymbolQuery {
  name_contains?: string;
  kind?: SymbolKind | null;
  limit?: number;
}

export interface SymbolHit {
  path: string;
  name: string;
  kind: SymbolKind;
  start_line: number;
  end_line: number;
  parent: string | null;
}

export interface ImpactReport {
  file: string;
  imports: string[];
  dependents: string[];
  /** File path to hop distance (up to three hops). */
  transitive_dependents: Record<string, number>;
  related_tests: string[];
}

export interface IndexStatus {
  root: string;
  index_version: number;
  files: number;
  chunks: number;
  symbols: number;
  imports: number;
  excluded: number;
  graph_files: number;
  graph_edges: number;
  git_repo: boolean;
  changed_files: number;
  git_warning: string | null;
}

export interface UpdateReport {
  scanned: number;
  reparsed: string[];
  removed: string[];
  unchanged: number;
  excluded: number;
  graph_recomputed: number;
}

/** Input to `context_recover`: a `chunk:` handle, or a query (optionally with `extra_budget`). */
export interface RecoverRequest {
  handle?: string | null;
  query?: string | null;
  task_kind?: TaskKind;
  extra_budget?: number;
}

export interface RecoveredItem {
  handle: string;
  path: string;
  start_line: number;
  end_line: number;
  text: string;
  tokens: number;
}

export interface RecoverResponse {
  items: RecoveredItem[];
  tokens_used: number;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Events

export const EVENTS = {
  serverState: "server://state",
  downloadProgress: "download://progress",
  downloadFinished: "download://finished",
  gatewayState: "gateway://state",
  chat: (requestId: string) => `chat://${requestId}`,
} as const;

function onEvent<T>(name: string, handler: (payload: T) => void): Promise<UnlistenFn> {
  return listen<T>(name, (event) => handler(event.payload));
}

/** Each subscriber returns an unlisten function. Call it on component unmount. */
export const subscribe = {
  /** Full engine status whenever the server state or GPU verdict changes. */
  serverState: (handler: (status: EngineStatus) => void) => onEvent(EVENTS.serverState, handler),
  downloadProgress: (handler: (p: DownloadProgress) => void) => onEvent(EVENTS.downloadProgress, handler),
  downloadFinished: (handler: (f: DownloadFinished) => void) => onEvent(EVENTS.downloadFinished, handler),
  gatewayState: (handler: (g: GatewayStatus) => void) => onEvent(EVENTS.gatewayState, handler),
  /** Stream for one chat request. Subscribe before calling `chat.start`. */
  chat: (requestId: string, handler: (e: ChatEvent) => void) => onEvent(EVENTS.chat(requestId), handler),
};

// ---------------------------------------------------------------------------
// Commands

/** One installed runtime and its newer build, from `dependencies_check`. */
export interface RuntimeUpdate {
  id: string;
  current_version: string;
  latest_version: string;
  available: boolean;
  note: string;
  url: string | null;
  sha256: string | null;
  size_bytes: number | null;
}

export interface HipSdkStatus {
  versions: string[];
  ok: boolean;
  required: string;
  guidance: string;
  download_page: string;
}

/** The Vulkan runtime: installed or not, and the verified download that adds or updates it. */
export interface VulkanOffer {
  installed_version: string | null;
  latest_version: string;
  available: boolean;
  note: string;
  url: string | null;
  sha256: string | null;
  size_bytes: number | null;
}

export interface DependencyReport {
  latest_tag: string | null;
  check_error: string | null;
  updates: RuntimeUpdate[];
  hip_sdk: HipSdkStatus | null;
  /** Null when the release could not be read. */
  vulkan: VulkanOffer | null;
}

export type BackendPref = "auto" | "rocm" | "cuda" | "sycl" | "vulkan";

/** The accelerator choice of one model. `effective` is what a load would use now. */
export interface BackendView {
  model_id: string;
  preference: BackendPref;
  effective: "rocm" | "cuda" | "sycl" | "vulkan" | "default" | "unavailable";
  rocm_installed: boolean;
  cuda_installed: boolean;
  sycl_installed: boolean;
  vulkan_installed: boolean;
  /** A sparse Mixture-of-experts model larger than VRAM: the kind Vulkan generated faster on. */
  spills: boolean;
  note: string | null;
  /** The measured comparison, quoted next to the selector. */
  measured: string;
}

export const api = {
  dependencies: {
    /** Reads release information only. Nothing is downloaded. */
    check: () => call<DependencyReport>("dependencies_check"),
    /** Installs the verified newer build of runtime `id`. Refused while a model is loaded. */
    installUpdate: (id: string) => call<RuntimeSelectionView>("dependencies_install_update", { id }),
    /** Installs the verified Vulkan runtime next to ROCm; the runtime in use does not change. Refused while a model is loaded. */
    installVulkan: () => call<string>("dependencies_install_vulkan"),
  },
  hardware: {
    info: () => call<HardwareInfo>("hardware_info"),
    refresh: () => call<HardwareInfo>("hardware_refresh"),
  },

  runtime: {
    status: () => call<EngineStatus>("runtime_status"),
    ensure: () => call<RuntimeSelectionView>("runtime_ensure"),
    /** Bytes of the runtime download written so far; 0 before it starts. */
    installProgress: () => call<number>("runtime_install_progress"),
    rollback: (id: string) => call<RuntimeRollbackView>("runtime_rollback", { id }),
    unpin: () => call<void>("runtime_unpin"),
    list: () => call<RuntimeListView>("runtime_list"),
    recommend: (modelId: string, context?: number) =>
      call<Recommendation>("runtime_recommend", { modelId, context }),
  },

  models: {
    search: (query: string, sort?: SearchSort, limit?: number) =>
      call<HfModelSummary[]>("models_search", { query, sort, limit }),
    repoFiles: (repo: string) => call<RepoFile[]>("models_repo_files", { repo }),
    repoInfo: (repo: string) => call<HfRepoInfo>("models_repo_info", { repo }),
    list: () => call<ModelEntry[]>("models_list"),
    current: () => call<ModelEntry | null>("models_current"),
    loadReport: () => call<StoredLoadReport | null>("models_load_report"),
    /** Stage of the load running now, read from the server log. `active` is false when idle. */
    loadProgress: () =>
      call<{ active: boolean; step: number; stage_count: number; label: string; elapsed_ms: number }>(
        "models_load_progress",
      ),
    storage: () => call<StorageUsage>("models_storage"),
    delete: (id: string) => call<void>("models_delete", { id }),
    import: (path: string, mode: ImportMode) => call<ModelEntry>("models_import", { path, mode }),
    /** Bytes hashed so far by the import running now, and the bytes it has to hash. */
    importProgress: () => call<{ done: number; total: number }>("models_import_progress"),
    scanFolder: (dir: string) => call<ScanItem[]>("models_scan_folder", { dir }),
    /** Models in other apps' folders (LM Studio, Jan, ...) and folders scanned before, not yet in the library. */
    discover: () => call<DiscoveredFolder[]>("models_discover"),
    /** Benchmarks a model on the GPU and saves the fastest settings. Takes minutes; the model must not be loaded. */
    tune: (modelId: string, deep = false) => call<TuneReport>("models_tune", { modelId, deep }),
    /** Speed and memory at growing context lengths (up to `maxDepth`, default 32,768). Takes minutes; the model must not be loaded. */
    contextMatrix: (modelId: string, maxDepth: number | null = null) =>
      call<MatrixReport>("models_context_matrix", { modelId, maxDepth }),
    /** The saved matrix for this model on this machine and runtime build, or null. */
    contextMatrixLast: (modelId: string) => call<MatrixReport | null>("models_context_matrix_last", { modelId }),
    /** The context windows the GPU can hold (default, large, maximum), from llama.cpp's own projection. Takes seconds. */
    contextOptions: (modelId: string) => call<ContextOption[]>("models_context_options", { modelId }),
    /** Which backend (ROCm or Vulkan) this model uses, and what a load would use now. */
    backend: (modelId: string) => call<BackendView>("models_backend", { modelId }),
    /** Whether the loaded model has a thinking-effort control, and what each level maps to. */
    thinking: () => call<ThinkingView>("models_thinking"),
    /** Chooses the backend for a model. Takes effect at the next load. */
    setBackend: (modelId: string, backend: BackendPref) =>
      call<BackendView>("models_set_backend", { modelId, backend }),
    /** Steps done and total of the tuning run in progress. */
    tuneProgress: () => call<{ done: number; total: number }>("models_tune_progress"),
    forgetTuning: (modelId: string) => call<boolean>("models_forget_tuning", { modelId }),
    download: {
      /** Returns the job key used by pause, resume, cancel and discard. */
      start: (repo: string, fileName: string, overwrite = false) =>
        call<string>("models_download_start", { repo, fileName, overwrite }),
      pause: (key: string) => call<void>("models_download_pause", { key }),
      resume: (key: string) => call<void>("models_download_resume", { key }),
      cancel: (key: string) => call<void>("models_download_cancel", { key }),
      queue: () => call<QueueEntry[]>("models_download_queue"),
      recover: () => call<RecoveryReport>("models_recover_downloads"),
      discardPartial: (key: string) => call<void>("models_discard_partial", { key }),
    },
    load: (modelId: string, context?: number) =>
      call<LoadModelResult>("models_load", { modelId, context }),
    unload: () => call<EngineStatus>("models_unload"),
  },

  gateway: {
    status: () => call<GatewayStatus>("gateway_status"),
  },

  settings: {
    get: () => call<SettingsView>("settings_get"),
    /** Takes effect on the next start. `null` restores the default location. */
    setStorageDir: (path: string | null) => call<SettingsView>("settings_set_storage_dir", { path }),
    setGatewayPort: (port: number) => call<SettingsView>("settings_set_gateway_port", { port }),
    setGatewayEnabled: (enabled: boolean) => call<GatewayStatus>("settings_set_gateway_enabled", { enabled }),
    /** Refused with `kind: "Unauthorized"` unless an API key exists. */
    setLan: (enabled: boolean) => call<SettingsView>("settings_set_lan", { enabled }),
    setContextTarget: (tokens: number) => call<SettingsView>("settings_set_context_target", { tokens }),
    rotateApiKey: () => call<ApiKeyReveal>("api_key_rotate"),
    revokeApiKey: () => call<SettingsView>("api_key_revoke"),
    registerWorkspace: (path: string) => call<WorkspaceInfo>("workspace_register", { path }),
    removeWorkspace: (id: string) => call<SettingsView>("workspace_remove", { id }),
  },

  chat: {
    /** Starts a stream. Text arrives on `subscribe.chat(requestId, ...)`. */
    start: (requestId: string, messages: ChatMessage[], options?: ChatOptions) =>
      call<void>("chat_start", { requestId, messages, options }),
    cancel: (requestId: string) => call<boolean>("chat_cancel", { requestId }),
  },

  conversations: {
    list: () => call<ConversationList>("conversations_list"),
    create: (title: string) => call<Conversation>("conversations_create", { title }),
    get: (id: string) => call<Conversation>("conversations_get", { id }),
    rename: (id: string, title: string) => call<Conversation>("conversations_rename", { id, title }),
    delete: (id: string) => call<void>("conversations_delete", { id }),
    append: (id: string, messages: NewMessage[]) =>
      call<Conversation>("conversations_append", { id, messages }),
    truncate: (id: string, count: number) =>
      call<Conversation>("conversations_truncate", { id, count }),
    replaceMessage: (id: string, messageId: string, content: string) =>
      call<Conversation>("conversations_replace_message", { id, messageId, content }),
    export: (id: string) => call<string>("conversations_export", { id }),
    import: (json: string) => call<Conversation>("conversations_import", { json }),
  },

  dialog: {
    pickFile: (filterExt?: string[]) => call<string | null>("dialog_pick_file", { filterExt }),
    pickFolder: () => call<string | null>("dialog_pick_folder"),
  },

  context: {
    index: (path: string) => call<UpdateReport>("context_index", { path }),
    indexStatus: (path: string) => call<IndexStatus>("context_index_status", { path }),
    search: (path: string, request: SearchRequest) => call<Hit[]>("context_search", { path, request }),
    build: (path: string, request: ContextRequest) => call<ContextResponse>("context_build", { path, request }),
    recover: (path: string, request: RecoverRequest) =>
      call<RecoverResponse>("context_recover", { path, request }),
    symbols: (path: string, query: SymbolQuery) => call<SymbolHit[]>("context_symbols", { path, query }),
    impact: (path: string, file: string) => call<ImpactReport>("context_impact", { path, file }),
  },

  diagnostics: {
    /** Last lines of the llama-server log (JSON lines). 1-2000, default 200. */
    logTail: (lines?: number) => call<string[]>("diagnostics_log_tail", { lines }),
    /** The log folder and its files. Files older than `keep_days` days are deleted automatically. */
    logFiles: () =>
      call<{
        dir: string;
        keep_days: number;
        files: { name: string; size_bytes: number; modified_unix: number }[];
      }>("diagnostics_log_files"),
  },
} as const;

// Added by the shell hardening pass: removes the runtime pin so runtime_ensure follows the
// newest compatible version again.
export const runtimeUnpin = (): Promise<void> => call<void>("runtime_unpin");

// ---------------------------------------------------------------------------
// Setup, performance memory and gateway TLS (added by the roadmap pass; additive only)

export type ConfigSource = "known_good" | "recommended";

export interface SetupCheck {
  id: "hardware" | "runtime" | "gpu" | "model" | "storage" | "performance_memory";
  label: string;
  ok: boolean;
  detail: string;
}

export interface AvailableRuntime {
  id: string;
  version: string;
  backend: Backend;
  /** True when only a CPU build is compatible. */
  degraded: boolean;
  reason: string;
}

/**
 * First-run checklist. Reads state only: `runtime_needs_confirmation` means a compatible
 * runtime exists but is not installed, and the user must confirm before `runtime_ensure`.
 */
export interface SetupStatus {
  checks: SetupCheck[];
  hardware: HardwareInfo | null;
  selected_runtime: RuntimeSummary | null;
  /** Installed on disk, as `id version`. */
  installed_runtimes: string[];
  available_runtime: AvailableRuntime | null;
  runtime_needs_confirmation: boolean;
  degraded: boolean;
  models_installed: number;
  storage_free_bytes: number | null;
  notes: string[];
  /** Device check of the selected runtime; a `failure` means the GPU backend cannot run. */
  device_preflight?: DevicePreflight | null;
}

export interface ConfigSnapshot {
  gpu_layers: number;
  ctx_size: number;
  batch_size: number;
  ubatch_size: number;
  threads: number;
  flash_attn: boolean;
  kv_cache: KvCacheType;
  speculative?: "ngram" | "mtp" | "dflash" | null;
  uses_draft_model?: boolean;
}

export interface BenchRecord {
  machine: string;
  model_hash: string;
  runtime_id: string;
  runtime_version: string;
  gpu_name?: string | null;
  gpu_driver?: string | null;
  quant_label?: string | null;
  config: ConfigSnapshot;
  /** Time to first token. */
  ttft_ms: number;
  prompt_tps: number;
  gen_tps: number;
  vram_bytes: number;
  ram_bytes: number;
  /** False for runs that crashed, timed out, lacked verified GPU offload or failed. */
  stable: boolean;
  samples: number;
  /** Unix seconds of the most recent run. */
  measured_at: number;
}

export interface PerfEntry {
  /** Stable id for this record. Pass it to `perfCompare`. */
  id: string;
  record: BenchRecord;
}

export interface PerfDeltas {
  gen_tps_delta_pct: number;
  prompt_tps_delta_pct: number;
  /** Negative means the candidate answers faster. */
  ttft_delta_pct: number;
  vram_delta_bytes: number;
}

export interface PerfComparison {
  base: PerfEntry;
  candidate: PerfEntry;
  comparison: PerfDeltas;
}

export interface BenchmarkOutcome {
  stable: boolean;
  /** What the store did with this run. */
  recorded: "inserted" | "merged" | "kept_stable" | "not_saved";
  /** This run, or null when the hardware is unknown. */
  entry: PerfEntry | null;
  /** Why the run is unstable, or the error that stopped it. Empty when stable. */
  reasons: string[];
  gpu_verdict: GpuVerdict | null;
  notes: string[];
}

export const setupStatus = (): Promise<SetupStatus> => call<SetupStatus>("setup_status");

/** Stored records, newest first. Pass a library model id to limit them to that model. */
export const perfHistory = (model?: string): Promise<PerfEntry[]> =>
  call<PerfEntry[]>("perf_history", { model });

export const perfCompare = (a: string, b: string): Promise<PerfComparison> =>
  call<PerfComparison>("perf_compare", { a, b });

export const perfRunBenchmark = (): Promise<BenchmarkOutcome> =>
  call<BenchmarkOutcome>("perf_run_benchmark");

export const perfBestKnown = (model: string): Promise<PerfEntry | null> =>
  call<PerfEntry | null>("perf_best_known", { model });

/** Replaces the LAN certificate. Returns the new status with the new fingerprint. */
export const gatewayRotateTls = (): Promise<GatewayStatus> =>
  call<GatewayStatus>("gateway_rotate_tls");

/** One accelerator llama-server lists with `--list-devices`, e.g. `ROCm0`. */
export interface RuntimeDevice {
  id: string;
  name: string;
  total_mib: number | null;
  free_mib: number | null;
  /** Backend implied by the id prefix; null for an unknown prefix. */
  backend: Backend | null;
}

/** Why a GPU backend has no usable device. */
export type FailureCause =
  /** The runtime needs `hipblas.dll`; the HIP SDK ships `libhipblas.dll` instead. */
  | "hipblas_name_mismatch"
  /** The runtime needs `hipblas.dll` and no folder provides it. */
  | "hipblas_missing"
  /** No HIP SDK bin folder was found. */
  | "hip_sdk_not_found"
  /** The DLLs resolve, but llama-server initializes no device of this backend. */
  | "no_device_visible";

/** Typed diagnosis shown instead of a silent CPU run. */
export interface BackendLoadFailure {
  backend: Backend;
  cause: FailureCause;
  /** One or two plain sentences naming the cause. */
  message: string;
  /** Concrete next steps, most direct first. */
  suggestions: string[];
}

/** DLL facts for a HIP runtime folder, read from disk without loading anything. */
export interface HipDependencies {
  runtime_dir: string;
  /** Whether ggml-hip.dll names hipblas.dll; null when ggml-hip.dll is absent. */
  backend_imports_hipblas: boolean | null;
  hipblas_in_runtime_dir: boolean;
  hipblas_in_sdk: string | null;
  libhipblas_in_sdk: string | null;
}

/** Result of `llama-server --list-devices` for the selected runtime. */
export interface DevicePreflight {
  backend: Backend;
  binary: string;
  devices: RuntimeDevice[];
  /** HIP SDK bin folders put first on the child's PATH. */
  sdk_bin_dirs: string[];
  hip_dependencies: HipDependencies | null;
  /** Set when the requested GPU backend has no device. Null means a usable device exists. */
  failure: BackendLoadFailure | null;
}

// ---------------------------------------------------------------------------
// Wave 6 additions. Everything below is additive: the interfaces and wrappers above keep their
// shapes. Each type mirrors the serde output of the shell command named beside it.
// See docs/specs/ui-contract.md for the commands.

/** `models_load_report`: the serving load's result plus the launch settings it used. */
export interface StoredLoadReport {
  result: LoadModelOutcome;
  ctx_size: number;
  gpu_layers: number;
}

/** Fields added to `LoadModelResult` by `models_load`. */
/** What a load did to GPU memory (Windows). `spilled` without `expected` is a speed problem. */
export interface GpuMemoryReport {
  dedicated_mib: number;
  shared_growth_mib: number;
  spilled: boolean;
  /** True when the spill is planned: a sparse Mixture-of-experts model larger than VRAM. */
  expected: boolean;
}

export interface LoadModelOutcome extends LoadModelResult {
  /** Absent when GPU memory could not be read. */
  gpu_memory?: GpuMemoryReport | null;
  /** Preset the load applied (its context length), or null. */
  preset: string | null;
  /**
   * True only when a GPU backend had no device and `allow_cpu_fallback` let the load run on
   * the CPU. Show it loudly: the run is not GPU-accelerated.
   */
  cpu_fallback: boolean;
}

/**
 * A rejected `models_load`. `kind` and `message` are as before. `backend_failure` is the typed
 * GPU diagnosis (with suggested actions) when the GPU backend had no device, otherwise null.
 */
export type LoadFailure = RaptorError & {
  backend_failure: BackendLoadFailure | null;
};

/** The GPU diagnosis carried by a rejected load, or null for any other failure. */
export function loadFailureOf(err: unknown): BackendLoadFailure | null {
  if (err instanceof RaptorApiError) {
    return (err.error as Partial<LoadFailure>).backend_failure ?? null;
  }
  return null;
}

/** `models_load` with a preset and an optional context. Null context uses the preset's, then the settings'. */
export const modelsLoadWithPreset = (
  modelId: string,
  context: number | null,
  preset: string | null,
  cacheType: KvCacheType | null = null,
  slots: number | null = null,
  speculation: boolean | null = null,
): Promise<LoadModelOutcome> =>
  call<LoadModelOutcome>("models_load", { modelId, context, preset, cacheType, slots, speculation });

/** A context window the GPU can hold for a model. Pass `ctx`, `kv_cache` and `slots` to a load to use it. */
export interface ContextOption {
  id: "default" | "large" | "maximum";
  label: string;
  ctx: number;
  kv_cache: KvCacheType;
  slots: number;
  /** False when speculative decoding is turned off for this window (the draft context needs memory). */
  speculation: boolean;
  /** Projected device memory at `ctx` including measured overhead, MiB. Null for the default option. */
  projected_mib: number | null;
  budget_mib: number;
  /** True when `ctx` is the model's trained maximum. */
  capped_by_model: boolean;
  note: string;
}

/** Extra chat options. `preset` names a preset of the loaded model; explicit values win over it. */
export interface ChatStartOptions extends ChatOptions {
  /** Thinking effort for this reply. Null uses the setting. Ignored by a model with no effort control. */
  thinking?: Effort | null;
  /**
   * Write the answer as parallel sections: a short plan, then every section at once. Faster for long
   * answers when the server has several request slots; thinking is off for these requests.
   */
  long_answer?: boolean;
  preset?: string | null;
  top_p?: number | null;
  top_k?: number | null;
  repeat_penalty?: number | null;
}

/** `chat_start` with the extended options. Subscribe to `chat://<requestId>` first. */
export const chatStartWithOptions = (
  requestId: string,
  messages: ChatMessage[],
  options: ChatStartOptions,
): Promise<void> => call<void>("chat_start", { requestId, messages, options });

// Presets ---------------------------------------------------------------------

export interface Preset {
  model_id: string;
  name: string;
  system_prompt: string | null;
  n_ctx: number;
  temperature: number;
  top_p: number;
  top_k: number;
  repeat_penalty: number;
  max_tokens: number | null;
}

export interface PresetList {
  model_id: string;
  presets: Preset[];
  /** Name of the default preset, or null. */
  default: string | null;
}

export const presetsList = (modelId: string): Promise<PresetList> =>
  call<PresetList>("presets_list", { modelId });

export const presetsGet = (modelId: string, name: string): Promise<Preset> =>
  call<Preset>("presets_get", { modelId, name });

/** Creates or replaces a preset. Returns the stored preset. */
export const presetsSave = (preset: Preset): Promise<Preset> =>
  call<Preset>("presets_save", { preset });

export const presetsDelete = (modelId: string, name: string): Promise<PresetList> =>
  call<PresetList>("presets_delete", { modelId, name });

/** `name` null clears the default. */
export const presetsSetDefault = (modelId: string, name: string | null): Promise<PresetList> =>
  call<PresetList>("presets_set_default", { modelId, name });

// Conversations: search and export ---------------------------------------------

export interface SearchHit {
  conversation_id: string;
  title: string;
  message_id: string;
  role: Role;
  /** Text around the match, whitespace collapsed; a leading or trailing ellipsis marks a cut. */
  snippet: string;
  updated_at: number;
}

export interface ConversationSearch {
  hits: SearchHit[];
  conversations_scanned: number;
  /** True when the limit or the read budget stopped the search early. */
  truncated: boolean;
  /** Conversation files that could not be read. */
  problems: string[];
}

export interface ExportedFile {
  path: string;
  bytes: number;
}

/** Case-insensitive search across all conversations. `limit` 1-200, default 50. */
export const conversationsSearch = (
  query: string,
  limit?: number | null,
): Promise<ConversationSearch> =>
  call<ConversationSearch>("conversations_search", { query, limit: limit ?? null });

/** Writes the conversation to `path` (absolute, `.json`), normally the path from `dialogSaveFile`. */
export const conversationsExportTo = (id: string, path: string): Promise<ExportedFile> =>
  call<ExportedFile>("conversations_export_to", { id, path });

/** Native save dialog. Returns the chosen path, or null when cancelled. Nothing is written. */
export const dialogSaveFile = (
  defaultName: string,
  filterExt?: string | null,
): Promise<string | null> =>
  call<string | null>("dialog_save_file", { defaultName, filterExt: filterExt ?? null });

// Runtime install preview -----------------------------------------------------

export interface RuntimeInstallPreview {
  id: string;
  version: string;
  backend: Backend;
  degraded: boolean;
  reason: string;
  gpu_arch: string | null;
  /** Source of the archive, from the runtime index (https). */
  url: string;
  /** Expected SHA-256 of the archive, from the runtime index. */
  sha256: string;
  /** Download size from the server's Content-Length. Null when unknown. */
  download_bytes: number | null;
  size_source: "content_length" | "unavailable";
  /** True when this version is already installed: nothing would be downloaded. */
  already_installed: boolean;
  rejected: string[];
  notes: string[];
}

/** What `runtime_ensure` would install. Confirm with the user before calling it. */
export const runtimeInstallPreview = (): Promise<RuntimeInstallPreview> =>
  call<RuntimeInstallPreview>("runtime_install_preview");

// Models: details, compare, rank, updates --------------------------------------

export interface ModelFile {
  file_name: string;
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface ModelDetails {
  entry: ModelEntry;
  /** Hub commit the model was downloaded from. Null for models installed before revisions were recorded. */
  revision: string | null;
  hub_tags: string[];
  /** Every file of the model; a split model lists all shards. */
  files: ModelFile[];
}

export const modelsDetails = (id: string): Promise<ModelDetails> =>
  call<ModelDetails>("models_details", { id });

/** Fit rating. Serialized as the variant name. */
export type CompatRating = "Great" | "Ok" | "Tight" | "WontFit";

/** What a model is good for. Heuristic: see the models README. */
export type ModelTask = "coding" | "reasoning" | "general" | "agentic";

export interface TaskTag {
  task: ModelTask;
  score: number;
  evidence: string[];
}

export interface HardwareSummary {
  ram_bytes: number;
  vram_bytes: number;
}

export interface ComparisonRow {
  id: string;
  name: string;
  architecture: string | null;
  quant_label: string | null;
  bits_per_weight: number | null;
  params_estimate: number | null;
  params_from_name_billions: number | null;
  file_bytes: number;
  context_train: number | null;
  chat_template: boolean;
  estimated_bytes_at_ctx: number | null;
  ctx_exceeds_train: boolean;
  compat: CompatRating | null;
  tags: TaskTag[];
}

export interface Comparison {
  ctx: number;
  hardware: HardwareSummary | null;
  rows: ComparisonRow[];
}

export interface RankedCandidate {
  id: string;
  score: number;
  task_score: number;
  rating: CompatRating | null;
  bits_per_weight: number | null;
  reasons: string[];
}

export type UpdateStatus = "up_to_date" | "changed" | "not_recorded";

export interface UpdateCheck {
  id: string;
  repo: string;
  pinned_revision: string | null;
  head_revision: string;
  status: UpdateStatus;
}

/** 1-8 installed model ids. `context` null uses the settings context target. */
export const modelsCompare = (ids: string[], context: number | null): Promise<Comparison> =>
  call<Comparison>("models_compare", { ids, context });

export const modelsRankFor = (
  task: ModelTask,
  context: number | null,
): Promise<RankedCandidate[]> =>
  call<RankedCandidate[]>("models_rank_for", { task, context });

/** Read-only check against Hugging Face. Fails with `Network` when offline. */
export const modelsCheckUpdates = (): Promise<UpdateCheck[]> =>
  call<UpdateCheck[]>("models_check_updates");

// Context: embeddings and project memory ---------------------------------------

export interface SemanticStatus {
  enabled: boolean;
  model_id: string | null;
  dimension: number;
  vectors: number;
  capacity: number;
  chunks_without_vector: number;
  last_error: string | null;
  store_notice: string | null;
}

export interface SemanticView {
  /** Embedding model in settings, or null. */
  configured_model: string | null;
  /** Why semantic retrieval is off for this workspace; null when active or unconfigured. */
  reason: string | null;
  status: SemanticStatus;
}

export type MemoryKind = "decision" | "convention" | "summary" | "completed_work" | "failed_attempt";

export interface ProjectMemoryAdded {
  id: string;
  evicted: string[];
}

/** Sets the embedding model (must be installed). Returns its id. */
export const contextSetEmbeddingModel = (modelId: string): Promise<string> =>
  call<string>("context_set_embedding_model", { modelId });

export const contextClearEmbeddingModel = (): Promise<void> =>
  call<void>("context_clear_embedding_model");

export const contextSemanticStatus = (path: string): Promise<SemanticView> =>
  call<SemanticView>("context_semantic_status", { path });

export const contextRememberProject = (
  path: string,
  kind: MemoryKind,
  text: string,
  paths: string[],
): Promise<ProjectMemoryAdded> =>
  call<ProjectMemoryAdded>("context_remember_project", { path, kind, text, paths });

/** One long-term memory entry of a workspace. `stale` means a file it cites has changed since. */
export interface ProjectMemoryEntry {
  id: string;
  kind: MemoryKind;
  text: string;
  sources: { path: string; hash: string }[];
  stale: boolean;
  stale_reason: string | null;
  seq: number;
}

export const contextProjectMemory = (path: string): Promise<ProjectMemoryEntry[]> =>
  call<ProjectMemoryEntry[]>("context_project_memory", { path });

export const contextForgetProjectMemory = (path: string, id: string): Promise<void> =>
  call<void>("context_forget_project_memory", { path, id });

// Settings: embedding model and CPU fallback ------------------------------------

/** Fields added to `SettingsView`. */
export interface SettingsViewV2 extends SettingsView {
  embedding_model: string | null;
  /** Off by default. */
  allow_cpu_fallback: boolean;
  /** Requests llama-server works on at once (1-8). Absent in older shells. */
  parallel_slots?: number;
  /** Default thinking effort for thinking models. Absent in older shells (Medium). */
  thinking_effort?: Effort;
}

export const settingsSetThinkingEffort = (effort: Effort): Promise<SettingsViewV2> =>
  call<SettingsViewV2>("settings_set_thinking_effort", { effort });

export const settingsSetParallelSlots = (slots: number): Promise<SettingsViewV2> =>
  call<SettingsViewV2>("settings_set_parallel_slots", { slots });

export const settingsSetAllowCpuFallback = (enabled: boolean): Promise<SettingsViewV2> =>
  call<SettingsViewV2>("settings_set_allow_cpu_fallback", { enabled });

/** `settings_get` as the shell returns it: includes `embedding_model` and `allow_cpu_fallback`. */
export const settingsGet = (): Promise<SettingsViewV2> => call<SettingsViewV2>("settings_get");

// Architecture support -------------------------------------------------------------

/**
 * `known`: in Raptor's tested list. `known_unsupported`: llama.cpp refuses it as a text model;
 * this blocks a download or an import. `unknown`: in neither list; never blocks a file, and
 * llama-server decides when the model loads.
 */
export type ArchSupport = "known" | "known_unsupported" | "unknown";

/** Support facts for a model's architecture. `note` is null for `known`. */
export interface ArchInfo {
  name: string | null;
  support: ArchSupport;
  note: string | null;
}

/**
 * Adds the architecture support facts returned by `models_details` (merged with `ModelDetails` above).
 * Optional in the type so existing fixtures keep compiling; the shell always sends it.
 */
export interface ModelDetails {
  architecture?: ArchInfo;
}
