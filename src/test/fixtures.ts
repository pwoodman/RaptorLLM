// Shared test fixtures that mirror the shapes in src/api.ts. Only the fields a test needs are
// overridden; the rest are realistic defaults.
import type {
  EngineStatus,
  LoadModelOutcome,
  ModelEntry,
  Preset,
  SettingsViewV2,
  ThinkingView,
} from "../api";

export function engineStatus(state: EngineStatus["server"]["state"] = { state: "ready" }): EngineStatus {
  return {
    hardware_detected: true,
    runtime: null,
    server: { state, endpoint: "http://127.0.0.1:8080", pid: 1, restarts: 0 },
    gpu_verdict: null,
  };
}

export function modelEntry(overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    id: "org/repo/model.gguf",
    file_name: "model.gguf",
    location: "models/org/repo/model.gguf",
    managed: true,
    size_bytes: 4 * 1024 * 1024 * 1024,
    sha256: "a".repeat(64),
    added_at: 0,
    source: { kind: "hugging_face", repo: "org/repo", file_name: "model.gguf" },
    gguf: {
      version: 3,
      kv_count: 1,
      architecture: "llama",
      name: "model",
      file_type: null,
      quant_label: "Q4_K_M",
      context_length: 8192,
      block_count: 32,
      embedding_length: 4096,
      head_count: 32,
      head_count_kv: 8,
      key_length: 128,
      tokenizer_model: null,
      has_chat_template: true,
    },
    ...overrides,
  };
}

export function loadOutcome(overrides: Partial<LoadModelOutcome> = {}): LoadModelOutcome {
  return {
    status: engineStatus(),
    rationale: [],
    estimate: { vram_bytes: 0, ram_bytes: 0, kv_bytes: 0, vram_budget: 0, ram_budget: 0 },
    model_id: "org/repo/model.gguf",
    config_source: "recommended",
    benchmark: {
      stable: false,
      recorded: "not_saved",
      entry: null,
      reasons: [],
      gpu_verdict: null,
      notes: [],
    },
    preset: null,
    cpu_fallback: false,
    ...overrides,
  };
}

export function preset(overrides: Partial<Preset> = {}): Preset {
  return {
    model_id: "org/repo/model.gguf",
    name: "Review",
    system_prompt: "You review diffs.",
    n_ctx: 4096,
    temperature: 0.2,
    top_p: 0.9,
    top_k: 20,
    repeat_penalty: 1.1,
    max_tokens: 256,
    ...overrides,
  };
}

export function settingsView(overrides: Partial<SettingsViewV2> = {}): SettingsViewV2 {
  return {
    storage_dir: null,
    default_storage_dir: "C:/data",
    active_storage_dir: "C:/data",
    restart_required: false,
    gateway_enabled: false,
    gateway_port: 8790,
    lan_enabled: false,
    api_key: null,
    context_target: 65536,
    workspaces: [],
    min_context_target: 65536,
    max_context_target: 200000,
    notes: [],
    embedding_model: null,
    allow_cpu_fallback: false,
    allow_ram_kv: false,
    ...overrides,
  };
}

/** Qwen3.8's effort control as llama-server reports it: low, medium, xhigh. High and Max share "xhigh". */
export function thinkingView(overrides: Partial<ThinkingView> = {}): ThinkingView {
  return {
    supported: true,
    model_levels: ["low", "medium", "xhigh"],
    choices: [
      { effort: "low", model_level: "low" },
      { effort: "medium", model_level: "medium" },
      { effort: "high", model_level: "xhigh" },
      { effort: "max", model_level: "xhigh" },
    ],
    default_effort: "medium",
    ...overrides,
  };
}

/** A model with no effort control. */
export function noThinking(): ThinkingView {
  return { supported: false, model_levels: [], choices: [], default_effort: "medium" };
}
