import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  perfBestKnown,
  perfCompare,
  perfHistory,
  perfRunBenchmark,
  RaptorApiError,
  type BenchRecord,
  type LoadModelOutcome,
  type ModelEntry,
  type PerfEntry,
} from "../../api";
import { engineStore } from "../../state/engine";
import { LoadReport } from "./LoadReport";
import { PerformancePanel } from "./PerformancePanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    perfHistory: vi.fn(),
    perfBestKnown: vi.fn(),
    perfCompare: vi.fn(),
    perfRunBenchmark: vi.fn(),
  };
});

function record(overrides: Partial<BenchRecord> = {}): BenchRecord {
  return {
    machine: "rx9060xt",
    model_hash: "abc",
    runtime_id: "llama-hip-win",
    runtime_version: "b12340",
    config: { gpu_layers: 99, ctx_size: 8192, batch_size: 512, ubatch_size: 256, threads: 8, flash_attn: true, kv_cache: "f16" },
    ttft_ms: 180,
    prompt_tps: 950.5,
    gen_tps: 61.25,
    vram_bytes: 9 * 1024 * 1024 * 1024,
    ram_bytes: 0,
    stable: true,
    samples: 3,
    measured_at: 1_700_000_000,
    ...overrides,
  };
}

const stable: PerfEntry = { id: "p-stable", record: record() };
const unstable: PerfEntry = { id: "p-bad", record: record({ stable: false, gen_tps: 4.0, ttft_ms: 900, samples: 1 }) };

const model: ModelEntry = {
  id: "owner/repo/model.gguf",
  file_name: "model.gguf",
  location: "models/model.gguf",
  managed: true,
  size_bytes: 1024,
  sha256: "abc",
  added_at: 0,
  source: { kind: "local_copy", original_path: "C:/m.gguf" },
  gguf: {
    version: 3,
    kv_count: 0,
    architecture: "llama",
    name: null,
    file_type: null,
    quant_label: "Q4_K_M",
    context_length: null,
    block_count: null,
    embedding_length: null,
    head_count: null,
    head_count_kv: null,
    key_length: null,
    tokenizer_model: null,
    has_chat_template: false,
  },
};

const loadResult: LoadModelOutcome = {
  status: {
    hardware_detected: true,
    runtime: null,
    server: { state: { state: "ready" }, endpoint: null, pid: null, restarts: 0 },
    gpu_verdict: { verdict: "verified", evidence: ["offloaded 33/33 layers"], offloaded_layers: 33, total_layers: 33 },
  },
  rationale: ["Fits in 16 GB VRAM with 33 layers offloaded."],
  estimate: { vram_bytes: 1, ram_bytes: 0, kv_bytes: 0, vram_budget: 2, ram_budget: 3 },
  model_id: model.id,
  config_source: "known_good",
  benchmark: {
    stable: false,
    recorded: "inserted",
    entry: unstable,
    reasons: ["The run timed out after 120 s."],
    gpu_verdict: null,
    notes: [],
  },
  preset: null,
  cpu_fallback: false,
};

describe("PerformancePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    engineStore.set({ loaded: null, busy: null, loadReport: null });
    vi.mocked(perfHistory).mockResolvedValue([stable, unstable]);
    vi.mocked(perfBestKnown).mockResolvedValue(stable);
  });

  it("shows the history, marks unstable runs, and names the best known run", async () => {
    render(<PerformancePanel models={[model]} />);

    expect(await screen.findByText("Best known on this machine")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: /Benchmark runs for model\.gguf/ });
    expect(within(table).getByText("Unstable, not a baseline")).toBeInTheDocument();
    expect(within(table).getAllByText("Stable")).toHaveLength(1);
    expect(perfHistory).toHaveBeenCalledWith(model.id);
  });

  it("says plainly when no stable run exists yet", async () => {
    vi.mocked(perfBestKnown).mockResolvedValue(null);
    vi.mocked(perfHistory).mockResolvedValue([]);
    render(<PerformancePanel models={[model]} />);
    expect(await screen.findByText(/No stable benchmark yet/)).toBeInTheDocument();
    expect(screen.getByText("No benchmark records for model.gguf yet.")).toBeInTheDocument();
  });

  it("compares two runs and warns when one of them is unstable", async () => {
    vi.mocked(perfCompare).mockResolvedValue({
      base: stable,
      candidate: unstable,
      comparison: { gen_tps_delta_pct: -93.5, prompt_tps_delta_pct: -40, ttft_delta_pct: 400, vram_delta_bytes: 0 },
    });
    const user = userEvent.setup();
    render(<PerformancePanel models={[model]} />);

    await user.click(await screen.findByRole("button", { name: "Compare" }));

    await waitFor(() => expect(perfCompare).toHaveBeenCalledWith("p-bad", "p-stable"));
    expect(await screen.findByText("One of these runs is unstable")).toBeInTheDocument();
    expect(screen.getByText("-93.5% (higher is better)")).toBeInTheDocument();
  });

  it("disables the benchmark button unless the selected model is the loaded one", async () => {
    render(<PerformancePanel models={[model]} />);
    const button = await screen.findByRole("button", { name: "Run benchmark" });
    expect(button).toBeDisabled();
    expect(screen.getByText(/Load model\.gguf first/)).toBeInTheDocument();
  });

  it("runs the benchmark on the loaded model and shows the refusal when none is ready", async () => {
    engineStore.set({ loaded: { modelId: model.id, ctxSize: 8192, plannedGpuLayers: 33, rationale: [], estimate: null } });
    vi.mocked(perfRunBenchmark).mockRejectedValue(new RaptorApiError({ kind: "Conflict", message: "No model is ready" }));
    const user = userEvent.setup();
    render(<PerformancePanel models={[model]} />);

    await user.click(await screen.findByRole("button", { name: "Run benchmark" }));

    expect(await screen.findByText("No model is ready")).toBeInTheDocument();
    expect(perfRunBenchmark).toHaveBeenCalledTimes(1);
  });
});

describe("LoadReport", () => {
  it("renders nothing until a load happens in this session", () => {
    const { container } = render(<LoadReport />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the config source, the rationale, the verified offload and the unstable benchmark", () => {
    engineStore.set({ loadReport: loadResult });
    render(<LoadReport />);

    expect(screen.getByText(/Known good: the fastest stable settings/)).toBeInTheDocument();
    expect(screen.getByText("33 / 33 layers on GPU (verified)")).toBeInTheDocument();
    expect(screen.getByText("Fits in 16 GB VRAM with 33 layers offloaded.")).toBeInTheDocument();
    expect(screen.getByText("Unstable")).toBeInTheDocument();
    expect(screen.getByText("The run timed out after 120 s.")).toBeInTheDocument();
  });
});
