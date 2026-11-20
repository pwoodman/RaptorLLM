import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { BenchRecord, ConfigSnapshot, LoadModelOutcome } from "../../api";
import { engineStatus, loadOutcome, modelEntry } from "../../test/fixtures";
import { engineStore } from "../../state/engine";
import { uiStore } from "../../state/ui";
import { LoadBenchmarkBar } from "./LoadBenchmarkBar";

const config: ConfigSnapshot = {
  gpu_layers: 41,
  ctx_size: 65_536,
  batch_size: 512,
  ubatch_size: 256,
  threads: 16,
  flash_attn: true,
  kv_cache: "q8_0",
};

const record: BenchRecord = {
  machine: "test",
  model_hash: "abc123",
  runtime_id: "hip",
  runtime_version: "b123",
  config,
  ttft_ms: 830,
  prompt_tps: 512.3,
  gen_tps: 33.36,
  vram_bytes: 12 * 1024 * 1024 * 1024,
  ram_bytes: 2 * 1024 * 1024 * 1024,
  prompt_tokens: 61_440,
  gen_tokens: 64,
  stable: true,
  samples: 3,
  measured_at: 1_700_000_000,
};

function stableReport(overrides: Partial<LoadModelOutcome> = {}): LoadModelOutcome {
  return loadOutcome({
    status: {
      ...engineStatus(),
      gpu_verdict: { verdict: "verified", evidence: ["loaded layers 0-40"], offloaded_layers: 41, total_layers: 41 },
    },
    config_source: "known_good",
    benchmark: {
      stable: true,
      recorded: "merged",
      reasons: [],
      gpu_verdict: null,
      notes: [],
      entry: { id: "entry-1", record },
    },
    ...overrides,
  });
}

function loadIntoChat(report: LoadModelOutcome | null) {
  engineStore.set({
    loaded: {
      modelId: "org/repo/model.gguf",
      ctxSize: 65_536,
      plannedGpuLayers: 41,
      rationale: [],
      estimate: null,
    },
    loadReport: report,
  });
}

describe("LoadBenchmarkBar", () => {
  beforeEach(() => {
    engineStore.set({ loaded: null, loadReport: null, status: engineStatus() });
    uiStore.set({ page: "chats", modelView: "library", focusModelImport: false });
  });

  it("renders nothing without a load report", () => {
    const { container } = render(<LoadBenchmarkBar />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the headline numbers collapsed and expands the full measurement on demand", async () => {
    loadIntoChat(stableReport());
    const user = userEvent.setup();
    render(<LoadBenchmarkBar />);

    // Collapsed: generation speed and first-token latency, nothing else yet.
    expect(screen.getByText("33.4 tok/s")).toBeInTheDocument();
    expect(screen.getByText("830 ms")).toBeInTheDocument();
    expect(screen.queryByText("Prompt speed")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Show load details/ }));
    expect(screen.getByText("Prompt speed")).toBeInTheDocument();
    expect(screen.getByText("512.3 tok/s")).toBeInTheDocument();
    expect(screen.getByText("61,440 in · 64 out")).toBeInTheDocument();
    expect(screen.getByText("41 / 41 layers on GPU (verified)")).toBeInTheDocument();
    expect(screen.getByText("65,536 tokens")).toBeInTheDocument();
    expect(screen.getByText("Known good")).toBeInTheDocument();
    expect(screen.getByText(/Stable · 3 samples/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Hide load details/ }));
    expect(screen.queryByText("Prompt speed")).not.toBeInTheDocument();
  });

  it("marks optimized loads and links to the performance history", async () => {
    loadIntoChat(stableReport({ tuning_applied: true }));
    const user = userEvent.setup();
    render(<LoadBenchmarkBar />);

    expect(screen.getByText("Optimized")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Show load details/ }));
    await user.click(screen.getByRole("button", { name: "Open performance history" }));
    expect(uiStore.get().page).toBe("models");
    expect(uiStore.get().modelView).toBe("performance");
  });

  it("surfaces unstable runs as a warning with the reason", async () => {
    loadIntoChat(
      loadOutcome({
        benchmark: {
          stable: false,
          recorded: "not_saved",
          reasons: ["generation speed varied by more than 15%"],
          gpu_verdict: null,
          notes: [],
          entry: null,
        },
      }),
    );
    const user = userEvent.setup();
    render(<LoadBenchmarkBar />);

    expect(screen.getByText("Unstable run")).toBeInTheDocument();
    expect(screen.getByText(/did not complete for this load/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Show load details/ }));
    expect(screen.getByText(/generation speed varied by more than 15%/)).toBeInTheDocument();
  });
});
