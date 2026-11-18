import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ModelEntry, TuneReport } from "../../api";

const mocks = vi.hoisted(() => ({ tune: vi.fn(), progress: vi.fn(), forget: vi.fn(), saved: vi.fn(), unload: vi.fn(), load: vi.fn() }));

vi.mock("../../state/engine", async (original) => ({ ...await original<typeof import("../../state/engine")>(), loadModel: mocks.load }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      models: { ...actual.api.models, tune: mocks.tune, tuneProgress: mocks.progress, forgetTuning: mocks.forget, savedTuning: mocks.saved, unload: mocks.unload },
    },
  };
});

import { longTuneSummary, tps, TuneSpeed, tuneSummary } from "./TuneSpeed";
import { tuningStore } from "../../state/tuning";
import { engineStore } from "../../state/engine";
import { chatStore } from "../../state/chat";

const entry = { id: "local/o.gguf", file_name: "o.gguf" } as ModelEntry;

function report(over: Partial<TuneReport> = {}): TuneReport {
  const best = { gpu_layers: 41, ubatch: 256, prompt_tps: 2234.7, gen_tps: 68.4, seconds: 5.6 };
  return {
    model_id: entry.id,
    results: [best, { gpu_layers: 39, ubatch: 512, prompt_tps: 1707, gen_tps: 47.8, seconds: 7.8 }],
    best,
    baseline: null,
    improvement_percent: 39,
    errors: [],
    long_results: [],
    long_best: null,
    long_improvement_percent: null,
    long_depth: 0,
    ...over,
  };
}

describe("tuneSummary", () => {
  it("says what was chosen and how much faster it is", () => {
    const text = tuneSummary(report());
    expect(text).toContain("41 layers on the GPU, micro-batch 256");
    expect(text).toContain("39%");
    expect(text).toContain("Saved");
  });

  it("says plainly when the default was already best", () => {
    expect(tuneSummary(report({ improvement_percent: 0 }))).toContain("already the fastest");
  });

  it("formats speeds", () => {
    expect(tps(68.4)).toBe("68.4 tok/s");
    expect(tps(2234.7)).toBe("2,235 tok/s");
  });
});

describe("longTuneSummary", () => {
  const long = { gpu_layers: 39, ubatch: 512, prompt_tps: 600, gen_tps: 30, seconds: 15.4 };

  it("is null when no deep tune ran or nothing worked", () => {
    expect(longTuneSummary(report())).toBeNull();
    expect(longTuneSummary(report({ long_depth: 16384, long_best: null }))).toBeNull();
  });

  it("names the long-prompt setting and its gain at that depth", () => {
    const text = longTuneSummary(report({ long_depth: 16384, long_best: long, long_improvement_percent: 17 }));
    expect(text).toContain("16,384 tokens");
    expect(text).toContain("39 layers on the GPU, micro-batch 512");
    expect(text).toContain("17%");
  });

  it("says when the quick-tune setup is also best at that depth", () => {
    const same = { gpu_layers: 41, ubatch: 256, prompt_tps: 900, gen_tps: 50, seconds: 9 };
    const text = longTuneSummary(report({ long_depth: 16384, long_best: same, long_improvement_percent: 0 }));
    expect(text).toContain("same setup is fastest");
  });
});

describe("TuneSpeed", () => {
  beforeEach(() => {
    engineStore.set({ busy: null, loaded: null, status: null });
    chatStore.set({ pending: null });
    tuningStore.set({ activeId: null, reports: {} });
    mocks.tune.mockReset();
    mocks.saved.mockResolvedValue([]);
    mocks.unload.mockResolvedValue({ server: { state: { state: "stopped" } } });
    mocks.load.mockReset();
    mocks.load.mockResolvedValue(undefined);
    mocks.progress.mockResolvedValue({ done: 0, total: 0 });
  });

  it("asks first, then tunes and shows the measured table", async () => {
    mocks.tune.mockResolvedValue(report());
    render(<TuneSpeed entry={entry} disabled={false} />);

    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    expect(mocks.tune).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));

    await waitFor(() => expect(mocks.tune).toHaveBeenCalledWith("local/o.gguf", false));
    await userEvent.click(await screen.findByText("Benchmark details and measured settings"));
    expect(screen.getByRole("table", { name: "Measured settings" })).toBeInTheDocument();
    expect(screen.getByText(/41 layers on the GPU/)).toBeInTheDocument();
  });

  it("tunes for long prompts only when asked, and shows that result", async () => {
    const long = { gpu_layers: 39, ubatch: 512, prompt_tps: 600, gen_tps: 30, seconds: 15.4 };
    mocks.tune.mockResolvedValue(report({ long_depth: 16384, long_best: long, long_improvement_percent: 17 }));
    render(<TuneSpeed entry={entry} disabled={false} />);

    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("checkbox", { name: /Also tune for long prompts/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));

    await waitFor(() => expect(mocks.tune).toHaveBeenCalledWith("local/o.gguf", true));
    expect(await screen.findByText(/39 layers on the GPU, micro-batch 512 is faster/)).toBeInTheDocument();
  });

  it("restores saved profiles after reopening and resets them", async () => {
    mocks.saved.mockResolvedValue([{ runtime_id: "hip", runtime_version: "b123", tuning: {
      gpu_layers: 30, ubatch_size: 512, measured_at: 1, prompt_tps: 500, gen_tps: 30, long: null,
    } }]);
    mocks.forget.mockResolvedValue(true);
    const view = render(<TuneSpeed entry={entry} disabled={false} />);
    expect(await screen.findByText("Speed settings saved")).toBeInTheDocument();
    view.unmount();
    render(<TuneSpeed entry={entry} disabled={false} />);
    expect(await screen.findByText(/30 GPU layers/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reset speed settings" }));
    await waitFor(() => expect(screen.queryByText("Speed settings saved")).not.toBeInTheDocument());
    expect(mocks.forget).toHaveBeenCalledWith(entry.id);
  });

  it("respects a disabled state while another model task is running", () => {
    render(<TuneSpeed entry={entry} disabled />);
    expect(screen.getByRole("button", { name: /Optimize speed/ })).toBeDisabled();
  });

  it("unloads a running model and resumes it with the same chat settings after tuning", async () => {
    const window = { ctx: 32768, cacheType: "q8_0" as const, slots: 1, speculation: false };
    engineStore.set({ loaded: { modelId: entry.id, ctxSize: window.ctx, plannedGpuLayers: 30, rationale: [], estimate: null, preset: "coding", window } });
    mocks.tune.mockResolvedValue(report());
    render(<TuneSpeed entry={entry} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));
    await waitFor(() => expect(mocks.load).toHaveBeenCalledWith(entry.id, "coding", window));
    const unloadCalls = mocks.unload.mock.invocationCallOrder;
    const tuneCalls = mocks.tune.mock.invocationCallOrder;
    expect(unloadCalls[unloadCalls.length - 1]).toBeLessThan(tuneCalls[tuneCalls.length - 1]);
  });

  it("keeps an optimization visible when leaving and reopening the page", async () => {
    let finish!: (value: TuneReport) => void;
    mocks.tune.mockReturnValue(new Promise<TuneReport>((resolve) => { finish = resolve; }));
    const view = render(<TuneSpeed entry={entry} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));
    view.unmount();
    render(<TuneSpeed entry={entry} disabled />);
    expect(screen.getByRole("button", { name: /Tuning/ })).toBeDisabled();
    finish(report());
    expect(await screen.findByText(/Saved: future loads/)).toBeInTheDocument();
  });

  it("shows why tuning failed", async () => {
    mocks.tune.mockRejectedValue(new Error("nothing could be measured: out of memory"));
    render(<TuneSpeed entry={entry} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));
    expect(await screen.findByText(/out of memory/)).toBeInTheDocument();
  });
});
