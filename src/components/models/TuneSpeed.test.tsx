import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
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

import { TuneSpeed, tuneSummary } from "./TuneSpeed";
import { tuningStore } from "../../state/tuning";
import { engineStore } from "../../state/engine";
import { chatStore } from "../../state/chat";

const entry = { id: "local/o.gguf", file_name: "o.gguf" } as ModelEntry;

function report(over: Partial<TuneReport> = {}): TuneReport {
  return {
    context_size: 65536, output_tokens: 64, finalist: 1, saved: true, retained_profile: false,
    results: [
      { gpu_layers: 41, ubatch: 256, kv_cache: "f16", seconds: 15, estimated_ttft_ms: 10_250, prompt_tps: 6400, gen_tps: 16 },
      { gpu_layers: 45, ubatch: 256, kv_cache: "q4_0", seconds: 10 },
    ],
    confirmation: [[15,10],[16,10],[15,10]], errors: [], ...over,
  };
}

describe("tuneSummary", () => {
  it("distinguishes a qualified winner from the baseline and a near miss", () => {
    expect(tuneSummary(report())).toContain("Qualified settings saved");
    expect(tuneSummary(report({saved:false,finalist:null,confirmation:[]}))).toContain("Default retained");
    expect(tuneSummary(report({saved:false,confirmation:[[15,10],[15,10],[14.99,10]]}))).toContain("49.9%");
    expect(tuneSummary(report({saved:false,retained_profile:true,finalist:null,confirmation:[]}))).toContain("Previous qualified settings retained");
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

    await waitFor(() => expect(mocks.tune).toHaveBeenCalledWith("local/o.gguf", false, null, null));
    await userEvent.click(await screen.findByText("Benchmark details and measured settings"));
    const table = within(screen.getByRole("table", { name: "Measured settings" }));
    expect(table.getByRole("columnheader", { name: "Time to first token (est.)" })).toBeInTheDocument();
    const baseline = within(table.getByRole("row", { name: /^Default/ }));
    expect(baseline.getByText("10.25 s")).toBeInTheDocument();
    expect(baseline.getByText(`${(65_472).toLocaleString()} tokens`, { exact: false })).toHaveTextContent("6400.0 tok/s");
    expect(baseline.getByText("64 tokens", { exact: false })).toHaveTextContent("16.0 tok/s");
    expect(baseline.getByText("4369.1 tok/s")).toBeInTheDocument();
    const candidate = within(table.getByRole("row", { name: /^Candidate 1/ }));
    expect(candidate.getAllByText("Unknown", { exact: false })).toHaveLength(3);
    expect(candidate.getByText("6553.6 tok/s")).toBeInTheDocument();
    expect(screen.getByRole("table", {name: "Independent confirmation"})).toBeInTheDocument();
  });

  it("tunes for long prompts only when asked, and shows that result", async () => {
    mocks.tune.mockResolvedValue(report());
    render(<TuneSpeed entry={entry} disabled={false} />);

    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("checkbox", { name: /Extended search/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));

    await waitFor(() => expect(mocks.tune).toHaveBeenCalledWith("local/o.gguf", true, null, null));
    expect(await screen.findByText(/Qualified settings saved/)).toBeInTheDocument();
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
    expect(mocks.tune).toHaveBeenCalledWith(entry.id, false, window.ctx, "coding");
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
    expect(await screen.findByText(/Qualified settings saved/)).toBeInTheDocument();
  });

  it("shows why tuning failed", async () => {
    mocks.tune.mockRejectedValue(new Error("nothing could be measured: out of memory"));
    render(<TuneSpeed entry={entry} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));
    expect(await screen.findByText(/out of memory/)).toBeInTheDocument();
  });
});
