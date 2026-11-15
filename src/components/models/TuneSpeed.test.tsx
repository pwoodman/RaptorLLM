import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ModelEntry, TuneReport } from "../../api";

const mocks = vi.hoisted(() => ({ tune: vi.fn(), progress: vi.fn(), forget: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      models: { ...actual.api.models, tune: mocks.tune, tuneProgress: mocks.progress, forgetTuning: mocks.forget },
    },
  };
});

import { longTuneSummary, tps, TuneSpeed, tuneSummary } from "./TuneSpeed";

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
    mocks.tune.mockReset();
    mocks.progress.mockResolvedValue({ done: 0, total: 0 });
  });

  it("asks first, then tunes and shows the measured table", async () => {
    mocks.tune.mockResolvedValue(report());
    render(<TuneSpeed entry={entry} disabled={false} />);

    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    expect(mocks.tune).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));

    await waitFor(() => expect(mocks.tune).toHaveBeenCalledWith("local/o.gguf", false));
    expect(await screen.findByRole("table", { name: "Measured settings" })).toBeInTheDocument();
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

  it("is unavailable while a model is loaded", () => {
    render(<TuneSpeed entry={entry} disabled />);
    expect(screen.getByRole("button", { name: /Optimize speed/ })).toBeDisabled();
  });

  it("shows why tuning failed", async () => {
    mocks.tune.mockRejectedValue(new Error("nothing could be measured: out of memory"));
    render(<TuneSpeed entry={entry} disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Optimize speed/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start tuning" }));
    expect(await screen.findByText(/out of memory/)).toBeInTheDocument();
  });
});
