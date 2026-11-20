import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { MatrixReport, MatrixRow } from "../../api";

const mocks = vi.hoisted(() => ({ matrix: vi.fn(), last: vi.fn(), progress: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      models: { ...actual.api.models, contextMatrix: mocks.matrix, contextMatrixLast: mocks.last, tuneProgress: mocks.progress },
    },
  };
});

import { ContextMatrix, depthLabel, latencyText, memoryText, summaryText, reportToMarkdown } from "./ContextMatrix";

function row(depth: number, over: Partial<MatrixRow> = {}): MatrixRow {
  return {
    depth,
    kv_cache: "f16",
    prompt_tps: 500,
    gen_tps: 20.5,
    turn_ms: 1024,
    peak_dedicated_mib: 13_312,
    peak_shared_growth_mib: 20,
    spilled: false,
    note: null,
    ...over,
  };
}

function report(rows: MatrixRow[]): MatrixReport {
  return { model_id: "local/m", measured_at: 1_794_000_000, gpu_layers: 66, ubatch: 256, max_depth: 32768, rows, prompt_seconds: [[0, 0], [16384, 32.8]] };
}

describe("formatting", () => {
  it("labels depths and latencies", () => {
    expect(depthLabel(0)).toBe("Empty");
    expect(depthLabel(32_768)).toBe("32K");
    expect(latencyText(800)).toBe("0.8 s");
    expect(latencyText(12_400)).toBe("12 s");
    expect(latencyText(80_000)).toBe("1 min 20 s");
    expect(latencyText(null)).toBe("-");
  });

  it("shows peak memory and says when part of it was shared", () => {
    expect(memoryText(row(0))).toBe("13.0 GB");
    expect(memoryText(row(0, { spilled: true, peak_shared_growth_mib: 2048 }))).toBe("13.0 GB + 2 GB shared");
    expect(memoryText(row(0, { peak_dedicated_mib: null }))).toBe("-");
  });

  it("summarises where generation stays at least half of the empty-context speed", () => {
    expect(summaryText([row(0), row(4096, { gen_tps: 12.0 }), row(8192, { gen_tps: 8.0 })])).toMatch(/Comfortable up to 4K/);
    expect(summaryText([row(0), row(4096, { gen_tps: 19.0 })])).toMatch(/every measured length/);
    expect(summaryText([row(0), row(4096, { gen_tps: 5.0 })])).toMatch(/below half/);
    expect(summaryText([row(0), row(4096, { spilled: true, gen_tps: 19.0 })])).toMatch(/spills into shared RAM/);
    expect(summaryText([row(0)])).toBeNull();
  });

  it("exports the measured report as a Markdown table", () => {
    const markdown = reportToMarkdown(report([row(0), row(65_536, { kv_cache: "q8_0", prompt_tps: null, note: "does not fit" })]));
    expect(markdown).toContain("| Context | Cache |");
    expect(markdown).toContain("| Empty | 16-bit |");
    expect(markdown).toContain("not measured: does not fit");
    expect(markdown).toContain("micro-batch 256");
  });
});

describe("ContextMatrix", () => {
  beforeEach(() => {
    mocks.matrix.mockReset();
    mocks.last.mockReset();
    mocks.last.mockResolvedValue(null);
    mocks.progress.mockResolvedValue({ done: 0, total: 0 });
  });

  it("shows the saved result without running anything", async () => {
    mocks.last.mockResolvedValue(report([row(0), row(4096, { gen_tps: 19.9 })]));
    render(<ContextMatrix modelId="local/m" disabled={false} />);
    expect(await screen.findByRole("table", { name: "Speed and memory by context length" })).toBeInTheDocument();
    expect(screen.getByText("19.9 tok/s")).toBeInTheDocument();
    expect(screen.getByText(/every measured length/)).toBeInTheDocument();
    expect(mocks.matrix).not.toHaveBeenCalled();
  });

  it("copies the measured report to the clipboard as Markdown", async () => {
    mocks.last.mockResolvedValue(report([row(0)]));
    const user = userEvent.setup();
    render(<ContextMatrix modelId="local/m" disabled={false} />);
    const writeText = vi.spyOn(navigator.clipboard, "writeText");
    await user.click(await screen.findByRole("button", { name: /Copy table/ }));
    expect(writeText).toHaveBeenCalledTimes(1);
    const markdown = writeText.mock.calls[0]?.[0] as unknown as string;
    expect(markdown).toContain("| Context | Cache |");
    expect(markdown).toContain("local/m");
  });

  it("asks first, runs the quick range by default and shows a depth that was not measured", async () => {
    mocks.matrix.mockResolvedValue(report([row(0), row(65_536, { prompt_tps: null, gen_tps: null, turn_ms: null, note: "does not fit in VRAM with any cache type" })]));
    render(<ContextMatrix modelId="local/m" disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Benchmark context lengths/ }));
    expect(mocks.matrix).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(mocks.matrix).toHaveBeenCalledWith("local/m", 32_768));
    expect(await screen.findByText(/Not measured: does not fit in VRAM/)).toBeInTheDocument();
  });

  it("can ask for the 200K range", async () => {
    mocks.matrix.mockResolvedValue(report([row(0)]));
    render(<ContextMatrix modelId="local/m" disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Benchmark context lengths/ }));
    await userEvent.click(screen.getByRole("radio", { name: /Up to 200K/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(mocks.matrix).toHaveBeenCalledWith("local/m", 200_000));
  });

  it("is unavailable while a model is loaded", () => {
    render(<ContextMatrix modelId="local/m" disabled />);
    expect(screen.getByRole("button", { name: /Benchmark context lengths/ })).toBeDisabled();
  });
});
