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

import { ContextMatrix, depthLabel, latencyText, memoryText } from "./ContextMatrix";

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
    expect(mocks.matrix).not.toHaveBeenCalled();
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

  it("can ask for the 128K range", async () => {
    mocks.matrix.mockResolvedValue(report([row(0)]));
    render(<ContextMatrix modelId="local/m" disabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: /Benchmark context lengths/ }));
    await userEvent.click(screen.getByRole("radio", { name: /Up to 128K/ }));
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(mocks.matrix).toHaveBeenCalledWith("local/m", 131_072));
  });

  it("is unavailable while a model is loaded", () => {
    render(<ContextMatrix modelId="local/m" disabled />);
    expect(screen.getByRole("button", { name: /Benchmark context lengths/ })).toBeDisabled();
  });
});
