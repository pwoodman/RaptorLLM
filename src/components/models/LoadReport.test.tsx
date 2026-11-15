import { afterEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { GpuMemoryReport, LoadModelOutcome } from "../../api";
import { engineStore } from "../../state/engine";
import { LoadReport } from "./LoadReport";

function outcome(gpu_memory: GpuMemoryReport | null): LoadModelOutcome {
  return {
    status: { hardware_detected: true, runtime: null, server: { state: { state: "ready" }, endpoint: null, pid: null, restarts: 0 }, gpu_verdict: null },
    rationale: [],
    estimate: null,
    model_id: "local/m.gguf",
    config_source: "recommended",
    benchmark: { stable: true, recorded: "inserted", entry: null, reasons: [], gpu_verdict: null, notes: [] },
    preset: null,
    cpu_fallback: false,
    gpu_memory,
  } as unknown as LoadModelOutcome;
}

describe("LoadReport GPU memory", () => {
  afterEach(() => engineStore.set({ loadReport: null }));

  it("warns when a dense load spilled into shared memory", () => {
    engineStore.set({ loadReport: outcome({ dedicated_mib: 15609, shared_growth_mib: 1798, spilled: true, expected: false }) });
    render(<LoadReport />);
    expect(screen.getByText("This load spilled into shared system memory")).toBeInTheDocument();
    expect(screen.getByText(/1,798 MiB did not fit in VRAM/)).toBeInTheDocument();
  });

  it("explains a planned spill without alarming", () => {
    engineStore.set({ loadReport: outcome({ dedicated_mib: 15000, shared_growth_mib: 9000, spilled: true, expected: true }) });
    render(<LoadReport />);
    expect(screen.getByText("Part of the model is in shared system memory")).toBeInTheDocument();
    expect(screen.queryByText("This load spilled into shared system memory")).not.toBeInTheDocument();
  });

  it("says nothing when it fit or could not be read", () => {
    engineStore.set({ loadReport: outcome({ dedicated_mib: 13000, shared_growth_mib: 100, spilled: false, expected: false }) });
    const { rerender } = render(<LoadReport />);
    expect(screen.queryByText(/shared system memory/)).not.toBeInTheDocument();
    engineStore.set({ loadReport: outcome(null) });
    rerender(<LoadReport />);
    expect(screen.queryByText(/shared system memory/)).not.toBeInTheDocument();
  });
});
