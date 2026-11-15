import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import type { GpuVerdict } from "../api";
import { GpuVerdictView, verdictSummary } from "./GpuVerdictView";

describe("GpuVerdictView", () => {
  it("shows a loud alert when the GPU is not running the model", () => {
    render(<GpuVerdictView verdict={{ verdict: "cpu_only", reason: "0 layers offloaded" }} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("CPU only");
    expect(alert).toHaveTextContent("0 layers offloaded");
    expect(screen.queryByText(/GPU verified/i)).toBeNull();
  });

  it("claims GPU acceleration only with positive evidence, and lists the evidence", () => {
    render(<GpuVerdictView verdict={{ verdict: "verified", evidence: ["offloaded 33/33 layers to ROCm0"] }} />);
    expect(screen.getByText("GPU verified")).toBeInTheDocument();
    expect(screen.getByText("offloaded 33/33 layers to ROCm0")).toBeInTheDocument();
  });

  it("never presents an unverified state as GPU use", () => {
    render(<GpuVerdictView verdict={{ verdict: "unverified", reason: "no offload line in the log" }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("GPU unverified");
    expect(screen.queryByText("GPU verified")).toBeNull();
  });

  it("summarises each verdict for the sidebar", () => {
    const verdicts: [GpuVerdict | null, string][] = [
      [null, "No GPU verdict yet"],
      [{ verdict: "verified", evidence: [] }, "GPU verified"],
      [{ verdict: "cpu_only", reason: "x" }, "CPU only"],
      [{ verdict: "unverified", reason: "x" }, "GPU unverified"],
    ];
    for (const [verdict, label] of verdicts) {
      expect(verdictSummary(verdict).label).toBe(label);
    }
  });
});
