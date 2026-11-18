import { afterEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { CurrentOptimizationBadge } from "./OptimizationBadge";
import { ReplySpeed } from "../chat/MessageList";
import { engineStore } from "../../state/engine";
import { loadOutcome } from "../../test/fixtures";

afterEach(() => engineStore.set({ loaded: null, loadReport: null }));
describe("optimization identity", () => {
  it("requires backend confirmation, rather than a saved or attempted profile", () => {
    const report = loadOutcome({ tuning_applied: false, rationale: ["Tuned on this machine: attempted"] });
    engineStore.set({ loaded: { modelId: report.model_id, ctxSize: 8192, plannedGpuLayers: 32, estimate: null, rationale: report.rationale }, loadReport: report });
    const view = render(<CurrentOptimizationBadge />);
    expect(screen.getByText("Standard")).toBeInTheDocument();
    engineStore.set({ loadReport: { ...report, tuning_applied: true } });
    view.rerender(<CurrentOptimizationBadge />);
    expect(screen.getByText("Raptor Optimized")).toBeInTheDocument();
  });

  it("colors historical speeds from that reply's captured status", () => {
    const view = render(<ReplySpeed stats={{ tokens_per_second: 28.4, raptor_optimized: true }} />);
    expect(screen.getByText(/28.4 tok\/s/)).toHaveClass("optimized-speed");
    expect(screen.getByText("Raptor Optimized")).toBeInTheDocument();
    view.rerender(<ReplySpeed stats={{ tokens_per_second: 28.4, raptor_optimized: false }} />);
    expect(screen.getByText(/28.4 tok\/s/)).toHaveClass("muted");
    expect(screen.getByText("Standard")).toBeInTheDocument();
  });
});
