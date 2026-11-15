import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { chatStore, paramsFor } from "../../state/chat";
import { engineStore } from "../../state/engine";
import { engineStatus } from "../../test/fixtures";
import { LongAnswerToggle } from "./LongAnswerToggle";

function loadModel(id: string | null) {
  engineStore.set({
    status: engineStatus(),
    statusError: null,
    loaded: id ? { modelId: id, ctxSize: 8192, plannedGpuLayers: null, rationale: [], estimate: null } : null,
    loadReport: null,
  });
}

describe("LongAnswerToggle", () => {
  beforeEach(() => {
    localStorage.clear();
    chatStore.set({ activeId: "c1", params: {} });
  });

  it("shows nothing while no model is loaded", () => {
    loadModel(null);
    render(<LongAnswerToggle />);
    expect(screen.queryByRole("button", { name: /Long answer/ })).toBeNull();
  });

  it("is off by default and turns long answers on and off for this conversation only", async () => {
    loadModel("m.gguf");
    render(<LongAnswerToggle />);
    const toggle = screen.getByRole("button", { name: /Long answer/ });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(toggle.getAttribute("title")).toMatch(/Requests at once/);
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(paramsFor("c1", chatStore.get()).longAnswer).toBe(true);
    expect(paramsFor("c2", chatStore.get()).longAnswer).toBe(false);
    await userEvent.click(toggle);
    expect(paramsFor("c1", chatStore.get()).longAnswer).toBe(false);
  });
});
