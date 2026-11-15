import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api } from "../../api";
import { chatStore, paramsFor } from "../../state/chat";
import { engineStore } from "../../state/engine";
import { engineStatus, noThinking, thinkingView } from "../../test/fixtures";
import { ThinkingEffort } from "./ThinkingEffort";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: { models: { thinking: vi.fn() } } };
});

function loadModel(id: string | null) {
  engineStore.set({
    status: engineStatus(),
    statusError: null,
    loaded: id ? { modelId: id, ctxSize: 8192, plannedGpuLayers: null, rationale: [], estimate: null } : null,
    loadReport: null,
  });
}

describe("ThinkingEffort", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    chatStore.set({ activeId: "c1", params: {} });
  });

  it("shows nothing and asks nothing while no model is loaded", () => {
    loadModel(null);
    render(<ThinkingEffort />);
    expect(screen.queryByRole("radiogroup")).toBeNull();
    expect(api.models.thinking).not.toHaveBeenCalled();
  });

  it("shows nothing for a model without an effort control", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(noThinking());
    render(<ThinkingEffort />);
    await waitFor(() => expect(api.models.thinking).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("radiogroup")).toBeNull();
    expect(screen.queryByText("Thinking")).toBeNull();
  });

  it("offers Low, Medium, High and Max for a thinking model, Medium selected by default", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView());
    render(<ThinkingEffort />);
    expect(await screen.findByRole("radiogroup", { name: /Thinking/ })).toBeInTheDocument();
    expect(screen.getAllByRole("radio").map((r) => r.getAttribute("aria-label"))).toEqual([
      "Low",
      "Medium",
      "High",
      "Max",
    ]);
    expect(screen.getByRole("radio", { name: "Medium" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "High" })).not.toBeChecked();
  });

  it("follows the default from Settings until the conversation chooses", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView({ default_effort: "high" }));
    render(<ThinkingEffort />);
    expect(await screen.findByRole("radio", { name: "High" })).toBeChecked();
    expect(paramsFor("c1").thinking).toBeNull();
  });

  it("choosing a level selects it for this conversation only", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView());
    const user = userEvent.setup();
    render(<ThinkingEffort />);

    await user.click(await screen.findByRole("radio", { name: "Low" }));

    expect(screen.getByRole("radio", { name: "Low" })).toBeChecked();
    expect(paramsFor("c1").thinking).toBe("low");
    expect(paramsFor("c2").thinking).toBeNull();
  });

  it("arrow keys move the choice", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView());
    const user = userEvent.setup();
    render(<ThinkingEffort />);

    const medium = await screen.findByRole("radio", { name: "Medium" });
    medium.focus();
    await user.keyboard("{ArrowRight}");
    expect(paramsFor("c1").thinking).toBe("high");
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(paramsFor("c1").thinking).toBe("low");
  });

  it("tells the user when High and Max are one setting on this model", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView());
    render(<ThinkingEffort />);
    const high = await screen.findByRole("radio", { name: "High" });
    expect(high).toHaveAttribute("title", expect.stringContaining("Max uses the same setting on this model"));
    expect(screen.getByRole("radio", { name: "Low" })).toHaveAttribute("title", 'Low sends the model\'s "low" setting.');
  });

  it("only the selected level is in the tab order", async () => {
    loadModel("m.gguf");
    vi.mocked(api.models.thinking).mockResolvedValue(thinkingView());
    render(<ThinkingEffort />);
    await screen.findByRole("radiogroup");
    expect(screen.getByRole("radio", { name: "Medium" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("radio", { name: "Low" })).toHaveAttribute("tabindex", "-1");
  });
});
