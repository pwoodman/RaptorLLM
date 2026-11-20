import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ModelEntry } from "../../api";

const mocks = vi.hoisted(() => ({ loadModel: vi.fn() }));

vi.mock("../../state/engine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../state/engine")>();
  return { ...actual, loadModel: mocks.loadModel };
});
vi.mock("../models/LoadProgressBar", () => ({ LoadProgressBar: () => null }));
vi.mock("./DiscoveredModels", () => ({ DiscoveredModels: () => <div data-testid="discovered" /> }));

import { engineStore } from "../../state/engine";
import { libraryStore } from "../../state/library";
import { orderModels, QuickLoadModels } from "./QuickLoadModels";

function model(id: string, file: string, size = 1_000_000_000): ModelEntry {
  return {
    id,
    file_name: file,
    location: file,
    managed: false,
    size_bytes: size,
    sha256: "x",
    added_at: 0,
    source: { kind: "local_reference", original_path: file } as unknown as ModelEntry["source"],
    gguf: { architecture: "qwen35" } as ModelEntry["gguf"],
  };
}

describe("orderModels", () => {
  it("puts the last used model first and sorts the rest by name", () => {
    const list = [model("b", "b.gguf"), model("c", "c.gguf"), model("a", "a.gguf")];
    expect(orderModels(list, "c").map((m) => m.id)).toEqual(["c", "a", "b"]);
    expect(orderModels(list, null).map((m) => m.id)).toEqual(["a", "b", "c"]);
  });
});

describe("QuickLoadModels", () => {
  beforeEach(() => {
    mocks.loadModel.mockReset();
    engineStore.set({ loaded: null });
    window.localStorage.clear();
  });

  it("offers installed models in one picker and loads the one chosen", async () => {
    libraryStore.set({ models: [model("local/a.gguf", "a.gguf"), model("local/b.gguf", "b.gguf")] });
    mocks.loadModel.mockImplementation(async (modelId) => {
      engineStore.set({ loaded: { modelId, ctxSize: null, plannedGpuLayers: null, rationale: [], estimate: null } });
    });
    render(<QuickLoadModels />);

    const picker = screen.getByRole("combobox", { name: "Model to load" });
    expect(screen.getAllByRole("option")).toHaveLength(2);
    await userEvent.selectOptions(picker, "local/b.gguf");
    await userEvent.click(screen.getByRole("button", { name: /Load model/ }));

    await waitFor(() => expect(mocks.loadModel).toHaveBeenCalledWith("local/b.gguf"));
    expect(window.localStorage.getItem("raptor.lastLoadedModel")).toBe("local/b.gguf");
  });

  it("shows why a load was refused and does not remember that model", async () => {
    libraryStore.set({ models: [model("local/a.gguf", "a.gguf")] });
    mocks.loadModel.mockRejectedValue(new Error("out of memory"));
    render(<QuickLoadModels />);

    await userEvent.click(screen.getByRole("button", { name: /Load model/ }));

    expect(await screen.findByText(/out of memory/)).toBeInTheDocument();
    expect(window.localStorage.getItem("raptor.lastLoadedModel")).toBeNull();
  });

  it("points to the Models page when nothing is installed", () => {
    libraryStore.set({ models: [] });
    render(<QuickLoadModels />);
    expect(screen.getByRole("button", { name: "Browse models" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add local models" })).toBeInTheDocument();
  });
});
