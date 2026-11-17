import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const mocks = vi.hoisted(() => ({ sync: vi.fn() }));
vi.mock("../../state/localModels", async (original) => ({
  ...await original<typeof import("../../state/localModels")>(), syncLocalModels: mocks.sync,
}));
import { localModelsStore } from "../../state/localModels";
import { baseName, DiscoveredModels } from "./DiscoveredModels";

beforeEach(() => {
  mocks.sync.mockReset();
  localModelsStore.set({ phase: "idle", current: null, done: 0, total: 0, added: 0, errors: [] });
});

describe("DiscoveredModels", () => {
  it("takes file names from either kind of path", () => {
    expect(baseName("C:\\a\\b\\m.gguf")).toBe("m.gguf");
    expect(baseName("/a/b/m.gguf")).toBe("m.gguf");
  });

  it("starts automatic discovery without presenting Add buttons", () => {
    localModelsStore.set({ phase: "indexing", current: "C:/models/a.gguf", done: 0, total: 13 });
    render(<DiscoveredModels />);
    expect(mocks.sync).toHaveBeenCalled();
    expect(screen.getByText(/0 of 13 checked/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Add/ })).not.toBeInTheDocument();
    expect(screen.getByText("a.gguf")).toBeInTheDocument();
  });

  it("shows nothing when no new models were found", () => {
    const { container } = render(<DiscoveredModels />);
    expect(container).toBeEmptyDOMElement();
  });

  it("offers a retry when automatic registration fails", async () => {
    localModelsStore.set({ errors: ["a.gguf: file unavailable"] });
    render(<DiscoveredModels />);
    await userEvent.click(screen.getByRole("button", { name: "Check folders again" }));
    expect(mocks.sync).toHaveBeenLastCalledWith(true);
  });
});
