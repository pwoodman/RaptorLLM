import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const mocks = vi.hoisted(() => ({ discover: vi.fn(), importModel: vi.fn(), importProgress: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      models: { ...actual.api.models, discover: mocks.discover, importProgress: mocks.importProgress },
    },
  };
});
vi.mock("../../state/library", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../state/library")>();
  return { ...actual, importModel: mocks.importModel };
});

import { baseName, DiscoveredModels } from "./DiscoveredModels";

const found = [
  {
    path: "C:\\Users\\me\\.lmstudio\\models",
    source: "LM Studio",
    items: [
      { path: "C:\\Users\\me\\.lmstudio\\models\\pub\\repo\\a.gguf", size_bytes: 2_000_000_000, status: { status: "new" } },
    ],
  },
];

describe("DiscoveredModels", () => {
  beforeEach(() => {
    mocks.discover.mockReset();
    mocks.importModel.mockReset();
    mocks.importProgress.mockResolvedValue({ done: 0, total: 0 });
  });

  it("takes the file name from either kind of path", () => {
    expect(baseName("C:\\a\\b\\m.gguf")).toBe("m.gguf");
    expect(baseName("/a/b/m.gguf")).toBe("m.gguf");
  });

  it("offers models from other apps and adds one by reference", async () => {
    mocks.discover.mockResolvedValueOnce(found).mockResolvedValue([]);
    mocks.importModel.mockResolvedValue({});
    render(<DiscoveredModels />);

    expect(await screen.findByText("a.gguf")).toBeInTheDocument();
    expect(screen.getByText(/LM Studio/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Add/ }));

    await waitFor(() =>
      expect(mocks.importModel).toHaveBeenCalledWith(
        "C:\\Users\\me\\.lmstudio\\models\\pub\\repo\\a.gguf",
        "reference",
      ),
    );
  });

  it("shows nothing when no other models are found", async () => {
    mocks.discover.mockResolvedValue([]);
    const { container } = render(<DiscoveredModels />);
    await waitFor(() => expect(mocks.discover).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });
});
