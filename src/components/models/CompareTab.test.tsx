import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  modelsCheckUpdates,
  modelsCompare,
  modelsDetails,
  modelsRankFor,
  RaptorApiError,
  type Comparison,
  type ModelDetails,
} from "../../api";
import { libraryStore } from "../../state/library";
import { modelEntry } from "../../test/fixtures";
import { CompareTab } from "./CompareTab";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    modelsCompare: vi.fn(),
    modelsRankFor: vi.fn(),
    modelsCheckUpdates: vi.fn(),
    modelsDetails: vi.fn(),
  };
});

const A = modelEntry({ id: "org/a/a.gguf", file_name: "a.gguf" });
const B = modelEntry({ id: "org/b/b.gguf", file_name: "b.gguf", gguf: { ...modelEntry().gguf, quant_label: "Q8_0" } });

const COMPARISON: Comparison = {
  ctx: 8192,
  hardware: { ram_bytes: 32 * 1024 ** 3, vram_bytes: 16 * 1024 ** 3 },
  rows: [
    {
      id: A.id,
      name: "a.gguf",
      architecture: "llama",
      quant_label: "Q4_K_M",
      bits_per_weight: 4.8,
      params_estimate: null,
      params_from_name_billions: 8,
      file_bytes: 4 * 1024 ** 3,
      context_train: 8192,
      chat_template: true,
      estimated_bytes_at_ctx: 5 * 1024 ** 3,
      ctx_exceeds_train: false,
      compat: "Great",
      tags: [{ task: "coding", score: 62, evidence: [] }],
    },
  ],
};

describe("CompareTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    libraryStore.set({ models: [A, B], storage: null, error: null });
  });

  it("compares the selected models and shows the table with its heuristic labels", async () => {
    vi.mocked(modelsCompare).mockResolvedValue(COMPARISON);
    const user = userEvent.setup();
    render(<CompareTab />);

    await user.click(screen.getByRole("checkbox", { name: /a\.gguf/ }));
    await user.click(screen.getByRole("button", { name: "Compare selected" }));

    expect(modelsCompare).toHaveBeenCalledWith([A.id], null);
    const table = await screen.findByRole("table");
    expect(within(table).getByRole("rowheader", { name: /a\.gguf/ })).toBeInTheDocument();
    expect(within(table).getByText("Fits well")).toBeInTheDocument();
    expect(screen.getByText(/heuristics from the file, not measured results/)).toBeInTheDocument();
  });

  it("does not compare anything until a model is chosen", () => {
    render(<CompareTab />);
    expect(screen.getByRole("button", { name: "Compare selected" })).toBeDisabled();
  });

  it("ranks installed models for a task and labels the result as heuristic", async () => {
    vi.mocked(modelsRankFor).mockResolvedValue([
      { id: B.id, score: 71.5, task_score: 71.5, rating: "Ok", bits_per_weight: 8, reasons: ["Large quantization"] },
    ]);
    const user = userEvent.setup();
    render(<CompareTab />);

    await user.selectOptions(screen.getByLabelText("Task"), "reasoning");
    await user.click(screen.getByRole("button", { name: "Rank installed models" }));

    expect(modelsRankFor).toHaveBeenCalledWith("reasoning", null);
    const list = await screen.findByRole("list", { name: "Models ranked for reasoning" });
    expect(within(list).getByText("b.gguf")).toBeInTheDocument();
    expect(within(list).getByText("Large quantization")).toBeInTheDocument();
    expect(screen.getByText(/Heuristic: it reads file names/)).toBeInTheDocument();
  });

  it("checks for updates and shows the pinned revision from the model details", async () => {
    vi.mocked(modelsCheckUpdates).mockResolvedValue([
      { id: A.id, repo: "org/a", pinned_revision: null, head_revision: "f".repeat(40), status: "changed" },
    ]);
    vi.mocked(modelsDetails).mockResolvedValue({
      entry: A,
      revision: "e".repeat(40),
      hub_tags: [],
      files: [],
    } as ModelDetails);
    const user = userEvent.setup();
    render(<CompareTab />);

    await user.click(screen.getByRole("button", { name: "Check for updates" }));

    const list = await screen.findByRole("list", { name: "Update check results" });
    expect(within(list).getByText("Head moved")).toBeInTheDocument();
    expect(within(list).getByText("e".repeat(12))).toBeInTheDocument();
    expect(modelsDetails).toHaveBeenCalledWith(A.id);
  });

  it("explains an offline update check instead of showing an empty result", async () => {
    vi.mocked(modelsCheckUpdates).mockRejectedValue(new RaptorApiError({ kind: "Network", message: "no route to host" }));
    const user = userEvent.setup();
    render(<CompareTab />);
    await user.click(screen.getByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText("Network error")).toBeInTheDocument();
    expect(screen.getByText(/Try again when you are online/)).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("list", { name: "Update check results" })).toBeNull());
  });
});
