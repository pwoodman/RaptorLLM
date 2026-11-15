import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, contextClearEmbeddingModel, contextSemanticStatus, contextSetEmbeddingModel, type SemanticView } from "../../api";
import { libraryStore } from "../../state/library";
import { modelEntry, settingsView } from "../../test/fixtures";
import { ContextPanel } from "./ContextPanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    contextSemanticStatus: vi.fn(),
    contextSetEmbeddingModel: vi.fn(),
    contextClearEmbeddingModel: vi.fn(),
    api: {
      models: { list: vi.fn(), storage: vi.fn() },
      context: { indexStatus: vi.fn() },
      settings: {},
    },
  };
});

const EMBED = modelEntry({ id: "org/embed/e.gguf", file_name: "e.gguf" });
const WS = { id: "ws-0123456789abcdef", path: "D:/Projects/app" };

function semantic(overrides: Partial<SemanticView> = {}): SemanticView {
  return {
    configured_model: null,
    reason: null,
    status: {
      enabled: false,
      model_id: null,
      dimension: 0,
      vectors: 0,
      capacity: 0,
      chunks_without_vector: 0,
      last_error: null,
      store_notice: null,
    },
    ...overrides,
  };
}

describe("ContextPanel embedding model and semantic status", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    libraryStore.set({ models: [EMBED], storage: null, error: null });
    vi.mocked(api.models.list).mockResolvedValue([EMBED]);
    vi.mocked(api.models.storage).mockResolvedValue({ models: 1, managed_bytes: 0, referenced_bytes: 0 });
    vi.mocked(api.context.indexStatus).mockResolvedValue({
      root: WS.path,
      index_version: 1,
      files: 3,
      chunks: 0,
      symbols: 0,
      imports: 0,
      excluded: 0,
      graph_files: 0,
      graph_edges: 0,
      git_repo: false,
      changed_files: 0,
      git_warning: null,
    });
  });

  it("says why semantic retrieval is off for a workspace", async () => {
    vi.mocked(contextSemanticStatus).mockResolvedValue(
      semantic({
        configured_model: EMBED.id,
        reason: "no model is loaded; load the embedding model to use semantic retrieval",
      }),
    );
    render(<ContextPanel settings={settingsView({ workspaces: [WS], embedding_model: EMBED.id })} onChange={() => {}} />);

    expect(await screen.findByText("Semantic off")).toBeInTheDocument();
    expect(screen.getByText(/no model is loaded/)).toBeInTheDocument();
    expect(screen.getByText(/Lexical and graph retrieval still run/)).toBeInTheDocument();
  });

  it("shows the vector count when semantic retrieval is active", async () => {
    vi.mocked(contextSemanticStatus).mockResolvedValue(
      semantic({
        configured_model: EMBED.id,
        status: { ...semantic().status, enabled: true, vectors: 1200, chunks_without_vector: 4 },
      }),
    );
    render(<ContextPanel settings={settingsView({ workspaces: [WS], embedding_model: EMBED.id })} onChange={() => {}} />);

    expect(await screen.findByText("Semantic on")).toBeInTheDocument();
    expect(screen.getByText(/1,200 vectors, 4 chunks without a vector/)).toBeInTheDocument();
  });

  it("sets the embedding model and reports the new setting back to the page", async () => {
    vi.mocked(contextSemanticStatus).mockResolvedValue(semantic());
    vi.mocked(contextSetEmbeddingModel).mockResolvedValue(EMBED.id);
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<ContextPanel settings={settingsView({ workspaces: [] })} onChange={onChange} />);

    await user.selectOptions(await screen.findByRole("combobox", { name: "Model for semantic retrieval" }), EMBED.id);

    await waitFor(() => expect(contextSetEmbeddingModel).toHaveBeenCalledWith(EMBED.id));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ embedding_model: EMBED.id }));
  });

  it("clears the embedding model back to off", async () => {
    vi.mocked(contextClearEmbeddingModel).mockResolvedValue(undefined);
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<ContextPanel settings={settingsView({ embedding_model: EMBED.id })} onChange={onChange} />);

    await user.selectOptions(await screen.findByRole("combobox", { name: "Model for semantic retrieval" }), "");

    await waitFor(() => expect(contextClearEmbeddingModel).toHaveBeenCalledTimes(1));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ embedding_model: null }));
  });
});
