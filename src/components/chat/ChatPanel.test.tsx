import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  conversationsExportTo,
  dialogSaveFile,
  presetsList,
  RaptorApiError,
  type Conversation,
  type PresetList,
} from "../../api";
import { chatStore } from "../../state/chat";
import { engineStore } from "../../state/engine";
import { engineStatus, loadOutcome, preset } from "../../test/fixtures";
import { ChatPanel } from "./ChatPanel";
import { ParamsPanel } from "./ParamsPanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    dialogSaveFile: vi.fn(),
    conversationsExportTo: vi.fn(),
    presetsList: vi.fn(),
    api: { conversations: { export: vi.fn() } },
  };
});

const conversation: Conversation = {
  version: 1,
  id: "c1",
  title: "Parser work",
  created_at: 0,
  updated_at: 0,
  messages: [],
};

function openChat() {
  chatStore.set({
    list: [],
    listProblems: [],
    listError: null,
    activeId: "c1",
    active: conversation,
    activeError: null,
    pending: null,
    params: {},
  });
}

describe("ChatPanel export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    engineStore.set({ status: engineStatus(), statusError: null });
    openChat();
  });

  it("writes nothing when the save dialog is cancelled", async () => {
    vi.mocked(dialogSaveFile).mockResolvedValue(null);
    const user = userEvent.setup();
    render(<ChatPanel />);

    await user.click(screen.getByRole("button", { name: /Export/ }));

    expect(await screen.findByText("Export cancelled. Nothing was saved.")).toBeInTheDocument();
    expect(dialogSaveFile).toHaveBeenCalledWith("Parser-work-c1.json", "json");
    expect(conversationsExportTo).not.toHaveBeenCalled();
  });

  it("writes the conversation to the chosen path and reports the size", async () => {
    vi.mocked(dialogSaveFile).mockResolvedValue("C:/Exports/parser.json");
    vi.mocked(conversationsExportTo).mockResolvedValue({ path: "C:/Exports/parser.json", bytes: 2048 });
    const user = userEvent.setup();
    render(<ChatPanel />);

    await user.click(screen.getByRole("button", { name: /Export/ }));

    await waitFor(() => expect(conversationsExportTo).toHaveBeenCalledWith("c1", "C:/Exports/parser.json"));
    expect(await screen.findByText(/Saved 2\.0 KB to C:\/Exports\/parser\.json/)).toBeInTheDocument();
  });

  it("shows a failed export as an error, not as saved", async () => {
    vi.mocked(dialogSaveFile).mockResolvedValue("C:/locked/parser.json");
    vi.mocked(conversationsExportTo).mockRejectedValue(new RaptorApiError({ kind: "Io", message: "access denied" }));
    const user = userEvent.setup();
    render(<ChatPanel />);

    await user.click(screen.getByRole("button", { name: /Export/ }));

    expect(await screen.findByText("Export failed")).toBeInTheDocument();
    expect(screen.getByText("access denied")).toBeInTheDocument();
    expect(screen.queryByText(/^Saved /)).toBeNull();
  });
});

describe("ParamsPanel presets", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    openChat();
    engineStore.set({ status: engineStatus(), statusError: null, loaded: { modelId: "org/repo/model.gguf", ctxSize: null, plannedGpuLayers: null, rationale: [], estimate: null }, loadReport: loadOutcome() });
  });

  it("copies a chosen preset's system prompt and temperature into the settings", async () => {
    const list: PresetList = { model_id: "org/repo/model.gguf", presets: [preset({ temperature: 0.2, max_tokens: 256 })], default: "Review" };
    vi.mocked(presetsList).mockResolvedValue(list);
    const user = userEvent.setup();
    render(<ParamsPanel conversationId="c1" />);

    await user.selectOptions(await screen.findByRole("combobox", { name: /Preset/ }), "Review");

    expect(chatStore.get().params.c1).toMatchObject({
      preset: "Review",
      systemPrompt: "You review diffs.",
      temperature: 0.2,
      maxTokens: 256,
    });
    expect(screen.getByText(/copied into the settings below/)).toBeInTheDocument();
  });

  it("asks for a loaded model when none is loaded", () => {
    engineStore.set({ loaded: null });
    render(<ParamsPanel conversationId="c1" />);
    expect(screen.getByText("Load a model to use its presets.")).toBeInTheDocument();
  });

  it("says so when the presets cannot be read", async () => {
    vi.mocked(presetsList).mockRejectedValue(new RaptorApiError({ kind: "Io", message: "presets file corrupt" }));
    render(<ParamsPanel conversationId="c1" />);
    expect(await screen.findByText("Presets unavailable")).toBeInTheDocument();
  });
});
