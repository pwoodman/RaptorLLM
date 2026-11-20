import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { api, RaptorApiError, type Conversation, type EngineStatus } from "../api";
import { chatStore } from "../state/chat";
import { engineStore } from "../state/engine";
import { libraryStore } from "../state/library";
import ChatsPage from "./Chats";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    api: {
      conversations: { list: vi.fn(), create: vi.fn(), get: vi.fn() },
      models: { list: vi.fn(), storage: vi.fn() },
    },
  };
});

function status(state: EngineStatus["server"]["state"]): EngineStatus {
  return {
    hardware_detected: true,
    runtime: null,
    server: { state, endpoint: null, pid: null, restarts: 0 },
    gpu_verdict: null,
  };
}

const conversation: Conversation = {
  version: 1,
  id: "c1",
  title: "Refactor the parser",
  created_at: 0,
  updated_at: 0,
  messages: [],
};

describe("ChatsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.conversations.list).mockResolvedValue({
      conversations: [{ id: "c1", title: "Refactor the parser", created_at: 0, updated_at: 0, message_count: 0 }],
      problems: [],
    });
    vi.mocked(api.conversations.get).mockResolvedValue(conversation);
    vi.mocked(api.models.list).mockResolvedValue([]);
    vi.mocked(api.models.storage).mockResolvedValue({ models: 0, managed_bytes: 0, referenced_bytes: 0 });
    libraryStore.set({ models: [], storage: null, error: null });
    chatStore.set({
      list: null,
      listProblems: [],
      listError: null,
      activeId: "c1",
      active: conversation,
      activeError: null,
      pending: null,
    });
  });

  it("shows the next model action when the server is stopped and disables sending", async () => {
    engineStore.set({ status: status({ state: "stopped" }), statusError: null });

    render(<ChatsPage />);

    expect(await screen.findByText(/not running/i)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Message" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Browse models" })).toBeInTheDocument();
  });

  it("enables the composer once the server is ready", async () => {
    engineStore.set({ status: status({ state: "ready" }), statusError: null });

    render(<ChatsPage />);

    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
  });

  it("shows a backend failure when the conversation list cannot be loaded", async () => {
    engineStore.set({ status: status({ state: "ready" }), statusError: null });
    vi.mocked(api.conversations.list).mockRejectedValue(new RaptorApiError({ kind: "Io", message: "disk unreadable" }));

    render(<ChatsPage />);

    expect(await screen.findByText("disk unreadable")).toBeInTheDocument();
  });
});
