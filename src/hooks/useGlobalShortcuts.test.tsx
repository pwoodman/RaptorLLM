import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { api, type Conversation, type EngineStatus, type ChatEvent } from "../api";
import { chatStore, sendMessage } from "../state/chat";
import { engineStore } from "../state/engine";
import { uiStore } from "../state/ui";
import { shortcutAction, useGlobalShortcuts } from "./useGlobalShortcuts";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    api: {
      conversations: { create: vi.fn(), append: vi.fn(), rename: vi.fn() },
      chat: { start: vi.fn(), cancel: vi.fn() },
    },
    subscribe: { chat: vi.fn() },
  };
});

function key(init: Partial<KeyboardEvent> & { key: string }): Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey" | "shiftKey" | "altKey"> {
  return { ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, ...init };
}

describe("shortcutAction", () => {
  it("maps the documented shortcuts", () => {
    expect(shortcutAction(key({ key: "n", ctrlKey: true }))).toBe("newChat");
    expect(shortcutAction(key({ key: "k", ctrlKey: true }))).toBe("focusSearch");
    expect(shortcutAction(key({ key: "/", ctrlKey: true }))).toBe("showHelp");
    expect(shortcutAction(key({ key: "Escape" }))).toBe("stop");
  });

  it("accepts Command on macOS and ignores plain letters and shifted variants", () => {
    expect(shortcutAction(key({ key: "n", metaKey: true }))).toBe("newChat");
    expect(shortcutAction(key({ key: "n" }))).toBeNull();
    expect(shortcutAction(key({ key: "n", ctrlKey: true, shiftKey: true }))).toBeNull();
    expect(shortcutAction(key({ key: "k", ctrlKey: true, altKey: true }))).toBeNull();
  });
});

function Harness({ onShowHelp }: { onShowHelp: () => void }) {
  useGlobalShortcuts({ onShowHelp });
  return null;
}

const conversation: Conversation = { version: 1, id: "c1", title: "Chat", created_at: 0, updated_at: 0, messages: [] };
const ready: EngineStatus = {
  hardware_detected: true,
  runtime: null,
  server: { state: { state: "ready" }, endpoint: null, pid: null, restarts: 0 },
  gpu_verdict: null,
};

describe("useGlobalShortcuts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    uiStore.set({ page: "models", anchor: null, flash: null });
    engineStore.set({ status: ready, statusError: null });
    chatStore.set({
      list: null,
      listProblems: [],
      listError: null,
      activeId: null,
      active: null,
      activeError: null,
      pending: null,
    });
  });

  it("Ctrl+/ opens the help list", () => {
    const onShowHelp = vi.fn();
    render(<Harness onShowHelp={onShowHelp} />);
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "/", ctrlKey: true, cancelable: true }));
    });
    expect(onShowHelp).toHaveBeenCalledTimes(1);
  });

  it("Ctrl+N opens Chats and starts a new conversation", async () => {
    vi.mocked(api.conversations.create).mockResolvedValue(conversation);
    render(<Harness onShowHelp={vi.fn()} />);
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "n", ctrlKey: true, cancelable: true }));
    });
    expect(api.conversations.create).toHaveBeenCalledWith("New chat");
    expect(uiStore.get().page).toBe("chats");
    expect(chatStore.get().activeId).toBe("c1");
  });

  it("reports a failed new chat in the app-level notice", async () => {
    vi.mocked(api.conversations.create).mockRejectedValue(new Error("disk full"));
    render(<Harness onShowHelp={vi.fn()} />);
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "n", ctrlKey: true, cancelable: true }));
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(uiStore.get().flash).toContain("disk full"));
  });

  it("Escape stops a streaming reply, and does nothing when no reply is running", async () => {
    let handler: ((event: ChatEvent) => void) | undefined;
    vi.mocked(api.conversations.append).mockResolvedValue(conversation);
    vi.mocked(api.chat.start).mockResolvedValue(undefined);
    vi.mocked(api.chat.cancel).mockResolvedValue(true);
    const { subscribe } = await import("../api");
    vi.mocked(subscribe.chat).mockImplementation(async (_id, h) => {
      handler = h;
      return () => undefined;
    });
    chatStore.set({ active: conversation, activeId: "c1" });

    render(<Harness onShowHelp={vi.fn()} />);

    // Nothing is running, so Escape is ignored.
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }));
    });
    expect(api.chat.cancel).not.toHaveBeenCalled();

    const sending = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());
    handler?.({ type: "chunk", text: "partial" });
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", cancelable: true }));
      await Promise.resolve();
    });
    expect(api.chat.cancel).toHaveBeenCalledTimes(1);
    handler?.({ type: "done", cancelled: true });
    await sending;
  });
});
