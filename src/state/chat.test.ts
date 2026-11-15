import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  api,
  RaptorApiError,
  subscribe,
  type ChatEvent,
  type Conversation,
  type EngineStatus,
  type StoredMessage,
} from "../api";
import { chatStore, NEW_CHAT_TITLE, regenerate, sendMessage, updateParams } from "./chat";
import { engineStore } from "./engine";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    api: {
      conversations: { append: vi.fn(), rename: vi.fn(), truncate: vi.fn() },
      chat: { start: vi.fn(), cancel: vi.fn() },
    },
    subscribe: { chat: vi.fn() },
  };
});

const handlers = new Map<string, (event: ChatEvent) => void>();
let saved: StoredMessage[] = [];

function stored(role: StoredMessage["role"], content: string, id: string): StoredMessage {
  return { id, role, content, created_at: 0 };
}

function convWith(messages: StoredMessage[], title = "Existing chat"): Conversation {
  return { version: 1, id: "c1", title, created_at: 0, updated_at: 0, messages };
}

function engine(state: EngineStatus["server"]["state"]): EngineStatus {
  return {
    hardware_detected: true,
    runtime: null,
    server: { state, endpoint: "http://127.0.0.1:8080", pid: 1, restarts: 0 },
    gpu_verdict: null,
  };
}

function requestId(): string {
  const call = vi.mocked(api.chat.start).mock.calls[0];
  if (!call) throw new Error("chat_start was not called");
  return call[0];
}

function emit(event: ChatEvent): void {
  handlers.get(requestId())?.(event);
}

beforeEach(() => {
  vi.clearAllMocks();
  handlers.clear();
  localStorage.clear();
  saved = [stored("user", "earlier question", "m0")];
  vi.mocked(api.conversations.truncate).mockImplementation(async (_id, count) => {
    saved = saved.slice(0, count);
    return convWith(saved);
  });
  vi.mocked(api.conversations.append).mockImplementation(async (_id, added) => {
    saved = [
      ...saved,
      ...added.map((m, i) => ({
        ...stored(m.role, m.content, `m${saved.length + i}`),
        ...(m.reasoning ? { reasoning: m.reasoning } : {}),
      })),
    ];
    return convWith(saved);
  });
  vi.mocked(subscribe.chat).mockImplementation(async (id, handler) => {
    handlers.set(id, handler);
    return () => {
      handlers.delete(id);
    };
  });
  vi.mocked(api.chat.start).mockResolvedValue(undefined);
  engineStore.set({ status: engine({ state: "ready" }), statusError: null });
  chatStore.set({
    list: null,
    listProblems: [],
    listError: null,
    activeId: "c1",
    active: convWith(saved),
    activeError: null,
    pending: null,
    params: {},
  });
});

describe("sendMessage", () => {
  it("makes no request and shows no reply while the server is not running", async () => {
    engineStore.set({ status: engine({ state: "stopped" }) });

    const result = await sendMessage("hello");

    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("not running");
    expect(api.conversations.append).not.toHaveBeenCalled();
    expect(api.chat.start).not.toHaveBeenCalled();
    expect(chatStore.get().pending).toBeNull();
  });

  it("saves the user message, streams chunks, then saves the reply on done", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    expect(vi.mocked(api.chat.start).mock.calls[0]?.[1]).toEqual([
      { role: "user", content: "earlier question" },
      { role: "user", content: "hello" },
    ]);

    emit({ type: "chunk", text: "Hel" });
    emit({ type: "chunk", text: "lo" });
    expect(chatStore.get().pending?.text).toBe("Hello");
    expect(chatStore.get().pending?.status).toBe("streaming");

    emit({ type: "done", cancelled: false });
    await vi.waitFor(() => expect(chatStore.get().pending).toBeNull());

    expect(await result).toEqual({ ok: true });
    expect(saved.map((m) => [m.role, m.content])).toEqual([
      ["user", "earlier question"],
      ["user", "hello"],
      ["assistant", "Hello"],
    ]);
    expect(chatStore.get().active?.messages).toHaveLength(3);
  });

  it("sends the system prompt and generation options with the request", async () => {
    updateParams("c1", { systemPrompt: "  Be terse.  ", temperature: 0.2, maxTokens: 64 });

    void sendMessage("hi");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    const [, payload, options] = vi.mocked(api.chat.start).mock.calls[0] ?? [];
    expect(payload?.[0]).toEqual({ role: "system", content: "Be terse." });
    expect(options).toEqual({ temperature: 0.2, max_tokens: 64 });
  });

  it("shows a failed stream as incomplete and never saves the partial text", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    emit({ type: "chunk", text: "partial" });
    emit({ type: "error", error: { kind: "Runtime", message: "llama-server crashed" } });
    await result;

    const pending = chatStore.get().pending;
    expect(pending?.status).toBe("failed");
    expect(pending?.text).toBe("partial");
    expect(pending?.error).toBe("llama-server crashed");
    expect(saved.some((m) => m.role === "assistant")).toBe(false);
  });

  it("keeps the text of a stopped reply and saves it", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    emit({ type: "chunk", text: "abc" });
    emit({ type: "done", cancelled: true });
    await result;
    await vi.waitFor(() => expect(chatStore.get().pending).toBeNull());

    expect(saved[saved.length - 1]).toMatchObject({ role: "assistant", content: "abc" });
  });

  it("reports a reply with no text instead of showing an empty answer", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    emit({ type: "done", cancelled: false });
    await result;

    expect(chatStore.get().pending?.status).toBe("failed");
    expect(chatStore.get().pending?.error).toContain("without producing any text");
    expect(saved.some((m) => m.role === "assistant")).toBe(false);
  });

  it("reports a refused start (no model loaded) as an error, not a reply", async () => {
    vi.mocked(api.chat.start).mockRejectedValue(new RaptorApiError({ kind: "Conflict", message: "No model is loaded" }));

    await sendMessage("hello");

    expect(chatStore.get().pending?.status).toBe("failed");
    expect(chatStore.get().pending?.error).toBe("No model is loaded");
    expect(saved.some((m) => m.role === "assistant")).toBe(false);
  });
});

describe("sendMessage with attachments", () => {
  it("sends the typed text followed by a delimited block for each attached file", async () => {
    const result = sendMessage("Why does it fail?", [{ name: "lib.ts", size: 9, content: "export {}" }]);
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());
    emit({ type: "done", cancelled: false });
    await result;

    const appended = vi.mocked(api.conversations.append).mock.calls[0]?.[1]?.[0]?.content ?? "";
    expect(appended).toContain("Why does it fail?");
    expect(appended).toContain("--- BEGIN ATTACHMENT: lib.ts (9 bytes) ---\nexport {}\n--- END ATTACHMENT: lib.ts ---");
  });

  it("accepts a file with no typed text and titles the chat from the file name", async () => {
    chatStore.set({ active: convWith([], NEW_CHAT_TITLE) });
    vi.mocked(api.conversations.rename).mockResolvedValue(convWith([], "Attachment: notes.md"));
    const result = sendMessage("", [{ name: "notes.md", size: 3, content: "abc" }]);
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());
    emit({ type: "done", cancelled: false });
    await result;

    expect(api.conversations.rename).toHaveBeenCalledWith("c1", "Attachment: notes.md");
  });

  it("still refuses an empty message with no files", async () => {
    const result = await sendMessage("   ", []);
    expect(result.ok).toBe(false);
    expect(api.conversations.append).not.toHaveBeenCalled();
  });
});

describe("regenerate", () => {
  it("answers the last user message using only the history up to it", async () => {
    saved = [stored("user", "question", "u1"), stored("assistant", "old answer", "a1")];
    chatStore.set({ active: convWith(saved) });

    await regenerate();

    expect(vi.mocked(api.chat.start).mock.calls[0]?.[1]).toEqual([{ role: "user", content: "question" }]);
  });

  it("truncates the earlier assistant reply and saves the replacement", async () => {
    saved = [stored("user", "question", "u1"), stored("assistant", "old answer", "a1")];
    chatStore.set({ active: convWith(saved) });

    await regenerate();
    emit({ type: "chunk", text: "new " });
    emit({ type: "chunk", text: "answer" });
    emit({ type: "done", cancelled: false });
    await Promise.resolve();

    expect(api.conversations.truncate).toHaveBeenCalledWith("c1", 1);
    expect(saved.map((m) => m.content)).toEqual(["question", "new answer"]);
  });
});

describe("reasoning and presets", () => {
  it("keeps reasoning out of the answer and saves it beside the answer", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    emit({ type: "reasoning", text: "Let me think. " });
    emit({ type: "reasoning", text: "Done." });
    expect(chatStore.get().pending?.reasoning).toBe("Let me think. Done.");
    expect(chatStore.get().pending?.text).toBe("");

    emit({ type: "chunk", text: "Hi" });
    emit({ type: "done", cancelled: false });
    await result;
    await vi.waitFor(() => expect(chatStore.get().pending).toBeNull());

    expect(vi.mocked(api.conversations.append).mock.calls[vi.mocked(api.conversations.append).mock.calls.length - 1]?.[1]).toEqual([
      { role: "assistant", content: "Hi", reasoning: "Let me think. Done." },
    ]);
    expect(saved[saved.length - 1]).toMatchObject({ role: "assistant", content: "Hi", reasoning: "Let me think. Done." });
  });

  it("does not count reasoning as an answer: reasoning alone is a failed reply and nothing is saved", async () => {
    const result = sendMessage("hello");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());

    emit({ type: "reasoning", text: "Only thinking." });
    emit({ type: "done", cancelled: false });
    await result;

    expect(chatStore.get().pending?.status).toBe("failed");
    expect(chatStore.get().pending?.error).toContain("reasoning but no answer");
    expect(saved.some((m) => m.role === "assistant")).toBe(false);
  });

  it("sends the chosen preset with the request, and omits it when none is chosen", async () => {
    updateParams("c1", { preset: "Review" });
    void sendMessage("hi");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());
    expect(vi.mocked(api.chat.start).mock.calls[0]?.[2]).toEqual({
      temperature: 0.7,
      max_tokens: null,
      preset: "Review",
    });
  });

  it("sends the conversation's thinking level, and nothing when it has not chosen one", async () => {
    updateParams("c1", { thinking: "high" });
    void sendMessage("hi");
    await vi.waitFor(() => expect(api.chat.start).toHaveBeenCalled());
    expect(vi.mocked(api.chat.start).mock.calls[0]?.[2]).toEqual({
      temperature: 0.7,
      max_tokens: null,
      thinking: "high",
    });
  });

  it("settings saved before thinking levels existed load with no choice", () => {
    localStorage.setItem(
      "raptor.chat.params.v1",
      JSON.stringify({ old: { systemPrompt: "x", temperature: 0.5, maxTokens: null, preset: null } }),
    );
    vi.resetModules();
    return import("./chat").then(({ paramsFor }) => {
      expect(paramsFor("old").thinking).toBeNull();
    });
  });
});
