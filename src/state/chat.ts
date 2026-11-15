import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  api,
  subscribe,
  type ChatEvent,
  type Effort,
  type ChatMessage,
  type Conversation,
  type ConversationSummary,
  type ReplyStats,
  type StoredMessage,
} from "../api";
import { composeMessage, type Attachment } from "../lib/attachments";
import { messageOf } from "../lib/errors";
import { engineStore, inferenceBlockReason } from "./engine";
import { createStore } from "./store";

export const NEW_CHAT_TITLE = "New chat";
const PARAMS_KEY = "raptor.chat.params.v1";
const TITLE_MAX_CHARS = 60;

export interface GenParams {
  systemPrompt: string;
  temperature: number;
  /** Null lets the model and runtime decide. */
  maxTokens: number | null;
  /** A preset of the loaded model, applied by the backend. Null applies none. */
  preset: string | null;
  /** Thinking effort for this conversation. Null uses the setting (Medium unless changed). */
  thinking: Effort | null;
  /** Long answers as parallel sections (plan first, then every section at once). Off by default. */
  longAnswer: boolean;
}

const DEFAULT_PARAMS: GenParams = {
  systemPrompt: "",
  temperature: 0.7,
  maxTokens: null,
  preset: null,
  thinking: null,
  longAnswer: false,
};

/**
 * A reply in progress. Its text is not in the conversation until the backend has saved it.
 * `failed` keeps the partial text visible, marked incomplete, and it is never saved.
 */
export interface PendingReply {
  requestId: string;
  conversationId: string;
  status: "streaming" | "saving" | "failed";
  /** The answer. Only this is shown as the reply and saved as its content. */
  text: string;
  /** The model's reasoning, shown in a separate Thinking block. Never counted as the answer. */
  reasoning: string;
  error: string | null;
  chunks: number;
  firstChunkAt: number | null;
}

export interface ChatState {
  list: ConversationSummary[] | null;
  listProblems: string[];
  listError: string | null;
  activeId: string | null;
  active: Conversation | null;
  activeError: string | null;
  pending: PendingReply | null;
  /** A message to scroll to once its conversation is shown (set by a content-search result). */
  focusMessageId: string | null;
  /** Generation settings per conversation id. Kept in this browser only (see the report). */
  params: Record<string, GenParams>;
}

export type ActionResult = { ok: true } | { ok: false; reason: string };

function fail(reason: string): ActionResult {
  return { ok: false, reason };
}

function readParams(): Record<string, GenParams> {
  try {
    const raw = globalThis.localStorage?.getItem(PARAMS_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object") return {};
    // Settings saved before presets existed have no `preset` field. The defaults are filled in once
    // here, so every stored entry is a stable object (a selector must not build a new one per read).
    const out: Record<string, GenParams> = {};
    for (const [id, value] of Object.entries(parsed as Record<string, Partial<GenParams>>)) {
      out[id] = { ...DEFAULT_PARAMS, ...value };
    }
    return out;
  } catch {
    return {};
  }
}

function persistParams(params: Record<string, GenParams>): void {
  try {
    globalThis.localStorage?.setItem(PARAMS_KEY, JSON.stringify(params));
  } catch {
    // Storage is unavailable (private window, blocked site data): the settings last for this session.
  }
}

export const chatStore = createStore<ChatState>({
  list: null,
  listProblems: [],
  listError: null,
  activeId: null,
  active: null,
  activeError: null,
  pending: null,
  focusMessageId: null,
  params: readParams(),
});

export function paramsFor(conversationId: string, state: ChatState = chatStore.get()): GenParams {
  return state.params[conversationId] ?? DEFAULT_PARAMS;
}

export function updateParams(conversationId: string, patch: Partial<GenParams>): void {
  const state = chatStore.get();
  const params = { ...state.params, [conversationId]: { ...paramsFor(conversationId, state), ...patch } };
  chatStore.set({ params });
  persistParams(params);
}

export function isGenerating(pending: PendingReply | null): boolean {
  return pending !== null && (pending.status === "streaming" || pending.status === "saving");
}

function summaryOf(conv: Conversation): ConversationSummary {
  return {
    id: conv.id,
    title: conv.title,
    created_at: conv.created_at,
    updated_at: conv.updated_at,
    message_count: conv.messages.length,
  };
}

function applyConversation(conv: Conversation): void {
  chatStore.set((state) => ({
    list: state.list?.map((item) => (item.id === conv.id ? summaryOf(conv) : item)) ?? state.list,
    active: state.activeId === conv.id ? conv : state.active,
  }));
}

function titleFrom(text: string): string {
  const firstLine = text.split("\n")[0]?.trim() ?? "";
  return firstLine.length > TITLE_MAX_CHARS ? `${firstLine.slice(0, TITLE_MAX_CHARS - 1)}...` : firstLine;
}

function findLastUserIndex(messages: StoredMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === "user") return i;
  }
  return -1;
}

function newRequestId(): string {
  return `req-${crypto.randomUUID()}`;
}

function updatePending(requestId: string, patch: (pending: PendingReply) => Partial<PendingReply>): void {
  chatStore.set((state) =>
    state.pending && state.pending.requestId === requestId
      ? { pending: { ...state.pending, ...patch(state.pending) } }
      : {},
  );
}

export async function refreshList(): Promise<void> {
  try {
    const result = await api.conversations.list();
    chatStore.set({ list: result.conversations, listProblems: result.problems, listError: null });
  } catch (e) {
    chatStore.set({ listError: messageOf(e) });
  }
}

export async function createConversation(): Promise<void> {
  const conv = await api.conversations.create(NEW_CHAT_TITLE);
  chatStore.set((state) => ({
    activeId: conv.id,
    active: conv,
    activeError: null,
    list: [summaryOf(conv), ...(state.list ?? [])],
  }));
}

export async function selectConversation(id: string): Promise<void> {
  chatStore.set({ activeId: id, active: null, activeError: null });
  try {
    const conv = await api.conversations.get(id);
    if (chatStore.get().activeId === id) chatStore.set({ active: conv });
  } catch (e) {
    if (chatStore.get().activeId === id) chatStore.set({ activeError: messageOf(e) });
  }
}

export async function renameConversation(id: string, title: string): Promise<void> {
  const conv = await api.conversations.rename(id, title.trim() || NEW_CHAT_TITLE);
  applyConversation(conv);
}

export async function deleteConversation(id: string): Promise<void> {
  const pending = chatStore.get().pending;
  if (pending && pending.conversationId === id) {
    // The reply is discarded, not saved. The stream's terminal event then finds nothing to persist.
    chatStore.set({ pending: null });
    if (pending.status === "streaming") {
      await api.chat.cancel(pending.requestId);
    }
  }
  await api.conversations.delete(id);
  chatStore.set((state) => ({
    list: state.list?.filter((item) => item.id !== id) ?? null,
    ...(state.activeId === id ? { activeId: null, active: null, activeError: null } : {}),
  }));
}

function blockedReason(): string | null {
  const engine = engineStore.get();
  return inferenceBlockReason(engine.status, engine.statusError);
}

/** Saves the user message, then streams the reply. Nothing is shown as a reply unless the model sent it. */
export async function sendMessage(text: string, attachments: readonly Attachment[] = []): Promise<ActionResult> {
  const trimmed = text.trim();
  if (trimmed === "" && attachments.length === 0) return fail("Type a message first.");
  const state = chatStore.get();
  if (state.activeId === null || state.active === null) return fail("Choose or create a conversation first.");
  if (isGenerating(state.pending)) return fail("Wait for the current reply to finish, or stop it.");
  const blocked = blockedReason();
  if (blocked) return fail(blocked);

  const conversationId = state.activeId;
  const isFirstMessage = state.active.messages.length === 0 && state.active.title === NEW_CHAT_TITLE;
  const content = composeMessage(trimmed, attachments);
  let conv: Conversation;
  try {
    conv = await api.conversations.append(conversationId, [{ role: "user", content }]);
  } catch (e) {
    return fail(messageOf(e));
  }
  applyConversation(conv);
  if (isFirstMessage) {
    try {
      const firstLine = trimmed !== "" ? trimmed : `Attachment: ${attachments[0]?.name ?? "file"}`;
      await renameConversation(conversationId, titleFrom(firstLine));
    } catch (e) {
      chatStore.set({ activeError: `The conversation could not be titled: ${messageOf(e)}` });
    }
  }
  await runGeneration(conversationId, conv.messages);
  return { ok: true };
}

/** Streams a new reply to the last user message, replacing subsequent replies. */
export async function regenerate(): Promise<ActionResult> {
  const state = chatStore.get();
  if (state.activeId === null || state.active === null) return fail("Choose a conversation first.");
  if (isGenerating(state.pending)) return fail("A reply is already in progress.");
  const blocked = blockedReason();
  if (blocked) return fail(blocked);
  const messages = state.active.messages;
  const last = findLastUserIndex(messages);
  if (last < 0) return fail("There is no message to answer yet.");
  await runGeneration(state.activeId, messages.slice(0, last + 1), last + 1);
  return { ok: true };
}

export async function stopGeneration(): Promise<ActionResult> {
  const pending = chatStore.get().pending;
  if (!pending || pending.status !== "streaming") return fail("No reply is streaming.");
  const cancelled = await api.chat.cancel(pending.requestId);
  return cancelled ? { ok: true } : fail("The reply had already finished.");
}

export function dismissPending(): void {
  chatStore.set({ pending: null });
}

async function runGeneration(conversationId: string, messages: StoredMessage[], truncateTo?: number): Promise<void> {
  const params = paramsFor(conversationId);
  const requestId = newRequestId();
  chatStore.set({
    pending: {
      requestId,
      conversationId,
      status: "streaming",
      text: "",
      reasoning: "",
      error: null,
      chunks: 0,
      firstChunkAt: null,
    },
  });

  const system = params.systemPrompt.trim();
  const payload: ChatMessage[] = [
    ...(system ? [{ role: "system" as const, content: system }] : []),
    ...messages.map((m) => ({ role: m.role, content: m.content })),
  ];

  let finished = false;
  let unlisten: UnlistenFn | undefined;
  const finish = () => {
    finished = true;
    unlisten?.();
  };

  const onEvent = (event: ChatEvent) => {
    if (finished) return;
    if (event.type === "chunk") {
      const text = event.text;
      updatePending(requestId, (p) => ({
        text: p.text + text,
        chunks: p.chunks + 1,
        firstChunkAt: p.firstChunkAt ?? Date.now(),
      }));
      return;
    }
    if (event.type === "reasoning") {
      const text = event.text;
      updatePending(requestId, (p) => ({
        reasoning: p.reasoning + text,
        chunks: p.chunks + 1,
        firstChunkAt: p.firstChunkAt ?? Date.now(),
      }));
      return;
    }
    finish();
    if (event.type === "error") {
      const message = event.error.message;
      updatePending(requestId, () => ({ status: "failed", error: message }));
      return;
    }
    void persistReply(requestId, conversationId, event.cancelled, truncateTo, event.stats ?? null);
  };

  try {
    // Subscribe before starting, so no chunk can be missed.
    const stop = await subscribe.chat(requestId, onEvent);
    unlisten = stop;
    if (finished) stop();
    await api.chat.start(requestId, payload, {
      temperature: params.temperature,
      max_tokens: params.maxTokens,
      // Only sent when chosen, so the backend's own defaults apply otherwise.
      ...(params.preset ? { preset: params.preset } : {}),
      // A model with no effort control ignores it, so it is sent whenever the conversation chose one.
      ...(params.thinking ? { thinking: params.thinking } : {}),
      ...(params.longAnswer ? { long_answer: true } : {}),
    });
  } catch (e) {
    finish();
    const message = messageOf(e);
    updatePending(requestId, () => ({ status: "failed", error: message }));
  }
}

async function persistReply(
  requestId: string,
  conversationId: string,
  cancelled: boolean,
  truncateTo?: number,
  stats: ReplyStats | null = null
): Promise<void> {
  const pending = chatStore.get().pending;
  if (!pending || pending.requestId !== requestId) return;
  const text = pending.text;
  const reasoning = pending.reasoning;

  if (text.trim() === "") {
    if (cancelled) {
      chatStore.set({ pending: null });
    } else {
      updatePending(requestId, () => ({
        status: "failed",
        error:
          reasoning.trim() !== ""
            ? "The model produced reasoning but no answer. Nothing was saved."
            : "The model finished without producing any text.",
      }));
    }
    return;
  }

  updatePending(requestId, () => ({ status: "saving" }));
  try {
    if (truncateTo !== undefined) {
      await api.conversations.truncate(conversationId, truncateTo);
    }
    // The reasoning is stored beside the answer. The answer itself stays the saved content.
    const saved = {
      role: "assistant" as const,
      content: text,
      ...(reasoning.trim() !== "" ? { reasoning } : {}),
      ...(stats ? { stats } : {}),
    };
    const conv = await api.conversations.append(conversationId, [saved]);
    applyConversation(conv);
    chatStore.set((state) => (state.pending?.requestId === requestId ? { pending: null } : {}));
  } catch (e) {
    const message = messageOf(e);
    updatePending(requestId, () => ({
      status: "failed",
      error: `The reply could not be saved: ${message}`,
    }));
  }
}
