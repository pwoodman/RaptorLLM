import { useEffect, useState, type RefObject } from "react";
import { MessageSquare } from "lucide-react";
import type { ReplyStats, StoredMessage } from "../../api";
import { useStore } from "../../hooks/useStore";
import { chatStore, dismissPending, isGenerating, regenerate, type PendingReply } from "../../state/chat";
import { CopyButton } from "../CopyButton";
import { EmptyState } from "../AsyncState";
import { LazyMarkdown } from "../markdown/LazyMarkdown";
import { Notice } from "../Notice";
import { ThinkingBlock } from "./ThinkingBlock";
import { OptimizationBadge } from "../models/OptimizationBadge";

const ROLE_LABEL: Record<StoredMessage["role"], string> = {
  user: "You",
  assistant: "Model",
  system: "System",
  tool: "Tool",
};

/** "21.3 tok/s · 0.41 s to first token", from the server's own timings. Nothing when unreported. */
export function ReplySpeed({ stats }: { stats: ReplyStats | null }) {
  const text = replySpeedText(stats);
  return text === null ? null : (
    <span className={`reply-speed small ${stats?.raptor_optimized ? "optimized-speed" : "muted"}`} title="Reply timings and context usage">
      {text}
      {typeof stats?.raptor_optimized === "boolean" && <OptimizationBadge optimized={stats.raptor_optimized} />}
    </span>
  );
}

export function replySpeedText(stats: ReplyStats | null): string | null {
  if (!stats) return null;
  const parts: string[] = [];
  if (typeof stats.tokens_per_second === "number") parts.push(`${stats.tokens_per_second.toFixed(1)} tok/s`);
  if (typeof stats.time_to_first_token_ms === "number") {
    const ms = stats.time_to_first_token_ms;
    parts.push(ms >= 1000 ? `${(ms / 1000).toFixed(2)} s to first token` : `${Math.round(ms)} ms to first token`);
  }
  const omitted = stats.omitted_history_turns ?? 0;
  if (omitted > 0) parts.push(`${omitted} earlier ${omitted === 1 ? "exchange" : "exchanges"} omitted for this reply`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

function MessageItem({ message }: { message: StoredMessage }) {
  const isUser = message.role === "user";
  return (
    <article className={`message message-${message.role}`} aria-label={`${ROLE_LABEL[message.role]} message`}>
      <div className="message-role">{ROLE_LABEL[message.role]}</div>
      {!isUser && message.reasoning && (
        <ThinkingBlock text={message.reasoning} streaming={false} answerStarted />
      )}
      <div className="message-body" data-find-block={message.id}>
        {isUser ? <p className="pre-wrap">{message.content}</p> : <LazyMarkdown text={message.content} />}
      </div>
      {!isUser && (
        <div className="message-actions">
          <ReplySpeed stats={message.stats ?? null} />
          <CopyButton label="Copy reply" getText={() => message.content} />
        </div>
      )}
    </article>
  );
}

function PendingItem({ pending, onRetry }: { pending: PendingReply; onRetry: () => void }) {
  // Speed is an estimate: each chunk is a piece of text, not necessarily one token.
  const elapsed = pending.firstChunkAt === null ? 0 : (Date.now() - pending.firstChunkAt) / 1000;
  const approxSpeed = pending.chunks > 1 && elapsed >= 0.5 ? (pending.chunks - 1) / elapsed : null;

  return (
    <article className="message message-assistant pending" aria-label="Model reply">
      <div className="message-role">Model {typeof pending.raptorOptimized === "boolean" && <OptimizationBadge optimized={pending.raptorOptimized} />}</div>
      {pending.reasoning !== "" && (
        <ThinkingBlock
          text={pending.reasoning}
          streaming={pending.status === "streaming"}
          answerStarted={pending.text !== ""}
        />
      )}
      <div className="message-body" data-find-block="pending">
        {pending.text ? (
          <LazyMarkdown text={pending.text} />
        ) : pending.reasoning !== "" ? (
          <span className="muted">Thinking. The answer has not started yet.</span>
        ) : (
          <span className="muted">Waiting for the first token...</span>
        )}
      </div>
      {pending.status === "streaming" && (
        <div className={`stream-meta ${pending.raptorOptimized ? "optimized-speed" : ""}`} role="status">
          Generating{approxSpeed !== null ? ` · about ${approxSpeed.toFixed(1)} tok/s (estimate)` : ""}
        </div>
      )}
      {pending.status === "saving" && <div className="stream-meta">Saving reply...</div>}
      {pending.status === "failed" && (
        <Notice
          tone="danger"
          title={pending.text ? "Incomplete reply, not saved" : "No reply"}
          actions={
            <>
              <button type="button" className="btn btn-sm" onClick={onRetry}>
                Retry
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={dismissPending}>
                Dismiss
              </button>
            </>
          }
        >
          {pending.error}
        </Notice>
      )}
    </article>
  );
}

/** The conversation's messages. `containerRef` is the scroll box, which the find bar searches. */
export function MessageList({ containerRef }: { containerRef: RefObject<HTMLDivElement> }) {
  const active = useStore(chatStore, (s) => s.active);
  const pending = useStore(chatStore, (s) => s.pending);
  const [actionError, setActionError] = useState<string | null>(null);

  const showPending = pending !== null && active !== null && pending.conversationId === active.id;
  const streaming = showPending && isGenerating(pending);
  const messageCount = active?.messages.length ?? 0;
  const pendingLength = showPending ? pending.text.length + pending.reasoning.length : 0;

  const focusMessageId = useStore(chatStore, (s) => s.focusMessageId);

  // A search result opens its conversation at the message that matched, not at the bottom.
  useEffect(() => {
    if (focusMessageId === null || active === null) return;
    const el = containerRef.current?.querySelector(`[data-find-block="${CSS.escape(focusMessageId)}"]`);
    if (el) {
      el.scrollIntoView?.({ block: "center" });
      el.classList.add("search-focus");
      window.setTimeout(() => el.classList.remove("search-focus"), 2200);
    }
    chatStore.set({ focusMessageId: null });
  }, [focusMessageId, active, containerRef]);

  useEffect(() => {
    if (chatStore.get().focusMessageId !== null) return;
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messageCount, pendingLength, containerRef]);

  if (active === null) return null;
  const last = active.messages[active.messages.length - 1];
  const canRegenerate = !streaming && last?.role === "assistant";

  async function retry() {
    setActionError(null);
    const result = await regenerate();
    if (!result.ok) setActionError(result.reason);
  }

  return (
    <div
      ref={containerRef}
      className="messages"
      role="log"
      aria-live="off"
      aria-busy={streaming}
      aria-label="Conversation messages"
    >
      {active.messages.length === 0 && !showPending && (
        <EmptyState icon={MessageSquare} title="No messages yet">
          Type below to start. Replies come only from the loaded model.
        </EmptyState>
      )}
      {active.messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
      {showPending && pending !== null && <PendingItem pending={pending} onRetry={() => void retry()} />}
      {canRegenerate && (
        <div className="regenerate-row">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void retry()}>
            Regenerate reply
          </button>
        </div>
      )}
      {actionError !== null && (
        <Notice tone="danger" title="Could not regenerate">
          {actionError}
        </Notice>
      )}
    </div>
  );
}
