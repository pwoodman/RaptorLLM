import { useId, useState, type DragEvent, type FormEvent } from "react";
import { Paperclip, Send, Square, X } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import {
  checkAttachmentFile,
  readAttachment,
  type Attachment,
} from "../../lib/attachments";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { chatStore, draftFor, isGenerating, sendMessage, stopGeneration, updateDraft } from "../../state/chat";
import { Notice } from "../Notice";
import { LongAnswerToggle } from "./LongAnswerToggle";
import { ThinkingEffort } from "./ThinkingEffort";

/** Reads the dropped or chosen files in order. Refused files are reported, the rest are kept. */
async function readFiles(files: File[], attached: number): Promise<{ added: Attachment[]; problems: string[] }> {
  const added: Attachment[] = [];
  const problems: string[] = [];
  for (const file of files) {
    const check = checkAttachmentFile(file, attached + added.length);
    if (!check.ok) {
      problems.push(check.reason);
      continue;
    }
    try {
      const result = await readAttachment(file);
      if (result.ok) added.push(result.attachment);
      else problems.push(result.reason);
    } catch (e) {
      problems.push(`${file.name} could not be read: ${messageOf(e)}`);
    }
  }
  return { added, problems };
}

export function Composer({ conversationId, blockedReason }: { conversationId: string; blockedReason: string | null }) {
  const generating = useStore(chatStore, (s) => isGenerating(s.pending));
  const sendingFromOtherChat = useStore(chatStore, (s) =>
    [...s.drafts].some(([id, draft]) => id !== conversationId && draft.sending),
  );
  const { text: draft, attachments, reading, sending } = useStore(chatStore, (s) => draftFor(conversationId, s));
  const [dragging, setDragging] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const textareaId = useId();
  const fileInputId = useId();
  const disabled = blockedReason !== null;

  async function addFiles(files: File[]) {
    const current = draftFor(conversationId);
    if (disabled || current.reading || current.sending || files.length === 0) return;
    updateDraft(conversationId, { reading: true });
    setProblems([]);
    try {
      const result = await readFiles(files, current.attachments.length);
      if (draftFor(conversationId).reading) {
        updateDraft(conversationId, { attachments: [...current.attachments, ...result.added] });
      }
      setProblems(result.problems);
    } finally {
      updateDraft(conversationId, { reading: false });
    }
  }

  function onDrop(event: DragEvent<HTMLFormElement>) {
    event.preventDefault();
    setDragging(false);
    void addFiles(Array.from(event.dataTransfer.files));
  }

  async function submit() {
    const current = draftFor(conversationId);
    if (disabled || current.sending || sendingFromOtherChat || generating || current.reading) return;
    if (current.text.trim() === "" && current.attachments.length === 0) return;
    updateDraft(conversationId, { sending: true });
    setError(null);
    try {
      const result = await sendMessage(current.text, current.attachments);
      if (result.ok) {
        updateDraft(conversationId, { text: "", attachments: [] });
        setProblems([]);
      } else {
        setError(result.reason);
      }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      updateDraft(conversationId, { sending: false });
    }
  }

  async function stop() {
    setError(null);
    try {
      const result = await stopGeneration();
      if (!result.ok) setError(result.reason);
    } catch (e) {
      setError(messageOf(e));
    }
  }

  const canSend = !disabled && !sending && !sendingFromOtherChat && !reading && (draft.trim() !== "" || attachments.length > 0);

  return (
    <form
      className={`composer ${dragging ? "composer-drop" : ""}`}
      onSubmit={(e: FormEvent) => {
        e.preventDefault();
        void submit();
      }}
      onDragOver={(e) => {
        if (!Array.from(e.dataTransfer.types).includes("Files")) return;
        e.preventDefault();
        if (!disabled && !reading && !sending) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
        setDragging(false);
      }}
      onDrop={onDrop}
      aria-label="Message composer"
    >
      {attachments.length > 0 && (
        <ul className="attach-list" aria-label="Attached files">
          {attachments.map((a, i) => (
            <li key={`${a.name}-${i}`} className="attach-chip">
              <span className="break">{a.name}</span>
              <span className="muted small">{formatBytes(a.size)}</span>
              <button
                type="button"
                className="icon-btn"
                aria-label={`Remove ${a.name}`}
                disabled={reading || sending}
                onClick={() => updateDraft(conversationId, { attachments: attachments.filter((_, j) => j !== i) })}
              >
                <X size={13} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}

      <label htmlFor={textareaId} className="sr-only">
        Message
      </label>
      <textarea
        id={textareaId}
        rows={2}
        value={draft}
        disabled={disabled}
        readOnly={sending}
        placeholder={
          disabled
            ? "Load a model to start chatting"
            : dragging
              ? "Drop text or code files here"
              : "Message your local model"
        }
        onChange={(e) => updateDraft(conversationId, { text: e.target.value })}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            void submit();
          }
        }}
      />

      <div className="composer-foot">
        <div className="composer-hints">
          <input
            id={fileInputId}
            type="file"
            multiple
            className="sr-only"
            accept=".txt,.md,.json,.csv,.yaml,.yml,.toml,.xml,.log,.ts,.tsx,.js,.jsx,.py,.rs,.go,.java,.c,.h,.cpp,.hpp,.cs,.rb,.php,.sh,.ps1,.sql,.html,.css,text/*"
            disabled={disabled || reading || sending}
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              void addFiles(files);
            }}
          />
          <label htmlFor={fileInputId} className="btn btn-ghost attach-button" aria-disabled={disabled || reading || sending}>
            <Paperclip size={14} aria-hidden="true" /> Attach text file
          </label>
          <ThinkingEffort />
          <LongAnswerToggle />
        </div>
        {generating ? (
          <button type="button" className="btn btn-danger" onClick={() => void stop()}>
            <Square size={14} aria-hidden="true" /> Stop
          </button>
        ) : (
          <button type="submit" className="btn btn-primary" disabled={!canSend}>
            <Send size={14} aria-hidden="true" /> {sending ? "Sending..." : "Send"}
          </button>
        )}
      </div>

      {reading && <p className="muted small" role="status">Reading files...</p>}
      {problems.map((problem) => (
        <Notice key={problem} tone="warn" title="File not attached">
          {problem}
        </Notice>
      ))}
      {error !== null && (
        <Notice tone="danger" title="Message not sent">
          {error}
        </Notice>
      )}
    </form>
  );
}
