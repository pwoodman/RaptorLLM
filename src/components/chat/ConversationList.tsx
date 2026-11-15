import { useEffect, useId, useState, type FormEvent } from "react";
import { Check, FileUp, MessageSquare, Pencil, Plus, Search, Trash2, X } from "lucide-react";
import { api, conversationsSearch, type ConversationSearch } from "../../api";
import { useStore } from "../../hooks/useStore";
import { TITLE_SEARCH_ID } from "../../hooks/useGlobalShortcuts";
import { MAX_IMPORT_BYTES } from "../../lib/conversationFile";
import { messageOf } from "../../lib/errors";
import { splitMatches } from "../../lib/snippet";
import {
  chatStore,
  createConversation,
  deleteConversation,
  refreshList,
  renameConversation,
  selectConversation,
} from "../../state/chat";
import { EmptyState, ErrorState, LoadingState } from "../AsyncState";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

/** Waiting this long after the last keystroke keeps the search from running on every letter. */
export const SEARCH_DEBOUNCE_MS = 250;
/** Matches requested per search. The backend also stops at its read budget and says so. */
export const SEARCH_LIMIT = 50;

type ContentSearch =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: ConversationSearch };

function Highlighted({ text, needle }: { text: string; needle: string }) {
  return (
    <>
      {splitMatches(text, needle).map((part, i) =>
        part.match ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
      )}
    </>
  );
}

export function ConversationList() {
  const list = useStore(chatStore, (s) => s.list);
  const listError = useStore(chatStore, (s) => s.listError);
  const problems = useStore(chatStore, (s) => s.listProblems);
  const activeId = useStore(chatStore, (s) => s.activeId);

  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [content, setContent] = useState<ContentSearch>({ status: "idle" });
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [actionError, setActionError] = useState<unknown>(null);
  const [importing, setImporting] = useState(false);
  const [importNotice, setImportNotice] = useState<string | null>(null);
  const importInputId = useId();

  const needle = query.trim();
  const titleNeedle = needle.toLowerCase();

  // Debounce the query, then search message content once per settled query. A result that
  // arrives after the query changed is ignored.
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(needle), SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [needle]);

  useEffect(() => {
    if (debounced === "") {
      setContent({ status: "idle" });
      return;
    }
    let current = true;
    setContent({ status: "loading" });
    conversationsSearch(debounced, SEARCH_LIMIT).then(
      (result) => {
        if (current) setContent({ status: "done", result });
      },
      (e: unknown) => {
        if (current) setContent({ status: "error", message: messageOf(e) });
      },
    );
    return () => {
      current = false;
    };
  }, [debounced]);

  const visible = (list ?? [])
    .filter((item) => item.title.toLowerCase().includes(titleNeedle))
    .sort((a, b) => b.updated_at - a.updated_at);
  const deleteTitle = list?.find((item) => item.id === deleteId)?.title ?? "this conversation";
  const contentPending = needle !== "" && (content.status === "loading" || debounced !== needle);

  async function run(action: () => Promise<void>) {
    setActionError(null);
    try {
      await action();
    } catch (e) {
      setActionError(e);
    }
  }

  async function commitRename(event: FormEvent, id: string) {
    event.preventDefault();
    const title = draftTitle;
    setRenamingId(null);
    await run(() => renameConversation(id, title));
  }

  /**
   * Imports a conversation JSON file. The file is read in the browser, because the picker only gives a
   * path and the backend has no command to read it. The backend stores the import under a new id.
   */
  async function importFile(file: File) {
    setImportNotice(null);
    setActionError(null);
    if (file.size > MAX_IMPORT_BYTES) {
      setActionError(new Error(`The file is larger than ${MAX_IMPORT_BYTES / (1024 * 1024)} MB.`));
      return;
    }
    setImporting(true);
    try {
      const text = await file.text();
      const conv = await api.conversations.import(text);
      await refreshList();
      await selectConversation(conv.id);
      setImportNotice(`Imported "${conv.title}".`);
    } catch (e) {
      setActionError(e);
    } finally {
      setImporting(false);
    }
  }

  async function confirmDelete() {
    if (deleteId === null) return;
    setDeleting(true);
    await run(() => deleteConversation(deleteId));
    setDeleting(false);
    setDeleteId(null);
  }

  return (
    <aside className="conv-list" aria-label="Conversations">
      <div className="conv-list-head">
        <h2>Conversations</h2>
        <button type="button" className="btn btn-primary btn-sm" onClick={() => void run(createConversation)}>
          <Plus size={15} aria-hidden="true" /> New chat
        </button>
      </div>

      <div className="conv-import">
        <label htmlFor={importInputId} className="btn btn-ghost btn-sm">
          <FileUp size={14} aria-hidden="true" /> {importing ? "Importing..." : "Import conversation (.json)"}
        </label>
        <input
          id={importInputId}
          type="file"
          accept=".json,application/json"
          className="sr-only"
          disabled={importing}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void importFile(file);
          }}
        />
      </div>
      {importNotice !== null && <p className="muted small" role="status">{importNotice}</p>}

      <label className="search-field">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Search conversation titles and messages</span>
        <input
          id={TITLE_SEARCH_ID}
          type="search"
          placeholder="Search titles and messages"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>

      {needle !== "" && (
        <section className="content-hits" aria-label="Matches in messages">
          <h3 className="subsection-title">Messages</h3>
          {contentPending && <LoadingState label="Searching messages..." />}
          {!contentPending && content.status === "error" && (
            <Notice tone="danger" title="Message search failed">
              {content.message}
            </Notice>
          )}
          {!contentPending && content.status === "done" && (
            <ContentResults result={content.result} needle={needle} onOpen={(id, messageId) =>
                void run(async () => {
                  chatStore.set({ focusMessageId: messageId });
                  await selectConversation(id);
                })
              } />
          )}
        </section>
      )}

      {problems.map((problem) => (
        <Notice key={problem} tone="warn">
          {problem}
        </Notice>
      ))}
      {actionError !== null && <ErrorState error={actionError} />}

      {list === null && listError === null && <LoadingState label="Loading conversations..." />}
      {listError !== null && (
        <Notice
          tone="danger"
          title="Could not load conversations"
          actions={
            <button type="button" className="btn btn-sm" onClick={() => void refreshList()}>
              Try again
            </button>
          }
        >
          {listError}
        </Notice>
      )}
      {list !== null && list.length === 0 && (
        <EmptyState icon={MessageSquare} title="No conversations yet">
          Start one to talk to your local model. Everything stays on this machine.
        </EmptyState>
      )}
      {needle !== "" && list !== null && list.length > 0 && visible.length === 0 && (
        <p className="muted small">No conversation titles match "{needle}".</p>
      )}

      <ul className="conv-items">
        {visible.map((item) => (
          <li key={item.id} className="conv-item">
            {renamingId === item.id ? (
              <form className="rename-form" onSubmit={(e) => void commitRename(e, item.id)}>
                <input
                  aria-label="Conversation title"
                  value={draftTitle}
                  autoFocus
                  maxLength={200}
                  onChange={(e) => setDraftTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") setRenamingId(null);
                  }}
                />
                <button type="submit" className="icon-btn" aria-label="Save title">
                  <Check size={15} aria-hidden="true" />
                </button>
                <button type="button" className="icon-btn" aria-label="Cancel rename" onClick={() => setRenamingId(null)}>
                  <X size={15} aria-hidden="true" />
                </button>
              </form>
            ) : (
              <>
                <button
                  type="button"
                  className="conv-select"
                  aria-current={item.id === activeId ? "true" : undefined}
                  onClick={() => {
                    if (item.id !== activeId) void run(() => selectConversation(item.id));
                  }}
                >
                  <span className="conv-title">{item.title}</span>
                  <span className="conv-meta">{item.message_count} messages</span>
                </button>
                <div className="conv-actions">
                  <button
                    type="button"
                    className="icon-btn"
                    aria-label={`Rename ${item.title}`}
                    onClick={() => {
                      setDraftTitle(item.title);
                      setRenamingId(item.id);
                    }}
                  >
                    <Pencil size={14} aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="icon-btn icon-btn-danger"
                    aria-label={`Delete ${item.title}`}
                    onClick={() => setDeleteId(item.id)}
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </div>
              </>
            )}
          </li>
        ))}
      </ul>

      <ConfirmDialog
        open={deleteId !== null}
        title="Delete conversation?"
        confirmLabel="Delete conversation"
        tone="danger"
        busy={deleting}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setDeleteId(null)}
      >
        <p>
          "{deleteTitle}" and its messages will be removed from this machine. This cannot be undone.
        </p>
      </ConfirmDialog>
    </aside>
  );
}

function ContentResults({
  result,
  needle,
  onOpen,
}: {
  result: ConversationSearch;
  needle: string;
  onOpen: (conversationId: string, messageId: string) => void;
}) {
  const count = result.hits.length;
  return (
    <>
      <p className="muted small" role="status">
        {count === 0
          ? `No messages contain "${needle}".`
          : `${count} ${count === 1 ? "match" : "matches"} in messages${result.truncated ? ". More exist: narrow the search." : "."}`}
      </p>
      {count > 0 && (
        <ul className="hit-list" aria-label="Message matches">
          {result.hits.map((hit) => (
            <li key={`${hit.conversation_id}-${hit.message_id}`}>
              <button type="button" className="conv-select hit" onClick={() => onOpen(hit.conversation_id, hit.message_id)}>
                <span className="conv-title">{hit.title}</span>
                <span className="conv-meta snippet">
                  <span className="muted">{hit.role === "user" ? "You" : hit.role === "assistant" ? "Model" : hit.role}: </span>
                  <Highlighted text={hit.snippet} needle={needle} />
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {result.problems.map((problem) => (
        <Notice key={problem} tone="warn">
          {problem}
        </Notice>
      ))}
    </>
  );
}
