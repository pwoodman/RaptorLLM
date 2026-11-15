import { useId, useState, type FormEvent } from "react";
import { Brain, Trash2 } from "lucide-react";
import {
  contextForgetProjectMemory,
  contextProjectMemory,
  contextRememberProject,
  type MemoryKind,
  type ProjectMemoryEntry,
} from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { messageOf } from "../../lib/errors";
import { ErrorState, LoadingState } from "../AsyncState";

export const KIND_LABEL: Record<MemoryKind, string> = {
  decision: "Decision",
  convention: "Convention",
  summary: "Summary",
  completed_work: "Completed work",
  failed_attempt: "Failed attempt",
};

/** Decisions and conventions are always sent to coding agents; the rest can be trimmed for space. */
export const PROTECTED_KINDS: MemoryKind[] = ["decision", "convention"];

/** Newest first, with stale entries after the current ones. Pure, for testing. */
export function orderMemory(entries: ProjectMemoryEntry[]): ProjectMemoryEntry[] {
  return [...entries].sort((a, b) => Number(a.stale) - Number(b.stale) || b.seq - a.seq);
}

/**
 * What the context engine remembers about one workspace beyond a single chat: decisions,
 * conventions and so on. Each entry shows whether it is protected and whether the files it cites
 * have changed (stale). Entries can be added with the files they depend on, and forgotten.
 */
export function ProjectMemoryView({ workspacePath }: { workspacePath: string }) {
  const memory = useAsyncResource(() => contextProjectMemory(workspacePath), [workspacePath]);
  const kindId = useId();
  const textId = useId();
  const pathsId = useId();
  const [kind, setKind] = useState<MemoryKind>("decision");
  const [text, setText] = useState("");
  const [paths, setPaths] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function add(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const cited = paths
        .split(/[\n,]/)
        .map((p) => p.trim())
        .filter((p) => p !== "");
      await contextRememberProject(workspacePath, kind, text, cited);
      setText("");
      setPaths("");
      memory.reload();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  async function forget(id: string) {
    setError(null);
    try {
      await contextForgetProjectMemory(workspacePath, id);
      memory.reload();
    } catch (e) {
      setError(messageOf(e));
    }
  }

  const entries = memory.data ? orderMemory(memory.data) : [];
  return (
    <details className="project-memory">
      <summary>
        <Brain size={13} aria-hidden="true" /> Project memory{memory.data ? ` (${memory.data.length})` : ""}
      </summary>
      {memory.loading && memory.data === null && <LoadingState label="Reading project memory..." />}
      {memory.error !== null && <ErrorState error={memory.error} onRetry={memory.reload} />}
      {memory.data !== null && entries.length === 0 && (
        <p className="muted small">Nothing remembered yet. Decisions and conventions added here are always sent to coding agents.</p>
      )}
      <ul className="memory-list">
        {entries.map((e) => (
          <li key={e.id} className={`memory-entry${e.stale ? " stale" : ""}`}>
            <div className="memory-head">
              <strong>{KIND_LABEL[e.kind]}</strong>
              {PROTECTED_KINDS.includes(e.kind) && <span className="muted small">always included</span>}
              {e.stale && <span className="muted small">stale: {e.stale_reason ?? "a cited file changed"}</span>}
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => void forget(e.id)}
                aria-label={`Forget ${KIND_LABEL[e.kind]}: ${e.text.slice(0, 40)}`}
              >
                <Trash2 size={13} aria-hidden="true" /> Forget
              </button>
            </div>
            <p className="pre-wrap">{e.text}</p>
            {e.sources.length > 0 && (
              <p className="muted small break">Cites: {e.sources.map((s) => s.path).join(", ")}</p>
            )}
          </li>
        ))}
      </ul>
      {error !== null && <p className="small error-text">{error}</p>}
      <form className="stack" onSubmit={(e) => void add(e)}>
        <div className="field">
          <label htmlFor={kindId} className="field-label">
            Remember
          </label>
          <select id={kindId} value={kind} onChange={(e) => setKind(e.target.value as MemoryKind)}>
            {(Object.keys(KIND_LABEL) as MemoryKind[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABEL[k]}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={textId} className="field-label">
            Text
          </label>
          <textarea id={textId} rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor={pathsId} className="field-label">
            Files it depends on (optional, relative paths, one per line)
          </label>
          <textarea id={pathsId} rows={2} value={paths} onChange={(e) => setPaths(e.target.value)} />
        </div>
        <div>
          <button type="submit" className="btn btn-sm" disabled={busy || text.trim() === ""}>
            Remember
          </button>
        </div>
      </form>
    </details>
  );
}
