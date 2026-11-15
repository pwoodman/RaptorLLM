import { useEffect, useId, useState, type FormEvent } from "react";
import { Activity, FolderPlus, RefreshCw, Trash2 } from "lucide-react";
import {
  api,
  settingsGet,
  contextClearEmbeddingModel,
  contextSemanticStatus,
  contextSetEmbeddingModel,
  type SemanticView,
  type SettingsView,
  type SettingsViewV2,
  type WorkspaceInfo,
} from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import { formatCount } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { libraryStore, refreshLibrary } from "../../state/library";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { ProjectMemoryView } from "./ProjectMemoryView";

interface ContextPanelProps {
  settings: SettingsViewV2;
  onChange: (view: SettingsView) => void;
}

export function ContextPanel({ settings, onChange }: ContextPanelProps) {
  const [target, setTarget] = useState(String(settings.context_target));
  const [workspacePath, setWorkspacePath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const targetId = useId();
  const workspaceId = useId();
  const embeddingId = useId();
  const installed = useStore(libraryStore, (s) => s.models);
  const libraryError = useStore(libraryStore, (s) => s.error);

  useEffect(() => {
    void refreshLibrary();
  }, []);

  const targetNumber = Number(target);
  const targetValid =
    Number.isInteger(targetNumber) &&
    targetNumber >= settings.min_context_target &&
    targetNumber <= settings.max_context_target;

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  async function applyTarget() {
    await run(async () => {
      onChange(await api.settings.setContextTarget(targetNumber));
    });
  }

  /** Sets or clears the embedding model. Semantic retrieval then follows whichever model is loaded. */
  async function chooseEmbedding(modelId: string) {
    await run(async () => {
      if (modelId === "") {
        await contextClearEmbeddingModel();
        const cleared: SettingsViewV2 = { ...settings, embedding_model: null };
        onChange(cleared);
      } else {
        const id = await contextSetEmbeddingModel(modelId);
        const set: SettingsViewV2 = { ...settings, embedding_model: id };
        onChange(set);
      }
    });
  }

  async function browseWorkspace() {
    try {
      const selected = await api.dialog.pickFolder();
      if (selected) setWorkspacePath(selected);
    } catch {
      // dialog cancelled or failed
    }
  }

  async function addWorkspace(event: FormEvent) {
    event.preventDefault();
    const path = workspacePath.trim();
    if (path === "") return;
    await run(async () => {
      await api.settings.registerWorkspace(path);
      // registerWorkspace returns only the new entry, so the full view is reloaded.
      onChange(await settingsGet());
      setWorkspacePath("");
    });
  }

  return (
    <Panel
      id="context"
      title="Context engine"
      icon={<Activity size={18} aria-hidden="true" />}
      description="Registered workspaces are the only folders the gateway can serve context from."
    >
      <div className="field-row">
        <div className="field">
          <label htmlFor={targetId} className="field-label">
            Context target (tokens)
          </label>
          <input
            id={targetId}
            type="number"
            inputMode="numeric"
            min={settings.min_context_target}
            max={settings.max_context_target}
            value={target}
            aria-invalid={!targetValid}
            aria-describedby={`${targetId}-hint`}
            onChange={(e) => setTarget(e.target.value)}
          />
          <span id={`${targetId}-hint`} className={targetValid ? "muted small" : "warn-text small"}>
            {targetValid
              ? `Between ${formatCount(settings.min_context_target)} and ${formatCount(settings.max_context_target)}.`
              : `Enter a whole number from ${formatCount(settings.min_context_target)} to ${formatCount(settings.max_context_target)}.`}
          </span>
        </div>
        <button
          type="button"
          className="btn"
          disabled={busy || !targetValid || targetNumber === settings.context_target}
          onClick={() => void applyTarget()}
        >
          Apply target
        </button>
      </div>

      <h3 className="subsection-title">Embedding model</h3>
      <div className="field">
        <label htmlFor={embeddingId} className="field-label">
          Model for semantic retrieval
        </label>
        <select
          id={embeddingId}
          value={settings.embedding_model ?? ""}
          disabled={busy || installed === null}
          aria-describedby={`${embeddingId}-hint`}
          onChange={(e) => void chooseEmbedding(e.target.value)}
        >
          <option value="">None: semantic retrieval off</option>
          {(installed ?? []).map((m) => (
            <option key={m.id} value={m.id}>
              {m.file_name}
            </option>
          ))}
          {settings.embedding_model !== null &&
            !(installed ?? []).some((m) => m.id === settings.embedding_model) && (
              <option value={settings.embedding_model}>{settings.embedding_model} (not installed)</option>
            )}
        </select>
        <span id={`${embeddingId}-hint`} className="muted small">
          Semantic search uses this model only while it is the loaded model, and its server must support embeddings.
          Lexical and graph retrieval always run.
        </span>
        {installed === null && libraryError === null && <LoadingState label="Reading installed models..." />}
        {libraryError !== null && (
          <Notice tone="warn" title="Installed models unavailable">
            {libraryError}
          </Notice>
        )}
      </div>

      <h3 className="subsection-title">Workspaces</h3>
      {settings.workspaces.length === 0 && (
        <p className="muted small">No workspaces registered. Add a project folder to index it.</p>
      )}
      <ul className="workspace-list">
        {settings.workspaces.map((ws) => (
          <WorkspaceRow
            key={ws.id}
            workspace={ws}
            embeddingModel={settings.embedding_model}
            onRemoved={(view) => onChange(view)}
            onError={(message) => setError(message)}
          />
        ))}
      </ul>

      <form className="scan-form" onSubmit={(e) => void addWorkspace(e)}>
        <div className="field grow">
          <label htmlFor={workspaceId} className="field-label">
            Add a workspace folder
          </label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              id={workspaceId}
              value={workspacePath}
              placeholder="Full path to a project folder"
              onChange={(e) => setWorkspacePath(e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="button" className="btn btn-ghost" onClick={() => void browseWorkspace()}>
              Browse...
            </button>
          </div>
        </div>
        <button type="submit" className="btn" disabled={busy || workspacePath.trim() === ""}>
          <FolderPlus size={14} aria-hidden="true" /> Add
        </button>
      </form>

      {error !== null && (
        <Notice tone="danger" title="Context engine action failed">
          {error}
        </Notice>
      )}
    </Panel>
  );
}

function WorkspaceRow({
  workspace,
  embeddingModel,
  onRemoved,
  onError,
}: {
  workspace: WorkspaceInfo;
  embeddingModel: string | null;
  onRemoved: (view: SettingsView) => void;
  onError: (message: string) => void;
}) {
  const index = useAsyncResource(() => api.context.indexStatus(workspace.path), [workspace.path]);
  const semantic = useAsyncResource(
    () => contextSemanticStatus(workspace.path),
    [workspace.path, embeddingModel],
  );
  const [indexing, setIndexing] = useState(false);

  async function reindex() {
    setIndexing(true);
    try {
      await api.context.index(workspace.path);
      index.reload();
    } catch (e) {
      onError(messageOf(e));
    } finally {
      setIndexing(false);
    }
  }

  async function remove() {
    try {
      onRemoved(await api.settings.removeWorkspace(workspace.id));
    } catch (e) {
      onError(messageOf(e));
    }
  }

  const status = index.data;
  return (
    <li className="workspace">
      <div className="workspace-head">
        <span className="break">{workspace.path}</span>
        <div className="button-row">
          <button type="button" className="btn btn-sm" disabled={indexing} onClick={() => void reindex()}>
            <RefreshCw size={13} aria-hidden="true" /> {indexing ? "Indexing..." : "Re-index"}
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void remove()} aria-label={`Remove workspace ${workspace.path}`}>
            <Trash2 size={13} aria-hidden="true" /> Remove
          </button>
        </div>
      </div>
      <ProjectMemoryView workspacePath={workspace.path} />
      {index.loading && <LoadingState label="Reading index status..." />}
      {index.error !== null && <ErrorState error={index.error} onRetry={index.reload} />}
      {status && (
        <dl className="facts compact">
          <div>
            <dt>Files</dt>
            <dd>{formatCount(status.files)} indexed, {formatCount(status.excluded)} excluded</dd>
          </div>
          <div>
            <dt>Symbols</dt>
            <dd>{formatCount(status.symbols)}</dd>
          </div>
          <div>
            <dt>Graph</dt>
            <dd>
              {formatCount(status.graph_files)} files, {formatCount(status.graph_edges)} edges
            </dd>
          </div>
          <div>
            <dt>Git</dt>
            <dd>
              {status.git_repo
                ? `${formatCount(status.changed_files)} changed files`
                : "Not a git repository"}
            </dd>
          </div>
        </dl>
      )}
      {status?.git_warning && (
        <Notice tone="warn" title="Git warning">
          {status.git_warning}
        </Notice>
      )}
      {semantic.loading && semantic.data === null && <LoadingState label="Reading semantic retrieval status..." />}
      {semantic.error !== null && <ErrorState error={semantic.error} onRetry={semantic.reload} />}
      {semantic.data && <SemanticLine view={semantic.data} />}
    </li>
  );
}

/** Says whether semantic retrieval is active for this workspace, and the reason when it is not. */
export function SemanticLine({ view }: { view: SemanticView }) {
  if (view.status.enabled) {
    return (
      <p className="small">
        <span className="badge tone-ok">Semantic on</span> Using {view.configured_model ?? "the embedding model"}:{" "}
        {formatCount(view.status.vectors)} vectors, {formatCount(view.status.chunks_without_vector)} chunks without a
        vector.
      </p>
    );
  }
  const reason =
    view.reason ?? (view.configured_model === null ? "no embedding model is set." : "the embedding engine is not active yet.");
  // Off because no embedding model is loaded is a normal state, so it stays a quiet line.
  // Only a real failure earns a warning.
  if (!view.status.last_error) {
    return (
      <p className="small muted">
        <span className="badge">Semantic off</span> {reason} Lexical and graph retrieval still run.
        {view.status.store_notice && <> {view.status.store_notice}</>}
      </p>
    );
  }
  return (
    <Notice tone="warn" title="Semantic retrieval failed">
      <p>Reason: {reason}</p>
      <p className="small">Last error: {view.status.last_error}</p>
      {view.status.store_notice && <p className="small">{view.status.store_notice}</p>}
      <p className="small">Lexical and graph retrieval still run.</p>
    </Notice>
  );
}
