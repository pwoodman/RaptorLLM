import { useState } from "react";
import { Bot, Play, Square, Trash2 } from "lucide-react";
import type { ContextOption, ModelEntry } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes, formatCount } from "../../lib/format";
import { deleteModel, libraryStore, refreshLibrary } from "../../state/library";
import { engineStore, loadModel, unloadModel } from "../../state/engine";
import { EmptyState, LoadingState } from "../AsyncState";
import { ConfirmDialog } from "../ConfirmDialog";
import { BackendChoice } from "./BackendChoice";
import { ContextChoice } from "./ContextChoice";
import { Notice } from "../Notice";
import { ImportPanel } from "./ImportPanel";
import { LoadFailureCard } from "./LoadFailureCard";
import { PresetsPanel } from "./PresetsPanel";
import { ContextMatrix } from "./ContextMatrix";
import { TuneSpeed } from "./TuneSpeed";

export function LibraryTab() {
  const models = useStore(libraryStore, (s) => s.models);
  const storage = useStore(libraryStore, (s) => s.storage);
  const libraryError = useStore(libraryStore, (s) => s.error);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const busy = useStore(engineStore, (s) => s.busy);
  const [error, setError] = useState<unknown>(null);
  const [deleteTarget, setDeleteTarget] = useState<ModelEntry | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [windows, setWindows] = useState<Record<string, ContextOption | null>>({});

  async function run(action: () => Promise<void>) {
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e);
    }
  }

  async function confirmDelete() {
    if (deleteTarget === null) return;
    setDeleting(true);
    await run(() => deleteModel(deleteTarget.id));
    setDeleting(false);
    setDeleteTarget(null);
  }

  return (
    <div className="library">
      {storage && (
        <dl className="facts storage-facts" aria-label="Storage usage">
          <div>
            <dt>Installed</dt>
            <dd>{formatCount(storage.models)} models</dd>
          </div>
          <div>
            <dt>In library folder</dt>
            <dd>{formatBytes(storage.managed_bytes)}</dd>
          </div>
          <div>
            <dt>Referenced outside library</dt>
            <dd>{formatBytes(storage.referenced_bytes)}</dd>
          </div>
        </dl>
      )}

      {libraryError !== null && (
        <Notice
          tone="danger"
          title="Library could not be read"
          actions={
            <button type="button" className="btn btn-sm" onClick={() => void refreshLibrary()}>
              Try again
            </button>
          }
        >
          {libraryError}
        </Notice>
      )}
      {error !== null && <LoadFailureCard error={error} />}

      {models === null && libraryError === null && <LoadingState label="Reading the library..." />}
      {models !== null && models.length === 0 && (
        <EmptyState icon={Bot} title="No models installed">
          Download one from Discover, or import a GGUF file below.
        </EmptyState>
      )}

      {models !== null && models.length > 0 && (
        <ul className="model-list" aria-label="Installed models">
          {models.map((entry) => {
            const isLoaded = loaded?.modelId === entry.id;
            return (
              <li key={entry.id} className="model-row">
                <div className="model-row-main">
                  <span className="model-name break">
                    {entry.file_name}
                    {isLoaded && <span className="badge tone-ok">Loaded</span>}
                  </span>
                  <span className="muted small">
                    {formatBytes(entry.size_bytes)} · {entry.gguf.quant_label ?? "Unknown quant"} ·{" "}
                    {entry.gguf.architecture ?? "unknown architecture"}
                    {" · "}
                    {entry.managed ? "In library" : "Referenced, not copied"}
                  </span>
                  {!entry.managed && (
                    <span className="muted small break">Location: {entry.location}</span>
                  )}
                  <BackendChoice modelId={entry.id} disabled={busy !== null} />
                  <ContextChoice
                    modelId={entry.id}
                    disabled={busy !== null}
                    value={windows[entry.id] ?? null}
                    onChange={(o) => setWindows((w) => ({ ...w, [entry.id]: o }))}
                  />
                  <TuneSpeed entry={entry} disabled={loaded !== null || busy !== null} />
                  <ContextMatrix modelId={entry.id} disabled={loaded !== null || busy !== null} />
                </div>
                <div className="model-row-actions">
                  {isLoaded ? (
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={busy !== null}
                      onClick={() => void run(unloadModel)}
                    >
                      <Square size={13} aria-hidden="true" /> Unload
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-sm btn-primary"
                      disabled={busy !== null}
                      onClick={() => {
                        const w = windows[entry.id];
                        void run(() =>
                          loadModel(
                            entry.id,
                            null,
                            w ? { ctx: w.ctx, cacheType: w.kv_cache, slots: w.slots, speculation: w.speculation } : null,
                          ),
                        );
                      }}
                    >
                      <Play size={13} aria-hidden="true" /> {busy === "loading" ? "Loading..." : "Load"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm icon-text-danger"
                    disabled={isLoaded}
                    title={isLoaded ? "Unload this model before deleting it" : `Delete ${entry.file_name}`}
                    onClick={() => setDeleteTarget(entry)}
                  >
                    <Trash2 size={13} aria-hidden="true" /> Delete
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <PresetsPanel />

      <ImportPanel />

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete this model?"
        confirmLabel="Delete"
        tone="danger"
        busy={deleting}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => void confirmDelete()}
      >
        {deleteTarget?.managed ? (
          <p>
            {deleteTarget.file_name} and its {formatBytes(deleteTarget.size_bytes)} file will be deleted from the
            library. This cannot be undone.
          </p>
        ) : (
          <p>
            {deleteTarget?.file_name} is removed from the library only. The file at its original location is not
            touched.
          </p>
        )}
      </ConfirmDialog>
    </div>
  );
}
