import { useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { api, type DiscoveredFolder } from "../../api";
import { formatBytes } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { importModel } from "../../state/library";
import { modelKind } from "../../lib/modelKind";

/** File name of a path from either kind of separator. */
export function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/**
 * Models that other apps (LM Studio, Jan, ...) or an earlier scan already have on this computer and
 * that are not in the library yet. Adding one references the file where it is: nothing is copied
 * and the original is never touched. Adding hashes the file, so it shows progress.
 */
export function DiscoveredModels({ showEmpty = false }: { showEmpty?: boolean } = {}) {
  const [folders, setFolders] = useState<DiscoveredFolder[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [adding, setAdding] = useState<string | null>(null);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function find() {
    setScanning(true);
    setError(null);
    try {
      setFolders(await api.models.discover());
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    void find();
  }, []);

  useEffect(() => {
    if (adding === null) {
      setProgress(null);
      return;
    }
    let current = true;
    const timer = setInterval(() => {
      api.models.importProgress().then(
        (p) => {
          if (current) setProgress(p);
        },
        () => undefined,
      );
    }, 500);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [adding]);

  async function add(path: string) {
    setError(null);
    setAdding(path);
    try {
      await importModel(path, "reference");
      await find();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setAdding(null);
    }
  }

  async function addAll() {
    setError(null);
    setBatch({ done: 0, total: items.length });
    const failures: string[] = [];
    let added = 0;
    for (const [index, { item }] of items.entries()) {
      setAdding(item.path);
      setBatch({ done: index, total: items.length });
      try {
        await importModel(item.path, "reference");
        added += 1;
      } catch (e) {
        failures.push(baseName(item.path) + ": " + messageOf(e));
      }
    }
    setAdding(null);
    setBatch(null);
    await find();
    if (failures.length > 0) {
      setError(
        (added > 0 ? "Added " + added + " models. " : "") +
          failures.length +
          " could not be added: " +
          failures.join("; "),
      );
    }
  }

  const items = (folders ?? []).flatMap((f) => f.items.map((i) => ({ folder: f, item: i })));
  if (items.length === 0 && !showEmpty) return null;

  return (
    <details className="discovered" open={showEmpty}>
      <summary className="small muted">
        {items.length > 0
          ? items.length + " existing " + (items.length === 1 ? "model" : "models") + " found on this computer"
          : "Find existing models on this computer"}
      </summary>
      <p className="small muted">
        Raptor checks its model and downloads folders plus common local AI folders. Adding a model references it where
        it is; the file is not moved or copied.
      </p>
      {items.length === 0 && (
        <div className="stack">
          {scanning && <p className="small muted">Checking model folders…</p>}
          {!scanning && folders !== null && error === null && (
            <p className="small muted">No unlisted GGUF models were found in the checked folders.</p>
          )}
          {error !== null && <p className="small error-text">{error}</p>}
          <button type="button" className="btn btn-sm" disabled={scanning} onClick={() => void find()}>
            {scanning ? "Checking…" : "Scan folders again"}
          </button>
        </div>
      )}
      {items.length > 0 && (
        <p className="small muted">
          Adding a model references it where it is; nothing is copied. Raptor verifies each file, so large models can
          take several minutes.
        </p>
      )}
      {items.length > 1 && (
        <div className="button-row">
          <button type="button" className="btn btn-primary" disabled={adding !== null} onClick={() => void addAll()}>
            {batch !== null
              ? "Adding " + Math.min(batch.done + 1, batch.total) + " of " + batch.total
              : "Add all " + items.length + " models"}
          </button>
        </div>
      )}
      <ul className="quick-load-list" aria-label="Models found on this computer">
        {items.map(({ folder, item }) => {
          const path = item.path;
          return (
            <li key={path} className="quick-load-row">
              <div className="quick-load-name">
                <strong>{baseName(path)}</strong>
                <span className="muted small">
                  {modelKind(item)} · {formatBytes(item.size_bytes)} · {folder.source}
                </span>
              </div>
              <button
                type="button"
                className="btn btn-sm"
                disabled={adding !== null}
                onClick={() => void add(path)}
              >
                <Plus size={13} aria-hidden="true" /> {adding === path ? "Adding..." : "Add"}
              </button>
            </li>
          );
        })}
      </ul>
      {adding !== null && (
        <div role="status" className="install-progress">
          <progress
            aria-label="Add progress"
            max={progress && progress.total > 0 ? progress.total : undefined}
            value={progress && progress.total > 0 ? progress.done : undefined}
          />
          <p className="small muted">
            {batch !== null ? "Adding model " + (batch.done + 1) + " of " + batch.total + ". " : ""}
            {progress && progress.total > 0
              ? `Checking the file (SHA-256): ${formatBytes(progress.done)} of ${formatBytes(progress.total)}. Large models take a few minutes.`
              : "Reading the file..."}
          </p>
        </div>
      )}
      {error !== null && <p className="small error-text">{error}</p>}
    </details>
  );
}
