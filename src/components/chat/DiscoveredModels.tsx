import { useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { api, type DiscoveredFolder } from "../../api";
import { formatBytes } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { importModel } from "../../state/library";

/** File name of a path from either kind of separator. */
export function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/**
 * Models that other apps (LM Studio, Jan, ...) or an earlier scan already have on this computer and
 * that are not in the library yet. Adding one references the file where it is: nothing is copied
 * and the original is never touched. Adding hashes the file, so it shows progress.
 */
export function DiscoveredModels() {
  const [folders, setFolders] = useState<DiscoveredFolder[] | null>(null);
  const [adding, setAdding] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function find() {
    try {
      setFolders(await api.models.discover());
    } catch {
      setFolders([]);
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

  const items = (folders ?? []).flatMap((f) => f.items.map((i) => ({ folder: f, item: i })));
  if (items.length === 0) return null;

  return (
    <details className="discovered">
      <summary className="small muted">
        {items.length} more {items.length === 1 ? "model" : "models"} found on this computer
      </summary>
      <p className="small muted">Adding one references the file where it is; nothing is copied.</p>
      <ul className="quick-load-list" aria-label="Models found on this computer">
        {items.map(({ folder, item }) => {
          const path = item.path;
          return (
            <li key={path} className="quick-load-row">
              <div className="quick-load-name">
                <strong>{baseName(path)}</strong>
                <span className="muted small">
                  {formatBytes(item.size_bytes)} · {folder.source}
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
