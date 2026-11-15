import { useEffect, useState } from "react";
import { FolderSearch, Plus } from "lucide-react";
import { api, type ScanItem } from "../../api";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { importModel } from "../../state/library";
import { ErrorState } from "../AsyncState";
import { Notice } from "../Notice";

/**
 * Adds models from disk. Pick a folder and every .gguf inside it, including nested folders, is found
 * and added in place: nothing is copied or moved, so the files stay where they are.
 */
export function ImportPanel() {
  const [folder, setFolder] = useState<string | null>(null);
  const [scan, setScan] = useState<ScanItem[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [adding, setAdding] = useState<{ done: number; total: number } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);

  const busy = scanning || adding !== null;

  // Adding hashes each file, which takes minutes for a large model. Show how far it is.
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

  async function scanDir(dir: string) {
    setFolder(dir);
    setScanning(true);
    setError(null);
    setNotice(null);
    try {
      setScan(await api.models.scanFolder(dir));
    } catch (e) {
      setScan(null);
      setError(e);
    } finally {
      setScanning(false);
    }
  }

  async function chooseFolder() {
    try {
      const selected = await api.dialog.pickFolder();
      if (selected) await scanDir(selected);
    } catch {
      // dialog cancelled or failed
    }
  }

  async function addFiles(paths: string[]) {
    setError(null);
    setNotice(null);
    let added = 0;
    const failures: string[] = [];
    for (const [i, path] of paths.entries()) {
      setAdding({ done: i, total: paths.length });
      try {
        await importModel(path, "reference");
        added += 1;
      } catch (e) {
        failures.push(`${path.split(/[\/]/).pop()}: ${messageOf(e)}`);
      }
    }
    setAdding(null);
    if (added > 0) setNotice(`Added ${added} ${added === 1 ? "model" : "models"}.`);
    if (failures.length > 0) setError(new Error(failures.join("\n")));
    if (folder !== null) await scanDir(folder);
  }

  async function addSingleFile() {
    try {
      const selected = await api.dialog.pickFile(["gguf"]);
      if (selected) await addFiles([selected]);
    } catch {
      // dialog cancelled or failed
    }
  }

  const fresh = (scan ?? []).filter((i) => i.status.status === "new");
  const known = (scan ?? []).filter((i) => i.status.status === "already_in_library").length;

  return (
    <section className="import-panel" aria-labelledby="import-heading">
      <h2 id="import-heading" className="section-title">
        Add models
      </h2>
      <p className="muted small">
        Choose the folder where your models live. Everything inside it, including subfolders, is found. Your files stay
        where they are.
      </p>

      <div className="import-actions">
        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void chooseFolder()}>
          <FolderSearch size={14} aria-hidden="true" /> {scanning ? "Looking..." : "Choose folder"}
        </button>
        <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => void addSingleFile()}>
          Add a single file
        </button>
      </div>

      {folder !== null && <p className="small break">Folder: {folder}</p>}

      {scan !== null && scan.length === 0 && <p className="muted small">No .gguf files were found in that folder.</p>}
      {scan !== null && scan.length > 0 && (
        <div className="stack">
          <p className="small">
            Found {scan.length} {scan.length === 1 ? "file" : "files"}
            {known > 0 ? `, ${known} already added` : ""}.
          </p>
          {fresh.length > 0 && (
            <div>
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy}
                onClick={() => void addFiles(fresh.map((i) => i.path))}
              >
                <Plus size={14} aria-hidden="true" />{" "}
                {adding !== null ? "Adding..." : `Add ${fresh.length} ${fresh.length === 1 ? "model" : "models"}`}
              </button>
            </div>
          )}
          <details>
            <summary className="small muted">Show files</summary>
            <ul className="plain-list">
              {scan.map((item) => (
                <li key={item.path} className="small break">
                  {item.path} · {formatBytes(item.size_bytes)}
                  {item.status.status === "already_in_library" && " · already added"}
                  {item.status.status === "invalid_gguf" && ` · not a valid GGUF (${item.status.reason})`}
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}

      {adding !== null && (
        <div role="status" className="install-progress">
          <progress
            aria-label="Add progress"
            max={progress && progress.total > 0 ? progress.total : undefined}
            value={progress && progress.total > 0 ? progress.done : undefined}
          />
          <p className="small muted">
            Adding {adding.done + 1} of {adding.total}.{" "}
            {progress && progress.total > 0
              ? `Checking the file: ${formatBytes(progress.done)} of ${formatBytes(progress.total)}. Large models take a few minutes.`
              : "Reading the file..."}
          </p>
        </div>
      )}

      {notice !== null && <Notice tone="success">{notice}</Notice>}
      {error !== null && <ErrorState error={error} />}
    </section>
  );
}
