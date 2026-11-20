import { useEffect, useId, useState } from "react";
import { Play } from "lucide-react";
import type { ModelEntry } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { engineStore, loadModel } from "../../state/engine";
import { libraryStore, refreshLibrary } from "../../state/library";
import { openModelImport, openModels } from "../../state/ui";
import { LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { LoadFailureCard } from "../models/LoadFailureCard";
import { LoadProgressBar } from "../models/LoadProgressBar";
import { TuneSpeed } from "../models/TuneSpeed";
import { DiscoveredModels } from "./DiscoveredModels";

const LAST_KEY = "raptor.lastLoadedModel";

function lastUsed(): string | null {
  try {
    return window.localStorage.getItem(LAST_KEY);
  } catch {
    return null;
  }
}

function remember(id: string): void {
  try {
    window.localStorage.setItem(LAST_KEY, id);
  } catch {
    // Not saved; the list is just not reordered next time.
  }
}

/** The last model that was loaded first, then the rest by name. Pure, for testing. */
export function orderModels(models: ModelEntry[], last: string | null): ModelEntry[] {
  return [...models].sort((a, b) => {
    if (a.id === last) return -1;
    if (b.id === last) return 1;
    return a.file_name.localeCompare(b.file_name);
  });
}

/**
 * One compact row to start chatting without leaving the page: pick an installed model (the last one
 * used is preselected) and load it. Loading uses the same path as the Models page: the app picks the
 * settings, verifies GPU use, and shows the stage progress. A refused load keeps its explanation.
 */
export function QuickLoadModels() {
  const models = useStore(libraryStore, (s) => s.models);
  const libraryError = useStore(libraryStore, (s) => s.error);
  const busy = useStore(engineStore, (s) => s.busy);
  const [choice, setChoice] = useState<string | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const selectId = useId();

  useEffect(() => {
    if (models === null) void refreshLibrary();
  }, [models]);

  async function load(id: string) {
    setError(null);
    try {
      await loadModel(id);
      if (engineStore.get().loaded?.modelId === id) remember(id);
    } catch (e) {
      setError(e);
    }
  }

  if (models === null) {
    return libraryError === null ? <LoadingState label="Reading the model library…" /> : (
      <Notice tone="warn" title="Model library unavailable" actions={
        <button type="button" className="btn btn-sm" onClick={() => void refreshLibrary()}>Try again</button>
      }>
        {libraryError}
      </Notice>
    );
  }
  if (models.length === 0) {
    return (
      <div className="quick-load">
        <p className="small">No models are installed yet.</p>
        <div className="button-row">
          <button type="button" className="btn btn-primary" onClick={() => openModels("discover")}>
            Browse models
          </button>
          <button type="button" className="btn" onClick={openModelImport}>Add local models</button>
        </div>
        <DiscoveredModels />
      </div>
    );
  }

  const ordered = orderModels(models, lastUsed());
  const selected = ordered.find((m) => m.id === choice) ?? ordered[0];

  return (
    <div className="quick-load">
      <div className="quick-load-bar">
        <label htmlFor={selectId} className="sr-only">
          Model to load
        </label>
        <select
          id={selectId}
          value={selected.id}
          disabled={busy !== null || optimizing}
          onChange={(e) => setChoice(e.target.value)}
        >
          {ordered.map((m) => (
            <option key={m.id} value={m.id}>
              {m.file_name} ({formatBytes(m.size_bytes)})
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy !== null || optimizing}
          onClick={() => void load(selected.id)}
        >
          <Play size={14} aria-hidden="true" /> {busy === "loading" ? "Loading..." : "Load model"}
        </button>
      </div>
      <LoadProgressBar active={busy === "loading"} />
      {error !== null && <LoadFailureCard error={error} />}
      <details className="model-options">
        <summary>Model options</summary>
        <TuneSpeed key={selected.id} entry={selected} disabled={busy !== null} onRunningChange={setOptimizing} />
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => openModels("library")}>
          Manage models
        </button>
      </details>
      <DiscoveredModels />
    </div>
  );
}
