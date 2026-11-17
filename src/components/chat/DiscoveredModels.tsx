import { useEffect, useState } from "react";
import { api } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { localModelsStore, syncLocalModels } from "../../state/localModels";

export function baseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** Automatic discovery status, shared across pages. No model-by-model registration step. */
export function DiscoveredModels({ showEmpty = false }: { showEmpty?: boolean } = {}) {
  const state = useStore(localModelsStore, (s) => s);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  useEffect(() => { void syncLocalModels(); }, []);
  useEffect(() => {
    setProgress(null);
    if (state.phase !== "indexing") return;
    let current = true;
    const timer = window.setInterval(() => {
      api.models.importProgress().then((p) => { if (current) setProgress(p); }, () => undefined);
    }, 500);
    return () => { current = false; window.clearInterval(timer); };
  }, [state.phase, state.current]);

  if (!showEmpty && state.phase === "idle" && state.errors.length === 0 && state.added === 0) return null;
  return (
    <section className="discovered" aria-label="Automatic model discovery">
      <div role="status" aria-live="polite">
        {state.phase === "scanning" && <p className="small muted">Checking your local model folders…</p>}
        {state.phase === "indexing" && <>
          <p className="small">Making local models available: {state.done} of {state.total} checked.</p>
          <p className="small muted">{state.current ? baseName(state.current) : "Reading model files…"}</p>
          <progress aria-label="Automatic model verification" max={progress?.total || undefined} value={progress?.total ? progress.done : undefined} />
          <p className="small muted">{progress?.total ? `${formatBytes(progress.done)} of ${formatBytes(progress.total)} verified. ` : ""}Models appear in the picker as they’re ready. Large files can take a few minutes the first time.</p>
        </>}
        {state.phase === "idle" && state.added > 0 && <p className="small">{state.added} local {state.added === 1 ? "model is" : "models are"} ready to use.</p>}
      </div>
      {showEmpty && <p className="small muted">Models in your usual folders appear automatically. Files stay where they are; Raptor does not move or copy them.</p>}
      {state.errors.length > 0 && <details>
        <summary className="small">Some files could not be made available ({state.errors.length})</summary>
        <ul className="small muted">{state.errors.map((error) => <li key={error}>{error}</li>)}</ul>
      </details>}
      {(showEmpty || state.errors.length > 0) && <button type="button" className="btn btn-sm btn-ghost" disabled={state.phase !== "idle"} onClick={() => void syncLocalModels(true)}>Check folders again</button>}
    </section>
  );
}
