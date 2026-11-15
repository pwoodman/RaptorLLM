import { useId, useState } from "react";
import { useStore } from "../../hooks/useStore";
import { isGenerating, chatStore } from "../../state/chat";
import { engineStore, loadModel, unloadModel } from "../../state/engine";
import { libraryStore } from "../../state/library";
import { uiStore } from "../../state/ui";
import { LoadFailureCard } from "../models/LoadFailureCard";
import { Notice } from "../Notice";

const UNKNOWN_OPTION = "__loaded_unknown__";

/** Lists installed models only. Choosing one loads it; choosing "No model loaded" unloads it. */
export function ModelPicker() {
  const models = useStore(libraryStore, (s) => s.models);
  const libraryError = useStore(libraryStore, (s) => s.error);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const status = useStore(engineStore, (s) => s.status);
  const busy = useStore(engineStore, (s) => s.busy);
  const generating = useStore(chatStore, (s) => isGenerating(s.pending));
  const [error, setError] = useState<unknown>(null);
  const selectId = useId();

  const serverReady = status?.server.state.state === "ready";
  const unknownLoaded = serverReady && loaded === null;
  const loadedId = loaded?.modelId ?? null;
  const loadedIsInstalled = loadedId !== null && (models ?? []).some((m) => m.id === loadedId);
  const value = loadedId ?? (unknownLoaded ? UNKNOWN_OPTION : "");

  async function choose(id: string) {
    setError(null);
    try {
      if (id === "") {
        await unloadModel();
      } else {
        await loadModel(id);
      }
    } catch (e) {
      setError(e);
    }
  }

  return (
    <div className="model-picker">
      <label htmlFor={selectId} className="field-label">
        Model
      </label>
      <select
        id={selectId}
        value={value}
        disabled={models === null || busy !== null || generating}
        onChange={(e) => void choose(e.target.value)}
      >
        <option value="">No model loaded</option>
        {unknownLoaded && (
          <option value={UNKNOWN_OPTION} disabled>
            Loaded model (not reported by the backend)
          </option>
        )}
        {loadedId !== null && !loadedIsInstalled && <option value={loadedId}>{loadedId}</option>}
        {(models ?? []).map((m) => (
          <option key={m.id} value={m.id}>
            {m.file_name}
          </option>
        ))}
      </select>
      {busy === "loading" && <span className="muted small">Loading model...</span>}
      {busy === "unloading" && <span className="muted small">Unloading...</span>}
      {libraryError !== null && (
        <Notice tone="warn" title="Model list unavailable">
          {libraryError}
        </Notice>
      )}
      {error !== null && <LoadFailureCard error={error} />}
      {models !== null && models.length === 0 && (
        <button type="button" className="btn btn-sm" onClick={() => uiStore.set({ page: "models" })}>
          Install a model
        </button>
      )}
    </div>
  );
}
