import { useEffect, useState } from "react";
import { api, type BackendPref, type BackendView } from "../../api";
import { messageOf } from "../../lib/errors";

const LABELS: Record<BackendPref, string> = {
  auto: "Auto",
  rocm: "ROCm",
  vulkan: "Vulkan",
};

/** What a load would use now, in a sentence. Pure, for testing. */
export function backendSummary(v: BackendView): string {
  switch (v.effective) {
    case "vulkan":
      return "Loads on the Vulkan runtime.";
    case "rocm":
      return "Loads on the ROCm runtime.";
    case "unavailable":
      return v.note ?? "The chosen runtime is not installed.";
    default:
      return v.spills && !v.vulkan_installed
        ? "Loads on the normal runtime. The Vulkan runtime is not installed."
        : "Loads on the normal runtime.";
  }
}

/**
 * Picks the llama.cpp backend (ROCm or Vulkan) for the next load of one model. Auto uses ROCm, and
 * uses Vulkan for a large sparse Mixture-of-experts model that does not fit in VRAM when the Vulkan
 * runtime is installed. A chosen backend that is not installed is shown here and refused at load; it
 * never quietly runs on the other one. The measured numbers are quoted so the choice is informed.
 */
export function BackendChoice({ modelId, disabled }: { modelId: string; disabled: boolean }) {
  const [view, setView] = useState<BackendView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    setView(null);
    api.models.backend(modelId).then(
      (v) => {
        if (current) setView(v);
      },
      (e) => {
        if (current) setError(messageOf(e));
      },
    );
    return () => {
      current = false;
    };
  }, [modelId]);

  async function choose(backend: BackendPref) {
    setError(null);
    try {
      setView(await api.models.setBackend(modelId, backend));
    } catch (e) {
      setError(messageOf(e));
    }
  }

  if (view === null) {
    return error === null ? null : <span className="small error-text block">{error}</span>;
  }

  return (
    <div className="backend-choice">
      <label className="small">
        Backend{" "}
        <select
          value={view.preference}
          disabled={disabled}
          onChange={(e) => void choose(e.target.value as BackendPref)}
        >
          {(Object.keys(LABELS) as BackendPref[]).map((b) => (
            <option key={b} value={b}>
              {LABELS[b]}
              {b === "vulkan" && !view.vulkan_installed ? " (not installed)" : ""}
              {b === "rocm" && !view.rocm_installed ? " (not installed)" : ""}
            </option>
          ))}
        </select>
      </label>
      <span className={`muted small block${view.effective === "unavailable" ? " error-text" : ""}`}>
        {backendSummary(view)}
      </span>
      <details className="small muted">
        <summary>Which is faster?</summary>
        {view.measured}
      </details>
      {error !== null && <span className="small error-text block">{error}</span>}
    </div>
  );
}
