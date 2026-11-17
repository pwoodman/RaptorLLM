import { useEffect, useState } from "react";
import { api, type BackendPref, type BackendView } from "../../api";
import { messageOf } from "../../lib/errors";

const LABELS: Record<BackendPref, string> = {
  auto: "Auto",
  rocm: "ROCm",
  cuda: "NVIDIA CUDA",
  sycl: "Intel SYCL",
  vulkan: "Vulkan",
};

/** What a load would use now, in a sentence. Pure, for testing. */
export function backendSummary(v: BackendView): string {
  if (v.preference === "auto" && v.spills && !v.vulkan_installed && v.effective === "default") {
    return "This sparse model is larger than VRAM, but the Vulkan runtime is not installed; Auto keeps the normal runtime.";
  }
  switch (v.effective) {
    case "vulkan":
      return "Loads on the Vulkan runtime.";
    case "rocm":
      return "Loads on the ROCm runtime.";
    case "cuda":
      return "Loads on the NVIDIA CUDA runtime.";
    case "sycl":
      return "Loads on the Intel SYCL runtime.";
    case "unavailable":
      return v.note ?? "The chosen runtime is not installed.";
    default:
      return "Loads on the normal runtime.";
  }
}

/**
 * Picks the llama.cpp backend for the next load of one model. Auto uses the installed native
 * runtime for the detected GPU; its measured Vulkan rule applies only to large AMD sparse-MoE
 * models. A forced backend that is not installed is shown here and refused at load.
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
              {b === "cuda" && !view.cuda_installed ? " (not installed)" : ""}
              {b === "sycl" && !view.sycl_installed ? " (not installed)" : ""}
            </option>
          ))}
        </select>
      </label>
      <span className={`muted small block${view.effective === "unavailable" ? " error-text" : ""}`}>
        {backendSummary(view)}
      </span>
      {view.preference === "auto" && view.note !== null && <span className="muted small block">{view.note}</span>}
      <details className="small muted">
        <summary>Which is faster?</summary>
        {view.measured}
      </details>
      {error !== null && <span className="small error-text block">{error}</span>}
    </div>
  );
}
