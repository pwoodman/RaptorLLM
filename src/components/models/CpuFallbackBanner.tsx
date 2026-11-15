import { useState } from "react";
import { AlertOctagon } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import { engineStore, unloadModel } from "../../state/engine";
import { openSettings } from "../../state/ui";

/**
 * Shown on every page while the loaded model runs on the CPU because a GPU backend had no device
 * and the user allowed CPU fallback. It stays until the model is unloaded or a load reports otherwise.
 */
export function CpuFallbackBanner() {
  const report = useStore(engineStore, (s) => s.loadReport);
  const busy = useStore(engineStore, (s) => s.busy);
  const [error, setError] = useState<string | null>(null);

  if (report === null || report.cpu_fallback !== true) return null;

  async function unload() {
    setError(null);
    try {
      await unloadModel();
    } catch (e) {
      setError(messageOf(e));
    }
  }

  return (
    <section className="cpu-fallback-banner" role="alert" aria-label="CPU fallback active">
      <AlertOctagon size={20} aria-hidden="true" />
      <div>
        <strong>CPU fallback: this model is NOT running on the GPU.</strong>
        <p>
          The GPU backend had no device, and CPU fallback is allowed in Settings. Replies will be much slower than
          GPU inference. Unload the model, or turn CPU fallback off, to stop this.
        </p>
        {error !== null && <p className="small">Could not unload: {error}</p>}
      </div>
      <div className="cpu-fallback-actions">
        <button type="button" className="btn btn-sm" disabled={busy !== null} onClick={() => void unload()}>
          Unload model
        </button>
        <button type="button" className="btn btn-sm btn-ghost" onClick={() => openSettings("runtime")}>
          Runtime settings
        </button>
      </div>
    </section>
  );
}
