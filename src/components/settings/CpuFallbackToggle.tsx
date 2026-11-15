import { useId, useState } from "react";
import { AlertOctagon } from "lucide-react";
import { settingsGet, settingsSetAllowCpuFallback, type SettingsViewV2 } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { messageOf } from "../../lib/errors";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";

/**
 * Off by default. When on, a model whose GPU backend has no device loads on the CPU, and the load
 * says so. Turning it on changes nothing that is already running.
 */
export function CpuFallbackToggle() {
  const settings = useAsyncResource<SettingsViewV2>(() => settingsGet(), []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();

  async function set(enabled: boolean) {
    setBusy(true);
    setError(null);
    try {
      const next = await settingsSetAllowCpuFallback(enabled);
      settings.setData(next);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  if (settings.loading && settings.data === null) return <LoadingState label="Reading the CPU fallback setting..." />;
  if (settings.error !== null) return <ErrorState error={settings.error} onRetry={settings.reload} />;
  const enabled = settings.data?.allow_cpu_fallback ?? false;

  return (
    <div className="cpu-fallback">
      <div className="toggle-row">
        <div>
          <label htmlFor={id} className="field-label">
            Allow CPU fallback when the GPU has no device
          </label>
          <span className="muted small block">
            Off (default): a load that finds no GPU device is refused with the reason. On: the model runs on the CPU
            instead, and the app shows a banner until you unload it.
          </span>
        </div>
        <input
          id={id}
          type="checkbox"
          role="switch"
          className="switch"
          checked={enabled}
          disabled={busy}
          onChange={(e) => void set(e.target.checked)}
        />
      </div>
      {enabled && (
        <Notice tone="warn" title="CPU fallback is on">
          <span className="cpu-fallback-warning">
            <AlertOctagon size={14} aria-hidden="true" /> Models can run on the CPU without GPU acceleration. Each such
            load is marked as CPU fallback, and it is much slower.
          </span>
        </Notice>
      )}
      {error !== null && (
        <Notice tone="danger" title="Could not change the setting">
          {error}
        </Notice>
      )}
    </div>
  );
}
