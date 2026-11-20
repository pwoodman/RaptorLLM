import { useId, useState } from "react";
import { MemoryStick } from "lucide-react";
import { settingsGet, settingsSetAllowRamKv, type SettingsViewV2 } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { messageOf } from "../../lib/errors";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";

/**
 * Off by default. When on, a model whose context does not fit in VRAM at any quantization
 * is loaded with its KV cache in system RAM (`--no-kv-offload`). Generation is slower because
 * the GPU reads the cache over the bus on every token. It changes nothing that is already loaded.
 */
export function RamKvToggle() {
  const settings = useAsyncResource<SettingsViewV2>(() => settingsGet(), []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();

  async function set(enabled: boolean) {
    setBusy(true);
    setError(null);
    try {
      const next = await settingsSetAllowRamKv(enabled);
      settings.setData(next);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  if (settings.loading && settings.data === null) return <LoadingState label="Reading the KV-cache setting..." />;
  if (settings.error !== null) return <ErrorState error={settings.error} onRetry={settings.reload} />;
  const enabled = settings.data?.allow_ram_kv ?? false;

  return (
    <div className="cpu-fallback">
      <div className="toggle-row">
        <div>
          <label htmlFor={id} className="field-label">
            Keep the KV cache in system RAM when VRAM is not enough
          </label>
          <span className="muted small block">
            Off (default): a load that does not fit in VRAM at any context is refused with the reason. On: the model
            loads with <code>--no-kv-offload</code>, all layers stay on the GPU and the KV cache lives in system RAM.
            Generation is slower; use it when a long context matters more than speed.
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
        <Notice tone="warn" title="KV cache in RAM is on">
          <span className="cpu-fallback-warning">
            <MemoryStick size={14} aria-hidden="true" /> Models whose context does not fit the GPU are loaded with a
            system-RAM KV cache. Expect slower generation, especially at long contexts.
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
