import { useId, useState } from "react";
import { settingsGet, settingsSetParallelSlots, type SettingsViewV2 } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { messageOf } from "../../lib/errors";

const CHOICES = [1, 2, 3, 4, 6, 8];

/**
 * How many requests the inference server works on at once. Each slot keeps its own state and prompt
 * cache, which costs memory (Qwen3.8-27B: about 450 MiB more at 4 slots than at 1) but not speed, so
 * fewer slots leave more room for context. The default is 2. Applies at the next model load.
 */
export function ParallelSlots() {
  const settings = useAsyncResource<SettingsViewV2>(() => settingsGet(), []);
  const id = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function set(n: number) {
    setBusy(true);
    setError(null);
    try {
      settings.setData(await settingsSetParallelSlots(n));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  const value = settings.data?.parallel_slots ?? 2;
  return (
    <div className="parallel-slots">
      <div className="toggle-row">
        <div>
          <label htmlFor={id} className="field-label">
            Requests at once
          </label>
          <span className="muted small block">
            Each extra slot uses memory (about 450 MiB on Qwen3.8-27B going from 1 to 4) and does not make a single
            reply faster. Use 1 to leave the most room for context. Applies at the next load.
          </span>
        </div>
        <select id={id} value={value} disabled={busy || settings.data === null} onChange={(e) => void set(Number(e.target.value))}>
          {(CHOICES.includes(value) ? CHOICES : [...CHOICES, value].sort((a, b) => a - b)).map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>
      {error !== null && <p className="small error-text">{error}</p>}
    </div>
  );
}
