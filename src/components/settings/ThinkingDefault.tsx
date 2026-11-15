import { useId, useState } from "react";
import { settingsGet, settingsSetThinkingEffort, type Effort, type SettingsViewV2 } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { messageOf } from "../../lib/errors";
import { EFFORT_NAME, EFFORTS } from "../../lib/thinking";

/**
 * The thinking effort used when a conversation or an API client does not choose one. It applies only to
 * models whose chat template has an effort control; every other model is left as it is. Medium by default.
 */
export function ThinkingDefault() {
  const settings = useAsyncResource<SettingsViewV2>(() => settingsGet(), []);
  const id = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function set(effort: Effort) {
    setBusy(true);
    setError(null);
    try {
      settings.setData(await settingsSetThinkingEffort(effort));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  const value: Effort = settings.data?.thinking_effort ?? "medium";
  return (
    <div className="thinking-default">
      <div className="toggle-row">
        <div>
          <label htmlFor={id} className="field-label">
            Default thinking effort
          </label>
          <span className="muted small block">
            How hard thinking models think when a chat or an API client does not choose. Lower is faster and uses
            less of the context window; higher reasons longer. Models that cannot change this are not affected.
            You can change it per chat next to the message box.
          </span>
        </div>
        <select
          id={id}
          value={value}
          disabled={busy || settings.data === null}
          onChange={(e) => void set(e.target.value as Effort)}
        >
          {EFFORTS.map((effort) => (
            <option key={effort} value={effort}>
              {EFFORT_NAME[effort]}
            </option>
          ))}
        </select>
      </div>
      {error !== null && <p className="small error-text">{error}</p>}
    </div>
  );
}
