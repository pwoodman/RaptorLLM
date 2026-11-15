import { useEffect, useState } from "react";
import { api, type ContextOption } from "../../api";
import { messageOf } from "../../lib/errors";

/** "8,192 tokens", "131,072 tokens (model maximum)". */
export function windowText(o: ContextOption): string {
  return `${o.ctx.toLocaleString()} tokens${o.capped_by_model ? " (model maximum)" : ""}`;
}

/**
 * Picks the context window for the next load of one model. The sizes come from llama.cpp's own memory
 * projection, so they take a few seconds to compute: they are read only when the user asks, not when the
 * list opens. Default means the size set in Settings.
 */
export function ContextChoice({
  modelId,
  disabled,
  value,
  onChange,
}: {
  modelId: string;
  disabled: boolean;
  value: ContextOption | null;
  onChange: (option: ContextOption | null) => void;
}) {
  const [options, setOptions] = useState<ContextOption[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setOptions(null);
    onChange(null);
    // The options belong to one model; a different model starts over.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  async function read() {
    setLoading(true);
    setError(null);
    try {
      setOptions(await api.models.contextOptions(modelId));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setLoading(false);
    }
  }

  if (options === null) {
    return (
      <div className="context-choice">
        <button type="button" className="btn btn-ghost btn-sm" disabled={disabled || loading} onClick={() => void read()}>
          {loading ? "Working out the sizes..." : "Context window..."}
        </button>
        {error !== null && <span className="small error-text"> {error}</span>}
      </div>
    );
  }

  const selected = value ?? options.find((o) => o.id === "default") ?? options[0];
  return (
    <div className="context-choice">
      <label className="small">
        Context window{" "}
        <select
          value={selected.id}
          disabled={disabled}
          onChange={(e) => {
            const next = options.find((o) => o.id === e.target.value) ?? null;
            onChange(next && next.id !== "default" ? next : null);
          }}
        >
          {options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}: {windowText(o)}
            </option>
          ))}
        </select>
      </label>
      <span className="muted small block">{selected.note}</span>
    </div>
  );
}
