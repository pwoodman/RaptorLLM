import { useEffect, useId, useState } from "react";
import { presetsList, type PresetList } from "../../api";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import { chatStore, paramsFor, updateParams } from "../../state/chat";
import { engineStore } from "../../state/engine";
import { Notice } from "../Notice";

/** Parses the max-tokens field. Returns undefined for input that is not a positive integer. */
function parseMaxTokens(text: string): number | null | undefined {
  if (text.trim() === "") return null;
  const n = Number(text);
  return Number.isInteger(n) && n > 0 ? n : undefined;
}

export function ParamsPanel({ conversationId }: { conversationId: string }) {
  const params = useStore(chatStore, (s) => paramsFor(conversationId, s));
  const [maxText, setMaxText] = useState(params.maxTokens === null ? "" : String(params.maxTokens));
  const promptId = useId();
  const temperatureId = useId();
  const maxId = useId();
  const maxHintId = useId();
  const presetId = useId();
  const loadedId = useStore(engineStore, (s) => s.loaded?.modelId ?? null);
  const [presets, setPresets] = useState<PresetList | null>(null);
  const [presetError, setPresetError] = useState<string | null>(null);

  const parsed = parseMaxTokens(maxText);
  const invalidMax = parsed === undefined;

  // Presets belong to the loaded model, so they are read again when another model loads.
  useEffect(() => {
    setPresets(null);
    setPresetError(null);
    if (loadedId === null) return;
    let current = true;
    presetsList(loadedId).then(
      (list) => {
        if (current) setPresets(list);
      },
      (e: unknown) => {
        if (current) setPresetError(messageOf(e));
      },
    );
    return () => {
      current = false;
    };
  }, [loadedId]);

  /**
   * Choosing a preset copies its system prompt, temperature and reply limit into these settings, so
   * what is shown is what is sent. Top-p, top-k and repeat penalty are not shown here; the backend
   * applies the preset's values for them.
   */
  function choosePreset(name: string) {
    const chosen = presets?.presets.find((p) => p.name === name);
    if (!chosen) {
      updateParams(conversationId, { preset: null });
      return;
    }
    updateParams(conversationId, {
      preset: chosen.name,
      systemPrompt: chosen.system_prompt ?? "",
      temperature: chosen.temperature,
      maxTokens: chosen.max_tokens,
    });
    setMaxText(chosen.max_tokens === null ? "" : String(chosen.max_tokens));
  }

  return (
    <section id="params-panel" className="params-panel" aria-label="Generation settings">
      <label htmlFor={promptId} className="field-label">
        System prompt <span className="muted">(sent with every message in this conversation)</span>
      </label>
      <textarea
        id={promptId}
        rows={3}
        value={params.systemPrompt}
        placeholder="For example: You are a careful code reviewer."
        onChange={(e) => updateParams(conversationId, { systemPrompt: e.target.value })}
      />

      <div className="field">
        <label htmlFor={presetId} className="field-label">
          Preset <span className="muted">(of the loaded model)</span>
        </label>
        {loadedId === null && <p className="muted small">Load a model to use its presets.</p>}
        {loadedId !== null && presetError !== null && (
          <Notice tone="warn" title="Presets unavailable">
            {presetError}
          </Notice>
        )}
        {loadedId !== null && presetError === null && (
          <select
            id={presetId}
            value={params.preset ?? ""}
            disabled={presets === null}
            onChange={(e) => choosePreset(e.target.value)}
          >
            <option value="">No preset</option>
            {(presets?.presets ?? []).map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
                {presets?.default === p.name ? " (default)" : ""}
              </option>
            ))}
          </select>
        )}
        {params.preset !== null && (
          <span className="muted small">
            Its system prompt, temperature and reply limit are copied into the settings below, and those are what is sent.
          </span>
        )}
      </div>

      <div className="params-grid">
        <div className="field">
          <label htmlFor={temperatureId} className="field-label">
            Temperature <output htmlFor={temperatureId}>{params.temperature.toFixed(2)}</output>
          </label>
          <input
            id={temperatureId}
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={params.temperature}
            onChange={(e) => updateParams(conversationId, { temperature: Number(e.target.value) })}
          />
        </div>

        <div className="field">
          <label htmlFor={maxId} className="field-label">
            Max reply tokens
          </label>
          <input
            id={maxId}
            type="number"
            inputMode="numeric"
            min={1}
            step={1}
            placeholder="Model default"
            value={maxText}
            aria-invalid={invalidMax}
            aria-describedby={maxHintId}
            onChange={(e) => {
              setMaxText(e.target.value);
              const next = parseMaxTokens(e.target.value);
              if (next !== undefined) updateParams(conversationId, { maxTokens: next });
            }}
          />
          <span id={maxHintId} className={invalidMax ? "warn-text small" : "muted small"}>
            {invalidMax ? "Enter a whole number above zero, or leave blank for the model default." : "Blank uses the model default."}
          </span>
        </div>
      </div>
      <p className="muted small">These settings are saved in this browser only.</p>
    </section>
  );
}
