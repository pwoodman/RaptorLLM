import { useId, type KeyboardEvent } from "react";
import { Brain } from "lucide-react";
import { api, type Effort, type ThinkingView } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import { describeEffort, EFFORT_LABEL, EFFORT_NAME, EFFORTS, effectiveEffort } from "../../lib/thinking";
import { chatStore, paramsFor, updateParams } from "../../state/chat";
import { engineStore } from "../../state/engine";

/**
 * How hard the loaded model thinks: Low, Med, High or Max, for this conversation. Shown only when the
 * loaded model's chat template has an effort control; any other model shows nothing and is sent nothing.
 * Unchosen, the conversation uses the default from Settings (Medium).
 */
export function ThinkingEffort() {
  const modelId = useStore(engineStore, (s) => s.loaded?.modelId ?? null);
  const loadReport = useStore(engineStore, (s) => s.loadReport);
  const conversationId = useStore(chatStore, (s) => s.activeId);
  const chosen = useStore(chatStore, (s) => (s.activeId ? paramsFor(s.activeId, s).thinking : null));
  const labelId = useId();
  // Asked again after each load: another model can be served at the same address.
  const view = useAsyncResource<ThinkingView | null>(
    () => (modelId === null ? Promise.resolve(null) : api.models.thinking()),
    [modelId, loadReport],
  );

  const info = view.data;
  if (modelId === null || conversationId === null || info === null || !info.supported || info.choices.length === 0) {
    return null;
  }
  const current = effectiveEffort(chosen, info);

  function choose(effort: Effort) {
    if (conversationId !== null) updateParams(conversationId, { thinking: effort });
  }

  /** Arrow keys move the choice, as in any radio group. */
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
    if (step === 0) return;
    event.preventDefault();
    const next = EFFORTS[(EFFORTS.indexOf(current) + step + EFFORTS.length) % EFFORTS.length];
    if (next) choose(next);
  }

  return (
    <div className="thinking-effort">
      <span id={labelId} className="thinking-effort-label muted small">
        <Brain size={13} aria-hidden="true" /> Thinking
      </span>
      <div role="radiogroup" aria-labelledby={labelId} className="effort-group" onKeyDown={onKeyDown}>
        {EFFORTS.map((effort) => (
          <button
            key={effort}
            type="button"
            role="radio"
            aria-checked={current === effort}
            tabIndex={current === effort ? 0 : -1}
            aria-label={EFFORT_NAME[effort]}
            title={describeEffort(info, effort)}
            className={`effort-option ${current === effort ? "effort-option-on" : ""}`}
            onClick={() => choose(effort)}
          >
            {EFFORT_LABEL[effort]}
          </button>
        ))}
      </div>
    </div>
  );
}
