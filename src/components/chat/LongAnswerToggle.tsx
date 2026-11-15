import { Layers } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { chatStore, paramsFor, updateParams } from "../../state/chat";
import { engineStore } from "../../state/engine";

const HELP =
  "Long answer: the model plans the answer, then writes every section at the same time. Measured on " +
  "Nanbeige4.3-33B with 8 request slots: whole answers at 32-45 tok/s instead of about 11-14. The " +
  "sections are written without seeing each other, so the structure differs from a single reply. " +
  "Fastest with Settings > Runtime > Requests at once set to 8. Thinking is off for these replies.";

/** Per-conversation switch for long answers written as parallel sections. Shown when a model is loaded. */
export function LongAnswerToggle() {
  const modelId = useStore(engineStore, (s) => s.loaded?.modelId ?? null);
  const conversationId = useStore(chatStore, (s) => s.activeId);
  const on = useStore(chatStore, (s) => (s.activeId ? paramsFor(s.activeId, s).longAnswer : false));
  if (modelId === null || conversationId === null) return null;

  return (
    <button
      type="button"
      className={`effort-option long-answer-toggle ${on ? "effort-option-on" : ""}`}
      aria-pressed={on}
      title={HELP}
      onClick={() => updateParams(conversationId, { longAnswer: !on })}
    >
      <Layers size={13} aria-hidden="true" /> Long answer
    </button>
  );
}
