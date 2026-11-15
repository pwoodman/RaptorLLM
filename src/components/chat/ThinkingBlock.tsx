import { useEffect, useId, useState } from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";
import { CopyButton } from "../CopyButton";

interface ThinkingBlockProps {
  /** The reasoning text. It is shown apart from the answer and is never the reply. */
  text: string;
  /** True while the model is still reasoning. */
  streaming: boolean;
  /** True once the answer has started. The block then collapses, unless the user opens it. */
  answerStarted: boolean;
}

/**
 * The model's reasoning. Open while it streams before the answer, collapsed once the answer
 * begins, and expandable at any time. Its text is rendered only while open, so it is not in the
 * page (or the find results) while collapsed. Copying the reply never includes it; the copy
 * button here copies only this reasoning, and only when the user opened it.
 */
export function ThinkingBlock({ text, streaming, answerStarted }: ThinkingBlockProps) {
  const [open, setOpen] = useState(streaming && !answerStarted);
  const regionId = useId();

  // Collapse at the moment the answer begins. After that the user's choice is kept.
  useEffect(() => {
    if (answerStarted) setOpen(false);
  }, [answerStarted]);

  const label = streaming && !answerStarted ? "Thinking..." : "Thinking";
  const size = text.length.toLocaleString();

  return (
    <section className="thinking" aria-label="Model reasoning">
      <button
        type="button"
        className="thinking-toggle"
        aria-expanded={open}
        aria-controls={regionId}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}
        <Brain size={14} aria-hidden="true" />
        <span>{label}</span>
        <span className="muted small">
          {streaming && !answerStarted ? " streaming" : ` · ${size} characters, not part of the answer`}
        </span>
      </button>
      {open && (
        <div id={regionId} className="thinking-body">
          <pre className="thinking-text">{text}</pre>
          <div className="message-actions">
            <CopyButton label="Copy thinking" getText={() => text} />
          </div>
        </div>
      )}
    </section>
  );
}
