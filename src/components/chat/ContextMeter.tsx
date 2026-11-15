import { useStore } from "../../hooks/useStore";
import { formatCount } from "../../lib/format";
import { estimateTokenTotal } from "../../lib/tokens";
import { engineStore } from "../../state/engine";
import { ProgressBar } from "../ProgressBar";

/** Estimated prompt size against the planned context. The estimate is not the model's tokenizer. */
export function ContextMeter({ texts }: { texts: string[] }) {
  const ctxSize = useStore(engineStore, (s) => s.loaded?.ctxSize ?? null);
  const used = estimateTokenTotal(texts);

  if (ctxSize === null) {
    return (
      <div className="context-meter">
        <div className="context-meter-top">
          <span>Context</span>
          <span className="muted">~{formatCount(used)} tokens (estimate), context size unknown</span>
        </div>
      </div>
    );
  }

  const nearLimit = used > ctxSize * 0.9;
  return (
    <div className="context-meter">
      <div className="context-meter-top">
        <span>Context</span>
        <span className="muted">
          ~{formatCount(used)} of {formatCount(ctxSize)} planned tokens (estimate)
        </span>
      </div>
      <ProgressBar label="Estimated context usage" value={used} max={ctxSize} />
      {nearLimit && (
        <p className="warn-text small">
          Near the context limit. The model may fail or drop older text. This figure is an estimate.
        </p>
      )}
    </div>
  );
}
