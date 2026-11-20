import { useEffect, useState } from "react";
import { Zap } from "lucide-react";
import { api } from "../../api";
import { useStore } from "../../hooks/useStore";
import { libraryStore } from "../../state/library";
import { tuningStore } from "../../state/tuning";

/**
 * Facts about what the tuner actually does, rotated slowly while it works. Every line is
 * something Raptor really measures or guarantees, so waiting teaches the product instead of
 * filling time with invented telemetry.
 */
const TIPS = [
  "Every candidate runs the same full-window request — the speedup you get is the speedup your chat gets.",
  "A candidate only wins after beating the default in three independent matched runs in a row.",
  "The KV cache is the model's attention scratch space — 8-bit or 4-bit precision keeps far more context resident on the GPU.",
  "Micro-batch size controls how much work the GPU gets per step. Smaller is not always slower.",
  "Weights never change. Tuning only rearranges how this GPU runs the model you already have.",
  "Time to first token is measured on every candidate, not just streaming speed.",
  "If no candidate wins, your current settings stay untouched.",
];

/** mm:ss (h:mm:ss past an hour), always counting up from the moment tuning started. */
export function formatElapsed(totalSeconds: number): string {
  const total = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const mm = hours > 0 ? String(minutes).padStart(2, "0") : String(minutes);
  const ss = String(seconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Elapsed h:mm:ss clock shown next to the title. Hidden from screen readers; the status role announces the rest. */
function ElapsedClock({ startedAt }: { startedAt: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(tick);
  }, []);
  if (startedAt === null) return null;
  return (
    <span className="optimization-elapsed" aria-hidden="true">
      {formatElapsed((now - startedAt) / 1000)}
    </span>
  );
}

/**
 * The optimization in progress, rendered identically wherever tuning can be started
 * (first-load prompt, model options). Polls the tuner's own progress file, counts the
 * elapsed time up from `tuningStore.startedAt`, and rotates one honest fact at a time.
 * Renders nothing when no optimization is running.
 */
export function OptimizationRun() {
  const activeId = useStore(tuningStore, (s) => s.activeId);
  const startedAt = useStore(tuningStore, (s) => s.startedAt);
  const fileName = useStore(libraryStore, (s) => s.models?.find((model) => model.id === activeId)?.file_name ?? null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [tip, setTip] = useState(0);

  useEffect(() => {
    setProgress(null);
    setTip(0);
    if (!activeId) return;
    let current = true;
    const read = () => {
      api.models.tuneProgress().then(
        (next) => {
          if (current) setProgress(next);
        },
        () => undefined,
      );
    };
    read();
    const poll = window.setInterval(read, 1000);
    return () => {
      current = false;
      window.clearInterval(poll);
    };
  }, [activeId]);

  useEffect(() => {
    if (!activeId) return;
    const rotate = window.setInterval(() => setTip((current) => (current + 1) % TIPS.length), 7000);
    return () => window.clearInterval(rotate);
  }, [activeId]);

  if (!activeId) return null;

  const total = progress && progress.total > 0 ? progress.total : 0;
  const done = progress ? Math.min(progress.done, total || progress.done) : 0;
  const percent = total > 0 ? Math.round((done / total) * 100) : null;
  const confirming = total > 0 && done >= total;

  return (
    <div className="optimization-run" role="status">
      <div className="optimization-orbit" aria-hidden="true">
        <Zap size={18} fill="currentColor" />
      </div>
      <div className="optimization-body">
        <div className="optimization-title-row">
          <strong className="optimization-title">
            Optimizing{fileName ? (
              <> <span className="optimization-model">{fileName}</span></>
            ) : (
              " this model"
            )}
          </strong>
          <ElapsedClock startedAt={startedAt} />
        </div>
        <p className="optimization-sub">
          Raptor is benchmarking full-window generations on this GPU. Nothing here is estimated.
        </p>
        <div className="optimization-progress">
          <progress
            aria-label="Optimization progress"
            max={total > 0 ? total : undefined}
            value={total > 0 ? done : undefined}
          />
          <span className="optimization-progress-label">
            {confirming
              ? "Confirming the winner across matched runs"
              : total > 0
                ? `Setup ${Math.min(done + 1, total)} of ${total}${percent !== null ? ` · ${percent}%` : ""}`
                : "Preparing the default setup"}
          </span>
        </div>
        <p className="optimization-tip" key={tip}>
          <span className="optimization-tip-tag">Good to know</span>
          {TIPS[tip]}
        </p>
      </div>
    </div>
  );
}
