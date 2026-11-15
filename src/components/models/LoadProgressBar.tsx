import { useEffect, useState } from "react";
import { api } from "../../api";

type Progress = Awaited<ReturnType<typeof api.models.loadProgress>>;

/**
 * Where a model load has got to. llama-server prints no percentage while it loads weights, so
 * this shows the stage it reported (step N of M) and the time elapsed, never an invented percent.
 * It polls only while `active` is true.
 */
export function LoadProgressBar({ active }: { active: boolean }) {
  const [progress, setProgress] = useState<Progress | null>(null);

  useEffect(() => {
    if (!active) {
      setProgress(null);
      return;
    }
    let current = true;
    const read = () =>
      api.models.loadProgress().then(
        (p) => {
          if (current && p.active) setProgress(p);
        },
        () => undefined,
      );
    void read();
    const timer = setInterval(() => void read(), 400);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [active]);

  if (!active) return null;
  const step = progress?.step ?? 1;
  const count = progress?.stage_count ?? 6;
  const seconds = progress ? Math.floor(progress.elapsed_ms / 1000) : 0;
  return (
    <div className="load-progress" role="status" aria-live="polite">
      <progress aria-label="Model loading progress" max={count} value={step - 1 + 0.5} />
      <p className="small">
        Step {step} of {count}: {progress?.label ?? "Starting the inference server"} ({seconds} s)
      </p>
    </div>
  );
}
