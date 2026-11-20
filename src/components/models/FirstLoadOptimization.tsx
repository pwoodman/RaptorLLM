import { useEffect, useState } from "react";
import { Zap } from "lucide-react";
import { api } from "../../api";
import { useStore } from "../../hooks/useStore";
import { firstLoadStore, chooseFirstLoadOptimization } from "../../state/firstLoad";
import { tuningStore } from "../../state/tuning";
import { ConfirmDialog } from "../ConfirmDialog";

export function FirstLoadOptimization() {
  const entry = useStore(firstLoadStore, (s) => s.entry);
  const activeId = useStore(tuningStore, (s) => s.activeId);
  const startedAt = useStore(tuningStore, (s) => s.startedAt);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  useEffect(() => {
    setProgress(null);
    if (!activeId) return;
    let current = true;
    const timer = window.setInterval(() => api.models.tuneProgress().then(
      (next) => { if (current) setProgress(next); }, () => undefined,
    ), 1000);
    return () => { current = false; window.clearInterval(timer); };
  }, [activeId]);
  return <>
    <ConfirmDialog open={entry !== null} title="Load this model"
      confirmLabel="Load model"
      secondaryAction={{ label: "Optimize first", onClick: () => chooseFirstLoadOptimization(true) }}
      onConfirm={() => chooseFirstLoadOptimization(false)} onCancel={() => chooseFirstLoadOptimization(null)}>
      <p className="break"><strong>{entry?.file_name}</strong></p>
      <p>Load the model with the recommended settings, or benchmark it first to find faster settings for this PC. Benchmarking large models can take hours.</p>
      <p className="small muted">You can optimize later from Model options and speed. Saved settings apply to compatible loads.</p>
    </ConfirmDialog>
    {activeId && <div className="optimization-run" role="status">
      <Zap size={18} aria-hidden="true" />
      <div><strong>Optimizing model speed</strong><p className="small">One-time tuning in progress{progress?.total ? ` · testing setup ${Math.min(progress.done + 1, progress.total)} of ${progress.total}` : ""}{startedAt ? ` · ${Math.floor((Date.now() - startedAt) / 1000)} s elapsed` : ""}. Keep Raptor open.</p></div>
      <progress aria-label="Optimization progress" max={progress?.total || undefined} value={progress?.total ? progress.done : undefined} />
    </div>}
  </>;
}
