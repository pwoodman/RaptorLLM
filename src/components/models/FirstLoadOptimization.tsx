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
    <ConfirmDialog open={entry !== null} title="Give this model the Raptor edge?"
      confirmLabel="Optimize & chat" cancelLabel="Skip and chat"
      onConfirm={() => chooseFirstLoadOptimization(true)} onCancel={() => chooseFirstLoadOptimization(false)}>
      <div className="optimization-intro"><Zap size={24} aria-hidden="true" /><strong>A one-time speed setup for {entry?.file_name}</strong></div>
      <p>Raptor tests at least 65,536 tokens in one request slot. It saves settings that prove at least 50% higher throughput in three matched tests, then loads the model for chat. Large models can take hours.</p>
      <p><strong>You only do this once per model on this PC.</strong> Saved settings apply automatically on future compatible loads. A new GPU or runtime can require a fresh tune.</p>
      <p className="small muted">Want to chat now? Skip this step. We won’t ask again for this model; Optimize speed stays available in Model options.</p>
    </ConfirmDialog>
    {activeId && <div className="optimization-run" role="status">
      <Zap size={18} aria-hidden="true" />
      <div><strong>Finding your Raptor edge</strong><p className="small">One-time tuning in progress{progress?.total ? ` · testing setup ${Math.min(progress.done + 1, progress.total)} of ${progress.total}` : ""}{startedAt ? ` · ${Math.floor((Date.now() - startedAt) / 1000)} s elapsed` : ""}. Keep Raptor open.</p></div>
      <progress aria-label="Optimization progress" max={progress?.total || undefined} value={progress?.total ? progress.done : undefined} />
    </div>}
  </>;
}
