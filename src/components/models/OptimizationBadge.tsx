import { Zap } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { engineStore } from "../../state/engine";

export function OptimizationBadge({ optimized }: { optimized: boolean }) {
  return optimized ? (
    <span className="optimization-badge" title="Raptor’s saved speed settings were applied to this load">
      <Zap size={12} fill="currentColor" aria-hidden="true" /> Raptor Optimized
    </span>
  ) : <span className="optimization-standard" title="Running automatic settings without a saved Raptor tuning profile">Standard</span>;
}

export function CurrentOptimizationBadge() {
  const loaded = useStore(engineStore, (s) => s.loaded);
  const report = useStore(engineStore, (s) => s.loadReport);
  if (!loaded) return null;
  return <OptimizationBadge optimized={report?.model_id === loaded.modelId && report.tuning_applied === true} />;
}
