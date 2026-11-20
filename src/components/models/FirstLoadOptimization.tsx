import { firstLoadStore, chooseFirstLoadOptimization } from "../../state/firstLoad";
import { useStore } from "../../hooks/useStore";
import { ConfirmDialog } from "../ConfirmDialog";
import { OptimizationRun } from "./OptimizationRun";

export function FirstLoadOptimization() {
  const entry = useStore(firstLoadStore, (s) => s.entry);
  return <>
    <ConfirmDialog open={entry !== null} title="Load this model"
      confirmLabel="Load model"
      secondaryAction={{ label: "Optimize first", onClick: () => chooseFirstLoadOptimization(true) }}
      onConfirm={() => chooseFirstLoadOptimization(false)} onCancel={() => chooseFirstLoadOptimization(null)}>
      <p className="break"><strong>{entry?.file_name}</strong></p>
      <p>Load the model with the recommended settings, or benchmark it first to find faster settings for this PC. Benchmarking large models can take hours.</p>
      <p className="small muted">You can optimize later from Model options and speed. Saved settings apply to compatible loads.</p>
    </ConfirmDialog>
    <OptimizationRun />
  </>;
}
