import { CheckCircle2, LoaderCircle, PackageCheck } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { runtimeSetupStore, retryRuntimeSetup } from "../../state/runtimeSetup";
import { formatBytes } from "../../lib/format";
import { Notice } from "../Notice";

const STEPS = ["Preparing", "Downloading", "Verifying", "Installing", "Ready"];
export function RuntimeSetupProgress({ onlyWhileRunning = false }: { onlyWhileRunning?: boolean }) {
  const state = useStore(runtimeSetupStore, (s) => s);
  if (!state.label || (onlyWhileRunning && !state.running)) return null;
  const index = ["preparing", "downloading", "verifying", "installing", "ready"].indexOf(state.stage);
  const downloading = state.stage === "downloading";
  const label = state.stage === "failed" ? "Setup needs attention" : STEPS[index] ?? "Preparing";
  return <div className="runtime-setup-progress">
    <div className="setup-progress-title" role="status">
      {state.running ? <LoaderCircle size={18} className="spin" aria-hidden="true" /> : state.stage === "ready" ? <CheckCircle2 size={18} aria-hidden="true" /> : <PackageCheck size={18} aria-hidden="true" />}
      <strong>{label}{state.running ? "…" : ""}</strong><span className="muted small">{state.label}</span>
    </div>
    <ol className="install-steps" aria-label="Dependency setup steps">{STEPS.map((step, i) => <li key={step} aria-current={i === index ? "step" : undefined} className={i < index ? "done" : ""}>{i < index ? "✓ " : ""}{step}</li>)}</ol>
    {state.running && <>
      <progress aria-label={downloading ? "Dependency download progress" : "Dependency setup progress"} max={downloading && state.total ? state.total : undefined} value={downloading && state.total ? Math.min(state.downloaded, state.total) : undefined} />
      <p className="small muted">{downloading ? `${formatBytes(state.downloaded)}${state.total ? ` of ${formatBytes(state.total)}` : ""} downloaded. ` : state.stage === "verifying" ? "Checking the download before using it. " : state.stage === "installing" ? "Unpacking and setting up the runtime files. " : "Getting the runtime ready. "}You can switch screens; setup continues in the background.</p>
    </>}
    {state.stage === "ready" && <p className="small tone-ok">Ready to use. You can load a model and start chatting.</p>}
    {state.error && <Notice tone="danger" title="Setup did not finish" actions={<button type="button" className="btn btn-sm" onClick={() => void retryRuntimeSetup()}>Try again</button>}>{state.error}</Notice>}
  </div>;
}
