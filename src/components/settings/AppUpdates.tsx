import { ArrowDownToLine, RefreshCw } from "lucide-react";
import { Notice } from "../Notice";
import { useStore } from "../../hooks/useStore";
import { appUpdatesStore, checkAppUpdates, installAppUpdate, setAutomaticUpdates } from "../../state/appUpdates";
import pkg from "../../../package.json";

export function AppUpdates() {
  const state = useStore(appUpdatesStore, (s) => s);
  const busy = ["checking", "downloading", "installing"].includes(state.phase);
  return (
    <section className="app-updates" aria-label="Application updates">
      <p>Version {pkg.version}. Updates are checked against signed GitHub Releases.</p>
      <div className="toggle-row">
        <div><label htmlFor="auto-updates" className="field-label">Update automatically while this PC is idle</label>
          <p className="small muted">While Raptor is open, install signed updates after at least 5 minutes without keyboard or mouse activity. Raptor may restart. Active chats, API requests, model tasks, downloads and unsent messages delay installation.</p>
        </div>
        <input id="auto-updates" className="switch" type="checkbox" role="switch" checked={state.enabled} disabled={state.phase === "installing"} onChange={(event) => setAutomaticUpdates(event.target.checked)} />
      </div>
      {state.message && <Notice tone="info">{state.message}</Notice>}
      {state.error && <Notice tone="danger">{state.error}</Notice>}
      {(state.error || state.version) && <p><a href="https://github.com/pwoodman/RaptorLLM/releases" target="_blank" rel="noreferrer">Open GitHub Releases</a></p>}
      {state.phase === "downloading" && <progress aria-label="Update download progress" max={100} value={state.progress ?? undefined} />}
      {state.notes && <p className="small muted">{state.notes}</p>}
      <div className="button-row">
        <button className="btn" onClick={() => void checkAppUpdates()} disabled={busy}>
          <RefreshCw size={15} aria-hidden="true" /> {state.phase === "checking" ? "Checking…" : "Check for updates"}
        </button>
        {state.version && <button className="btn btn-primary" onClick={() => void installAppUpdate()} disabled={busy}>
          <ArrowDownToLine size={15} aria-hidden="true" /> {state.phase === "installing" ? "Installing…" : `Install ${state.version}`}
        </button>}
      </div>
    </section>
  );
}
