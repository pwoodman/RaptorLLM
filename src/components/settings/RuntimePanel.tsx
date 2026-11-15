import { useEffect, useState, type FormEvent } from "react";
import { History, Wrench } from "lucide-react";
import { api, type RuntimeListView, type RuntimeRollbackView, type RuntimeSelectionView } from "../../api";
import { useStore } from "../../hooks/useStore";
import { engineStore, refreshEngine } from "../../state/engine";
import { ConfirmDialog } from "../ConfirmDialog";
import { GpuVerdictView } from "../GpuVerdictView";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { ErrorState, LoadingState } from "../AsyncState";
import { InstallRuntime } from "../setup/InstallRuntime";
import { CpuFallbackToggle } from "./CpuFallbackToggle";
import { ParallelSlots } from "./ParallelSlots";
import { ThinkingDefault } from "./ThinkingDefault";

export function RuntimePanel() {
  const status = useStore(engineStore, (s) => s.status);
  const statusError = useStore(engineStore, (s) => s.statusError);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const [busy, setBusy] = useState<"rollback" | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [selection, setSelection] = useState<RuntimeSelectionView | null>(null);
  const [rolledBack, setRolledBack] = useState<RuntimeRollbackView | null>(null);
  const [rollbackId, setRollbackId] = useState("");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [runtimeList, setRuntimeList] = useState<RuntimeListView | null>(null);

  async function loadRuntimes() {
    try {
      const list = await api.runtime.list();
      setRuntimeList(list);
    } catch {
      // Non-critical if list fails
    }
  }

  useEffect(() => {
    void loadRuntimes();
  }, [status]);

  async function rollbackTo(id: string) {
    setError(null);
    setBusy("rollback");
    try {
      setRolledBack(await api.runtime.rollback(id));
      setSelection(null);
      await refreshEngine();
      await loadRuntimes();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
      setConfirmOpen(false);
    }
  }

  async function rollback() {
    await rollbackTo(rollbackId.trim());
  }

  function submitRollback(event: FormEvent) {
    event.preventDefault();
    if (rollbackId.trim() !== "") setConfirmOpen(true);
  }

  const modelLoaded = loaded !== null;

  return (
    <Panel
      id="runtime"
      title="Inference runtime"
      icon={<Wrench size={18} aria-hidden="true" />}
      description="The llama.cpp build that Raptor runs. Installs are verified before they are used."
    >
      {statusError !== null && (
        <Notice tone="danger" title="Runtime status unavailable">
          {statusError}
        </Notice>
      )}
      {status === null && statusError === null && <LoadingState label="Reading runtime status..." />}

      {status && (
        <>
          <dl className="facts">
            <div>
              <dt>Installed</dt>
              <dd>
                {status.runtime
                  ? `${status.runtime.id} ${status.runtime.version} (${status.runtime.backend.toUpperCase()})`
                  : "None"}
              </dd>
            </div>
          </dl>
          {status.runtime?.degraded && (
            <Notice tone="warn" title="CPU build in use">
              {status.runtime.reason}
            </Notice>
          )}
          <GpuVerdictView verdict={status.gpu_verdict} />
        </>
      )}

      {modelLoaded && (
        <Notice tone="info">Install and rollback are refused while a model is loaded. Unload it first.</Notice>
      )}

      <CpuFallbackToggle />

      <ParallelSlots />

      <ThinkingDefault />

      <div className="runtime-actions">
        <InstallRuntime
          disabled={busy !== null || modelLoaded}
          onInstalled={(next) => {
            setSelection(next);
            setRolledBack(null);
          }}
        />
      </div>

      {runtimeList && runtimeList.runtimes.length > 0 && (
        <div className="runtime-inventory" style={{ marginTop: "1rem" }}>
          <h4>Runtimes</h4>
          <ul className="runtime-list" style={{ listStyle: "none", padding: 0 }}>
            {runtimeList.runtimes.map((r) => (
              <li
                key={`${r.id}-${r.version}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.5rem 0",
                  borderBottom: "1px solid var(--border-color, #333)",
                }}
              >
                <div>
                  <strong>{r.id}</strong> <span className="muted small">{r.version}</span>
                  <span className="badge" style={{ marginLeft: "0.5rem" }}>
                    {r.backend.toUpperCase()}
                  </span>
                  {r.current && (
                    <span className="badge badge-success" style={{ marginLeft: "0.5rem" }}>
                      In use
                    </span>
                  )}
                  {r.installed && !r.current && (
                    <span className="badge" style={{ marginLeft: "0.5rem" }}>
                      Installed
                    </span>
                  )}
                </div>
                {r.installed && !r.current && (
                  <button
                    type="button"
                    className="btn btn-sm btn-ghost"
                    disabled={busy !== null || modelLoaded}
                    onClick={() => {
                      setRollbackId(r.id);
                      setConfirmOpen(true);
                    }}
                  >
                    <History size={12} aria-hidden="true" /> Roll back
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {selection && (
        <Notice tone={selection.degraded ? "warn" : "success"} title={`Selected ${selection.id} ${selection.version}`}>
          <p>
            Backend {selection.backend.toUpperCase()}
            {selection.gpu_arch ? `, GPU ${selection.gpu_arch}` : ""}. {selection.reason}
          </p>
          {selection.rejected.length > 0 && (
            <>
              <p className="small">Not used:</p>
              <ul className="evidence">
                {selection.rejected.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </>
          )}
        </Notice>
      )}

      <form className="rollback-form" onSubmit={submitRollback}>
        <div className="field grow">
          <label htmlFor="rollback-id" className="field-label">
            Roll back by runtime id
          </label>
          <input
            id="rollback-id"
            value={rollbackId}
            placeholder="Runtime id (e.g. llama-hip-win)"
            onChange={(e) => setRollbackId(e.target.value)}
          />
        </div>
        <button type="submit" className="btn" disabled={busy !== null || modelLoaded || rollbackId.trim() === ""}>
          <History size={14} aria-hidden="true" /> Roll back
        </button>
      </form>

      {rolledBack && (
        <Notice tone="success" title={`Rolled back to ${rolledBack.id} ${rolledBack.version}`}>
          Backend {rolledBack.backend.toUpperCase()}.
        </Notice>
      )}
      {error !== null && <ErrorState error={error} />}

      <ConfirmDialog
        open={confirmOpen}
        title="Roll back the runtime?"
        confirmLabel="Roll back"
        tone="danger"
        busy={busy === "rollback"}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => void rollback()}
      >
        <p>The current runtime becomes "{rollbackId.trim()}". Rollback is refused while a model is loaded.</p>
      </ConfirmDialog>
    </Panel>
  );
}
