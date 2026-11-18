import { useState } from "react";
import { Cpu, Square } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { formatBytes, formatCount } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { engineStore, unloadModel } from "../../state/engine";
import { libraryStore } from "../../state/library";
import { EmptyState, LoadingState } from "../AsyncState";
import { GpuVerdictView } from "../GpuVerdictView";
import { Notice } from "../Notice";
import { LoadProgressBar } from "./LoadProgressBar";
import { Panel } from "../Panel";
import { CurrentOptimizationBadge } from "./OptimizationBadge";

export function LoadedModelPanel() {
  const status = useStore(engineStore, (s) => s.status);
  const statusError = useStore(engineStore, (s) => s.statusError);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const busy = useStore(engineStore, (s) => s.busy);
  const models = useStore(libraryStore, (s) => s.models);
  const [error, setError] = useState<string | null>(null);

  const serverState = status?.server.state ?? null;
  const name = loaded ? (models?.find((m) => m.id === loaded.modelId)?.file_name ?? loaded.modelId) : null;
  const canUnload = loaded !== null || serverState?.state === "ready";

  async function unload() {
    setError(null);
    try {
      await unloadModel();
    } catch (e) {
      setError(messageOf(e));
    }
  }

  return (
    <Panel
      title="Loaded model"
      icon={<Cpu size={18} aria-hidden="true" />}
      description="What is running now, and whether the GPU is doing the work."
      actions={
        canUnload && (
          <button type="button" className="btn btn-sm" onClick={() => void unload()} disabled={busy !== null}>
            <Square size={13} aria-hidden="true" /> {busy === "unloading" ? "Unloading..." : "Unload"}
          </button>
        )
      }
    >
      <LoadProgressBar active={busy === "loading"} />
      {statusError !== null && (
        <Notice tone="danger" title="Engine status unavailable">
          {statusError}
        </Notice>
      )}
      {status === null && statusError === null && <LoadingState label="Reading engine status..." />}

      {status !== null && (
        <dl className="facts">
          <div>
            <dt>Server</dt>
            <dd>{serverLabel(serverState?.state ?? "stopped")}</dd>
          </div>
          <div>
            <dt>Runtime</dt>
            <dd>
              {status.runtime ? `${status.runtime.id} ${status.runtime.version} (${status.runtime.backend.toUpperCase()})` : "No runtime installed"}
            </dd>
          </div>
          {loaded && (
            <>
              <div>
                <dt>Model</dt>
                <dd className="break">{name} <CurrentOptimizationBadge /></dd>
              </div>
              <div>
                <dt>Planned context</dt>
                <dd>{loaded.ctxSize === null ? "Unknown" : `${formatCount(loaded.ctxSize)} tokens`}</dd>
              </div>
              <div>
                <dt>GPU layers</dt>
                <dd>
                  {status.gpu_verdict?.verdict === "verified" && status.gpu_verdict.offloaded_layers != null ? (
                    <span>
                      Offloaded {formatCount(status.gpu_verdict.offloaded_layers)}
                      {status.gpu_verdict.total_layers != null
                        ? ` / ${formatCount(status.gpu_verdict.total_layers)}`
                        : ""}{" "}
                      layers to GPU
                      {loaded.plannedGpuLayers !== null && (
                        <span className="muted small"> (planned: {formatCount(loaded.plannedGpuLayers)})</span>
                      )}
                    </span>
                  ) : (
                    <span>
                      {loaded.plannedGpuLayers === null ? "Unknown" : `${formatCount(loaded.plannedGpuLayers)} planned`}
                      <span className="muted small"> (offload not active)</span>
                    </span>
                  )}
                </dd>
              </div>
              {loaded.estimate && (
                <div>
                  <dt>Estimated memory</dt>
                  <dd>
                    VRAM {formatBytes(loaded.estimate.vram_bytes)} of {formatBytes(loaded.estimate.vram_budget)}, RAM{" "}
                    {formatBytes(loaded.estimate.ram_bytes)} of {formatBytes(loaded.estimate.ram_budget)}
                  </dd>
                </div>
              )}
            </>
          )}
        </dl>
      )}

      {status?.runtime?.degraded && (
        <Notice tone="warn" title="CPU build selected">
          {status.runtime.reason}
        </Notice>
      )}
      {serverState?.state === "crashed" && (
        <Notice tone="danger" title="Server crashed">
          {serverState.reason}
        </Notice>
      )}
      {serverState?.state === "degraded" && (
        <Notice tone="warn" title="Server degraded">
          {serverState.reason}
        </Notice>
      )}

      {/* While this session's own load is still benchmarking, the server is ready but `loaded` is not set yet. */}
      {status !== null && loaded === null && busy !== "loading" && serverState?.state === "ready" && (
        <Notice tone="warn" title="A model is running, but this session did not load it">
          The backend does not report which model it is serving. Unload it, or load a model below to replace it.
        </Notice>
      )}
      {status !== null && loaded === null && serverState?.state !== "ready" && (
        <EmptyState icon={Cpu} title="No model loaded">
          Load one from the Library tab below.
        </EmptyState>
      )}

      {loaded && loaded.rationale.length > 0 && (
        <details className="rationale">
          <summary>Why these settings</summary>
          <ul>
            {loaded.rationale.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </details>
      )}

      {status !== null && <GpuVerdictView verdict={status.gpu_verdict} />}
      {error !== null && (
        <Notice tone="danger" title="Could not unload">
          {error}
        </Notice>
      )}
    </Panel>
  );
}

function serverLabel(state: string): string {
  const labels: Record<string, string> = {
    stopped: "Stopped",
    starting: "Starting",
    ready: "Ready",
    degraded: "Degraded",
    crashed: "Crashed",
  };
  return labels[state] ?? state;
}
