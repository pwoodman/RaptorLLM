import { Activity, Boxes, Keyboard, MessageSquare, Settings, type LucideIcon } from "lucide-react";
import { useStore } from "../hooks/useStore";
import { CurrentOptimizationBadge } from "./models/OptimizationBadge";
import { engineStore } from "../state/engine";
import { libraryStore } from "../state/library";
import { openModels, uiStore, type Page } from "../state/ui";
import { verdictSummary } from "./GpuVerdictView";

const NAV: { id: Page; label: string; icon: LucideIcon }[] = [
  { id: "chats", label: "Chats", icon: MessageSquare },
  { id: "models", label: "Models", icon: Boxes },
  { id: "settings", label: "Settings", icon: Settings },
];

const SERVER_LABEL = {
  stopped: "No model loaded",
  starting: "Server starting",
  ready: "Server ready",
  degraded: "Server degraded",
  crashed: "Server crashed",
} as const;

export function Sidebar({ onShowShortcuts }: { onShowShortcuts: () => void }) {
  const page = useStore(uiStore, (s) => s.page);
  const status = useStore(engineStore, (s) => s.status);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const loadReport = useStore(engineStore, (s) => s.loadReport);
  const modelName = useStore(libraryStore, (s) => s.models?.find((m) => m.id === loaded?.modelId)?.file_name);
  const serverState = status?.server.state.state ?? null;
  const verdict = verdictSummary(status?.gpu_verdict ?? null);
  const backend = status?.runtime?.backend ?? null;
  const loadRecord = loadReport?.benchmark.stable ? loadReport.benchmark.entry?.record : null;

  return (
    <aside className="sidebar">
      <div className="brand">
        <img className="brand-logo" src="/raptor.png" alt="" width={32} height={32} />
        <span className="brand-name">Raptor LLM</span>
      </div>

      <nav aria-label="Main">
        <ul className="nav-list">
          {NAV.map(({ id, label, icon: Icon }) => (
            <li key={id}>
              <button
                type="button"
                className="nav-item"
                aria-label={label}
                title={label}
                aria-current={page === id ? "page" : undefined}
                onClick={() => uiStore.set({ page: id })}
              >
                <Icon size={18} aria-hidden="true" />
                <span>{label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className="runtime-card" aria-label="Runtime status">
          <div className="runtime-row">
            <span className={`dot dot-${serverState ?? "unknown"}`} aria-hidden="true" />
            <span>{serverState ? SERVER_LABEL[serverState] : "Checking server"}</span>
          </div>
          {loaded ? (
            <div className="runtime-model" title={modelName ?? loaded.modelId}>
              {modelName ?? loaded.modelId.split("/").pop()}
            </div>
          ) : serverState === "ready" && <div className="muted small">Model not reported</div>}
          {backend !== null && <div className="muted small">{backendLabel(backend)} runtime</div>}
          {loadReport?.tuning_applied && <CurrentOptimizationBadge />}
          {status?.gpu_verdict && <div className={`runtime-verdict tone-${verdict.tone}`}>{verdict.label}</div>}
          {loadRecord != null && (
            <div
              className={`small ${loadReport?.tuning_applied ? "optimized-speed" : "muted"}`}
              title={`Measured after load: ${loadRecord.prompt_tps.toFixed(1)} prompt tokens/s; ${loadRecord.gen_tps.toFixed(1)} generation tokens/s; ${Math.round(loadRecord.ttft_ms)} ms to first token`}
            >
              {loadRecord.gen_tps.toFixed(1)} tok/s · {Math.round(loadRecord.ttft_ms)} ms first token
            </div>
          )}
        </div>
        <div className="sidebar-tools">
          <button type="button" className="icon-btn" aria-label="Model performance" title="Model performance"
            onClick={() => openModels("performance")}>
            <Activity size={17} aria-hidden="true" />
          </button>
          <button type="button" className="icon-btn" aria-label="Keyboard shortcuts" title="Keyboard shortcuts"
            onClick={onShowShortcuts}>
            <Keyboard size={17} aria-hidden="true" />
          </button>
        </div>
      </div>
    </aside>
  );
}

function backendLabel(backend: string): string {
  const labels: Record<string, string> = {
    hip: "ROCm",
    cuda: "CUDA",
    sycl: "SYCL",
    vulkan: "Vulkan",
    cpu: "CPU",
  };
  return labels[backend] ?? backend.toUpperCase();
}
