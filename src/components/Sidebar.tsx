import { Boxes, MessageSquare, Settings, type LucideIcon } from "lucide-react";
import { useStore } from "../hooks/useStore";
import { engineStore } from "../state/engine";
import { uiStore, type Page } from "../state/ui";
import { verdictSummary } from "./GpuVerdictView";

const NAV: { id: Page; label: string; icon: LucideIcon }[] = [
  { id: "chats", label: "Chats", icon: MessageSquare },
  { id: "models", label: "Models", icon: Boxes },
  { id: "settings", label: "Settings", icon: Settings },
];

const SERVER_LABEL = {
  stopped: "Server stopped",
  starting: "Server starting",
  ready: "Server ready",
  degraded: "Server degraded",
  crashed: "Server crashed",
} as const;

export function Sidebar() {
  const page = useStore(uiStore, (s) => s.page);
  const status = useStore(engineStore, (s) => s.status);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const serverState = status?.server.state.state ?? null;
  const verdict = verdictSummary(status?.gpu_verdict ?? null);

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

      <div className="runtime-card" aria-label="Runtime status">
        <div className="runtime-row">
          <span className={`dot dot-${serverState ?? "unknown"}`} aria-hidden="true" />
          <span>{serverState ? SERVER_LABEL[serverState] : "Checking server"}</span>
        </div>
        <div className="runtime-model" title={loaded?.modelId ?? undefined}>
          {loaded ? loaded.modelId.split("/").pop() : serverState === "ready" ? "Model not reported" : "No model loaded"}
        </div>
        <div className={`runtime-verdict tone-${verdict.tone}`}>{verdict.label}</div>
      </div>
    </aside>
  );
}
