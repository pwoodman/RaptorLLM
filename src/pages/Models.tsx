import { useEffect, type KeyboardEvent } from "react";
import { ArrowUpRight } from "lucide-react";
import { DiscoverTab } from "../components/models/DiscoverTab";
import { CompareTab } from "../components/models/CompareTab";
import { DownloadsTab } from "../components/models/DownloadsTab";
import { LibraryTab } from "../components/models/LibraryTab";
import { LoadedModelPanel } from "../components/models/LoadedModelPanel";
import { LoadMemoryWarning, LoadReport } from "../components/models/LoadReport";
import { LoadProgressBar } from "../components/models/LoadProgressBar";
import { PerformancePanel } from "../components/models/PerformancePanel";
import { SetupGuide } from "../components/setup/SetupGuide";
import { PageHeader } from "../components/PageHeader";
import { useStore } from "../hooks/useStore";
import { refreshDownloads } from "../state/downloads";
import { libraryStore, refreshLibrary } from "../state/library";
import { engineStore } from "../state/engine";
import { uiStore, type ModelView } from "../state/ui";

const TABS: { id: ModelView; label: string }[] = [
  { id: "library", label: "Library" },
  { id: "discover", label: "Discover" },
  { id: "downloads", label: "Downloads" },
  { id: "compare", label: "Compare" },
  { id: "performance", label: "Performance" },
];

export default function ModelsPage() {
  const tab = useStore(uiStore, (s) => s.modelView);
  const models = useStore(libraryStore, (s) => s.models);
  const serverState = useStore(engineStore, (s) => s.status?.server.state.state);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const report = useStore(engineStore, (s) => s.loadReport);
  const busy = useStore(engineStore, (s) => s.busy);
  const chatReady = loaded !== null && (serverState === "ready" || serverState === "degraded");

  useEffect(() => {
    void refreshLibrary();
    void refreshDownloads();
  }, []);

  function setTab(modelView: ModelView) {
    uiStore.set({ modelView, focusModelImport: false });
  }

  function moveTab(event: KeyboardEvent<HTMLDivElement>) {
    const current = TABS.findIndex((item) => item.id === tab);
    let next: number;
    switch (event.key) {
      case "ArrowLeft": next = (current + TABS.length - 1) % TABS.length; break;
      case "ArrowRight": next = (current + 1) % TABS.length; break;
      case "Home": next = 0; break;
      case "End": next = TABS.length - 1; break;
      default: return;
    }
    event.preventDefault();
    const id = TABS[next].id;
    setTab(id);
    event.currentTarget.querySelector<HTMLButtonElement>(`#models-tab-${id}`)?.focus();
  }

  return (
    <div className="page">
      <PageHeader
        title="Models"
        actions={chatReady && (
          <button type="button" className="btn btn-primary" onClick={() => uiStore.set({ page: "chats" })}>
            Go to Chats <ArrowUpRight size={15} aria-hidden="true" />
          </button>
        )}
      />

      <SetupGuide onModelsPage />

      <div className="tabs" role="tablist" aria-label="Model views" onKeyDown={moveTab}>
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`models-tab-${item.id}`}
            aria-selected={tab === item.id}
            aria-controls="models-tabpanel"
            tabIndex={tab === item.id ? 0 : -1}
            className={`tab ${tab === item.id ? "tab-active" : ""}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <LoadProgressBar active={busy === "loading"} />
      {tab !== "performance" && <LoadMemoryWarning report={report} />}
      <div id="models-tabpanel" role="tabpanel" tabIndex={0} aria-labelledby={`models-tab-${tab}`} className="tabpanel">
        {tab === "discover" && <DiscoverTab onQueued={() => setTab("downloads")} />}
        {tab === "library" && <LibraryTab />}
        {tab === "downloads" && <DownloadsTab />}
        {tab === "performance" && (
          <div className="model-performance">
            <LoadedModelPanel />
            <LoadReport />
            <PerformancePanel models={models} />
          </div>
        )}
        {tab === "compare" && <CompareTab />}
      </div>
    </div>
  );
}
