import { useEffect, useState } from "react";
import { DiscoverTab } from "../components/models/DiscoverTab";
import { CompareTab } from "../components/models/CompareTab";
import { DownloadsTab } from "../components/models/DownloadsTab";
import { LibraryTab } from "../components/models/LibraryTab";
import { LoadedModelPanel } from "../components/models/LoadedModelPanel";
import { LoadReport } from "../components/models/LoadReport";
import { PerformancePanel } from "../components/models/PerformancePanel";
import { SetupGuide } from "../components/setup/SetupGuide";
import { PageHeader } from "../components/PageHeader";
import { useStore } from "../hooks/useStore";
import { refreshDownloads } from "../state/downloads";
import { libraryStore, refreshLibrary } from "../state/library";

type Tab = "discover" | "library" | "downloads" | "performance" | "compare";

const TABS: { id: Tab; label: string }[] = [
  { id: "discover", label: "Discover" },
  { id: "library", label: "Library" },
  { id: "downloads", label: "Downloads" },
  { id: "performance", label: "Performance" },
  { id: "compare", label: "Compare and rank" },
];

export default function ModelsPage() {
  const [tab, setTab] = useState<Tab>("library");
  const models = useStore(libraryStore, (s) => s.models);

  useEffect(() => {
    void refreshLibrary();
    void refreshDownloads();
  }, []);

  return (
    <div className="page">
      <PageHeader
        eyebrow="Model library"
        title="Models"
        description="Find GGUF models on Hugging Face, download them with verification, and load one for chat."
      />

      <SetupGuide />

      <LoadedModelPanel />
      <LoadReport />

      <div className="tabs" role="tablist" aria-label="Model views">
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

      <div id="models-tabpanel" role="tabpanel" aria-labelledby={`models-tab-${tab}`} className="tabpanel">
        {tab === "discover" && <DiscoverTab onQueued={() => setTab("downloads")} />}
        {tab === "library" && <LibraryTab />}
        {tab === "downloads" && <DownloadsTab />}
        {tab === "performance" && <PerformancePanel models={models} />}
        {tab === "compare" && <CompareTab />}
      </div>
    </div>
  );
}
