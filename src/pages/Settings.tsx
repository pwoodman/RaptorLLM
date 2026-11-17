import { useEffect } from "react";
import { Info } from "lucide-react";
import { settingsGet, type SettingsView, type SettingsViewV2 } from "../api";
import { useStore } from "../hooks/useStore";
import { uiStore } from "../state/ui";
import { useAsyncResource } from "../hooks/useAsyncResource";
import { ContextPanel } from "../components/settings/ContextPanel";
import { DependenciesPanel } from "../components/settings/DependenciesPanel";
import { DiagnosticsPanel } from "../components/settings/DiagnosticsPanel";
import { HardwarePanel } from "../components/settings/HardwarePanel";
import { RuntimePanel } from "../components/settings/RuntimePanel";
import { ServerPanel } from "../components/settings/ServerPanel";
import { StoragePanel } from "../components/settings/StoragePanel";
import { AppUpdates } from "../components/settings/AppUpdates";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { Notice } from "../components/Notice";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/Panel";
import { AppearancePanel } from "../components/settings/AppearancePanel";

const SECTIONS = [
  { id: "appearance", label: "Appearance" },
  { id: "hardware", label: "Hardware" },
  { id: "runtime", label: "Inference runtime" },
  { id: "dependencies", label: "Dependencies" },
  { id: "server", label: "Server and API" },
  { id: "storage", label: "Storage" },
  { id: "context", label: "Context engine" },
  { id: "diagnostics", label: "Diagnostics" },
  { id: "about", label: "About" },
];

export default function SettingsPage() {
  const settings = useAsyncResource<SettingsViewV2>(() => settingsGet(), []);
  const view = settings.data;
  // Every settings command returns the full view, including the V2 fields, so the page keeps it whole.
  const adopt = (next: SettingsView) => settings.setData(next as SettingsViewV2);
  const anchor = useStore(uiStore, (s) => s.anchor);

  // Sections are shown only after their data loads, so the jump waits for the page to settle.
  useEffect(() => {
    if (anchor === null) return;
    const target = document.getElementById(anchor);
    if (target === null) return;
    target.scrollIntoView({ block: "start" });
    uiStore.set({ anchor: null });
  }, [anchor, view]);

  return (
    <div className="page">
      <PageHeader
        eyebrow="Preferences and runtime"
        title="Settings"
        description="Make Raptor your own. Appearance changes apply right away; other controls explain when they take effect."
      />

      {view?.notes.map((note) => (
        <p key={note} className="muted small">
          {note}
        </p>
      ))}

      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Settings sections">
          {SECTIONS.map((section) => (
            <a key={section.id} href={`#${section.id}`}>
              {section.label}
            </a>
          ))}
        </nav>

        <div className="settings-content">
          <AppearancePanel />
          <HardwarePanel />
          <RuntimePanel />
          <DependenciesPanel />

          {view ? (
            <>
              <ServerPanel settings={view} onChange={adopt} />
              <StoragePanel settings={view} onChange={adopt} />
              <ContextPanel settings={view} onChange={adopt} />
            </>
          ) : settings.error !== null ? (
            <ErrorState error={settings.error} onRetry={settings.reload} />
          ) : (
            <LoadingState label="Loading server, storage and context settings..." />
          )}

          <DiagnosticsPanel />

          <Panel id="about" title="About" icon={<Info size={18} aria-hidden="true" />}>
            <p>
              Local AI for llama.cpp GGUF models, with a context engine for coding
              agents. Chats, model files and logs stay on this machine. Model search and downloads use Hugging Face.
            </p>
            <AppUpdates />
          </Panel>
        </div>
      </div>
    </div>
  );
}
