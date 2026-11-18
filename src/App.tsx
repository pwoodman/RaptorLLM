import { useEffect, useState, type ReactElement } from "react";
import { Notice } from "./components/Notice";
import { ShortcutsDialog } from "./components/ShortcutsDialog";
import { CpuFallbackBanner } from "./components/models/CpuFallbackBanner";
import { Sidebar } from "./components/Sidebar";
import { FirstLoadOptimization } from "./components/models/FirstLoadOptimization";
import { IdleUpdateNotice } from "./components/settings/IdleUpdateNotice";
import { startIdleUpdates } from "./state/appUpdates";
import { RuntimeSetupProgress } from "./components/settings/RuntimeSetupProgress";
import { useGlobalShortcuts } from "./hooks/useGlobalShortcuts";
import { useStore } from "./hooks/useStore";
import ChatsPage from "./pages/Chats";
import ModelsPage from "./pages/Models";
import SettingsPage from "./pages/Settings";
import { startSync } from "./state/sync";
import { uiStore, type Page } from "./state/ui";

const PAGES: Record<Page, () => ReactElement> = {
  chats: ChatsPage,
  models: ModelsPage,
  settings: SettingsPage,
};

export default function App() {
  const page = useStore(uiStore, (s) => s.page);
  const flash = useStore(uiStore, (s) => s.flash);
  const Current = PAGES[page];
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => startSync(), []);
  useEffect(() => startIdleUpdates(), []);
  useGlobalShortcuts({ onShowHelp: () => setHelpOpen(true) });

  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Sidebar />
      <main id="main" className={`main ${page === "chats" ? "main-fill" : ""}`} tabIndex={-1}>
        <CpuFallbackBanner />
        <FirstLoadOptimization />
        <IdleUpdateNotice />
        <RuntimeSetupProgress onlyWhileRunning />
        {flash !== null && (
          <div className="flash">
            <Notice
              tone="danger"
              title="Something went wrong"
              actions={
                <button type="button" className="btn btn-sm" onClick={() => uiStore.set({ flash: null })}>
                  Dismiss
                </button>
              }
            >
              {flash}
            </Notice>
          </div>
        )}
        <Current />
      </main>
      <ShortcutsDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
