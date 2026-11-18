import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { appUpdatesStore, postponeUpdate, setAutomaticUpdates } from "../../state/appUpdates";

export function IdleUpdateNotice() {
  const state = useStore(appUpdatesStore, (s) => s);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!state.countdownAt) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [state.countdownAt]);
  if (state.countdownAt === null || !state.enabled) return null;
  return <div className="idle-update-notice" role="status">
    <RefreshCw size={18} aria-hidden="true" />
    <div><strong>Raptor {state.version} is ready</strong><p className="small">Your PC has been idle. Restarting to update in {Math.max(0, Math.ceil((state.countdownAt - now) / 1000))} seconds if it stays idle.</p></div>
    <button type="button" className="btn btn-sm" onClick={postponeUpdate}>Later</button>
    <button type="button" className="btn btn-ghost btn-sm" onClick={() => setAutomaticUpdates(false)}>Turn off auto updates</button>
  </div>;
}
