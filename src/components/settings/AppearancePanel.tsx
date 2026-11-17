import { useState } from "react";
import { Palette } from "lucide-react";
import { Panel } from "../Panel";
import { useStore } from "../../hooks/useStore";
import { appearanceStore, saveAppearance, type Appearance, type Theme } from "../../state/appearance";

export function AppearancePanel() {
  const appearance = useStore(appearanceStore, (s) => s);
  const [error, setError] = useState(false);
  function change(update: Partial<Appearance>) {
    try { saveAppearance(update); setError(false); }
    catch { setError(true); }
  }
  return (
    <Panel id="appearance" title="Appearance" icon={<Palette size={18} aria-hidden="true" />} description="Make Raptor comfortable for you. Changes apply immediately and are saved on this device.">
      <div className="toggle-row">
        <div><label htmlFor="theme" className="field-label">Color theme</label><span className="muted small block">Choose a look, or follow your device’s light and dark settings.</span></div>
        <select id="theme" value={appearance.theme} onChange={(e) => change({ theme: e.target.value as Theme })}>
          <option value="system">Use device setting</option><option value="light">Light</option><option value="dark">Dark</option>
        </select>
      </div>
      <div className="toggle-row">
        <div><label htmlFor="glass" className="field-label">Glass effect</label><span className="muted small block">Soft, translucent panels with background blur. Turn off for solid surfaces.</span></div>
        <input id="glass" type="checkbox" role="switch" className="switch" checked={appearance.glass} onChange={(e) => change({ glass: e.target.checked })} />
      </div>
      {error && <p role="alert">Your appearance choice could not be saved. Try again.</p>}
    </Panel>
  );
}
