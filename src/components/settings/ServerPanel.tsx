import { useId, useState } from "react";
import { Network } from "lucide-react";
import { api, type SettingsView } from "../../api";
import { useStore } from "../../hooks/useStore";
import { isErrorKind } from "../../lib/errors";
import { engineStore } from "../../state/engine";
import { ConfirmDialog } from "../ConfirmDialog";
import { ErrorState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { ApiKeyPanel } from "./ApiKeyPanel";
import { GatewayTls } from "./GatewayTls";

const MIN_PORT = 1;
const MAX_PORT = 65535;

interface ServerPanelProps {
  settings: SettingsView;
  onChange: (view: SettingsView) => void;
}

export function ServerPanel({ settings, onChange }: ServerPanelProps) {
  const gateway = useStore(engineStore, (s) => s.gateway);
  const [portText, setPortText] = useState(String(settings.gateway_port));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [lanConfirmOpen, setLanConfirmOpen] = useState(false);
  const enabledId = useId();
  const portId = useId();
  const lanId = useId();

  const port = Number(portText);
  const portValid = Number.isInteger(port) && port >= MIN_PORT && port <= MAX_PORT;

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  function setGatewayEnabled(enabled: boolean) {
    void run(async () => {
      const status = await api.settings.setGatewayEnabled(enabled);
      engineStore.set({ gateway: status });
      onChange({ ...settings, gateway_enabled: enabled });
    });
  }

  function applyPort() {
    void run(async () => {
      onChange(await api.settings.setGatewayPort(port));
    });
  }

  function setLan(enabled: boolean) {
    void run(async () => {
      onChange(await api.settings.setLan(enabled));
    });
  }

  function requestLan(enabled: boolean) {
    if (enabled) {
      setLanConfirmOpen(true);
    } else {
      setLan(false);
    }
  }

  return (
    <Panel
      id="server"
      title="Server and API"
      icon={<Network size={18} aria-hidden="true" />}
      description="The OpenAI-compatible endpoint that other apps and coding agents call."
    >
      <div className="toggle-row">
        <div>
          <label htmlFor={enabledId} className="field-label">
            Gateway enabled
          </label>
          <span className="muted small block">
            {gateway
              ? gateway.running
                ? `Running${gateway.addr ? ` on ${gateway.addr}` : ""}`
                : "Not running"
              : "Reading gateway status..."}
          </span>
        </div>
        <input
          id={enabledId}
          type="checkbox"
          role="switch"
          className="switch"
          checked={settings.gateway_enabled}
          disabled={busy}
          onChange={(e) => setGatewayEnabled(e.target.checked)}
        />
      </div>

      {gateway?.last_error && (
        <Notice tone="warn" title="Gateway error">
          {gateway.last_error}
        </Notice>
      )}

      <div className="field-row">
        <div className="field">
          <label htmlFor={portId} className="field-label">
            Port
          </label>
          <input
            id={portId}
            type="number"
            inputMode="numeric"
            min={MIN_PORT}
            max={MAX_PORT}
            value={portText}
            aria-invalid={!portValid}
            aria-describedby={`${portId}-hint`}
            onChange={(e) => setPortText(e.target.value)}
          />
          <span id={`${portId}-hint`} className={portValid ? "muted small" : "warn-text small"}>
            {portValid
              ? "Applied immediately. Reverted if the gateway refuses it."
              : `Enter a port from ${MIN_PORT} to ${MAX_PORT}.`}
          </span>
        </div>
        <button
          type="button"
          className="btn"
          disabled={busy || !portValid || port === settings.gateway_port}
          onClick={applyPort}
        >
          Apply port
        </button>
      </div>

      <div className="toggle-row">
        <div>
          <label htmlFor={lanId} className="field-label">
            Allow LAN connections
          </label>
          <span className="muted small block">
            Off keeps the endpoint on this computer only. On makes it reachable from other devices on your network.
          </span>
        </div>
        <input
          id={lanId}
          type="checkbox"
          role="switch"
          className="switch"
          checked={settings.lan_enabled}
          disabled={busy}
          onChange={(e) => requestLan(e.target.checked)}
        />
      </div>
      {settings.lan_enabled && (
        <Notice tone="warn" title="LAN requires an API key">
          Anyone on your network who holds the key can run requests on this machine. Enable LAN only on networks you trust.
        </Notice>
      )}

      <GatewayTls />

      <ApiKeyPanel settings={settings} onChange={onChange} />

      {error !== null &&
        (isErrorKind(error, "Unauthorized") ? (
          <Notice tone="danger" title="Not allowed right now">
            LAN needs an API key, and the key cannot be revoked while LAN is on. Generate a key first, or turn LAN off.
          </Notice>
        ) : (
          <ErrorState error={error} />
        ))}

      <ConfirmDialog
        open={lanConfirmOpen}
        title="Allow LAN connections?"
        confirmLabel="Allow LAN"
        tone="danger"
        busy={busy}
        onCancel={() => setLanConfirmOpen(false)}
        onConfirm={() => {
          setLanConfirmOpen(false);
          setLan(true);
        }}
      >
        <p>The endpoint will accept connections from other devices on your network, using your API key.</p>
      </ConfirmDialog>
    </Panel>
  );
}
