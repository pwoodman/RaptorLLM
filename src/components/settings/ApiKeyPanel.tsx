import { useId, useState } from "react";
import { KeyRound } from "lucide-react";
import { api, type ApiKeyReveal, type SettingsView } from "../../api";
import { messageOf } from "../../lib/errors";
import { CopyButton } from "../CopyButton";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

interface ApiKeyPanelProps {
  settings: SettingsView;
  onChange: (view: SettingsView) => void;
}

/**
 * Generates, rotates and revokes the gateway API key. The secret is shown once, held only in this
 * component's state, and dropped when the user dismisses it or leaves the page.
 */
export function ApiKeyPanel({ settings, onChange }: ApiKeyPanelProps) {
  const [reveal, setReveal] = useState<ApiKeyReveal | null>(null);
  const [rotateOpen, setRotateOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const secretId = useId();
  const hasKey = settings.api_key !== null;

  async function refreshSettings() {
    onChange(await api.settings.get());
  }

  async function generate() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.settings.rotateApiKey();
      setReveal(result);
      await refreshSettings();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
      setRotateOpen(false);
    }
  }

  async function revoke() {
    setBusy(true);
    setError(null);
    try {
      onChange(await api.settings.revokeApiKey());
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  function startGenerate() {
    if (hasKey) {
      setRotateOpen(true);
    } else {
      void generate();
    }
  }

  return (
    <section id="api-key" className="api-key" aria-labelledby={`${secretId}-heading`}>
      <h3 id={`${secretId}-heading`} className="subsection-title">
        <KeyRound size={15} aria-hidden="true" /> API key
      </h3>
      <p className="muted small">
        {hasKey ? "A key is set. Clients send it as a Bearer token or in the x-api-key header." : "No key is set. LAN needs one."}
      </p>

      {reveal !== null && (
        <div className="secret-reveal">
          <Notice tone="warn" title="Copy this key now. It will not be shown again.">
            <div className="secret-row">
              <input id={secretId} readOnly value={reveal.secret} aria-label="New API key" onFocus={(e) => e.currentTarget.select()} />
              <CopyButton label="Copy key" getText={() => reveal.secret} />
            </div>
            <button type="button" className="btn btn-sm" onClick={() => setReveal(null)}>
              I have saved the key
            </button>
          </Notice>
        </div>
      )}

      <div className="button-row">
        <button type="button" className="btn btn-primary" disabled={busy} onClick={startGenerate}>
          {hasKey ? "Rotate key" : "Generate key"}
        </button>
        <button
          type="button"
          className="btn btn-ghost"
          disabled={busy || !hasKey || settings.lan_enabled}
          title={settings.lan_enabled ? "Turn LAN off before revoking the key" : "Remove the key"}
          onClick={() => void revoke()}
        >
          Revoke key
        </button>
      </div>
      {settings.lan_enabled && hasKey && (
        <p className="muted small">The key cannot be revoked while LAN is on.</p>
      )}
      {error !== null && (
        <Notice tone="danger" title="API key action failed">
          {error}
        </Notice>
      )}

      <ConfirmDialog
        open={rotateOpen}
        title="Replace the API key?"
        confirmLabel="Replace key"
        tone="danger"
        busy={busy}
        onCancel={() => setRotateOpen(false)}
        onConfirm={() => void generate()}
      >
        <p>Clients using the current key stop working until they receive the new one.</p>
      </ConfirmDialog>
    </section>
  );
}
