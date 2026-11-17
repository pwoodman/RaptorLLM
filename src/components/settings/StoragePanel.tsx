import { useId, useState, type FormEvent } from "react";
import { HardDrive } from "lucide-react";
import { api, type SettingsView } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { libraryStore } from "../../state/library";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { ConfirmDialog } from "../ConfirmDialog";

interface StoragePanelProps {
  settings: SettingsView;
  onChange: (view: SettingsView) => void;
}

export function StoragePanel({ settings, onChange }: StoragePanelProps) {
  const storage = useStore(libraryStore, (s) => s.storage);
  const [path, setPath] = useState(settings.storage_dir ?? "");
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [pending, setPending] = useState<{ value: string | null } | null>(null);
  const inputId = useId();

  async function applyStorageChange(value: string | null) {
    setBusy(true);
    setError(null);
    setSuccess(false);
    try {
      const view = await api.settings.setStorageDir(value);
      setPath(view.storage_dir ?? "");
      onChange(view);
      setSuccess(true);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
      setPending(null);
    }
  }

  async function browseDir() {
    setPicking(true);
    setError(null);
    try {
      const selected = await api.dialog.pickFolder();
      if (selected) {
        setPath(selected);
        setPending({ value: selected });
      }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setPicking(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = path.trim();
    setPending({ value: trimmed === "" ? null : trimmed });
  }

  return (
    <Panel
      id="storage"
      title="Storage"
      icon={<HardDrive size={18} aria-hidden="true" />}
      description="Where models, runtimes and downloads are kept."
    >
      <dl className="facts">
        <div>
          <dt>In use now</dt>
          <dd className="break">{settings.active_storage_dir}</dd>
        </div>
        <div>
          <dt>Default location</dt>
          <dd className="break">{settings.default_storage_dir}</dd>
        </div>
        {storage && (
          <div>
            <dt>Models on disk</dt>
            <dd>
              {formatBytes(storage.managed_bytes)} in library, {formatBytes(storage.referenced_bytes)} referenced
            </dd>
          </div>
        )}
      </dl>

      {success && (
        <Notice tone="success" title="Raptor is reopening">
          It will use the new folder automatically. Windows does not need to restart.
        </Notice>
      )}
      {settings.restart_required && !success && (
        <Notice tone="info" title="Raptor will apply this automatically">
          The app reopens itself after you confirm. You do not need to restart Windows.
        </Notice>
      )}

      <form className="stack" onSubmit={submit}>
        <div className="field">
          <label htmlFor={inputId} className="field-label">
            Storage folder
          </label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              id={inputId}
              value={path}
              placeholder={settings.default_storage_dir}
              onChange={(e) => setPath(e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="button" className="btn btn-ghost" disabled={busy || picking} onClick={() => void browseDir()}>
              {picking ? "Selecting..." : "Browse..."}
            </button>
          </div>
          <span className="muted small">Choose where future models, runtimes, downloads, chats and logs are kept. Raptor reopens automatically to apply it; existing files are not copied. Leave empty and save to use the default.</span>
        </div>
        <div className="button-row">
          <button type="submit" className="btn btn-primary" disabled={busy}>
            Save folder
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy || settings.storage_dir === null}
            onClick={() => setPending({ value: null })}
          >
            Use default location
          </button>
        </div>
      </form>

      {error !== null && (
        <Notice tone="danger" title="Storage folder not saved">
          {error}
        </Notice>
      )}

      <ConfirmDialog
        open={pending !== null}
        title="Switch storage folder now?"
        confirmLabel="Switch folder now"
        busy={busy}
        onCancel={() => setPending(null)}
        onConfirm={() => pending && void applyStorageChange(pending.value)}
      >
        <p>Raptor will save this folder and reopen automatically. The local model engine and API service briefly stop, then start again on the new folder.</p>
        <p>Files already in the previous folder, including models and chats, will stay there and will not be moved. You can switch back to see them again.</p>
        <p className="small"><strong>New folder:</strong> <code className="break">{pending?.value ?? settings.default_storage_dir}</code></p>
      </ConfirmDialog>
    </Panel>
  );
}
