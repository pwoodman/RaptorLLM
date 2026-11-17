import { useId, useState, type FormEvent } from "react";
import { HardDrive } from "lucide-react";
import { api, type SettingsView } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { messageOf } from "../../lib/errors";
import { libraryStore } from "../../state/library";
import { Notice } from "../Notice";
import { Panel } from "../Panel";

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
  const inputId = useId();

  async function save(value: string | null) {
    setBusy(true);
    setError(null);
    try {
      const view = await api.settings.setStorageDir(value);
      setPath(view.storage_dir ?? "");
      onChange(view);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }

  async function browseDir() {
    setPicking(true);
    setError(null);
    try {
      const selected = await api.dialog.pickFolder();
      if (selected) {
        setPath(selected);
        await save(selected);
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
    void save(trimmed === "" ? null : trimmed);
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

      {settings.restart_required && (
        <Notice tone="warn" title="Restart required">
          The new storage folder is used after Raptor restarts.
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
          <span className="muted small">Browse saves the selected folder immediately. Typed paths need Save folder. Changes apply after restart; models already stored elsewhere are not moved. Leave empty and save to use the default.</span>
        </div>
        <div className="button-row">
          <button type="submit" className="btn btn-primary" disabled={busy}>
            Save folder
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy || settings.storage_dir === null}
            onClick={() => void save(null)}
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
    </Panel>
  );
}
