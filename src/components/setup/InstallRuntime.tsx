import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { api, runtimeInstallPreview, type AvailableRuntime, type RuntimeInstallPreview, type RuntimeSelectionView } from "../../api";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { refreshEngine } from "../../state/engine";
import { ConfirmDialog } from "../ConfirmDialog";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";

interface InstallRuntimeProps {
  /** The compatible runtime the backend found, when known. */
  candidate?: AvailableRuntime | null;
  disabled?: boolean;
  onInstalled: (selection: RuntimeSelectionView) => void;
}

/**
 * Installs the inference runtime only after an explicit click and a confirmation. Nothing is
 * downloaded at startup. Opening the dialog reads the install preview (source, SHA-256 and the size
 * the server reports). The install button stays disabled until that preview has loaded.
 */
export function InstallRuntime({ candidate = null, disabled = false, onInstalled }: InstallRuntimeProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [preview, setPreview] = useState<RuntimeInstallPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [written, setWritten] = useState(0);

  // While installing, poll how many bytes have been written. It only reads a counter.
  useEffect(() => {
    if (!busy) {
      setWritten(0);
      return;
    }
    let current = true;
    const timer = setInterval(() => {
      api.runtime.installProgress().then(
        (n) => {
          if (current) setWritten(n);
        },
        () => undefined,
      );
    }, 500);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [busy]);

  // Read the preview each time the dialog opens, so the size and source are current.
  useEffect(() => {
    if (!open) return;
    let current = true;
    setPreview(null);
    setPreviewError(null);
    runtimeInstallPreview().then(
      (next) => {
        if (current) setPreview(next);
      },
      (e: unknown) => {
        if (current) setPreviewError(messageOf(e));
      },
    );
    return () => {
      current = false;
    };
  }, [open]);

  async function install() {
    setBusy(true);
    setError(null);
    try {
      const selection = await api.runtime.ensure();
      setOpen(false);
      await refreshEngine();
      onInstalled(selection);
    } catch (e) {
      setError(e);
      setOpen(false);
    } finally {
      setBusy(false);
    }
  }

  const label = candidate ? `Install ${candidate.id} ${candidate.version}` : "Install best runtime";
  const confirmReady = preview !== null && previewError === null;

  return (
    <div className="install-runtime">
      <button type="button" className="btn btn-primary" disabled={disabled || busy} onClick={() => setOpen(true)}>
        <Download size={14} aria-hidden="true" /> {busy ? "Installing..." : label}
      </button>
      {error !== null && <ErrorState error={error} />}

      <ConfirmDialog
        open={open}
        title="Install the inference runtime?"
        confirmLabel={preview?.already_installed ? "Use installed runtime" : "Download and install"}
        busy={busy}
        confirmDisabled={!confirmReady}
        onCancel={() => setOpen(false)}
        onConfirm={() => void install()}
      >
        {candidate ? (
          <p>
            Raptor will download and verify {candidate.id} {candidate.version} ({candidate.backend.toUpperCase()}).{" "}
            {candidate.reason}
          </p>
        ) : (
          <p>Raptor will download and verify the best compatible runtime for this machine.</p>
        )}
        {candidate?.degraded && (
          <Notice tone="warn" title="CPU build">
            No compatible GPU build was found. Models will run on the CPU, which is much slower.
          </Notice>
        )}

        {previewError !== null && (
          <Notice tone="danger" title="Install details unavailable">
            {previewError}. Installation stays disabled until the source and checksum can be read.
          </Notice>
        )}
        {previewError === null && preview === null && <LoadingState label="Reading the download details..." />}
        {preview !== null && <PreviewFacts preview={preview} />}
        {busy && <InstallProgress written={written} total={preview?.download_bytes ?? null} />}

        <p className="small">
          The download source is the runtime index in config/runtime-manifest.json. Each file is checked against its
          published SHA-256 before it is used.
        </p>
      </ConfirmDialog>
    </div>
  );
}

function InstallProgress({ written, total }: { written: number; total: number | null }) {
  const done = total !== null && total > 0 && written >= total;
  const text =
    written === 0
      ? "Starting the download..."
      : done
        ? "Downloaded. Verifying the checksum and extracting..."
        : total !== null && total > 0
          ? `Downloaded ${formatBytes(written)} of ${formatBytes(total)}`
          : `Downloaded ${formatBytes(written)}`;
  return (
    <div className="install-progress" role="status">
      {total !== null && total > 0 ? (
        <progress max={total} value={Math.min(written, total)} aria-label="Runtime download progress" />
      ) : (
        <progress aria-label="Runtime download progress" />
      )}
      <p className="small">{text}</p>
    </div>
  );
}

function PreviewFacts({ preview }: { preview: RuntimeInstallPreview }) {
  const sizeText =
    preview.download_bytes !== null
      ? `${formatBytes(preview.download_bytes)} (reported by the download server)`
      : "Unknown. The download server did not report a size, so the download size is not known in advance.";
  return (
    <>
      {preview.already_installed && (
        <Notice tone="info" title="Already installed">
          This version is on disk. Nothing would be downloaded; the installed copy would be selected.
        </Notice>
      )}
      <dl className="facts compact">
        <div>
          <dt>Runtime</dt>
          <dd>
            {preview.id} {preview.version} ({preview.backend.toUpperCase()})
          </dd>
        </div>
        <div>
          <dt>Download size</dt>
          <dd>{preview.already_installed ? "None (already installed)" : sizeText}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>
            <code className="break">{preview.url}</code>
          </dd>
        </div>
        <div>
          <dt>SHA-256</dt>
          <dd>
            <code className="break">{preview.sha256}</code>
          </dd>
        </div>
      </dl>
      {preview.rejected.length > 0 && (
        <>
          <p className="small">Index entries that will not be used:</p>
          <ul className="evidence">
            {preview.rejected.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </>
      )}
      {preview.notes.map((note) => (
        <p key={note} className="muted small">
          {note}
        </p>
      ))}
    </>
  );
}
