import { useState } from "react";
import { Download, Pause, Play, X } from "lucide-react";
import { api, type JobState } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { discardPartial, downloadsStore, refreshDownloads, startDownload, type DownloadRow } from "../../state/downloads";
import { EmptyState, ErrorState } from "../AsyncState";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";
import { ProgressBar } from "../ProgressBar";

const STATE_LABEL: Record<JobState, string> = {
  queued: "Queued",
  running: "Downloading",
  paused: "Paused",
  verifying: "Verifying SHA-256",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

function rowState(row: DownloadRow): string {
  if (row.finished) return STATE_LABEL[row.finished.status];
  return row.progress ? STATE_LABEL[row.progress.state] : "Queued";
}

export function DownloadsTab() {
  const order = useStore(downloadsStore, (s) => s.order);
  const rows = useStore(downloadsStore, (s) => s.rows);
  const partials = useStore(downloadsStore, (s) => s.partials);
  const problems = useStore(downloadsStore, (s) => s.problems);
  const loadError = useStore(downloadsStore, (s) => s.loadError);
  const [error, setError] = useState<unknown>(null);
  const [cancelKey, setCancelKey] = useState<string | null>(null);
  const [discardKey, setDiscardKey] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  async function run(action: () => Promise<unknown>) {
    setError(null);
    setWorking(true);
    try {
      await action();
    } catch (e) {
      setError(e);
    } finally {
      setWorking(false);
    }
  }

  const cancelRow = cancelKey ? rows[cancelKey] : undefined;
  const discardPart = partials.find((p) => p.key === discardKey);
  const queueRows = order.map((key) => rows[key]).filter((row): row is DownloadRow => row !== undefined);

  return (
    <div className="downloads">
      {error !== null && <ErrorState error={error} />}
      {problems.map((problem) => (
        <Notice key={problem} tone="warn">
          {problem}
        </Notice>
      ))}

      <section aria-labelledby="queue-heading">
        <h2 id="queue-heading" className="section-title">
          This session
        </h2>
        {queueRows.length === 0 ? (
          <EmptyState icon={Download} title="No downloads in this session">
            Choose a file on the Discover tab. Downloads are verified against the published SHA-256 before they
            appear in your library.
          </EmptyState>
        ) : (
          <ul className="job-list">
            {queueRows.map((row) => {
              const progress = row.progress;
              const active = row.finished === null && progress !== null;
              const canPause = active && progress.state === "running";
              const canResume = active && progress.state === "paused";
              const canCancel = row.finished === null && progress?.state !== "verifying";
              return (
                <li key={row.key} className="job">
                  <div className="job-head">
                    <div className="job-name">
                      <span className="break">{row.fileName || "Unknown file"}</span>
                      <span className="muted small break">{row.repo}</span>
                    </div>
                    <span className={`badge ${row.finished?.status === "failed" ? "tone-danger" : row.finished?.status === "completed" ? "tone-ok" : "tone-muted"}`}>
                      {rowState(row)}
                    </span>
                  </div>
                  {active && progress && (
                    <>
                      <ProgressBar label={`Progress for ${row.fileName}`} value={progress.downloaded} max={progress.total} />
                      <span className="muted small">
                        {formatBytes(progress.downloaded)} of {formatBytes(progress.total)}
                        {progress.attempts > 1 ? ` · attempt ${progress.attempts}` : ""}
                      </span>
                    </>
                  )}
                  {row.finished?.status === "completed" && row.finished.path && (
                    <span className="muted small break">Saved to {row.finished.path}</span>
                  )}
                  {row.finished?.status === "failed" && row.finished.error && (
                    <Notice tone="danger" title="Download failed">
                      {row.finished.error}
                    </Notice>
                  )}
                  {(canPause || canResume || canCancel) && (
                    <div className="job-actions">
                      {canPause && (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={working}
                          onClick={() => void run(() => api.models.download.pause(row.key))}
                        >
                          <Pause size={13} aria-hidden="true" /> Pause
                        </button>
                      )}
                      {canResume && (
                        <button
                          type="button"
                          className="btn btn-sm"
                          disabled={working}
                          onClick={() => void run(() => api.models.download.resume(row.key))}
                        >
                          <Play size={13} aria-hidden="true" /> Resume
                        </button>
                      )}
                      {canCancel && (
                        <button type="button" className="btn btn-ghost btn-sm" disabled={working} onClick={() => setCancelKey(row.key)}>
                          <X size={13} aria-hidden="true" /> Cancel
                        </button>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {partials.length > 0 && (
        <section aria-labelledby="partials-heading">
          <h2 id="partials-heading" className="section-title">
            Unfinished from earlier sessions
          </h2>
          <ul className="job-list">
            {partials.map((partial) => (
              <li key={partial.key} className="job">
                <div className="job-head">
                  <div className="job-name">
                    <span className="break">{partial.file_name}</span>
                    <span className="muted small break">{partial.repo}</span>
                  </div>
                  <span className="muted small">
                    {formatBytes(partial.downloaded_bytes)} of {formatBytes(partial.expected_size)}
                  </span>
                </div>
                <div className="job-actions">
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={working}
                    onClick={() => void run(() => startDownload(partial.repo, partial.file_name))}
                  >
                    <Play size={13} aria-hidden="true" /> Resume
                  </button>
                  <button type="button" className="btn btn-ghost btn-sm" disabled={working} onClick={() => setDiscardKey(partial.key)}>
                    Discard partial file
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <ConfirmDialog
        open={cancelRow !== undefined}
        title="Cancel this download?"
        confirmLabel="Cancel download"
        tone="danger"
        busy={working}
        onCancel={() => setCancelKey(null)}
        onConfirm={() => {
          const key = cancelKey;
          if (key === null) return;
          void run(() => api.models.download.cancel(key)).then(() => setCancelKey(null));
        }}
      >
        <p>
          {cancelRow?.fileName ?? "This file"} will stop, and its partial file is deleted. The next attempt starts
          from zero.
        </p>
      </ConfirmDialog>

      <ConfirmDialog
        open={discardPart !== undefined}
        title="Discard the partial file?"
        confirmLabel="Discard"
        tone="danger"
        busy={working}
        onCancel={() => setDiscardKey(null)}
        onConfirm={() => {
          const key = discardKey;
          if (key === null) return;
          void run(() => discardPartial(key)).then(() => setDiscardKey(null));
        }}
      >
        <p>
          The {formatBytes(discardPart?.downloaded_bytes ?? 0)} already downloaded for {discardPart?.file_name} will be
          deleted. This cannot be undone.
        </p>
      </ConfirmDialog>

      {loadError !== null && (
        <Notice
          tone="warn"
          title="Download state could not be refreshed"
          actions={
            <button type="button" className="btn btn-sm" onClick={() => void refreshDownloads()}>
              Try again
            </button>
          }
        >
          {loadError}
        </Notice>
      )}
    </div>
  );
}
