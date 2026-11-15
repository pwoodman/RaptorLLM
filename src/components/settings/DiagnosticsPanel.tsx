import { useId, useState } from "react";
import { ScrollText } from "lucide-react";
import { api } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { ErrorState, LoadingState } from "../AsyncState";
import { formatBytes } from "../../lib/format";
import { Panel } from "../Panel";

const LINE_OPTIONS = [50, 200, 500, 2000] as const;

/** The backend sends each line as `{"stream","text"}` JSON; show the line as the server wrote it. */
export function logLineText(line: string): string {
  try {
    const parsed: unknown = JSON.parse(line);
    if (typeof parsed === "object" && parsed !== null && typeof (parsed as { text?: unknown }).text === "string") {
      return (parsed as { text: string }).text;
    }
  } catch {
    // Not JSON: already a plain line.
  }
  return line;
}

export function DiagnosticsPanel() {
  const [lines, setLines] = useState<number>(200);
  const log = useAsyncResource(() => api.diagnostics.logTail(lines), [lines]);
  const files = useAsyncResource(() => api.diagnostics.logFiles(), []);
  const selectId = useId();

  return (
    <Panel
      id="diagnostics"
      title="Diagnostics"
      icon={<ScrollText size={18} aria-hidden="true" />}
      description="The most recent lines of the llama-server log, exactly as written."
      actions={
        <div className="log-controls">
          <label htmlFor={selectId} className="sr-only">
            Number of lines
          </label>
          <select id={selectId} value={lines} onChange={(e) => setLines(Number(e.target.value))}>
            {LINE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                Last {n} lines
              </option>
            ))}
          </select>
          <button type="button" className="btn btn-sm" onClick={log.reload} disabled={log.loading}>
            Refresh
          </button>
        </div>
      }
    >
      {files.data !== null && (
        <details className="log-files">
          <summary>
            Log files: the last {files.data.keep_days} days are kept automatically ({files.data.files.length} files)
          </summary>
          <p className="small muted">
            Folder: <code>{files.data.dir}</code>. Send this folder when reporting a problem.
          </p>
          <ul className="small">
            {files.data.files.map((f) => (
              <li key={f.name}>
                {f.name} · {formatBytes(f.size_bytes)}
              </li>
            ))}
          </ul>
        </details>
      )}
      {log.loading && log.data === null && <LoadingState label="Reading the log..." />}
      {log.error !== null && <ErrorState error={log.error} onRetry={log.reload} />}
      {log.data !== null && log.data.length === 0 && <p className="muted small">The log is empty.</p>}
      {log.data !== null && log.data.length > 0 && (
        <pre className="log" tabIndex={0} aria-label="llama-server log lines">
          {log.data.map(logLineText).join("\n")}
        </pre>
      )}
    </Panel>
  );
}
