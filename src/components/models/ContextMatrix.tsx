import { useEffect, useState } from "react";
import { Ruler } from "lucide-react";
import { api, type MatrixReport, type MatrixRow } from "../../api";
import { messageOf } from "../../lib/errors";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

/** "4K", "32K", "128K", "baseline" for 0. */
export function depthLabel(depth: number): string {
  return depth === 0 ? "Empty" : `${Math.round(depth / 1024)}K`;
}

/** Milliseconds as "0.8 s" / "12 s" / "1 min 20 s"; "-" when unknown. */
export function latencyText(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "-";
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function memoryText(row: MatrixRow): string {
  if (row.peak_dedicated_mib === null || row.peak_dedicated_mib === undefined) return "-";
  const gb = (row.peak_dedicated_mib / 1024).toFixed(1);
  return row.spilled ? `${gb} GB + ${Math.round((row.peak_shared_growth_mib ?? 0) / 1024 * 10) / 10} GB shared` : `${gb} GB`;
}

const CACHE_LABEL: Record<string, string> = { f16: "16-bit", q8_0: "8-bit", q4_0: "4-bit" };

/**
 * Measures how the model behaves as the context grows: prefill and decode speed, the wait for a
 * 512-token turn, and GPU memory, at 0 to 128K tokens, each with the cheapest cache type that fits.
 * It runs a benchmark on the GPU for minutes (a 128K run can take 20+), so it asks first and is
 * offered only while no model is loaded. The numbers are plain decoding, without speculation.
 */
export function ContextMatrix({ modelId, disabled }: { modelId: string; disabled: boolean }) {
  const [asking, setAsking] = useState(false);
  const [deep, setDeep] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [report, setReport] = useState<MatrixReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    api.models.contextMatrixLast(modelId).then(
      (r) => {
        if (current) setReport(r);
      },
      () => undefined,
    );
    return () => {
      current = false;
    };
  }, [modelId]);

  useEffect(() => {
    if (!running) {
      setProgress(null);
      return;
    }
    let current = true;
    const timer = setInterval(() => {
      api.models.tuneProgress().then(
        (p) => {
          if (current) setProgress(p);
        },
        () => undefined,
      );
    }, 2000);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [running]);

  async function start() {
    setAsking(false);
    setRunning(true);
    setError(null);
    try {
      setReport(await api.models.contextMatrix(modelId, deep ? 131_072 : 32_768));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="context-matrix">
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        disabled={disabled || running}
        title={disabled && !running ? "Unload the model first: the benchmark needs the GPU" : "Measure speed and memory at growing context lengths"}
        onClick={() => setAsking(true)}
      >
        <Ruler size={13} aria-hidden="true" /> {running ? "Measuring..." : "Benchmark context lengths"}
      </button>
      {running && (
        <div role="status" className="install-progress">
          <progress
            aria-label="Context benchmark progress"
            max={progress && progress.total > 0 ? progress.total : undefined}
            value={progress && progress.total > 0 ? progress.done : undefined}
          />
          <p className="small muted">
            Measuring{progress && progress.total > 0 ? ` (${progress.done} of ${progress.total} lengths done)` : ""}. Keep the app
            open; longer contexts take longer.
          </p>
        </div>
      )}
      {error !== null && (
        <Notice tone="danger" title="Benchmark failed">
          {error}
        </Notice>
      )}
      {report !== null && (
        <div className="matrix-report">
          <p className="small muted">
            Measured {new Date(report.measured_at * 1000).toLocaleString()} with {report.gpu_layers} layers on the GPU. Plain
            decoding: the speeds with speculative decoding are higher. The wait is for a 512-token turn arriving at that
            context length.
          </p>
          <table className="small matrix-table" aria-label="Speed and memory by context length">
            <thead>
              <tr>
                <th>Context</th>
                <th>Cache</th>
                <th>Prompt</th>
                <th>Generation</th>
                <th>Wait for a turn</th>
                <th>GPU memory</th>
              </tr>
            </thead>
            <tbody>
              {report.rows.map((r) =>
                r.prompt_tps === null || r.prompt_tps === undefined ? (
                  <tr key={r.depth} className="matrix-skipped">
                    <td>{depthLabel(r.depth)}</td>
                    <td colSpan={5} className="muted">
                      Not measured: {r.note ?? "no result"}
                    </td>
                  </tr>
                ) : (
                  <tr key={r.depth} className={r.spilled ? "matrix-spilled" : ""}>
                    <td>{depthLabel(r.depth)}</td>
                    <td>{CACHE_LABEL[r.kv_cache] ?? r.kv_cache}</td>
                    <td>{Math.round(r.prompt_tps).toLocaleString()} tok/s</td>
                    <td>{(r.gen_tps ?? 0).toFixed(1)} tok/s</td>
                    <td>{latencyText(r.turn_ms)}</td>
                    <td>{memoryText(r)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
          {report.rows.some((r) => r.spilled) && (
            <p className="small muted">
              A row with shared memory used more than the GPU holds: the model works but generation slows down. Pick a smaller
              window or cache for that length.
            </p>
          )}
          {report.prompt_seconds.length > 1 && (
            <p className="small muted">
              Reading a whole prompt from empty:{" "}
              {report.prompt_seconds
                .filter(([d]) => d > 0)
                .map(([d, s]) => `${depthLabel(d)} in ${latencyText(s * 1000)}`)
                .join(", ")}
              .
            </p>
          )}
        </div>
      )}
      <ConfirmDialog
        open={asking}
        title="Benchmark context lengths?"
        confirmLabel="Start"
        onCancel={() => setAsking(false)}
        onConfirm={() => void start()}
      >
        <p>
          Raptor will run a benchmark of this model on the GPU at growing context lengths and record speed, the wait for a
          turn and GPU memory. It loads the model once per length and uses the GPU and a lot of memory meanwhile, so close
          other heavy programs.
        </p>
        <label className="radio">
          <input type="radio" name="matrix-depth" checked={!deep} onChange={() => setDeep(false)} />
          <span>
            <strong>Up to 32K</strong>
            <span className="muted small block">A few minutes.</span>
          </span>
        </label>
        <label className="radio">
          <input type="radio" name="matrix-depth" checked={deep} onChange={() => setDeep(true)} />
          <span>
            <strong>Up to 128K</strong>
            <span className="muted small block">Can take 20 minutes or more: every length first reads its whole context.</span>
          </span>
        </label>
      </ConfirmDialog>
    </div>
  );
}
