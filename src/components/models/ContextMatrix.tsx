import { useEffect, useState } from "react";
import { ClipboardCheck, Copy, Ruler } from "lucide-react";
import { api, type MatrixReport, type MatrixRow } from "../../api";
import { messageOf } from "../../lib/errors";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

/** "4K", "32K", "128K", "192K", "baseline" for 0. */
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

/** Generation speed of `row` relative to the empty-context row, in percent. Null when either side is missing. */
export function generationShare(rows: MatrixRow[], row: MatrixRow): number | null {
  const empty = rows.find((r) => r.depth === 0);
  if (!empty?.gen_tps || !row.gen_tps) return null;
  return (row.gen_tps / empty.gen_tps) * 100;
}

/**
 * Plain-English summary of the measured rows: the deepest length where generation stays at
 * least half of the empty-context speed without spilling, or a warning otherwise.
 */
export function summaryText(rows: MatrixRow[]): string | null {
  const nonEmpty = rows.filter((r) => r.depth > 0 && r.gen_tps != null && r.gen_tps > 0);
  if (nonEmpty.length === 0) return null;
  const comfortable = nonEmpty.filter((r) => {
    const share = generationShare(rows, r);
    return share !== null && share >= 50 && !r.spilled;
  });
  if (comfortable.length === 0) {
    const fastButSpilled = nonEmpty.some((r) => {
      const share = generationShare(rows, r);
      return r.spilled && share !== null && share >= 50;
    });
    return fastButSpilled
      ? "Generation stays high, but memory spills into shared RAM at some lengths; pick a shorter window or a smaller cache."
      : "Generation falls below half of the empty-context speed at the measured lengths; keep prompts short or pick a smaller model.";
  }
  const deepest = Math.max(...comfortable.map((r) => r.depth));
  const maxDepth = Math.max(...nonEmpty.map((r) => r.depth));
  return deepest >= maxDepth
    ? "Generation stays at least half of the empty-context speed at every measured length."
    : `Comfortable up to ${depthLabel(deepest)} of context (generation at ≥50% of the empty-context speed); slower beyond that.`;
}

/** A Markdown table of the report, for pasting into issues and documents. */
export function reportToMarkdown(report: MatrixReport): string {
  const lines = [
    `### Context length benchmark — ${report.model_id}`,
    "",
    `Measured ${new Date(report.measured_at * 1000).toISOString()} · ${report.gpu_layers} layers on the GPU · micro-batch ${report.ubatch} · plain decoding (no speculation)`,
    "",
    "| Context | Cache | Prompt tok/s | Generation tok/s | Turn wait | GPU memory |",
    "| --- | --- | --- | --- | --- | --- |",
  ];
  for (const r of report.rows) {
    const memory = memoryText(r);
    if (r.prompt_tps === null || r.prompt_tps === undefined) {
      lines.push(`| ${depthLabel(r.depth)} | - | not measured: ${r.note ?? "no result"} ||||`);
      continue;
    }
    lines.push(
      `| ${depthLabel(r.depth)} | ${CACHE_LABEL[r.kv_cache] ?? r.kv_cache} | ${Math.round(r.prompt_tps).toLocaleString()} | ${(r.gen_tps ?? 0).toFixed(1)} | ${latencyText(r.turn_ms)} | ${memory} |`,
    );
  }
  return lines.join("\n");
}

/**
 * Measures how the model behaves as the context grows: prefill and decode speed, the wait for a
 * 512-token turn, and GPU memory, at 0 to 192K tokens, each with the cheapest cache type that fits.
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
  const [copied, setCopied] = useState(false);

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
      setReport(await api.models.contextMatrix(modelId, deep ? 200_000 : 32_768));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setRunning(false);
    }
  }

  async function copyReport() {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(reportToMarkdown(report));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Could not copy to the clipboard.");
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
          <div className="matrix-header">
            <p className="small muted">
              Measured {new Date(report.measured_at * 1000).toLocaleString()} with {report.gpu_layers} layers on the GPU. Plain
              decoding: the speeds with speculative decoding are higher. The wait is for a 512-token turn arriving at that
              context length.
            </p>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => void copyReport()}
              title="Copy this table as Markdown, ready to paste into an issue or document"
            >
              {copied ? <ClipboardCheck size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
              {copied ? " Copied" : " Copy table"}
            </button>
          </div>
          {summaryText(report.rows) && <p className="small matrix-summary">{summaryText(report.rows)}</p>}
          <table className="small matrix-table" aria-label="Speed and memory by context length">
            <thead>
              <tr>
                <th>Context</th>
                <th>Cache</th>
                <th>Prompt</th>
                <th>Generation</th>
                <th title="Generation speed relative to the empty-context row">% of empty</th>
                <th>Wait for a turn</th>
                <th>GPU memory</th>
              </tr>
            </thead>
            <tbody>
              {report.rows.map((r) =>
                r.prompt_tps === null || r.prompt_tps === undefined ? (
                  <tr key={r.depth} className="matrix-skipped">
                    <td>{depthLabel(r.depth)}</td>
                    <td colSpan={6} className="muted">
                      Not measured: {r.note ?? "no result"}
                    </td>
                  </tr>
                ) : (
                  <tr key={r.depth} className={r.spilled ? "matrix-spilled" : ""}>
                    <td>{depthLabel(r.depth)}</td>
                    <td>{CACHE_LABEL[r.kv_cache] ?? r.kv_cache}</td>
                    <td>{Math.round(r.prompt_tps).toLocaleString()} tok/s</td>
                    <td>{(r.gen_tps ?? 0).toFixed(1)} tok/s</td>
                    <td>{generationShare(report.rows, r)?.toFixed(0) ?? "-"}{generationShare(report.rows, r) !== null ? "%" : ""}</td>
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
            <strong>Up to 200K</strong>
            <span className="muted small block">Measures through 192K plus turn reserve; can take 20 minutes or more because each length reads its whole context.</span>
          </span>
        </label>
      </ConfirmDialog>
    </div>
  );
}
