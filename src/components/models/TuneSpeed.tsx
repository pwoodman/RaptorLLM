import { useEffect, useState } from "react";
import { Gauge } from "lucide-react";
import { api, type ModelEntry, type TuneReport } from "../../api";
import { messageOf } from "../../lib/errors";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

/** "34.7 tok/s", "1,707 tok/s". */
export function tps(n: number): string {
  return `${n >= 100 ? Math.round(n).toLocaleString() : n.toFixed(1)} tok/s`;
}

/** What the result means in a sentence. Pure, for testing. */
export function tuneSummary(r: TuneReport): string {
  const b = r.best;
  const where = `${b.gpu_layers} layers on the GPU, micro-batch ${b.ubatch}`;
  if (r.improvement_percent < 3) {
    return `The default setup was already the fastest found (${where}): ${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation.`;
  }
  return `${where}: ${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation. About ${Math.round(r.improvement_percent)}% less time for a long prompt plus a reply than the default setup. Saved: every load of this model now uses it.`;
}

/** The long-prompt result of a deep tune in a sentence, or null when none ran or none worked. Pure. */
export function longTuneSummary(r: TuneReport): string | null {
  const b = r.long_best;
  if (b === null || r.long_depth === 0) return null;
  const depth = r.long_depth.toLocaleString("en-US");
  const where = `${b.gpu_layers} layers on the GPU, micro-batch ${b.ubatch}`;
  const gain = r.long_improvement_percent ?? 0;
  const same = b.gpu_layers === r.best.gpu_layers && b.ubatch === r.best.ubatch;
  if (same || gain < 3) {
    return `With ${depth} tokens already in the context the same setup is fastest (${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation), so windows of that size use it too.`;
  }
  return `With ${depth} tokens already in the context, ${where} is faster: ${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation, about ${Math.round(gain)}% less time than the setup above at that depth. Saved: used for context windows of ${depth} tokens or more.`;
}

/**
 * Measures one model at the layer counts and micro-batch sizes worth trying on this machine and
 * remembers the fastest, so loads use it. It runs a benchmark on the GPU for a few minutes, so it
 * is offered only while no model is loaded, and it asks first.
 */
export function TuneSpeed({ entry, disabled }: { entry: ModelEntry; disabled: boolean }) {
  const [asking, setAsking] = useState(false);
  const [deep, setDeep] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [report, setReport] = useState<TuneReport | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    }, 1000);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [running]);

  async function start() {
    setAsking(false);
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      setReport(await api.models.tune(entry.id, deep));
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setRunning(false);
    }
  }

  async function forget() {
    try {
      await api.models.forgetTuning(entry.id);
      setReport(null);
    } catch (e) {
      setError(messageOf(e));
    }
  }

  return (
    <div className="tune-speed">
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        disabled={disabled || running}
        title={disabled && !running ? "Unload the model first: tuning needs the GPU" : "Find the fastest settings for this model"}
        onClick={() => setAsking(true)}
      >
        <Gauge size={13} aria-hidden="true" /> {running ? "Tuning..." : "Optimize speed"}
      </button>
      {running && (
        <div role="status" className="install-progress">
          <progress
            aria-label="Tuning progress"
            max={progress && progress.total > 0 ? progress.total : undefined}
            value={progress && progress.total > 0 ? progress.done : undefined}
          />
          <p className="small muted">
            Measuring{progress && progress.total > 0 ? ` (${progress.done} of ${progress.total} setups done)` : ""}. This
            takes {deep ? "several minutes, longer than a quick tune" : "a few minutes"}. Keep the app open.
          </p>
        </div>
      )}
      {error !== null && (
        <Notice tone="danger" title="Tuning failed">
          {error}
        </Notice>
      )}
      {report !== null && (
        <div className="tune-report">
          <p className="small">{tuneSummary(report)}</p>
          {longTuneSummary(report) !== null && <p className="small">{longTuneSummary(report)}</p>}
          <table className="small tune-table" aria-label="Measured settings">
            <thead>
              <tr>
                <th>GPU layers</th>
                <th>Micro-batch</th>
                <th>Prompt</th>
                <th>Generation</th>
              </tr>
            </thead>
            <tbody>
              {report.results.map((m) => (
                <tr key={`${m.gpu_layers}-${m.ubatch}`} className={m === report.results.find((x) => x.gpu_layers === report.best.gpu_layers && x.ubatch === report.best.ubatch) ? "best" : ""}>
                  <td>{m.gpu_layers}</td>
                  <td>{m.ubatch}</td>
                  <td>{tps(m.prompt_tps)}</td>
                  <td>{tps(m.gen_tps)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {report.errors.map((e) => (
            <p key={e} className="small muted">
              Not measured: {e}
            </p>
          ))}
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => void forget()}>
            Forget this tuning
          </button>
        </div>
      )}
      <ConfirmDialog
        open={asking}
        title={`Optimize ${entry.file_name}?`}
        confirmLabel="Start tuning"
        onCancel={() => setAsking(false)}
        onConfirm={() => void start()}
      >
        <p>
          Raptor will run a short benchmark of this model on the GPU at a few settings and keep the fastest. It takes a
          few minutes, loads the model several times, and uses the GPU and a lot of memory meanwhile, so close other
          heavy programs. Nothing is changed in the model file.
        </p>
        <label className="small">
          <input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} /> Also tune for long
          prompts
        </label>
        <p className="small muted">
          Adds one more run for every setup measured above (usually two to four), each reading 16,384 tokens first, so it
          takes noticeably longer. The progress bar counts every run. The result is used for context windows of 16,384
          tokens or more; smaller windows keep the quick-tune setting.
        </p>
      </ConfirmDialog>
    </div>
  );
}
