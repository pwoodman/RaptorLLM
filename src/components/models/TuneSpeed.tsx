import { useEffect, useState } from "react";
import { Gauge } from "lucide-react";
import { api, type ModelEntry, type TuneReport, type SavedTuning } from "../../api";
import { engineStore } from "../../state/engine";
import { tuningStore } from "../../state/tuning";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";

/** "34.7 tok/s", "1,707 tok/s". */
export function tps(n: number): string {
  return `${n >= 100 ? Math.round(n).toLocaleString() : n.toFixed(1)} tok/s`;
}

function percentChange(base: number, candidate: number): string {
  if (base <= 0) return "n/a";
  const delta = Math.round(((candidate - base) / base) * 100);
  return `${delta > 0 ? "+" : ""}${delta}%`;
}

function formatTtft(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** What the result means in a sentence. Pure, for testing. */
export function tuneSummary(r: TuneReport): string {
  const b = r.best;
  const where = `${b.gpu_layers} layers on the GPU, micro-batch ${b.ubatch}`;
  if (r.improvement_percent < 3) {
    return `The default setup was already the fastest found (${where}): ${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation. Saved for future loads with the same runtime and cache settings.`;
  }
  return `${where}: ${tps(b.prompt_tps)} prompt, ${tps(b.gen_tps)} generation. About ${Math.round(r.improvement_percent)}% less time for a long prompt plus a reply than the default setup. Saved: future loads with the same runtime and cache settings use it.`;
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
export function TuneSpeed({ entry, disabled, onRunningChange, prominent = false }: { entry: ModelEntry; disabled: boolean; onRunningChange?: (running: boolean) => void; prominent?: boolean }) {
  const running = useStore(tuningStore, (s) => s.activeId === entry.id);
  const report = useStore(tuningStore, (s) => s.reports[entry.id] ?? null);
  const setReport = (next: TuneReport | null) => tuningStore.set((s) => ({ reports: { ...s.reports, [entry.id]: next } }));
  const [saved, setSaved] = useState<SavedTuning[]>([]);
  const [savedError, setSavedError] = useState(false);
  const [resetting, setResetting] = useState(false);
  useEffect(() => {
    let current = true;
    api.models.savedTuning(entry.id).then(
      (profiles) => { if (current) { setSaved(profiles); setSavedError(false); } },
      () => { if (current) setSavedError(true); },
    );
    return () => { current = false; };
  }, [entry.id, running]);
  const [asking, setAsking] = useState(false);
  const [deep, setDeep] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
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
    tuningStore.set({ activeId: entry.id });
    engineStore.set({ busy: "optimizing" });
    onRunningChange?.(true);
    setError(null);
    setReport(null);
    try {
      setReport(await api.models.tune(entry.id, deep));
      try { setSaved(await api.models.savedTuning(entry.id)); setSavedError(false); }
      catch { setSavedError(true); }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      tuningStore.set({ activeId: null });
      engineStore.set({ busy: null });
      onRunningChange?.(false);
    }
  }

  async function forget() {
    setResetting(true);
    setError(null);
    try {
      await api.models.forgetTuning(entry.id);
      setReport(null);
      setSaved([]);
    } catch (e) {
      setError(messageOf(e));
    } finally { setResetting(false); }
  }

  return (
    <div className="tune-speed">
      <button
        type="button"
        className={prominent ? "btn btn-primary" : "btn btn-ghost btn-sm"}
        disabled={disabled || running || resetting}
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
            Measuring{progress && progress.total > 0 ? ` (${progress.done} of ${progress.total} benchmark steps done)` : ""}. This
            takes {deep ? "several minutes, longer than a quick tune" : "a few minutes"}. Keep the app open.
          </p>
        </div>
      )}
      {error !== null && (
        <Notice tone="danger" title="Tuning failed">
          {error}
        </Notice>
      )}
      {saved.length > 0 && !running && (
        <div className="saved-tuning" role="status">
          <strong className="small">Speed settings saved</strong>
          <p className="small muted">Applied automatically when the runtime and cache settings match. You can load the model now.</p>
          {saved.map((profile) => <p className="small" key={`${profile.runtime_id}-${profile.runtime_version}`}>
            {profile.runtime_id} {profile.runtime_version}: {profile.tuning.gpu_layers} GPU layers · micro-batch {profile.tuning.ubatch_size}
            {profile.tuning.long && ` · long-context profile from ${profile.tuning.long.depth.toLocaleString()} tokens`}
          </p>)}
          <button type="button" className="btn btn-ghost btn-sm" disabled={disabled || resetting} onClick={() => void forget()}>{resetting ? "Resetting…" : "Reset speed settings"}</button>
        </div>
      )}
      {savedError && <p role="status" className="small muted">Saved speed settings could not be read. Reopen this page to try again.</p>}
      {report !== null && (
        <div className="tune-report">
          <p className="small">{tuneSummary(report)}</p>
          <details className="tune-details">
          <summary>Benchmark details and measured settings</summary>
          {report.baseline !== null && (
            <div className="table-wrap">
              <table className="small tune-table" aria-label="Default and Raptor optimized speed comparison">
                <thead>
                  <tr><th>Configuration</th><th>Prompt</th><th>Generation</th><th>Time to first token</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Default</th>
                    <td>{report.baseline_live ? tps(report.baseline_live.prompt_tps) : tps(report.baseline.prompt_tps)}</td>
                    <td>{report.baseline_live ? tps(report.baseline_live.gen_tps) : tps(report.baseline.gen_tps)}</td>
                    <td>{report.baseline_live ? formatTtft(report.baseline_live.ttft_ms) : "Not measured"}</td>
                  </tr>
                  <tr className="best">
                    <th scope="row">Raptor optimized</th>
                    <td>
                      {report.best_live ? tps(report.best_live.prompt_tps) : tps(report.best.prompt_tps)}
                      {report.baseline_live && report.best_live && (
                        <span className="muted small"> ({percentChange(report.baseline_live.prompt_tps, report.best_live.prompt_tps)})</span>
                      )}
                    </td>
                    <td>
                      {report.best_live ? tps(report.best_live.gen_tps) : tps(report.best.gen_tps)}
                      {report.baseline_live && report.best_live && (
                        <span className="muted small"> ({percentChange(report.baseline_live.gen_tps, report.best_live.gen_tps)})</span>
                      )}
                    </td>
                    <td>
                      {report.best_live ? formatTtft(report.best_live.ttft_ms) : "Not measured"}
                      {report.baseline_live && report.best_live && (
                        <span className="muted small"> ({percentChange(report.baseline_live.ttft_ms, report.best_live.ttft_ms)})</span>
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
              {report.baseline_live && report.best_live ? (
                <p className="muted small">
                  {entry.gguf?.quant_label ?? "Unknown quantization"} model on {report.gpu_name ?? "unknown GPU"}
                  {report.gpu_driver ? ` (driver ${report.gpu_driver})` : " (driver unknown)"}; matched streamed requests on {report.runtime_id ?? "the selected runtime"} {report.runtime_version ?? ""}
                  {report.backend ? ` (${report.backend.toUpperCase()})` : ""}, {report.context_size?.toLocaleString() ?? "same"} context,
                  {report.baseline_live.prompt_tokens_exact ? " exact" : " estimated"} {report.baseline_live.prompt_tokens} prompt tokens,
                  and a 64-token output limit. Values are medians after one warm-up and 3 measured samples per configuration.
                  The percentage is candidate vs default; negative TTFT change is faster. KV cache {report.baseline_live.kv_cache},
                  flash attention {report.baseline_live.flash_attn ? "on" : "off"}, speculation {report.baseline_live.speculative ? "on" : "off"}. Default used {report.baseline_live.gpu_layers} GPU layers and micro-batch {report.baseline_live.ubatch_size}; optimized used {report.best_live.gpu_layers} layers and micro-batch {report.best_live.ubatch_size}.
                  {report.baseline_live.parallel_slots ? ` Both runs used ${report.baseline_live.parallel_slots} request slots.` : ""}
                </p>
              ) : (
                <p className="muted small">
                  Prompt and generation use 3 llama-bench repeats at {report.context_size?.toLocaleString() ?? "the selected"} context, with the same cache and flash-attention mode as the default load, a 2,048-token prompt and 64-token reply. TTFT needs a live
                  streamed comparison and was not measured for this run; llama-bench does not report it. The default/optimized
                  request-time estimate is {report.baseline.seconds.toFixed(1)} s / {report.best.seconds.toFixed(1)} s
                  ({Math.round(report.improvement_percent)}% faster by the tuning score).
                </p>
              )}
            </div>
          )}
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
          </details>
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
          Raptor will try a few setups and save the fastest for this model on your PC. This takes a few minutes.
          Keep Raptor open and pause other GPU-heavy work. When it finishes, load your model as usual; compatible saved
          settings apply automatically.
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
