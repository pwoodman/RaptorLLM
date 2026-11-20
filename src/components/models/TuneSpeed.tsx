import { useEffect, useState } from "react";
import { Gauge } from "lucide-react";
import { api, type ModelEntry, type TuneReport, type SavedTuning } from "../../api";
import { engineStore, loadModel } from "../../state/engine";
import { chatStore } from "../../state/chat";
import { tuningStore } from "../../state/tuning";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";
import { formatTps } from "./LoadReport";

export function tuneSummary(r: TuneReport): string {
  const window = r.context_size.toLocaleString("en-US");
  const outcome = r.saved ? "Qualified settings saved." : r.retained_profile ? "Previous qualified settings retained." : "Default retained.";
  if (r.results.length === 1) return `${outcome} Only the default completed the ${window}-token test.`;
  if (r.finalist !== null && r.confirmation.length < 3) return `${outcome} Candidate ${r.finalist} completed ${r.confirmation.length} of three required confirmation pairs at ${window} tokens.`;
  if (!r.confirmation.length) return `${outcome} No candidate advanced to confirmation at ${window} tokens.`;
  const gain = Math.min(...r.confirmation.map(([base, candidate]) => (base / candidate - 1) * 100));
  const minimum = (Math.floor(gain * 10) / 10).toFixed(1);
  return `${outcome} Candidate ${r.finalist}. Lowest speed change across three matched ${window}-token tests: ${minimum}%. Qualification requires at least 50% in every pair.`;
}

export function TuneSpeed({ entry, disabled, onRunningChange, prominent = false }: { entry: ModelEntry; disabled: boolean; onRunningChange?: (running: boolean) => void; prominent?: boolean }) {
  const running = useStore(tuningStore, (s) => s.activeId === entry.id);
  const startedAt = useStore(tuningStore, (s) => s.startedAt);
  const pending = useStore(chatStore, (s) => s.pending);
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
    if (engineStore.get().busy !== null || chatStore.get().pending !== null) return;
    setAsking(false);
    const previous = engineStore.get().loaded;
    const resumeChat = previous?.modelId === entry.id;
    tuningStore.set({ activeId: entry.id, startedAt: Date.now() });
    engineStore.set({ busy: "optimizing" });
    onRunningChange?.(true);
    setError(null);
    setReport(null);
    try {
      if (engineStore.get().loaded || (engineStore.get().status && engineStore.get().status?.server.state.state !== "stopped")) {
        const status = await api.models.unload();
        engineStore.set({ status, loaded: null, loadReport: null });
      }
      setReport(await api.models.tune(entry.id, deep, resumeChat ? previous.window?.ctx ?? null : null, resumeChat ? previous.preset ?? null : null));
      try { setSaved(await api.models.savedTuning(entry.id)); setSavedError(false); }
      catch { setSavedError(true); }
      if (resumeChat) {
        tuningStore.set({ activeId: null, startedAt: null });
        engineStore.set({ busy: null });
        await loadModel(entry.id, previous?.preset ?? null, previous?.window ?? null);
      }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      tuningStore.set({ activeId: null, startedAt: null });
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
        disabled={disabled || running || resetting || pending !== null}
        title={pending ? "Finish the current reply first" : disabled && !running ? "Wait for the current model task to finish" : "Find the fastest settings for this model"}
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
            Testing{progress && progress.total > 0 ? ` setup ${Math.min(progress.done + 1, progress.total)} of ${progress.total}` : " the first setup"}{startedAt ? ` · ${Math.floor((Date.now() - startedAt) / 1000)} s elapsed` : ""}. This
            may take a long time at full context. Keep the app open.
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
          <p className="small muted">Qualified profiles apply to matching single-request windows.</p>
          {saved.map((profile) => <div className="small" key={`${profile.runtime_id}-${profile.runtime_version}`}>
            {profile.runtime_id} {profile.runtime_version}: {profile.tuning.gpu_layers} GPU layers · micro-batch {profile.tuning.ubatch_size}
            {profile.tuning.qualification ? ` · qualified at ${profile.tuning.qualification.context_size.toLocaleString()} tokens · ${profile.tuning.profile?.kv_cache} KV cache` : " · legacy measurement, run Optimize speed to qualify"}
            {profile.tuning.qualification && <details>
              <summary>Saved benchmark evidence</summary>
              <p>Single-slot requests with 64 generated tokens. Elapsed times in default and qualified order.</p>
              <ul>{profile.tuning.qualification.pairs.map(([baseline, candidate], index) =>
                <li key={index}>{baseline.toFixed(2)} s and {candidate.toFixed(2)} s ({(baseline / candidate).toFixed(3)}×)</li>)}</ul>
            </details>}
          </div>)}
          <button type="button" className="btn btn-ghost btn-sm" disabled={disabled || resetting} onClick={() => void forget()}>{resetting ? "Resetting…" : "Reset speed settings"}</button>
        </div>
      )}
      {savedError && <p role="status" className="small muted">Saved speed settings could not be read. Reopen this page to try again.</p>}
      {report !== null && (
        <div className="tune-report">
          <p className="small">{tuneSummary(report)}</p>
          <details className="tune-details">
          <summary>Benchmark details and measured settings</summary>
          <p className="small">Each run uses one slot and fresh synthetic input. First-token time is estimated from the server’s timings. Combined speed counts input and output tokens over the full request.</p>
          <div className="table-wrap">
            <table className="small tune-table" aria-label="Measured settings">
              <thead><tr><th scope="col">Setup</th><th scope="col">GPU layers</th><th scope="col">Micro-batch</th><th scope="col">KV cache</th><th scope="col">Request time</th><th scope="col">Time to first token (est.)</th><th scope="col">Input</th><th scope="col">Output</th><th scope="col">Combined speed</th></tr></thead>
              <tbody>{report.results.map((trial, i) => (
                <tr key={i}>
                  <th scope="row">{i === 0 ? "Default" : `Candidate ${i}`}</th>
                  <td>{trial.gpu_layers}</td>
                  <td>{trial.ubatch}</td>
                  <td>{trial.kv_cache}</td>
                  <td>{trial.seconds.toFixed(2)} s</td>
                  <td>{trial.estimated_ttft_ms == null ? "Unknown" : `${(trial.estimated_ttft_ms / 1000).toFixed(2)} s`}</td>
                  <td>{(report.context_size - report.output_tokens).toLocaleString()} tokens<br />{formatTps(trial.prompt_tps ?? NaN)}</td>
                  <td>{report.output_tokens.toLocaleString()} tokens<br />{formatTps(trial.gen_tps ?? NaN)}</td>
                  <td>{formatTps(trial.seconds > 0 ? report.context_size / trial.seconds : NaN)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          {report.confirmation.length > 0 && <table className="small tune-table" aria-label="Independent confirmation">
            <caption>Independent runs of the default and candidate {report.finalist}</caption>
            <thead><tr><th>Pair</th><th>Default</th><th>Candidate</th><th>Speed ratio</th></tr></thead>
            <tbody>{report.confirmation.map(([base, candidate], i) => <tr key={i}><td>{i + 1}</td><td>{base.toFixed(2)} s</td><td>{candidate.toFixed(2)} s</td><td>{(base / candidate).toFixed(3)}×</td></tr>)}</tbody>
          </table>}
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
          Raptor compares GPU settings at your configured context target, with a minimum of 65,536 tokens in one request slot.
          The profile uses one slot on compatible loads. Candidates can change KV-cache precision while keeping the same model weights. Each candidate processes the entire prompt. On large models this can take hours. A profile is saved when three fresh matched runs each reach 1.50× baseline throughput.
          Tuning unloads the current model and reloads it afterward if it was selected for this chat.
        </p>
        <label className="small">
          <input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} /> Extended search
        </label>
        <p className="small muted">Also try intermediate offload, micro-batches from 128 to 1,024, and 4-bit and 8-bit KV caches.</p>
      </ConfirmDialog>
    </div>
  );
}
