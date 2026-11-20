import { useState, type FormEvent } from "react";
import { Activity, Play } from "lucide-react";
import {
  perfBestKnown,
  perfCompare,
  perfHistory,
  perfRunBenchmark,
  type BenchmarkOutcome,
  type ModelEntry,
  type PerfComparison,
  type PerfEntry,
} from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import { formatBytes, formatCount } from "../../lib/format";
import { engineStore } from "../../state/engine";
import { EmptyState, ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { BenchmarkResult, formatMs, formatTps } from "./LoadReport";

interface PerfData {
  history: PerfEntry[];
  best: PerfEntry | null;
}

export function PerformancePanel({ models }: { models: ModelEntry[] | null }) {
  const loaded = useStore(engineStore, (s) => s.loaded);
  const [selected, setSelected] = useState<string | null>(null);
  const modelId = selected ?? loaded?.modelId ?? models?.[0]?.id ?? null;
  const picked = models?.find((m) => m.id === modelId) ?? null;

  return (
    <Panel
      id="performance"
      title="Performance history"
      icon={<Activity size={18} aria-hidden="true" />}
      description="Benchmarks recorded on this machine. Unstable runs are marked and should not be used as baselines."
    >
      {models === null && <LoadingState label="Reading the library..." />}
      {models !== null && models.length === 0 && (
        <EmptyState icon={Activity} title="No models installed">
          Benchmarks are recorded for installed models.
        </EmptyState>
      )}
      {models !== null && models.length > 0 && modelId !== null && (
        <>
          <div className="field">
            <label htmlFor="perf-model" className="field-label">
              Model
            </label>
            <select id="perf-model" value={modelId} onChange={(e) => setSelected(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.file_name}
                </option>
              ))}
            </select>
          </div>
          <ModelHistory key={modelId} modelId={modelId} modelName={picked?.file_name ?? modelId} />
        </>
      )}
    </Panel>
  );
}

function ModelHistory({ modelId, modelName }: { modelId: string; modelName: string }) {
  const data = useAsyncResource<PerfData>(
    async () => {
      const [history, best] = await Promise.all([perfHistory(modelId), perfBestKnown(modelId)]);
      return { history, best };
    },
    [modelId],
  );

  return (
    <div className="perf-model">
      {data.loading && data.data === null && <LoadingState label="Reading benchmark history..." />}
      {data.error !== null && <ErrorState error={data.error} onRetry={data.reload} />}
      {data.data && (
        <>
          <BestKnown best={data.data.best} />
          <RunBenchmark modelId={modelId} modelName={modelName} onDone={data.reload} />
          <History entries={data.data.history} modelName={modelName} />
          {data.data.history.length >= 2 && <Compare entries={data.data.history} />}
        </>
      )}
    </div>
  );
}

function BestKnown({ best }: { best: PerfEntry | null }) {
  if (best === null) {
    return (
      <p className="muted small">No stable benchmark yet. Run one after loading this model.</p>
    );
  }
  return (
    <Notice tone="success" title="Best known on this machine">
      Generation {formatTps(best.record.gen_tps)}, prompt {formatTps(best.record.prompt_tps)}, first token{" "}
      {formatMs(best.record.ttft_ms)} ({best.record.runtime_id} {best.record.runtime_version}).
    </Notice>
  );
}

function RunBenchmark({ modelId, modelName, onDone }: { modelId: string; modelName: string; onDone: () => void }) {
  const loaded = useStore(engineStore, (s) => s.loaded);
  const busy = useStore(engineStore, (s) => s.busy);
  const [running, setRunning] = useState(false);
  const [outcome, setOutcome] = useState<BenchmarkOutcome | null>(null);
  const [error, setError] = useState<unknown>(null);
  const isLoaded = loaded?.modelId === modelId;

  async function run() {
    setRunning(true);
    setError(null);
    setOutcome(null);
    try {
      setOutcome(await perfRunBenchmark());
      onDone();
    } catch (e) {
      setError(e);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="perf-run">
      <button type="button" className="btn btn-primary" disabled={!isLoaded || running || busy !== null} onClick={() => void run()}>
        <Play size={14} aria-hidden="true" /> {running ? "Running benchmark..." : "Run benchmark"}
      </button>
      {!isLoaded && (
        <span className="muted small">
          Load {modelName} first. The benchmark runs on the model that is loaded now.
        </span>
      )}
      {outcome && <BenchmarkResult outcome={outcome} />}
      {error !== null && <ErrorState error={error} />}
    </div>
  );
}

function History({ entries, modelName }: { entries: PerfEntry[]; modelName: string }) {
  if (entries.length === 0) {
    return <p className="muted small">No benchmark records for {modelName} yet.</p>;
  }
  const best = entries.reduce((m, e) => (e.record.stable && e.record.gen_tps > m ? e.record.gen_tps : m), 0);
  const stable = entries.filter((e) => e.record.stable).length;
  return (
    <div className="table-wrap">
      <p className="muted small">
        {entries.length} {entries.length === 1 ? "run" : "runs"} · {stable} stable
        {best > 0 ? ` · best ${best.toFixed(1)} tok/s generation` : " · none stable yet"}
      </p>
      <table className="data-table perf-table">
        <caption className="sr-only">Benchmark runs for {modelName}, newest first</caption>
        <thead>
          <tr>
            <th scope="col">Measured</th>
            <th scope="col">Runtime</th>
            <th scope="col">GPU layers</th>
            <th scope="col">First token</th>
            <th scope="col">Prompt</th>
            <th scope="col">Generation</th>
            <th scope="col">vs best</th>
            <th scope="col">VRAM</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => {
            const r = entry.record;
            const hardware = [
              r.gpu_name ?? "GPU unknown",
              r.gpu_driver ? `driver ${r.gpu_driver}` : null,
              r.quant_label ?? "quant unknown",
            ].filter(Boolean).join(" · ");
            const acceleration = [
              `${formatCount(r.config.ctx_size)} context`,
              `${r.config.kv_cache} KV`,
              `flash attention ${r.config.flash_attn ? "on" : "off"}`,
              r.config.speculative ? `speculation ${r.config.speculative}` : "no speculation",
              r.config.uses_draft_model ? "draft model" : null,
            ].filter(Boolean).join(" · ");
            return (
              <tr key={entry.id} className={r.stable ? undefined : "perf-unstable"}>
                <td>{new Date(r.measured_at * 1000).toLocaleString()}</td>
                <td>
                  {r.runtime_id} {r.runtime_version}
                  <div className="muted small">{hardware}</div>
                  <div className="muted small">{acceleration}</div>
                  {(r.prompt_tokens > 0 || r.gen_tokens > 0) && (
                    <div className="muted small" title="Prompt tokens processed · tokens generated (averaged over merged runs)">
                      {r.prompt_tokens} in · {r.gen_tokens} out
                    </div>
                  )}
                </td>
                <td>{r.config.gpu_layers}</td>
                <td>{formatMs(r.ttft_ms)}</td>
                <td>{formatTps(r.prompt_tps)}</td>
                <td>{formatTps(r.gen_tps)}</td>
                <td className="num">
                  {!r.stable || best <= 0
                    ? "—"
                    : (best - r.gen_tps) < 0.005 * best
                      ? "best"
                      : `${(((r.gen_tps - best) / best) * 100).toFixed(1)}%`}
                </td>
                <td>{r.vram_bytes > 0 ? formatBytes(r.vram_bytes) : "Unknown"}</td>
                <td>
                  {r.stable ? (
                    <span className="badge tone-ok">Stable</span>
                  ) : (
                    <span className="badge tone-warn">Unstable, not a baseline</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Compare({ entries }: { entries: PerfEntry[] }) {
  const [baseId, setBaseId] = useState(entries[entries.length - 1]?.id ?? "");
  const [candidateId, setCandidateId] = useState(entries[0]?.id ?? "");
  const [result, setResult] = useState<PerfComparison | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function compare(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await perfCompare(baseId, candidateId));
    } catch (e) {
      setResult(null);
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const label = (entry: PerfEntry) =>
    `${new Date(entry.record.measured_at * 1000).toLocaleString()} · ${entry.record.stable ? "stable" : "unstable"}`;

  return (
    <form className="perf-compare" onSubmit={(e) => void compare(e)} aria-label="Compare two runs">
      <h3 className="subsection-title">Compare two runs</h3>
      <div className="field-row">
        <div className="field">
          <label htmlFor="perf-base" className="field-label">
            Baseline
          </label>
          <select id="perf-base" value={baseId} onChange={(e) => setBaseId(e.target.value)}>
            {entries.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {label(entry)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="perf-candidate" className="field-label">
            Compare with
          </label>
          <select id="perf-candidate" value={candidateId} onChange={(e) => setCandidateId(e.target.value)}>
            {entries.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {label(entry)}
              </option>
            ))}
          </select>
        </div>
        <button type="submit" className="btn" disabled={busy || baseId === "" || candidateId === "" || baseId === candidateId}>
          {busy ? "Comparing..." : "Compare"}
        </button>
      </div>
      {baseId === candidateId && <p className="muted small">Choose two different runs.</p>}
      {error !== null && <ErrorState error={error} />}
      {result && <ComparisonView result={result} />}
    </form>
  );
}

function signed(pct: number): string {
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

function ComparisonView({ result }: { result: PerfComparison }) {
  const unstable = !result.base.record.stable || !result.candidate.record.stable;
  const c = result.comparison;
  return (
    <div className="perf-result">
      {unstable && (
        <Notice tone="warn" title="One of these runs is unstable">
          The differences may reflect a failed or partial run rather than the settings. Compare stable runs for a fair result.
        </Notice>
      )}
      <dl className="facts compact">
        <div>
          <dt>Generation speed</dt>
          <dd>{signed(c.gen_tps_delta_pct)} (higher is better)</dd>
        </div>
        <div>
          <dt>Prompt speed</dt>
          <dd>{signed(c.prompt_tps_delta_pct)} (higher is better)</dd>
        </div>
        <div>
          <dt>First token</dt>
          <dd>
            {signed(c.ttft_delta_pct)} ({c.ttft_delta_pct < 0 ? "candidate answers faster" : "candidate answers slower or the same"})
          </dd>
        </div>
        <div>
          <dt>VRAM</dt>
          <dd>{c.vram_delta_bytes === 0 ? "No change" : `${c.vram_delta_bytes > 0 ? "+" : "-"}${formatBytes(Math.abs(c.vram_delta_bytes))}`}</dd>
        </div>
      </dl>
    </div>
  );
}
