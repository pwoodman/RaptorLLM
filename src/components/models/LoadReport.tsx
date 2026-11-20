import { ClipboardList } from "lucide-react";
import type { BenchmarkOutcome, ConfigSource, LoadModelOutcome } from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatCount } from "../../lib/format";
import { engineStore } from "../../state/engine";
import { GpuVerdictView } from "../GpuVerdictView";
import { Notice } from "../Notice";
import { Panel } from "../Panel";

const SOURCE_LABEL: Record<ConfigSource, string> = {
  known_good: "Known good: the fastest stable settings measured on this machine for this model",
  recommended: "Recommended: no measured settings yet, so Raptor estimated them",
};

export function LoadMemoryWarning({ report }: { report: LoadModelOutcome | null }) {
  const memory = report?.gpu_memory;
  if (!memory?.spilled || memory.expected) return null;
  return (
    <Notice tone="warn" title="This load spilled into shared system memory">
      {memory.shared_growth_mib.toLocaleString()} MiB did not fit in VRAM and went into shared system
      memory. The model works, but generation can be several times slower. Choose a smaller context window or
      cache, or a smaller model.
    </Notice>
  );
}

/** What the last load in this session used, how the GPU behaved, and the benchmark taken right after. */
export function LoadReport() {
  const report = useStore(engineStore, (s) => s.loadReport);
  if (report === null) return null;

  const verdict = report.status.gpu_verdict;
  const offloaded = verdict?.verdict === "verified" ? verdict.offloaded_layers : null;
  const total = verdict?.verdict === "verified" ? verdict.total_layers : null;

  return (
    <Panel
      id="load-report"
      title="Last load report"
      icon={<ClipboardList size={18} aria-hidden="true" />}
      description="Settings used for the most recent load in this session, and the benchmark taken after it became ready."
    >
      {report.cpu_fallback && (
        <Notice tone="danger" title="CPU fallback: not GPU accelerated">
          The GPU backend had no device, and CPU fallback is allowed. This model runs on the CPU.
        </Notice>
      )}
      <LoadMemoryWarning report={report} />
      {report.gpu_memory?.spilled && report.gpu_memory.expected && (
        <Notice tone="info" title="Part of the model is in shared system memory">
          {report.gpu_memory.shared_growth_mib.toLocaleString()} MiB sits in shared memory, as planned: only a few
          experts of this Mixture-of-experts model are read per token, so this is faster than splitting layers.
        </Notice>
      )}
      <dl className="facts">
        <div>
          <dt>Config source</dt>
          <dd>{report.tuning_applied === true
            ? "Optimized: saved speed settings applied for this model and runtime"
            : SOURCE_LABEL[report.config_source]}</dd>
        </div>
        {report.preset !== null && (
          <div>
            <dt>Preset</dt>
            <dd>{report.preset}</dd>
          </div>
        )}
        <div>
          <dt>GPU offload</dt>
          <dd>
            {offloaded != null
              ? `${formatCount(offloaded)}${total != null ? ` / ${formatCount(total)}` : ""} layers on GPU (verified)`
              : "Not verified, so no layers are counted as GPU"}
          </dd>
        </div>
      </dl>

      <GpuVerdictView verdict={verdict} />

      {report.rationale.length > 0 && (
        <details className="rationale" open>
          <summary>Why these settings ({report.rationale.length})</summary>
          <ul>
            {report.rationale.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </details>
      )}

      <BenchmarkResult outcome={report.benchmark} optimized={report.tuning_applied === true} />
    </Panel>
  );
}

export function BenchmarkResult({ outcome, optimized = false }: { outcome: BenchmarkOutcome; optimized?: boolean }) {
  const record = outcome.entry?.record ?? null;
  return (
    <section className="benchmark-result" aria-label="Benchmark result">
      <h3 className="subsection-title">Benchmark after load</h3>
      {!outcome.stable && (
        <Notice tone="warn" title="Unstable run. These numbers are not a reliable baseline.">
          {outcome.reasons.length > 0 ? (
            <ul className="evidence">
              {outcome.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : (
            "The run did not finish cleanly."
          )}
        </Notice>
      )}
      {record ? (
        <dl className="facts compact">
          <div>
            <dt>Time to first token</dt>
            <dd>{formatMs(record.ttft_ms)}</dd>
          </div>
          <div>
            <dt>Prompt speed</dt>
            <dd className={optimized ? "optimized-speed" : undefined}>{formatTps(record.prompt_tps)}</dd>
          </div>
          <div>
            <dt>Generation speed</dt>
            <dd className={optimized ? "optimized-speed" : undefined}>{formatTps(record.gen_tps)}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>
              <span className={`badge ${outcome.stable ? "tone-ok" : "tone-warn"}`}>
                {outcome.stable ? "Stable" : "Unstable"}
              </span>{" "}
              <span className="muted small">
                {record.samples} samples · {recordedLabel(outcome.recorded)}
              </span>
            </dd>
          </div>
        </dl>
      ) : (
        <p className="muted small">No benchmark record was made. The hardware may be unknown, or the run failed.</p>
      )}
      {outcome.notes.map((note) => (
        <p key={note} className="muted small">
          {note}
        </p>
      ))}
    </section>
  );
}

export function formatMs(ms: number): string {
  return Number.isFinite(ms) ? `${Math.round(ms)} ms` : "Unknown";
}

export function formatTps(tps: number): string {
  return Number.isFinite(tps) ? `${tps.toFixed(1)} tok/s` : "Unknown";
}

function recordedLabel(recorded: BenchmarkOutcome["recorded"]): string {
  switch (recorded) {
    case "inserted":
      return "saved as a new record";
    case "merged":
      return "merged into an existing record";
    case "kept_stable":
      return "kept the earlier stable record";
    case "not_saved":
      return "not saved";
  }
}
