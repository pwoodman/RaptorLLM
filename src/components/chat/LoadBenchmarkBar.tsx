import { useId, useState } from "react";
import { ChevronDown, ChevronUp, Gauge, TriangleAlert, Zap } from "lucide-react";
import { useStore } from "../../hooks/useStore";
import { openModels } from "../../state/ui";
import { engineStore } from "../../state/engine";
import { formatCount } from "../../lib/format";
import { formatMs, formatTps, recordedLabel } from "../models/LoadReport";

const SOURCE_LABEL = {
  known_good: "Known good",
  recommended: "Recommended",
} as const;

/**
 * The benchmark of the load that brought the current model up, surfaced on the chat page.
 * Collapsed it is one quiet line (speed and first-token latency); "Show load details" expands
 * the full measurement — prompt speed, tokens, samples, GPU offload and stability — without
 * leaving the conversation. Renders nothing when no load is active or the load never made it.
 */
export function LoadBenchmarkBar() {
  const report = useStore(engineStore, (s) => s.loadReport);
  const loaded = useStore(engineStore, (s) => s.loaded);
  const [open, setOpen] = useState(false);
  const detailsId = useId();

  // A load that is still running (no report) or that the session owns nothing of shows nothing.
  if (!report || !loaded) return null;

  const outcome = report.benchmark;
  const record = outcome.entry?.record ?? null;
  const stable = outcome.stable;
  const optimized = report.tuning_applied === true;
  const verdict = report.status.gpu_verdict;
  const offloaded = verdict?.verdict === "verified" ? verdict.offloaded_layers : null;
  const totalLayers = verdict?.verdict === "verified" ? verdict.total_layers : null;
  const gpuLine =
    verdict?.verdict === "verified"
      ? offloaded != null
        ? `${formatCount(offloaded)}${totalLayers != null ? ` / ${formatCount(totalLayers)}` : ""} layers on GPU (verified)`
        : "GPU verified"
      : verdict?.verdict === "cpu_only"
        ? "CPU only — GPU is not running this model"
        : "GPU offload not verified";

  return (
    <div className={`load-benchmark${stable ? "" : " load-benchmark-unstable"}`}>
      <div className="load-benchmark-row">
        <span className={`load-benchmark-glyph${optimized ? " optimized" : ""}`} aria-hidden="true">
          {optimized ? <Zap size={15} fill="currentColor" /> : <Gauge size={15} />}
        </span>
        <span className="load-benchmark-headline">
          {record ? (
            <>
              <strong>{formatTps(record.gen_tps)}</strong>
              <span className="muted"> generation · </span>
              <strong>{formatMs(record.ttft_ms)}</strong>
              <span className="muted"> first token</span>
            </>
          ) : (
            <span className="muted">The after-load benchmark did not complete for this load.</span>
          )}
          {optimized && <span className="load-benchmark-chip">Optimized</span>}
          {!stable && (
            <span className="load-benchmark-chip warn">
              <TriangleAlert size={11} aria-hidden="true" /> Unstable run
            </span>
          )}
        </span>
        <button
          type="button"
          className="load-benchmark-toggle"
          aria-expanded={open}
          aria-controls={detailsId}
          onClick={() => setOpen((current) => !current)}
        >
          {open ? "Hide load details" : "Show load details"}
          {open ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
        </button>
      </div>

      {open && (
        <div id={detailsId} className="load-benchmark-details">
          <dl className="facts compact">
            <div>
              <dt>Prompt speed</dt>
              <dd>{record ? formatTps(record.prompt_tps) : "Unknown"}</dd>
            </div>
            <div>
              <dt>Tokens</dt>
              <dd>
                {record ? `${formatCount(record.prompt_tokens)} in · ${formatCount(record.gen_tokens)} out` : "Unknown"}
              </dd>
            </div>
            <div>
              <dt>GPU offload</dt>
              <dd>{gpuLine}</dd>
            </div>
            <div>
              <dt>Context window</dt>
              <dd>{loaded.ctxSize === null ? "Unknown" : `${formatCount(loaded.ctxSize)} tokens`}</dd>
            </div>
            <div>
              <dt>Config</dt>
              <dd>
                {optimized
                  ? "Optimized — Raptor's saved settings for this model"
                  : SOURCE_LABEL[report.config_source]}
              </dd>
            </div>
            <div>
              <dt>Benchmark</dt>
              <dd>
                {stable
                  ? record
                    ? `Stable · ${record.samples} ${record.samples === 1 ? "sample" : "samples"} · ${recordedLabel(outcome.recorded)}`
                    : `Stable · ${recordedLabel(outcome.recorded)}`
                  : `Unstable${outcome.reasons.length > 0 ? ` — ${outcome.reasons[0]}` : " — not a reliable baseline"}`}
              </dd>
            </div>
          </dl>
          <div className="load-benchmark-foot">
            <span className="small muted">Measured on this machine right after the model became ready.</span>
            <button type="button" className="btn btn-sm" onClick={() => openModels("performance")}>
              Open performance history
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
