import { useEffect, useState } from "react";
import { GitCompare, RefreshCw, Trophy } from "lucide-react";
import {
  modelsCheckUpdates,
  modelsCompare,
  modelsDetails,
  modelsRankFor,
  type CompatRating,
  type Comparison,
  type ModelTask,
  type RankedCandidate,
  type UpdateCheck,
} from "../../api";
import { useStore } from "../../hooks/useStore";
import { formatBytes } from "../../lib/format";
import { describeError } from "../../lib/errors";
import { libraryStore } from "../../state/library";
import { EmptyState, ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";

/** The most models one comparison may hold (the backend limit). */
const MAX_COMPARE = 8;

const TASKS: { id: ModelTask; label: string }[] = [
  { id: "coding", label: "Coding" },
  { id: "reasoning", label: "Reasoning" },
  { id: "general", label: "General" },
  { id: "agentic", label: "Agentic" },
];

const FIT_LABEL: Record<CompatRating, string> = {
  Great: "Fits well",
  Ok: "Fits",
  Tight: "Tight on memory",
  WontFit: "Will not fit",
};

interface UpdateRow {
  check: UpdateCheck;
  /** The revision pinned at download, from `models_details`. Null when not recorded. */
  pinned: string | null;
}

/** Compare, rank and update-check for installed models. Every figure here is heuristic or a read-only check. */
export function CompareTab() {
  const models = useStore(libraryStore, (s) => s.models);
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<unknown>(null);

  const [task, setTask] = useState<ModelTask>("coding");
  const [ranking, setRanking] = useState<RankedCandidate[] | null>(null);
  const [rankingBusy, setRankingBusy] = useState(false);
  const [rankError, setRankError] = useState<unknown>(null);

  const [updates, setUpdates] = useState<UpdateRow[] | null>(null);
  const [checking, setChecking] = useState(false);
  const [updateError, setUpdateError] = useState<unknown>(null);

  // Forget a comparison when the library changes under it, so no row names a model that is gone.
  useEffect(() => {
    if (models === null) return;
    const installed = new Set(models.map((m) => m.id));
    setSelected((current) => current.filter((id) => installed.has(id)));
    setComparison(null);
  }, [models]);

  const nameOf = (id: string) => models?.find((m) => m.id === id)?.file_name ?? id;

  function toggle(id: string) {
    setSelected((current) => {
      if (current.includes(id)) return current.filter((x) => x !== id);
      return current.length >= MAX_COMPARE ? current : [...current, id];
    });
  }

  async function compare() {
    setComparing(true);
    setCompareError(null);
    try {
      setComparison(await modelsCompare(selected, null));
    } catch (e) {
      setCompareError(e);
      setComparison(null);
    } finally {
      setComparing(false);
    }
  }

  async function rank() {
    setRankingBusy(true);
    setRankError(null);
    try {
      setRanking(await modelsRankFor(task, null));
    } catch (e) {
      setRankError(e);
      setRanking(null);
    } finally {
      setRankingBusy(false);
    }
  }

  async function checkUpdates() {
    setChecking(true);
    setUpdateError(null);
    try {
      const checks = await modelsCheckUpdates();
      // The pinned revision comes from each model's details, which is the record of what was downloaded.
      const details = await Promise.all(checks.map((c) => modelsDetails(c.id).catch(() => null)));
      setUpdates(checks.map((check, i) => ({ check, pinned: details[i]?.revision ?? check.pinned_revision ?? null })));
    } catch (e) {
      setUpdateError(e);
      setUpdates(null);
    } finally {
      setChecking(false);
    }
  }

  if (models === null) return <LoadingState label="Reading installed models..." />;
  if (models.length === 0) {
    return (
      <EmptyState icon={GitCompare} title="Nothing to compare yet">
        Install at least one model in Library or Discover. Comparison, ranking and update checks use installed models.
      </EmptyState>
    );
  }

  return (
    <div className="compare-tab">
      <Panel
        id="compare-models"
        title="Compare models"
        icon={<GitCompare size={18} aria-hidden="true" />}
        description={`Pick 1 to ${MAX_COMPARE} installed models. Estimates use this machine's RAM and VRAM and the context in Settings.`}
      >
        <fieldset className="check-group">
          <legend className="sr-only">Models to compare</legend>
          {models.map((m) => {
            const checked = selected.includes(m.id);
            const disabled = !checked && selected.length >= MAX_COMPARE;
            return (
              <label key={m.id} className="check-row">
                <input type="checkbox" checked={checked} disabled={disabled} onChange={() => toggle(m.id)} />
                <span className="break">{m.file_name}</span>
                <span className="muted small">
                  {m.gguf.quant_label ?? "Unknown quant"} · {formatBytes(m.size_bytes)}
                </span>
              </label>
            );
          })}
        </fieldset>
        <p className="muted small" role="status">
          {selected.length} of {MAX_COMPARE} selected.
        </p>
        <button type="button" className="btn btn-primary" disabled={selected.length === 0 || comparing} onClick={() => void compare()}>
          {comparing ? "Comparing..." : "Compare selected"}
        </button>

        {compareError !== null && <ErrorState error={compareError} />}
        {comparison !== null && (
          <div className="compare-result">
            <p className="muted small">
              Context {comparison.ctx.toLocaleString()} tokens.{" "}
              {comparison.hardware
                ? `This machine: ${formatBytes(comparison.hardware.ram_bytes)} RAM, ${formatBytes(comparison.hardware.vram_bytes)} VRAM.`
                : "Hardware is unknown, so fit ratings are not available."}{" "}
              Sizes and tags are heuristics from the file, not measured results.
            </p>
            <div className="table-wrap">
              <table className="data-table">
                <caption className="sr-only">Side-by-side comparison of the selected models</caption>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">Quant</th>
                    <th scope="col">Bits per weight</th>
                    <th scope="col">Size at context</th>
                    <th scope="col">Fit</th>
                    <th scope="col">Likely strengths (heuristic)</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row) => (
                    <tr key={row.id}>
                      <th scope="row" className="break">
                        {row.name}
                        {row.ctx_exceeds_train && <span className="muted small"> · context above training length</span>}
                      </th>
                      <td>{row.quant_label ?? "Unknown"}</td>
                      <td>{row.bits_per_weight === null ? "Unknown" : row.bits_per_weight.toFixed(2)}</td>
                      <td>{row.estimated_bytes_at_ctx === null ? "Unknown" : formatBytes(row.estimated_bytes_at_ctx)}</td>
                      <td>{row.compat === null ? "Unknown" : FIT_LABEL[row.compat]}</td>
                      <td>
                        {row.tags.length === 0
                          ? "None suggested"
                          : row.tags.map((t) => `${t.task} (${t.score})`).join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {comparison === null && selected.length > 0 && (
          <p className="muted small">Selected: {selected.map(nameOf).join(", ")}</p>
        )}
      </Panel>

      <Panel
        id="best-for"
        title="Best for a task"
        icon={<Trophy size={18} aria-hidden="true" />}
        description="Ranks installed models for a kind of work. Heuristic: it reads file names, quantization and size, not benchmark results."
      >
        <div className="field-row">
          <div className="field">
            <label htmlFor="rank-task" className="field-label">
              Task
            </label>
            <select id="rank-task" value={task} onChange={(e) => setTask(e.target.value as ModelTask)}>
              {TASKS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <button type="button" className="btn" disabled={rankingBusy} onClick={() => void rank()}>
            {rankingBusy ? "Ranking..." : "Rank installed models"}
          </button>
        </div>
        {rankError !== null && <ErrorState error={rankError} />}
        {ranking !== null && ranking.length === 0 && <p className="muted small">No installed models to rank.</p>}
        {ranking !== null && ranking.length > 0 && (
          <ol className="rank-list" aria-label={`Models ranked for ${task}`}>
            {ranking.map((candidate) => (
              <li key={candidate.id} className="rank-item">
                <strong className="break">{nameOf(candidate.id)}</strong>{" "}
                <span className="muted small">
                  score {candidate.score.toFixed(1)}
                  {candidate.rating ? ` · ${FIT_LABEL[candidate.rating]}` : ""}
                </span>
                {candidate.reasons.length > 0 && (
                  <ul className="evidence">
                    {candidate.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ol>
        )}
      </Panel>

      <Panel
        id="model-updates"
        title="Check for updates"
        icon={<RefreshCw size={18} aria-hidden="true" />}
        description="Compares each Hugging Face model's pinned commit with the repository's current head. Read-only: nothing is downloaded or changed."
      >
        <button type="button" className="btn" disabled={checking} onClick={() => void checkUpdates()}>
          <RefreshCw size={14} aria-hidden="true" /> {checking ? "Checking..." : "Check for updates"}
        </button>
        {updateError !== null && (
          <Notice tone="danger" title={describeError(updateError).title}>
            {describeError(updateError).message}
            {" "}Update checks need the network and Hugging Face. Try again when you are online.
          </Notice>
        )}
        {updates !== null && updates.length === 0 && (
          <p className="muted small">No Hugging Face models are installed, so there is nothing to check.</p>
        )}
        {updates !== null && updates.length > 0 && (
          <ul className="update-list" aria-label="Update check results">
            {updates.map(({ check, pinned }) => (
              <li key={check.id} className="update-item">
                <strong className="break">{check.repo}</strong>{" "}
                <span className={`badge ${check.status === "changed" ? "tone-warn" : check.status === "up_to_date" ? "tone-ok" : ""}`}>
                  {check.status === "up_to_date" ? "Up to date" : check.status === "changed" ? "Head moved" : "Revision not recorded"}
                </span>
                <dl className="facts compact">
                  <div>
                    <dt>Pinned revision</dt>
                    <dd>
                      <code>{pinned ? pinned.slice(0, 12) : "Not recorded"}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Repository head</dt>
                    <dd>
                      <code>{check.head_revision.slice(0, 12)}</code>
                    </dd>
                  </div>
                </dl>
                {check.status === "changed" && (
                  <p className="muted small">
                    The repository head moved. That does not mean this file changed; the file&apos;s checksum is what matters.
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
