import type { GpuVerdict } from "../api";
import { Notice } from "./Notice";

/** Only `verified` may be presented as GPU acceleration. Everything else is said plainly. */
export function GpuVerdictView({ verdict }: { verdict: GpuVerdict | null }) {
  // Nothing to say until a model has started; the sidebar already shows the pending state.
  if (verdict === null) return null;
  if (verdict.verdict === "verified") {
    return (
      <Notice tone="success" title="GPU verified">
        <p>The server log shows the GPU ran this model.</p>
        {verdict.evidence.length > 0 && (
          <ul className="evidence">
            {verdict.evidence.map((line) => (
              <li key={line}>
                <code>{line}</code>
              </li>
            ))}
          </ul>
        )}
      </Notice>
    );
  }
  if (verdict.verdict === "cpu_only") {
    return (
      <Notice tone="danger" title="CPU only: the GPU is not running this model">
        <p>{verdict.reason}</p>
        <p>Generation runs on the CPU and will be much slower than expected.</p>
      </Notice>
    );
  }
  return (
    <Notice tone="warn" title="GPU unverified">
      <p>{verdict.reason}</p>
      <p>Raptor cannot confirm that a GPU is accelerating this model.</p>
    </Notice>
  );
}

export function verdictSummary(verdict: GpuVerdict | null): { label: string; tone: "ok" | "danger" | "warn" | "muted" } {
  if (verdict === null) return { label: "No GPU verdict yet", tone: "muted" };
  if (verdict.verdict === "verified") return { label: "GPU verified", tone: "ok" };
  if (verdict.verdict === "cpu_only") return { label: "CPU only", tone: "danger" };
  return { label: "GPU unverified", tone: "warn" };
}
