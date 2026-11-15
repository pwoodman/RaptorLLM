import { CircleAlert } from "lucide-react";
import { loadFailureOf, type BackendLoadFailure, type FailureCause } from "../../api";
import { describeError } from "../../lib/errors";
import { ErrorState } from "../AsyncState";

const CAUSE_TITLE: Record<FailureCause, string> = {
  hipblas_name_mismatch: "The GPU runtime asks for a DLL name the HIP SDK does not ship",
  hipblas_missing: "The GPU runtime cannot find hipblas.dll",
  hip_sdk_not_found: "The AMD HIP SDK was not found",
  no_device_visible: "The GPU backend started, but it sees no GPU device",
};

/**
 * Shows why a GPU load was refused, with the suggested next steps. A refused load is never shown
 * as success: nothing is loaded and no CPU run was started. Errors without a GPU diagnosis fall back
 * to the plain error message.
 */
export function LoadFailureCard({ error }: { error: unknown }) {
  const diagnosis: BackendLoadFailure | null = loadFailureOf(error);
  if (diagnosis === null) return <ErrorState error={error} />;

  const title = CAUSE_TITLE[diagnosis.cause] ?? "The GPU backend has no usable device";
  return (
    <section className="load-failure" role="alert" aria-labelledby="load-failure-title">
      <h3 id="load-failure-title" className="load-failure-title">
        <CircleAlert size={18} aria-hidden="true" /> Model not loaded: {title}
      </h3>
      <p>{diagnosis.message}</p>
      <p className="small">
        Nothing is loaded and no CPU run was started. Raptor does not run on the CPU unless you allow it in Settings.
      </p>
      {diagnosis.suggestions.length > 0 && (
        <>
          <h4 className="subsection-title">What to try</h4>
          <ol className="load-failure-steps">
            {diagnosis.suggestions.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </>
      )}
      <p className="muted small">Backend {diagnosis.backend.toUpperCase()}. Error: {describeError(error).message}</p>
    </section>
  );
}
