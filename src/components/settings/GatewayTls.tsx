import { useId, useState } from "react";
import { ShieldCheck, RotateCw } from "lucide-react";
import { api, gatewayRotateTls, type GatewayStatus } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import { engineStore } from "../../state/engine";
import { CopyButton } from "../CopyButton";
import { ConfirmDialog } from "../ConfirmDialog";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";

/**
 * Gateway transport: LAN exposure, the TLS certificate fingerprint clients pin, and certificate
 * rotation. Shows what the listener is doing now, not what the settings say it should do.
 */
export function GatewayTls() {
  const stored = useStore(engineStore, (s) => s.gateway);
  // The shared snapshot can be missing when the engine read failed, so this section reads the gateway itself.
  const own = useAsyncResource(() => (stored === null ? api.gateway.status() : Promise.resolve(stored)), [stored]);
  const gateway: GatewayStatus | null = stored ?? own.data;
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [rotated, setRotated] = useState<GatewayStatus | null>(null);
  const fingerprintId = useId();
  const headingId = useId();

  async function rotate() {
    setBusy(true);
    setError(null);
    try {
      const next = await gatewayRotateTls();
      engineStore.set({ gateway: next });
      setRotated(next);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      setConfirmOpen(false);
    }
  }

  if (gateway === null) {
    if (own.error !== null) return <ErrorState error={own.error} onRetry={own.reload} />;
    return <LoadingState label="Reading gateway transport..." />;
  }

  const canRotate = gateway.running && gateway.tls;
  const fingerprint = gateway.tls_fingerprint_sha256;
  const lanWithoutTls = gateway.lan && !gateway.tls;

  return (
    <section className="gateway-tls" aria-labelledby={headingId}>
      <h3 id={headingId} className="subsection-title">
        <ShieldCheck size={15} aria-hidden="true" /> Transport and certificate
      </h3>

      {gateway.insecure && (
        <Notice tone="danger" title="Insecure mode is on">
          Requests and API keys are not protected in transit. Turn insecure mode off before exposing this endpoint.
        </Notice>
      )}
      {lanWithoutTls && (
        <Notice tone="danger" title="LAN is on without TLS">
          Other devices connect without encryption. Restart the gateway, or turn LAN off.
        </Notice>
      )}
      {gateway.last_error && (
        <Notice tone="warn" title="Last gateway error">
          {gateway.last_error}
        </Notice>
      )}

      <dl className="facts compact">
        <div>
          <dt>Listener</dt>
          <dd>{gateway.running ? (gateway.addr ?? "Running") : "Not running"}</dd>
        </div>
        <div>
          <dt>LAN</dt>
          <dd>
            {gateway.lan ? "On: listening on all interfaces" : "Off: this computer only"}
          </dd>
        </div>
        <div>
          <dt>TLS</dt>
          <dd>{gateway.tls ? "On" : "Off"}</dd>
        </div>
      </dl>

      <div className="fingerprint-row">
        <div className="field grow">
          <label htmlFor={fingerprintId} className="field-label">
            Certificate fingerprint (SHA-256)
          </label>
          <input
            id={fingerprintId}
            readOnly
            className="mono"
            value={fingerprint ?? "No certificate: TLS is off"}
            onFocus={(e) => e.currentTarget.select()}
          />
          <span className="muted small">Pin this value in clients that connect over LAN.</span>
        </div>
        <CopyButton label="Copy fingerprint" getText={() => fingerprint ?? ""} />
      </div>

      {rotated && (
        <Notice tone="success" title="Certificate rotated">
          New fingerprint: <code>{rotated.tls_fingerprint_sha256 ?? "none"}</code>. Clients must pin it again.
        </Notice>
      )}

      <div className="button-row">
        <button
          type="button"
          className="btn"
          disabled={!canRotate || busy}
          title={canRotate ? "Replace the LAN certificate" : "Needs a running gateway with TLS"}
          onClick={() => setConfirmOpen(true)}
        >
          <RotateCw size={14} aria-hidden="true" /> Rotate certificate
        </button>
        {!canRotate && <span className="muted small">Available while the gateway runs with TLS.</span>}
      </div>

      {error !== null && <ErrorState error={error} />}

      <ConfirmDialog
        open={confirmOpen}
        title="Rotate the certificate?"
        confirmLabel="Rotate certificate"
        tone="danger"
        busy={busy}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => void rotate()}
      >
        <p>
          A new self-signed certificate replaces the current one on the running listener. Every client that pinned the
          old fingerprint will fail to connect until it pins the new one.
        </p>
      </ConfirmDialog>
    </section>
  );
}
