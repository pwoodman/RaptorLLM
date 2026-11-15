import { Cpu, RefreshCw } from "lucide-react";
import { api } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { formatBytes, formatCount } from "../../lib/format";
import { ErrorState, LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";

export function HardwarePanel() {
  const hardware = useAsyncResource(() => api.hardware.info(), []);
  const hw = hardware.data;

  return (
    <Panel
      id="hardware"
      title="Hardware"
      icon={<Cpu size={18} aria-hidden="true" />}
      description="Detected at startup. Used for fit ratings and launch settings."
      actions={
        <button type="button" className="btn btn-sm" onClick={hardware.reload} disabled={hardware.loading}>
          <RefreshCw size={13} aria-hidden="true" /> Refresh
        </button>
      }
    >
      {hardware.loading && hw === null && <LoadingState label="Detecting hardware..." />}
      {hardware.error !== null && <ErrorState error={hardware.error} onRetry={hardware.reload} />}
      {hw && (
        <>
          <dl className="facts">
            <div>
              <dt>CPU</dt>
              <dd>
                {hw.cpu.brand} · {formatCount(hw.cpu.cores)} cores, {formatCount(hw.cpu.threads)} threads
              </dd>
            </div>
            <div>
              <dt>CPU features</dt>
              <dd>{hw.cpu.features.length > 0 ? hw.cpu.features.join(", ") : "None reported"}</dd>
            </div>
            <div>
              <dt>System RAM</dt>
              <dd>
                {formatBytes(hw.ram_avail)} available of {formatBytes(hw.ram_total)}
              </dd>
            </div>
            <div>
              <dt>OS</dt>
              <dd>{hw.os}</dd>
            </div>
          </dl>

          {hw.gpus.length === 0 ? (
            <p className="muted small">No GPU detected. Models will run on the CPU, which is much slower.</p>
          ) : (
            <ul className="gpu-list" aria-label="Detected GPUs">
              {hw.gpus.map((gpu) => (
                <li key={`${gpu.vendor}-${gpu.name}`} className="gpu-card">
                  <strong>{gpu.name}</strong>
                  <dl className="facts compact">
                    <div>
                      <dt>Vendor</dt>
                      <dd>{gpu.vendor.toUpperCase()}</dd>
                    </div>
                    <div>
                      <dt>Architecture</dt>
                      <dd>{gpu.arch ?? "Unknown"}</dd>
                    </div>
                    <div>
                      <dt>VRAM</dt>
                      <dd>
                        {formatBytes(gpu.vram_total)} total,{" "}
                        {gpu.vram_free === null ? "free amount not reported" : `${formatBytes(gpu.vram_free)} free`}
                      </dd>
                    </div>
                    <div>
                      <dt>Driver</dt>
                      <dd>{gpu.driver ?? "Unknown"}</dd>
                    </div>
                  </dl>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </Panel>
  );
}
