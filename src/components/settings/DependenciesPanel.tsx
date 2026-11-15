import { useEffect, useState } from "react";
import { PackageCheck } from "lucide-react";
import { api, type DependencyReport, type RuntimeUpdate } from "../../api";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { refreshEngine } from "../../state/engine";
import { ConfirmDialog } from "../ConfirmDialog";
import { Notice } from "../Notice";
import { Panel } from "../Panel";

/**
 * Keeps the inference runtime up to date. Checking reads release information only. Installing an
 * update asks for confirmation first, shows the source and checksum, and is refused while a model
 * is loaded. The AMD HIP SDK is only detected: AMD's installer is never run for the user.
 */
export function DependenciesPanel() {
  const [report, setReport] = useState<DependencyReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<RuntimeUpdate | null>(null);
  const [installing, setInstalling] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmingVulkan, setConfirmingVulkan] = useState(false);

  async function check() {
    setChecking(true);
    setError(null);
    try {
      setReport(await api.dependencies.check());
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => {
    void check();
  }, []);

  async function install(update: RuntimeUpdate) {
    setInstalling(true);
    setError(null);
    try {
      const selection = await api.dependencies.installUpdate(update.id);
      setNotice(`Installed ${selection.id} ${selection.version}.`);
      setConfirming(null);
      await refreshEngine();
      await check();
    } catch (e) {
      setError(messageOf(e));
      setConfirming(null);
    } finally {
      setInstalling(false);
    }
  }

  async function installVulkan() {
    setInstalling(true);
    setError(null);
    try {
      setNotice(await api.dependencies.installVulkan());
      setConfirmingVulkan(false);
      await check();
    } catch (e) {
      setError(messageOf(e));
      setConfirmingVulkan(false);
    } finally {
      setInstalling(false);
    }
  }

  const available = report?.updates.filter((u) => u.available) ?? [];

  return (
    <Panel
      id="dependencies"
      title="Dependencies and updates"
      icon={<PackageCheck size={18} aria-hidden="true" />}
      description="Keep the llama.cpp runtime current and check the AMD HIP SDK it relies on."
      actions={
        <button type="button" className="btn btn-sm" onClick={() => void check()} disabled={checking || installing}>
          {checking ? "Checking..." : "Check for updates"}
        </button>
      }
    >
      {error !== null && (
        <Notice tone="danger" title="Update problem">
          {error}
        </Notice>
      )}
      {notice !== null && (
        <Notice tone="info" title="Updated">
          {notice}
        </Notice>
      )}
      {report?.check_error && (
        <Notice tone="warn" title="Could not check for updates">
          {report.check_error}. Everything keeps working with what is installed.
        </Notice>
      )}

      {report !== null && report.updates.length === 0 && report.check_error === null && (
        <p className="muted small">No runtime is installed yet. Install one from the checklist on the Models page.</p>
      )}
      {report?.updates.map((u) => (
        <div key={u.id} className="dep-row">
          <div>
            <strong>{u.id}</strong>
            <span className="muted small block">
              Installed {u.current_version}
              {u.available ? `, ${u.latest_version} is available` : ` · ${u.note}`}
            </span>
          </div>
          {u.available && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setConfirming(u)}
              disabled={installing}
            >
              Update to {u.latest_version}
            </button>
          )}
        </div>
      ))}
      {report !== null && available.length === 0 && report.latest_tag !== null && report.updates.length > 0 && (
        <p className="muted small">Newest llama.cpp build: {report.latest_tag}. (llama.cpp publishes every build as a pre-release.)</p>
      )}

      {report?.hip_sdk && (
        <div className="dep-row">
          <div>
            <strong>AMD HIP SDK</strong>
            <span className="muted small block">
              {report.hip_sdk.versions.length > 0 ? `Found ${report.hip_sdk.versions.join(", ")}. ` : ""}
              {report.hip_sdk.guidance}
            </span>
          </div>
          {!report.hip_sdk.ok && (
            <a className="btn btn-sm" href={report.hip_sdk.download_page} target="_blank" rel="noreferrer noopener">
              Open AMD download page
            </a>
          )}
        </div>
      )}

      {report?.vulkan && (
        <div className="dep-row">
          <div>
            <strong>Vulkan runtime (optional)</strong>
            <span className="muted small block">
              {report.vulkan.installed_version !== null
                ? `Installed ${report.vulkan.installed_version}${report.vulkan.available ? `, ${report.vulkan.latest_version} is available` : ` · ${report.vulkan.note}`}. `
                : report.vulkan.available
                  ? "Not installed. "
                  : `Not installed · ${report.vulkan.note}. `}
              Needs no HIP SDK. Choose it per model in the Library: it generated faster on large sparse
              Mixture-of-experts models that do not fit in VRAM, while ROCm reads prompts faster.
            </span>
          </div>
          {report.vulkan.available && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setConfirmingVulkan(true)}
              disabled={installing}
            >
              {report.vulkan.installed_version !== null ? `Update to ${report.vulkan.latest_version}` : "Install Vulkan runtime"}
            </button>
          )}
        </div>
      )}

      <ConfirmDialog
        open={confirmingVulkan}
        title="Install the Vulkan runtime?"
        confirmLabel="Download and install"
        busy={installing}
        onCancel={() => setConfirmingVulkan(false)}
        onConfirm={() => void installVulkan()}
      >
        {report?.vulkan && (
          <>
            <p>
              Raptor will download and verify the Vulkan build of llama.cpp {report.vulkan.latest_version}. It is
              installed next to ROCm; the runtime in use does not change, and each model picks its own backend.
            </p>
            <dl className="facts compact">
              <div>
                <dt>Source</dt>
                <dd>{report.vulkan.url}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{report.vulkan.size_bytes ? formatBytes(report.vulkan.size_bytes) : "Not reported"}</dd>
              </div>
              <div>
                <dt>SHA-256</dt>
                <dd>{report.vulkan.sha256}</dd>
              </div>
            </dl>
            <p className="small">{report.vulkan.note}. Refused while a model is loaded.</p>
          </>
        )}
      </ConfirmDialog>

      <ConfirmDialog
        open={confirming !== null}
        title="Install this runtime update?"
        confirmLabel="Download and install"
        busy={installing}
        onCancel={() => setConfirming(null)}
        onConfirm={() => confirming && void install(confirming)}
      >
        {confirming && (
          <>
            <p>
              Raptor will download and verify {confirming.id} {confirming.latest_version}. Your current version stays on
              disk, and you can roll back from the runtime section.
            </p>
            <dl className="facts compact">
              <div>
                <dt>Source</dt>
                <dd>{confirming.url}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{confirming.size_bytes ? formatBytes(confirming.size_bytes) : "Not reported"}</dd>
              </div>
              <div>
                <dt>SHA-256</dt>
                <dd>{confirming.sha256}</dd>
              </div>
            </dl>
            <p className="small">{confirming.note}. Updates are refused while a model is loaded.</p>
          </>
        )}
      </ConfirmDialog>
    </Panel>
  );
}
