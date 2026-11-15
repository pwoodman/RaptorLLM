import { Boxes, CheckCircle2, CircleAlert, Cpu, HardDrive, RotateCw, Wrench, type LucideIcon } from "lucide-react";
import { setupStatus, type SetupCheck, type SetupStatus } from "../../api";
import { useAsyncResource } from "../../hooks/useAsyncResource";
import { useStore } from "../../hooks/useStore";
import { messageOf } from "../../lib/errors";
import { formatBytes } from "../../lib/format";
import { engineStore } from "../../state/engine";
import { openSettings, uiStore } from "../../state/ui";
import { LoadingState } from "../AsyncState";
import { Notice } from "../Notice";
import { Panel } from "../Panel";
import { InstallRuntime } from "./InstallRuntime";

/** The checks a user must clear before chat works. GPU and performance memory are informational. */
const REQUIRED: SetupCheck["id"][] = ["hardware", "runtime", "model", "storage"];

const ICONS: Record<SetupCheck["id"], LucideIcon> = {
  hardware: Cpu,
  runtime: Wrench,
  gpu: Cpu,
  model: Boxes,
  storage: HardDrive,
  performance_memory: Wrench,
};

/** True when the guide should be shown: a required check is unmet, or the runtime is a CPU build. */
export function setupNeedsAttention(status: SetupStatus): boolean {
  const unmet = status.checks.some((c) => REQUIRED.includes(c.id) && !c.ok);
  return unmet || status.degraded || status.runtime_needs_confirmation;
}

/**
 * First-run checklist, shown on Chats and Models while something is missing. It reads state only.
 * Installing the runtime needs an explicit click and a confirmation. Nothing starts on its own.
 */
export function SetupGuide() {
  const status = useStore(engineStore, (s) => s.status);
  const page = useStore(uiStore, (s) => s.page);
  const setup = useAsyncResource(() => setupStatus(), [status]);

  if (setup.loading && setup.data === null && setup.error === null) {
    return <LoadingState label="Checking setup..." />;
  }
  if (setup.error !== null) {
    return (
      <Notice
        tone="warn"
        title="Setup check unavailable"
        actions={
          <button type="button" className="btn btn-sm" onClick={setup.reload}>
            <RotateCw size={14} aria-hidden="true" /> Try again
          </button>
        }
      >
        {messageOf(setup.error)}
      </Notice>
    );
  }
  const data = setup.data;
  if (data === null || !setupNeedsAttention(data)) return null;

  const checks = data.checks.filter((c) => REQUIRED.includes(c.id) || !c.ok);

  return (
    <Panel
      id="setup-guide"
      title="Finish setup"
      icon={<Wrench size={18} aria-hidden="true" />}
      description="Chat needs these. Nothing is downloaded or installed until you choose to."
    >
      {data.degraded && data.selected_runtime && (
        <Notice tone="danger" title="CPU build in use: generation will be slow">
          {data.selected_runtime.reason}
        </Notice>
      )}
      <ol className="setup-list">
        {checks.map((check) => (
          <SetupItem key={check.id} check={check} data={data} onChanged={setup.reload} onModelsPage={page === "models"} />
        ))}
      </ol>
      {data.notes.map((note) => (
        <p key={note} className="muted small">
          {note}
        </p>
      ))}
    </Panel>
  );
}

interface SetupItemProps {
  check: SetupCheck;
  data: SetupStatus;
  onChanged: () => void;
  onModelsPage: boolean;
}

function SetupItem({ check, data, onChanged, onModelsPage }: SetupItemProps) {
  const Icon = ICONS[check.id];
  const StatusIcon = check.ok ? CheckCircle2 : CircleAlert;
  const tone = check.ok ? "tone-ok" : "tone-warn";
  const required = REQUIRED.includes(check.id);

  return (
    <li className="setup-item" data-check={check.id}>
      <div className="setup-item-head">
        <StatusIcon size={18} aria-hidden="true" className={tone} />
        <Icon size={16} aria-hidden="true" />
        <strong>{check.label}</strong>
        <span className={`badge ${tone}`}>{check.ok ? "Done" : required ? "Needed" : "Note"}</span>
      </div>
      <div className="setup-item-body">
        <p className="muted small">{detailFor(check, data)}</p>
        {!check.ok && <ActionFor check={check} data={data} onChanged={onChanged} onModelsPage={onModelsPage} />}
      </div>
    </li>
  );
}

function detailFor(check: SetupCheck, data: SetupStatus): string {
  if (check.id === "runtime" && data.selected_runtime) {
    const rt = data.selected_runtime;
    const base = `${rt.id} ${rt.version}, ${rt.backend.toUpperCase()}${rt.degraded ? " (CPU build)" : ""}.`;
    return `${base} ${check.detail}`;
  }
  if (check.id === "model") return `${data.models_installed} installed. ${check.detail}`;
  if (check.id === "storage" && data.storage_free_bytes !== null) {
    return `${formatBytes(data.storage_free_bytes)} free. ${check.detail}`;
  }
  return check.detail;
}

function ActionFor({ check, data, onChanged, onModelsPage }: SetupItemProps) {
  switch (check.id) {
    case "runtime":
      if (data.runtime_needs_confirmation && data.available_runtime) {
        return <InstallRuntime candidate={data.available_runtime} onInstalled={() => onChanged()} />;
      }
      return (
        <button type="button" className="btn btn-sm" onClick={() => openSettings("runtime")}>
          Open runtime settings
        </button>
      );
    case "model":
      if (onModelsPage) return <p className="small">Download one from Discover, or import a GGUF file in Library.</p>;
      return (
        <button type="button" className="btn btn-sm btn-primary" onClick={() => uiStore.set({ page: "models" })}>
          Go to Models
        </button>
      );
    case "hardware":
      return (
        <button type="button" className="btn btn-sm" onClick={() => openSettings("hardware")}>
          Open hardware settings
        </button>
      );
    case "storage":
      return (
        <button type="button" className="btn btn-sm" onClick={() => openSettings("storage")}>
          Open storage settings
        </button>
      );
    default:
      return null;
  }
}
