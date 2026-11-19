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
export function SetupGuide({ onModelAction }: { onModelAction?: (action: "discover" | "library") => void } = {}) {
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
  const byId = new Map(checks.map((check) => [check.id, check]));
  const nextCheck = (["hardware", "storage", "runtime", "model", "gpu"] as const)
    .map((id) => byId.get(id))
    .find((check) => check && !check.ok);

  return (
    <Panel
      id="setup-guide"
      title="Let's get Raptor ready"
      icon={<Wrench size={18} aria-hidden="true" />}
      description="We'll walk you through what you need. Downloads only start when you choose them."
    >
      {data.degraded && data.selected_runtime && (
        <Notice tone="danger" title="CPU build in use: generation will be slow">
          {data.selected_runtime.reason}
        </Notice>
      )}
      {nextCheck && (
        <div className="setup-next-step">
          <span className="setup-step-label">Your next step</span>
          <h3>{stepTitle(nextCheck.id)}</h3>
          <p className="muted small">{stepHelp(nextCheck, data)}</p>
          <ActionFor
            check={nextCheck}
            data={data}
            onChanged={setup.reload}
            onModelsPage={page === "models"}
            onModelAction={onModelAction}
          />
        </div>
      )}
      <details className="setup-details">
        <summary>See all setup details</summary>
        <ol className="setup-list">
          {checks.map((check) => (
            <SetupItem
              key={check.id}
              check={check}
              data={data}
              onChanged={setup.reload}
              onModelsPage={page === "models"}
              onModelAction={onModelAction}
              hasStepAbove={check.id === nextCheck?.id}
            />
          ))}
        </ol>
      </details>
      {data.notes.map((note) => (
        <p key={note} className="muted small">
          {note}
        </p>
      ))}
    </Panel>
  );
}

function stepTitle(id: SetupCheck["id"]): string {
  switch (id) {
    case "hardware": return "Check this computer";
    case "storage": return "Make room before downloading";
    case "runtime": return "Set up the model engine";
    case "model": return "Choose your first model";
    case "gpu": return "Review graphics acceleration";
    case "performance_memory": return "Review performance storage";
  }
}

function stepHelp(check: SetupCheck, data: SetupStatus): string {
  switch (check.id) {
    case "hardware": return `${check.detail}. Raptor uses this to choose a compatible setup.`;
    case "storage": return `${check.detail}. Choose a drive with more free space before downloading large files.`;
    case "runtime": return data.available_runtime?.degraded
      ? "This small download lets Raptor run models. The available option uses your CPU and will be slower."
      : "This small one-time download lets Raptor run models. We'll select the best available option for this computer.";
    case "model": return "Models are the AI brains you chat with. Pick a recommended download or bring a GGUF file you already have.";
    case "gpu": return `${check.detail}. You can review the runtime choice in Settings.`;
    case "performance_memory": return check.detail;
  }
}

interface SetupItemProps {
  check: SetupCheck;
  data: SetupStatus;
  onChanged: () => void;
  onModelsPage: boolean;
  onModelAction?: (action: "discover" | "library") => void;
  /** The same check is the panel's "next step" above, so repeating the action here duplicates it. */
  hasStepAbove: boolean;
}

function SetupItem({
  check,
  data,
  onChanged,
  onModelsPage,
  onModelAction,
  hasStepAbove,
}: SetupItemProps) {
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
        {!check.ok && !hasStepAbove && (
          <ActionFor
            check={check}
            data={data}
            onChanged={onChanged}
            onModelsPage={onModelsPage}
            onModelAction={onModelAction}
          />
        )}
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

function ActionFor({
  check,
  data,
  onChanged,
  onModelsPage,
  onModelAction,
}: Omit<SetupItemProps, "hasStepAbove">) {
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
      if (onModelsPage) {
        return (
          <div className="button-row">
            <button type="button" className="btn btn-sm btn-primary" onClick={() => onModelAction?.("discover")}>
              Browse recommended models
            </button>
            <button type="button" className="btn btn-sm" onClick={() => onModelAction?.("library")}>
              I already have a model file
            </button>
          </div>
        );
      }
      return (
        <button type="button" className="btn btn-sm btn-primary" onClick={() => uiStore.set({ page: "models" })}>
          Choose a model
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
    case "gpu":
      return (
        <button type="button" className="btn btn-sm" onClick={() => openSettings("runtime")}>
          Review runtime settings
        </button>
      );
    default:
      return null;
  }
}
