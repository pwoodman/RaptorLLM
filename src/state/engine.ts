import {
  api,
  modelsLoadWithPreset,
  type EngineStatus,
  type GatewayStatus,
  type KvCacheType,
  type LoadModelOutcome,
  type MemoryEstimate,
} from "../api";
import { messageOf } from "../lib/errors";
import { createStore } from "./store";
import { libraryStore } from "./library";
import { offerFirstLoadOptimization, rememberOptimizationOffer } from "./firstLoad";
import { tuningStore } from "./tuning";

/** What this session loaded. The backend does not report the loaded model's id in EngineStatus. */
export interface LoadedModel {
  modelId: string;
  /** Planned context size from the launch recommendation. Null when that could not be read. */
  ctxSize: number | null;
  /** Planned GPU layers from the launch recommendation. The offloaded count is not reported. */
  plannedGpuLayers: number | null;
  rationale: string[];
  estimate: MemoryEstimate | null;
  preset?: string | null;
  window?: LoadWindow | null;
}

export interface EngineState {
  status: EngineStatus | null;
  statusError: string | null;
  gateway: GatewayStatus | null;
  loaded: LoadedModel | null;
  /**
   * The full result of the last load in this session (config source, benchmark, CPU fallback).
   * Null after unload, a stop, or a refused load.
   */
  loadReport: LoadModelOutcome | null;
  busy: "preparing" | "loading" | "unloading" | "optimizing" | "installing" | null;
}

export const engineStore = createStore<EngineState>({
  status: null,
  statusError: null,
  gateway: null,
  loaded: null,
  loadReport: null,
  busy: null,
});

export async function refreshEngine(): Promise<void> {
  try {
    const [status, gateway, currentModel] = await Promise.all([
      api.runtime.status(),
      api.gateway.status(),
      api.models.current().catch(() => null),
    ]);
    const currentLoaded = engineStore.get().loaded;
    let loaded = currentLoaded;
    let loadReport = engineStore.get().loadReport;
    if (status.server.state.state === "stopped") {
      loaded = null;
      loadReport = null;
    } else if (currentModel) {
      if (!loaded || loaded.modelId !== currentModel.id) {
        // This window did not see the load (it was reloaded). Ask the backend what it chose.
        const stored = await api.models.loadReport().catch(() => null);
        if (stored && stored.result.model_id === currentModel.id) {
          loaded = {
            modelId: currentModel.id,
            ctxSize: stored.ctx_size,
            plannedGpuLayers: stored.gpu_layers,
            rationale: stored.result.rationale,
            estimate: stored.result.estimate,
          };
          loadReport = stored.result;
        } else {
          loaded = {
            modelId: currentModel.id,
            ctxSize: null,
            plannedGpuLayers: null,
            rationale: [],
            estimate: null,
          };
        }
      }
    }
    // A load in this window sets its own state when it finishes; do not overwrite it mid-load.
    if (engineStore.get().busy !== null) {
      engineStore.set({ status, gateway, statusError: null });
      return;
    }
    engineStore.set({ status, gateway, loaded, loadReport, statusError: null });
  } catch (e) {
    engineStore.set({ statusError: messageOf(e) });
  }
}

/** Why chat cannot run right now, or null when the server is ready. Never fabricates a reply. */
export function inferenceBlockReason(status: EngineStatus | null, statusError: string | null): string | null {
  if (statusError) return `The engine status could not be read: ${statusError}`;
  if (status === null) return "Checking the inference server...";
  const state = status.server.state;
  switch (state.state) {
    case "ready":
    case "degraded":
      return null;
    case "stopped":
      return "The inference server is not running. Load a model from Models to start it.";
    case "starting":
      return "The inference server is starting. Wait for it to report ready.";
    case "crashed":
      return `The inference server crashed: ${state.reason}. Load a model again from Models.`;
  }
}

/**
 * Loads a model, optionally with one of its presets. A refused load is rethrown unchanged (its
 * `backend_failure` diagnosis is read by `loadFailureOf`). Nothing is reported as loaded then: the
 * status is read again, because a refused load can still have stopped the previous model.
 */
/** A context window picked from `api.models.contextOptions`. */
export interface LoadWindow {
  ctx: number;
  cacheType: KvCacheType;
  slots: number;
  speculation: boolean;
}

export async function loadModel(
  modelId: string,
  preset: string | null = null,
  window: LoadWindow | null = null,
): Promise<void> {
  if (engineStore.get().busy !== null) throw new Error("Wait for the current model task to finish.");
  engineStore.set({ busy: "preparing" });
  try {
    const entry = libraryStore.get().models?.find((model) => model.id === modelId);
    if (entry && await offerFirstLoadOptimization(entry)) {
      engineStore.set({ busy: "optimizing" });
      // Optimize & chat includes replacing the currently running model, just like a normal load.
      const status = await api.models.unload();
      engineStore.set({ status, loaded: null, loadReport: null });
      tuningStore.set({ activeId: modelId, startedAt: Date.now() });
      try {
        const report = await api.models.tune(modelId, false);
        tuningStore.set((state) => ({ reports: { ...state.reports, [modelId]: report } }));
        rememberOptimizationOffer(entry);
      } finally {
        tuningStore.set({ activeId: null, startedAt: null });
      }
    }
    engineStore.set({ busy: "loading" });
    const result = await modelsLoadWithPreset(
      modelId,
      window?.ctx ?? null,
      preset,
      window?.cacheType ?? null,
      window?.slots ?? null,
      window ? window.speculation : null,
    );
    // What the backend actually launched, so a chosen window shows its real size. The plan is only a
    // fallback; if neither can be read, the size is shown as unknown.
    const report = await api.models.loadReport().catch(() => null);
    const plan = report ? null : await api.runtime.recommend(modelId).catch(() => null);
    engineStore.set({
      status: result.status,
      statusError: null,
      loadReport: result,
      loaded: {
        modelId,
        preset,
        window,
        ctxSize: report?.ctx_size ?? plan?.config.ctx_size ?? null,
        plannedGpuLayers: report?.gpu_layers ?? plan?.config.gpu_layers ?? null,
        rationale: result.rationale,
        estimate: result.estimate,
      },
    });
  } catch (e) {
    engineStore.set({ loadReport: null, loaded: null });
    await refreshEngine().catch(() => undefined);
    throw e;
  } finally {
    engineStore.set({ busy: null });
  }
}

export async function unloadModel(): Promise<void> {
  engineStore.set({ busy: "unloading" });
  try {
    const status = await api.models.unload();
    engineStore.set({ status, loaded: null, loadReport: null });
  } finally {
    engineStore.set({ busy: null });
  }
}
