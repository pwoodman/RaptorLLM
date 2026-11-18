import { api } from "../api";
import { messageOf } from "../lib/errors";
import { createStore } from "./store";
import { engineStore, refreshEngine } from "./engine";

export type SetupStage = "idle" | "preparing" | "downloading" | "verifying" | "installing" | "ready" | "failed";
export const runtimeSetupStore = createStore<{
  running: boolean; label: string; stage: SetupStage; downloaded: number; total: number | null; error: string | null;
}>({ running: false, label: "", stage: "idle", downloaded: 0, total: null, error: null });
let retry: (() => Promise<unknown>) | null = null;

/** A single persistent setup task, so progress survives navigation between settings screens. */
export async function runRuntimeSetup<T>(label: string, total: number | null, action: () => Promise<T>): Promise<T> {
  if (runtimeSetupStore.get().running || engineStore.get().busy) throw new Error("Wait for the current model task to finish.");
  retry = () => runRuntimeSetup(label, total, action);
  runtimeSetupStore.set({ running: true, label, stage: "preparing", downloaded: 0, total, error: null });
  engineStore.set({ busy: "installing" });
  let current = true;
  const poll = async () => {
    try {
      const status = await api.runtime.installStatus();
      if (!current) return;
    // A previous installation may still be the last backend status while the new request
    // resolves its release. Only the command's success can mark this operation Ready.
    const stage = ["idle", "ready", "failed"].includes(status.stage) ? "preparing" : status.stage;
      runtimeSetupStore.set({ stage: stage as SetupStage, downloaded: stage === "preparing" ? 0 : status.downloaded });
    } catch { /* The installation itself reports failures; missing progress does not stop it. */ }
  };
  const timer = window.setInterval(() => void poll(), 500);
  void poll();
  try {
    const result = await action();
    current = false;
    runtimeSetupStore.set({ stage: "ready" });
    retry = null;
    await refreshEngine();
    return result;
  } catch (error) {
    current = false;
    runtimeSetupStore.set({ stage: "failed", error: messageOf(error) });
    throw error;
  } finally {
    current = false;
    window.clearInterval(timer);
    runtimeSetupStore.set({ running: false });
    engineStore.set({ busy: null });
  }
}

export async function retryRuntimeSetup(): Promise<void> {
  try { await retry?.(); } catch { /* The persistent status already describes the failure. */ }
}
