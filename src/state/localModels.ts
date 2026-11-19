import { api } from "../api";
import { messageOf } from "../lib/errors";
import { refreshLibrary } from "./library";
import { createStore } from "./store";

export const localModelsStore = createStore<{
  phase: "idle" | "scanning" | "indexing";
  current: string | null;
  done: number;
  total: number;
  added: number;
  errors: string[];
}>({ phase: "idle", current: null, done: 0, total: 0, added: 0, errors: [] });

let inFlight: Promise<void> | null = null;
let rescanRequested = false;
let paused = false;
// Do not repeatedly hash duplicate copies or failed files on every automatic scan.
const attempted = new Set<string>();
const failures = new Map<string, string>();

/** Keep discovery from starting a model import while an app update is preparing to restart. */
export function pauseLocalModelSync(): () => void {
  paused = true;
  return () => { paused = false; };
}

/** One shared queue, independent of pages. A verified model becomes available as soon as it finishes. */
export function syncLocalModels(retry = false): Promise<void> {
  if (retry) {
    for (const key of failures.keys()) attempted.delete(key);
    failures.clear();
  }
  if (paused) return inFlight ?? Promise.resolve();
  if (inFlight) {
    if (retry) rescanRequested = true;
    return inFlight;
  }
  inFlight = (async () => {
    do {
      rescanRequested = false;
      await indexModels();
    } while (rescanRequested && !paused);
  })().finally(() => {
    localModelsStore.set({ phase: "idle", current: null });
    inFlight = null;
  });
  return inFlight;
}

async function indexModels(): Promise<void> {
  localModelsStore.set({ phase: "scanning" });
  try {
    const folders = await api.models.discover();
    if (paused) return;
    const unique = new Map<string, string>();
    for (const folder of folders) {
      for (const item of folder.items) {
        const normalized = item.path.replace(/\\/g, "/");
        const path = /^[a-z]:|^\/\//i.test(normalized) ? normalized.toLowerCase() : normalized;
        const key = JSON.stringify([path, item.size_bytes]);
        if (!attempted.has(key)) unique.set(key, item.path);
      }
    }
    if (unique.size === 0) {
      localModelsStore.set({ errors: [...failures.values()] });
      return;
    }
    localModelsStore.set({ phase: "indexing", done: 0, total: unique.size, added: 0, errors: [...failures.values()] });
    let done = 0;
    let added = 0;
    for (const [key, path] of unique) {
      if (paused) return;
      localModelsStore.set({ current: path });
      attempted.add(key);
      try {
        await api.models.importDiscovered(path);
        added += 1;
        await refreshLibrary();
      } catch (e) {
        failures.set(key, `${path.split(/[\\/]/).pop()}: ${messageOf(e)}`);
      }
      localModelsStore.set({ done: ++done, added, errors: [...failures.values()] });
    }
  } catch (e) {
    localModelsStore.set({ errors: [...failures.values(), `Could not check local model folders: ${messageOf(e)}`] });
  }
}

export function startLocalModelSync(): () => void {
  void syncLocalModels();
  const timer = window.setInterval(() => void syncLocalModels(), 60_000);
  const onFocus = () => void syncLocalModels();
  window.addEventListener("focus", onFocus);
  return () => { window.clearInterval(timer); window.removeEventListener("focus", onFocus); };
}
