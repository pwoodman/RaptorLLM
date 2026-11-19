import { api, type ModelEntry } from "../api";
import { createStore } from "./store";

const KEY = "raptor.optimization-intro.v1";
const seen = new Set<string>();
export const firstLoadStore = createStore<{ entry: ModelEntry | null }>({ entry: null });
let resolveChoice: ((optimize: boolean) => void) | null = null;
const identity = (entry: ModelEntry) => entry.sha256 || entry.id;

function storedIdentities(): string[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(stored) ? stored.filter((id): id is string => typeof id === "string") : [];
  } catch { return []; }
}

function wasOffered(entry: ModelEntry): boolean {
  return seen.has(identity(entry)) || storedIdentities().includes(identity(entry));
}

export function rememberOptimizationOffer(entry: ModelEntry): void {
  seen.add(identity(entry));
  try {
    localStorage.setItem(KEY, JSON.stringify([...new Set([...storedIdentities(), ...seen])]));
  } catch { /* The session still remembers the choice when local storage is unavailable. */ }
}

/** Only the first unoptimized load offers tuning. Skipping is remembered per model content. */
export async function offerFirstLoadOptimization(entry: ModelEntry): Promise<boolean> {
  if (wasOffered(entry) || entry.gguf.architecture === "dflash") return false;
  try {
    const saved = await api.models.savedTuning(entry.id);
    if (saved.some((profile) => profile.tuning.qualification != null)) {
      rememberOptimizationOffer(entry);
      return false;
    }
  } catch {
    // An optional speed check must not prevent the normal model load from diagnosing problems.
    return false;
  }
  return new Promise<boolean>((resolve) => {
    resolveChoice = resolve;
    firstLoadStore.set({ entry });
  });
}

export function chooseFirstLoadOptimization(optimize: boolean): void {
  const entry = firstLoadStore.get().entry;
  if (!entry || !resolveChoice) return;
  if (!optimize) rememberOptimizationOffer(entry);
  const resolve = resolveChoice;
  resolveChoice = null;
  firstLoadStore.set({ entry: null });
  resolve(optimize);
}
