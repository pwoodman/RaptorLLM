import { check, type Update, type DownloadEvent } from "@tauri-apps/plugin-updater";
import { api } from "../api";
import { createStore } from "./store";
import { engineStore } from "./engine";
import { chatStore } from "./chat";
import { downloadsStore } from "./downloads";
import { localModelsStore } from "./localModels";
import { firstLoadStore } from "./firstLoad";
import { uiStore } from "./ui";

const PREF_KEY = "raptor.autoUpdates";
const CHECK_INTERVAL = 60 * 60_000;
export const IDLE_SECONDS = 5 * 60;
export const UPDATE_COUNTDOWN_MS = 30_000;
function readEnabled(): boolean {
  try { return localStorage.getItem(PREF_KEY) !== "false"; } catch { return true; }
}

export const appUpdatesStore = createStore<{
  enabled: boolean;
  phase: "idle" | "checking" | "downloading" | "ready" | "installing";
  version: string | null;
  notes: string;
  message: string;
  error: string;
  progress: number | null;
  countdownAt: number | null;
  snoozedUntil: number;
}>({ enabled: readEnabled(), phase: "idle", version: null, notes: "", message: "", error: "", progress: null, countdownAt: null, snoozedUntil: 0 });

let candidate: Update | null = null;
let downloaded = false;
let operating = false;
let cycling = false;
let lastCheck = -Infinity;
let retryAfter = 0;

export function setAutomaticUpdates(enabled: boolean): void {
  try {
    localStorage.setItem(PREF_KEY, String(enabled));
    appUpdatesStore.set({ enabled, countdownAt: null, error: "" });
  } catch { appUpdatesStore.set({ error: "Your automatic update preference could not be saved. Try again." }); }
}

export function postponeUpdate(): void {
  appUpdatesStore.set({ countdownAt: null, snoozedUntil: Date.now() + CHECK_INTERVAL, message: "Automatic installation postponed for one hour." });
}

export function appHasUpdateWork(): boolean {
  if (engineStore.get().busy || chatStore.get().pending || firstLoadStore.get().entry || localModelsStore.get().phase !== "idle") return true;
  if (Object.values(downloadsStore.get().rows).some((row) => !row.finished && ["queued", "running", "verifying"].includes(row.progress?.state ?? "queued"))) return true;
  const ui = uiStore.get();
  // Settings forms may have unsaved edits. About contains only immediate-save update controls.
  if (ui.page === "settings" && ui.settingsSection !== "about") return true;
  return document.querySelector('[data-unsaved-work="true"], .modal-backdrop') !== null;
}

async function eligibleForAutomaticInstall(): Promise<boolean> {
  if (!appUpdatesStore.get().enabled || appHasUpdateWork() || appUpdatesStore.get().snoozedUntil > Date.now()) return false;
  try {
    const idle = await api.runtime.idleStatus();
    return idle.idle_seconds !== null && idle.idle_seconds >= IDLE_SECONDS && !idle.work_active
      && appUpdatesStore.get().enabled && !appHasUpdateWork() && appUpdatesStore.get().snoozedUntil <= Date.now();
  } catch { return false; }
}

export async function checkAppUpdates(fromIdle = false): Promise<void> {
  if (operating || (cycling && !fromIdle)) return;
  operating = true;
  lastCheck = Date.now();
  appUpdatesStore.set({ phase: "checking", message: "", error: "", countdownAt: null, progress: null });
  try {
    if (candidate) await candidate.close().catch(() => undefined);
    candidate = null;
    downloaded = false;
    const update = await check({ timeout: 15_000 });
    candidate = update;
    appUpdatesStore.set({ version: update?.version ?? null, notes: update?.body ?? "", message: update ? `Version ${update.version} is available.` : "You're up to date." });
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    const missing = /\b404\b|not found|no release found|valid release json/i.test(detail);
    appUpdatesStore.set({ version: null, notes: "", error: missing
      ? "Update information is unavailable. The signed release feed may be missing or invalid. You can get the latest installer from GitHub Releases."
      : `Couldn't reach the update service. Try again or get the latest installer from GitHub Releases. Details: ${detail}` });
  } finally { operating = false; appUpdatesStore.set({ phase: "idle" }); }
}

async function downloadCandidate(): Promise<void> {
  if (!candidate || downloaded) return;
  appUpdatesStore.set({ phase: "downloading", message: "Downloading signed update…", error: "", progress: null });
  let received = 0;
  let total = 0;
  await candidate.download((event: DownloadEvent) => {
    if (event.event === "Started") { received = 0; total = event.data.contentLength ?? 0; }
    if (event.event === "Progress") received += event.data.chunkLength;
    appUpdatesStore.set({ progress: event.event === "Finished" ? 100 : total ? Math.min(100, Math.round(received / total * 100)) : null });
  });
  downloaded = true;
  appUpdatesStore.set({ phase: "ready", message: "Update downloaded. Waiting for a quiet moment to restart Raptor." });
}

export async function installAppUpdate(manual = true): Promise<void> {
  if (!candidate || operating || (cycling && manual)) return;
  if (appHasUpdateWork()) { appUpdatesStore.set({ countdownAt: null, message: "Update is ready. Finish the current task or save your draft before installing." }); return; }
  operating = true;
  try {
    await downloadCandidate();
    // Downloading may take minutes. Check activity and the preference again immediately before restart.
    if (appHasUpdateWork() || (!manual && !await eligibleForAutomaticInstall())) {
      appUpdatesStore.set({ countdownAt: null });
      return;
    }
    appUpdatesStore.set({ phase: "installing", countdownAt: null, message: "Installing update. Raptor will restart…" });
    await candidate.install({ restartAfterInstall: true });
  } catch (cause) {
    retryAfter = Date.now() + CHECK_INTERVAL;
    appUpdatesStore.set({ phase: downloaded ? "ready" : "idle", countdownAt: null, error: `Update installation failed: ${cause instanceof Error ? cause.message : String(cause)}` });
  } finally { operating = false; }
}

/** Checks metadata hourly. Only downloads/installs after proven PC idle time and no Raptor work. */
export async function runIdleUpdateCycle(): Promise<void> {
  const state = appUpdatesStore.get();
  if (cycling || operating || !state.enabled || state.phase === "installing" || state.snoozedUntil > Date.now() || retryAfter > Date.now()) return;
  cycling = true;
  try {
    if (!candidate && Date.now() - lastCheck >= CHECK_INTERVAL) await checkAppUpdates(true);
    if (!candidate) return;
    if (!await eligibleForAutomaticInstall()) { appUpdatesStore.set({ countdownAt: null }); return; }
    if (!downloaded) {
      operating = true;
      try { await downloadCandidate(); }
      finally { operating = false; }
      if (!await eligibleForAutomaticInstall()) { appUpdatesStore.set({ countdownAt: null }); return; }
    }
    const countdown = appUpdatesStore.get().countdownAt;
    if (countdown === null) appUpdatesStore.set({ countdownAt: Date.now() + UPDATE_COUNTDOWN_MS });
    else if (Date.now() >= countdown) await installAppUpdate(false);
  } catch (cause) {
    retryAfter = Date.now() + CHECK_INTERVAL;
    appUpdatesStore.set({ phase: "idle", countdownAt: null, error: `Automatic update will retry later: ${cause instanceof Error ? cause.message : String(cause)}` });
  } finally { cycling = false; }
}

export function startIdleUpdates(): () => void {
  void runIdleUpdateCycle();
  const timer = window.setInterval(() => void runIdleUpdateCycle(), 30_000);
  return () => window.clearInterval(timer);
}
