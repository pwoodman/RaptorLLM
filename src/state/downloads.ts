import { api, type DownloadFinished, type DownloadProgress, type PartialDownload } from "../api";
import { messageOf } from "../lib/errors";
import { createStore } from "./store";
import { refreshLibrary } from "./library";

export interface DownloadRow {
  key: string;
  repo: string;
  fileName: string;
  progress: DownloadProgress | null;
  finished: DownloadFinished | null;
}

export interface DownloadsState {
  rows: Record<string, DownloadRow>;
  /** Keys in the order they were started in this session. */
  order: string[];
  /** Partial files on disk that are not part of this session's queue. */
  partials: PartialDownload[];
  problems: string[];
  loadError: string | null;
}

export const downloadsStore = createStore<DownloadsState>({
  rows: {},
  order: [],
  partials: [],
  problems: [],
  loadError: null,
});

function upsert(key: string, patch: Partial<Omit<DownloadRow, "key">>): void {
  const state = downloadsStore.get();
  const previous = state.rows[key];
  const row: DownloadRow = {
    key,
    repo: patch.repo ?? previous?.repo ?? "",
    fileName: patch.fileName ?? previous?.fileName ?? "",
    progress: patch.progress ?? previous?.progress ?? null,
    finished: patch.finished ?? previous?.finished ?? null,
  };
  downloadsStore.set({
    rows: { ...state.rows, [key]: row },
    order: previous ? state.order : [...state.order, key],
  });
}

/** Loads the session queue and leftover partial files. Live progress arrives through events. */
export async function refreshDownloads(): Promise<void> {
  try {
    const [queue, recovery] = await Promise.all([api.models.download.queue(), api.models.download.recover()]);
    const state = downloadsStore.get();
    const rows = { ...state.rows };
    const order = [...state.order];
    for (const entry of queue) {
      if (!rows[entry.key]) order.push(entry.key);
      rows[entry.key] = {
        key: entry.key,
        repo: entry.repo,
        fileName: entry.file_name,
        progress: entry.progress,
        finished: entry.finished,
      };
    }
    downloadsStore.set({
      rows,
      order,
      partials: recovery.partials.filter((p) => !rows[p.key]),
      problems: recovery.problems,
      loadError: null,
    });
  } catch (e) {
    downloadsStore.set({ loadError: messageOf(e) });
  }
}

/** Starts a verified download. Throws the backend error (for example Conflict when already installed). */
export async function startDownload(repo: string, fileName: string): Promise<string> {
  const key = await api.models.download.start(repo, fileName);
  upsert(key, { repo, fileName });
  return key;
}

export async function discardPartial(key: string): Promise<void> {
  await api.models.download.discardPartial(key);
  await refreshDownloads();
}

export function applyProgress(progress: DownloadProgress): void {
  if (!downloadsStore.get().rows[progress.key]) {
    void refreshDownloads();
  }
  upsert(progress.key, { progress });
}

export function applyFinished(finished: DownloadFinished): void {
  if (!downloadsStore.get().rows[finished.key]) {
    void refreshDownloads();
  }
  upsert(finished.key, { finished });
  if (finished.status === "completed") {
    void refreshLibrary();
  }
}
