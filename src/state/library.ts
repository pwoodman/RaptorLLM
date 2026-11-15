import { api, type ImportMode, type ModelEntry, type StorageUsage } from "../api";
import { messageOf } from "../lib/errors";
import { createStore } from "./store";

export interface LibraryState {
  models: ModelEntry[] | null;
  storage: StorageUsage | null;
  error: string | null;
}

export const libraryStore = createStore<LibraryState>({ models: null, storage: null, error: null });

export async function refreshLibrary(): Promise<void> {
  try {
    const [models, storage] = await Promise.all([api.models.list(), api.models.storage()]);
    libraryStore.set({ models, storage, error: null });
  } catch (e) {
    libraryStore.set({ error: messageOf(e) });
  }
}

export async function importModel(path: string, mode: ImportMode): Promise<ModelEntry> {
  const entry = await api.models.import(path, mode);
  await refreshLibrary();
  return entry;
}

export async function deleteModel(id: string): Promise<void> {
  await api.models.delete(id);
  await refreshLibrary();
}
