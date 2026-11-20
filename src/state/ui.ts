import { createStore } from "./store";

export type Page = "chats" | "models" | "settings";
export type ModelView = "library" | "discover" | "downloads" | "performance" | "compare";

export interface UiState {
  page: Page;
  /** Settings section to open when navigating from another page. */
  anchor: string | null;
  /** A short message shown at the top of the app, such as a failed keyboard shortcut. */
  flash: string | null;
  /** Remember the settings screen when switching between app pages. */
  settingsSection: string;
  modelView: ModelView;
  focusModelImport: boolean;
}

export const uiStore = createStore<UiState>({
  page: "chats", anchor: null, flash: null, settingsSection: "appearance",
  modelView: "library", focusModelImport: false,
});

/** Opens Settings at a section (for example "runtime" or "storage"). */
export function openSettings(anchor?: string): void {
  uiStore.set({ page: "settings", anchor: anchor ?? null });
}

export function openModels(view: ModelView): void {
  uiStore.set({ page: "models", modelView: view, focusModelImport: false });
}

export function openModelImport(): void {
  uiStore.set({ page: "models", modelView: "library", focusModelImport: true });
}
