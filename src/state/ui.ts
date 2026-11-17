import { createStore } from "./store";

export type Page = "chats" | "models" | "settings";

export interface UiState {
  page: Page;
  /** Settings section (panel id) to scroll to once the Settings page shows. */
  anchor: string | null;
  /** A short message shown at the top of the app, such as a failed keyboard shortcut. */
  flash: string | null;
  /** Remember the settings screen when switching between app pages. */
  settingsSection: string;
}

export const uiStore = createStore<UiState>({ page: "chats", anchor: null, flash: null, settingsSection: "appearance" });

/** Opens Settings and scrolls to a section (for example "runtime" or "storage"). */
export function openSettings(anchor?: string): void {
  uiStore.set({ page: "settings", anchor: anchor ?? null });
}
