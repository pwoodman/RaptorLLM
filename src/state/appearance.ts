import { createStore } from "./store";

export type Theme = "system" | "light" | "dark";
export interface Appearance { theme: Theme; glass: boolean }
const KEY = "raptor.appearance";

export function readAppearance(): Appearance {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? "null");
    return {
      theme: ["system", "light", "dark"].includes(saved?.theme) ? saved.theme : "system",
      glass: saved?.glass === true,
    };
  } catch {
    return { theme: "system", glass: false };
  }
}

export const appearanceStore = createStore<Appearance>(readAppearance());

export function saveAppearance(update: Partial<Appearance>): void {
  const next = { ...appearanceStore.get(), ...update };
  // Commit only after persistence succeeds, so the UI never claims an unsaved choice.
  localStorage.setItem(KEY, JSON.stringify(next));
  appearanceStore.set(next);
}

export function startAppearance(): () => void {
  const media = window.matchMedia("(prefers-color-scheme: light)");
  const apply = () => {
    const { theme, glass } = appearanceStore.get();
    const resolved = theme === "system" ? (media.matches ? "light" : "dark") : theme;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.glass = String(glass);
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", resolved === "light" ? "#f3f6fa" : "#0a0d13");
  };
  apply();
  const unsubscribe = appearanceStore.subscribe(apply);
  media.addEventListener("change", apply);
  return () => { unsubscribe(); media.removeEventListener("change", apply); };
}
