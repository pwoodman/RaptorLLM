import type { TuneReport } from "../api";
import { createStore } from "./store";

// Keep an in-flight optimization visible when the user switches pages.
export const tuningStore = createStore<{
  activeId: string | null;
  reports: Record<string, TuneReport | null>;
}>({ activeId: null, reports: {} });
