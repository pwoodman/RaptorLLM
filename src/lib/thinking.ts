import type { Effort, EffortChoice, ThinkingView } from "../api";

export const EFFORTS: readonly Effort[] = ["low", "medium", "high", "max"];

export const EFFORT_LABEL: Record<Effort, string> = {
  low: "Low",
  medium: "Med",
  high: "High",
  max: "Max",
};

export const EFFORT_NAME: Record<Effort, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  max: "Max",
};

/** The level in effect for a conversation: its own choice, else the setting. */
export function effectiveEffort(chosen: Effort | null, view: ThinkingView): Effort {
  return chosen ?? view.default_effort;
}

/** The app levels that map to the same model level as `effort` (including it), lowest first. */
export function sharedWith(view: ThinkingView, effort: Effort): EffortChoice[] {
  const mine = view.choices.find((c) => c.effort === effort);
  if (!mine) return [];
  return view.choices.filter((c) => c.model_level === mine.model_level);
}

/**
 * What a level means for this model, for a tooltip: the model's own level, and which other app levels
 * are the same setting. Says so plainly when two levels are one setting.
 */
export function describeEffort(view: ThinkingView, effort: Effort): string {
  const mine = view.choices.find((c) => c.effort === effort);
  if (!mine) return `${EFFORT_NAME[effort]}: not available for this model.`;
  const same = sharedWith(view, effort).filter((c) => c.effort !== effort);
  const base = `${EFFORT_NAME[effort]} sends the model's "${mine.model_level}" setting.`;
  if (same.length === 0) return base;
  const names = same.map((c) => EFFORT_NAME[c.effort]).join(" and ");
  return `${base} ${names} ${same.length === 1 ? "uses" : "use"} the same setting on this model.`;
}
