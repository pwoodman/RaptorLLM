import type { Preset } from "../api";

/**
 * Form state for one preset. Numbers stay as text until they are checked, so a half-typed value is
 * reported as an error instead of being guessed at.
 */
export interface PresetForm {
  name: string;
  systemPrompt: string;
  nCtx: string;
  temperature: string;
  topP: string;
  topK: string;
  repeatPenalty: string;
  /** Blank means no reply limit. */
  maxTokens: string;
}

export type PresetField = keyof PresetForm;
export type PresetErrors = Partial<Record<PresetField, string>>;

/** The same ranges the model manager checks on every save (raptor-models presets). */
export const PRESET_LIMITS = {
  nameChars: 64,
  systemPromptBytes: 32 * 1024,
  nCtx: { min: 256, max: 1_048_576 },
  temperature: { min: 0, max: 2 },
  topK: { min: 0, max: 1000 },
  repeatPenalty: { min: 0.5, max: 2 },
  maxTokens: { min: 1, max: 1_000_000 },
} as const;

export function emptyPresetForm(): PresetForm {
  return {
    name: "",
    systemPrompt: "",
    nCtx: "4096",
    temperature: "0.7",
    topP: "0.95",
    topK: "40",
    repeatPenalty: "1.1",
    maxTokens: "",
  };
}

export function formFromPreset(p: Preset): PresetForm {
  return {
    name: p.name,
    systemPrompt: p.system_prompt ?? "",
    nCtx: String(p.n_ctx),
    temperature: String(p.temperature),
    topP: String(p.top_p),
    topK: String(p.top_k),
    repeatPenalty: String(p.repeat_penalty),
    maxTokens: p.max_tokens === null ? "" : String(p.max_tokens),
  };
}

function numberOf(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

function inRange(n: number | null, min: number, max: number, integer: boolean): boolean {
  return n !== null && n >= min && n <= max && (!integer || Number.isInteger(n));
}

/** Checks a form the way the backend will, so most mistakes are shown before a save is attempted. */
export function validatePresetForm(f: PresetForm): PresetErrors {
  const errors: PresetErrors = {};
  const name = f.name.trim();
  if (name === "") {
    errors.name = "Give the preset a name.";
  } else if (name.length > PRESET_LIMITS.nameChars || /[\u0000-\u001f\u007f]/.test(name)) {
    errors.name = `Use 1 to ${PRESET_LIMITS.nameChars} printable characters.`;
  }
  if (new TextEncoder().encode(f.systemPrompt).length > PRESET_LIMITS.systemPromptBytes) {
    errors.systemPrompt = "The system prompt is larger than 32 KiB.";
  }
  const ctx = numberOf(f.nCtx);
  if (!inRange(ctx, PRESET_LIMITS.nCtx.min, PRESET_LIMITS.nCtx.max, true)) {
    errors.nCtx = `Enter a whole number from ${PRESET_LIMITS.nCtx.min} to ${PRESET_LIMITS.nCtx.max.toLocaleString()}.`;
  }
  const temperature = numberOf(f.temperature);
  if (!inRange(temperature, PRESET_LIMITS.temperature.min, PRESET_LIMITS.temperature.max, false)) {
    errors.temperature = "Enter a number from 0 to 2.";
  }
  const topP = numberOf(f.topP);
  if (topP === null || topP <= 0 || topP > 1) {
    errors.topP = "Enter a number above 0 and at most 1.";
  }
  const topK = numberOf(f.topK);
  if (!inRange(topK, PRESET_LIMITS.topK.min, PRESET_LIMITS.topK.max, true)) {
    errors.topK = "Enter a whole number from 0 to 1000. 0 turns top-k off.";
  }
  const repeat = numberOf(f.repeatPenalty);
  if (!inRange(repeat, PRESET_LIMITS.repeatPenalty.min, PRESET_LIMITS.repeatPenalty.max, false)) {
    errors.repeatPenalty = "Enter a number from 0.5 to 2.";
  }
  if (f.maxTokens.trim() !== "") {
    const max = numberOf(f.maxTokens);
    if (!inRange(max, PRESET_LIMITS.maxTokens.min, PRESET_LIMITS.maxTokens.max, true)) {
      errors.maxTokens = "Enter a whole number from 1 to 1,000,000, or leave blank for no limit.";
    }
  }
  return errors;
}

/** Builds the preset to save. Call only after `validatePresetForm` returned no errors. */
export function presetFromForm(f: PresetForm, modelId: string): Preset {
  return {
    model_id: modelId,
    name: f.name.trim(),
    system_prompt: f.systemPrompt.trim() === "" ? null : f.systemPrompt,
    n_ctx: Number(f.nCtx),
    temperature: Number(f.temperature),
    top_p: Number(f.topP),
    top_k: Number(f.topK),
    repeat_penalty: Number(f.repeatPenalty),
    max_tokens: f.maxTokens.trim() === "" ? null : Number(f.maxTokens),
  };
}

/** One line that summarises a preset for a list. */
export function presetSummary(p: Preset): string {
  const limit = p.max_tokens === null ? "no reply limit" : `reply limit ${p.max_tokens}`;
  return `Context ${p.n_ctx.toLocaleString()} · temperature ${p.temperature} · top-p ${p.top_p} · top-k ${p.top_k} · repeat ${p.repeat_penalty} · ${limit}`;
}
