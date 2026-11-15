import { describe, expect, it } from "vitest";
import { emptyPresetForm, presetFromForm, validatePresetForm } from "./presets";

describe("validatePresetForm", () => {
  it("accepts the defaults once a name is given", () => {
    expect(validatePresetForm({ ...emptyPresetForm(), name: "Review" })).toEqual({});
  });

  it("requires a name and rejects one with control characters or over 64 characters", () => {
    expect(validatePresetForm(emptyPresetForm()).name).toBeDefined();
    expect(validatePresetForm({ ...emptyPresetForm(), name: "   " }).name).toBeDefined();
    expect(validatePresetForm({ ...emptyPresetForm(), name: "a".repeat(65) }).name).toBeDefined();
    expect(validatePresetForm({ ...emptyPresetForm(), name: "bad\u0007name" }).name).toBeDefined();
  });

  it("checks each number against the backend range, including half-typed values", () => {
    const f = { ...emptyPresetForm(), name: "x" };
    expect(validatePresetForm({ ...f, nCtx: "10" }).nCtx).toMatch(/256/);
    expect(validatePresetForm({ ...f, nCtx: "4096.5" }).nCtx).toBeDefined();
    expect(validatePresetForm({ ...f, nCtx: "" }).nCtx).toBeDefined();
    expect(validatePresetForm({ ...f, temperature: "2.5" }).temperature).toBeDefined();
    expect(validatePresetForm({ ...f, topP: "0" }).topP).toBeDefined();
    expect(validatePresetForm({ ...f, topP: "1.2" }).topP).toBeDefined();
    expect(validatePresetForm({ ...f, topK: "1001" }).topK).toBeDefined();
    expect(validatePresetForm({ ...f, topK: "0" }).topK).toBeUndefined();
    expect(validatePresetForm({ ...f, repeatPenalty: "0.4" }).repeatPenalty).toBeDefined();
  });

  it("treats a blank reply limit as no limit, and rejects zero or a non-number", () => {
    const f = { ...emptyPresetForm(), name: "x" };
    expect(validatePresetForm({ ...f, maxTokens: "" }).maxTokens).toBeUndefined();
    expect(validatePresetForm({ ...f, maxTokens: "0" }).maxTokens).toBeDefined();
    expect(validatePresetForm({ ...f, maxTokens: "lots" }).maxTokens).toBeDefined();
  });
});

describe("presetFromForm", () => {
  it("builds the backend shape, trimming the name and mapping blanks to null", () => {
    const preset = presetFromForm({ ...emptyPresetForm(), name: "  Review  ", systemPrompt: "  " }, "m1");
    expect(preset).toEqual({
      model_id: "m1",
      name: "Review",
      system_prompt: null,
      n_ctx: 4096,
      temperature: 0.7,
      top_p: 0.95,
      top_k: 40,
      repeat_penalty: 1.1,
      max_tokens: null,
    });
  });
});
