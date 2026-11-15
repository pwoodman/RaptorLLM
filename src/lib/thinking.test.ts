import { describe, expect, it } from "vitest";
import { noThinking, thinkingView } from "../test/fixtures";
import { describeEffort, effectiveEffort, sharedWith } from "./thinking";

describe("thinking helpers", () => {
  it("a conversation's own choice wins, else the setting applies", () => {
    expect(effectiveEffort("low", thinkingView({ default_effort: "high" }))).toBe("low");
    expect(effectiveEffort(null, thinkingView({ default_effort: "high" }))).toBe("high");
    expect(effectiveEffort(null, thinkingView())).toBe("medium");
  });

  it("finds the levels that are one setting on this model", () => {
    const view = thinkingView();
    expect(sharedWith(view, "high").map((c) => c.effort)).toEqual(["high", "max"]);
    expect(sharedWith(view, "max").map((c) => c.effort)).toEqual(["high", "max"]);
    expect(sharedWith(view, "low").map((c) => c.effort)).toEqual(["low"]);
    expect(sharedWith(noThinking(), "low")).toEqual([]);
  });

  it("says plainly when two levels are the same setting", () => {
    const view = thinkingView();
    expect(describeEffort(view, "high")).toBe(
      'High sends the model\'s "xhigh" setting. Max uses the same setting on this model.',
    );
    expect(describeEffort(view, "max")).toContain("High uses the same setting");
    expect(describeEffort(view, "low")).toBe('Low sends the model\'s "low" setting.');
  });

  it("describes a level the model does not offer without inventing a setting", () => {
    expect(describeEffort(noThinking(), "medium")).toBe("Medium: not available for this model.");
  });
});
