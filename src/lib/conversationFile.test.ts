import { describe, expect, it } from "vitest";
import { exportFileName } from "./conversationFile";

describe("exportFileName", () => {
  it("builds a safe, readable name from the title and a short id", () => {
    expect(exportFileName("Refactor the parser: v2 / final?", "c1234567890")).toBe("Refactor-the-parser-v2-final-c1234567.json");
  });

  it("falls back when the title has no usable characters", () => {
    expect(exportFileName("???", "abcdef12")).toBe("conversation-abcdef12.json");
  });
});
