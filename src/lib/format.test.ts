import { describe, expect, it } from "vitest";
import { formatBytes, formatCount } from "./format";

describe("formatBytes", () => {
  it("uses binary units with sensible precision", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1024 ** 3 * 4.5)).toBe("4.5 GB");
    expect(formatBytes(1024 ** 3 * 250)).toBe("250 GB");
  });

  it("does not invent a size for invalid input", () => {
    expect(formatBytes(-1)).toBe("unknown size");
    expect(formatBytes(Number.NaN)).toBe("unknown size");
  });
});

describe("formatCount", () => {
  it("groups thousands", () => {
    expect(formatCount(1234567)).toBe("1,234,567");
  });
});
