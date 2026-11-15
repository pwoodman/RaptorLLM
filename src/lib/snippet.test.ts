import { describe, expect, it } from "vitest";
import { splitMatches } from "./snippet";

describe("splitMatches", () => {
  it("marks every case-insensitive occurrence and keeps the rest of the text", () => {
    expect(splitMatches("Use the Parser, then parse again", "parse")).toEqual([
      { text: "Use the ", match: false },
      { text: "Parse", match: true },
      { text: "r, then ", match: false },
      { text: "parse", match: true },
      { text: " again", match: false },
    ]);
  });

  it("returns the whole text unmarked when there is no search term or no match", () => {
    expect(splitMatches("abc", "  ")).toEqual([{ text: "abc", match: false }]);
    expect(splitMatches("abc", "zzz")).toEqual([{ text: "abc", match: false }]);
  });

  it("joins back to the original text", () => {
    const text = "…needle in a haystack needle…";
    expect(splitMatches(text, "needle").map((p) => p.text).join("")).toBe(text);
  });
});
