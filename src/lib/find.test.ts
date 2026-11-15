import { describe, expect, it } from "vitest";
import { findHits, supportsHighlights } from "./find";

function messages(blocks: Record<string, string>): HTMLElement {
  const root = document.createElement("div");
  for (const [id, html] of Object.entries(blocks)) {
    const block = document.createElement("div");
    block.setAttribute("data-find-block", id);
    block.innerHTML = html;
    root.appendChild(block);
  }
  return root;
}

describe("findHits", () => {
  it("finds every case-insensitive occurrence across message blocks", () => {
    const root = messages({ a: "Parser error, then another ERROR", b: "no match here" });
    const hits = findHits(root, "error");
    expect(hits).toHaveLength(2);
    expect(hits.map((h) => h.range.toString())).toEqual(["error", "ERROR"]);
  });

  it("matches text that is split across inline elements inside one block", () => {
    const root = messages({ a: "use <strong>bold</strong> text" });
    const hits = findHits(root, "bold text");
    expect(hits).toHaveLength(1);
    expect(hits[0]?.range.toString()).toBe("bold text");
  });

  it("does not match across two different message blocks", () => {
    const root = messages({ a: "first part", b: "second part" });
    expect(findHits(root, "part second")).toHaveLength(0);
  });

  it("returns nothing for an empty query", () => {
    expect(findHits(messages({ a: "anything" }), "")).toHaveLength(0);
  });

  it("reports that highlighting is unavailable in this environment", () => {
    expect(supportsHighlights()).toBe(false);
  });
});
