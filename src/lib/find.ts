/**
 * Find-in-page for the open conversation. Matches are searched in the rendered text (what the
 * reader sees), not in the Markdown source, and are returned as DOM ranges so nothing in the React
 * tree is modified. Highlighting uses the CSS Custom Highlight API where the browser has it.
 */

export interface FindHit {
  range: Range;
}

/** Element attribute that marks one message's rendered body. Matches never cross message bodies. */
export const FIND_BLOCK_ATTR = "data-find-block";
export const HIGHLIGHT_ALL = "raptor-find";
export const HIGHLIGHT_CURRENT = "raptor-find-current";

interface Segment {
  node: Text;
  start: number;
}

/** Finds every case-insensitive occurrence of `query` inside each `[data-find-block]` under `root`. */
export function findHits(root: HTMLElement, query: string): FindHit[] {
  const needle = query.toLowerCase();
  if (needle === "") return [];
  const hits: FindHit[] = [];
  root.querySelectorAll<HTMLElement>(`[${FIND_BLOCK_ATTR}]`).forEach((block) => {
    const { text, segments } = collectText(block);
    const haystack = text.toLowerCase();
    // Lower-casing can change the length of some Unicode text. Such blocks are skipped, not mis-highlighted.
    if (haystack.length !== text.length) return;
    let at = haystack.indexOf(needle);
    while (at !== -1) {
      const range = rangeFor(segments, at, at + needle.length);
      if (range) hits.push({ range });
      at = haystack.indexOf(needle, at + needle.length);
    }
  });
  return hits;
}

function collectText(block: HTMLElement): { text: string; segments: Segment[] } {
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
  const segments: Segment[] = [];
  let text = "";
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    const textNode = node as Text;
    segments.push({ node: textNode, start: text.length });
    text += textNode.data;
  }
  return { text, segments };
}

function locate(segments: Segment[], offset: number, preferEnd: boolean): { node: Text; offset: number } | null {
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (!seg) continue;
    const end = seg.start + seg.node.data.length;
    const inside = preferEnd ? offset > seg.start && offset <= end : offset >= seg.start && offset < end;
    if (inside) return { node: seg.node, offset: offset - seg.start };
  }
  return null;
}

function rangeFor(segments: Segment[], start: number, end: number): Range | null {
  const from = locate(segments, start, false);
  const to = locate(segments, end, true);
  if (!from || !to) return null;
  const range = document.createRange();
  range.setStart(from.node, from.offset);
  range.setEnd(to.node, to.offset);
  return range;
}

interface HighlightRegistry {
  set: (name: string, highlight: unknown) => void;
  delete: (name: string) => void;
}
type HighlightCtor = new (...ranges: Range[]) => unknown;

function registry(): HighlightRegistry | null {
  const css = (globalThis as { CSS?: { highlights?: HighlightRegistry } }).CSS;
  return css?.highlights ?? null;
}

function highlightCtor(): HighlightCtor | null {
  return (globalThis as { Highlight?: HighlightCtor }).Highlight ?? null;
}

/** True when this browser can paint matches without changing the DOM. */
export function supportsHighlights(): boolean {
  return registry() !== null && highlightCtor() !== null;
}

/** Paints all matches and the current one. Does nothing where the CSS Custom Highlight API is missing. */
export function paintHits(hits: FindHit[], current: number): void {
  const reg = registry();
  const Ctor = highlightCtor();
  if (!reg || !Ctor) return;
  if (hits.length === 0) {
    clearHits();
    return;
  }
  reg.set(HIGHLIGHT_ALL, new Ctor(...hits.map((h) => h.range)));
  const focus = hits[current];
  if (focus) reg.set(HIGHLIGHT_CURRENT, new Ctor(focus.range));
}

export function clearHits(): void {
  const reg = registry();
  reg?.delete(HIGHLIGHT_ALL);
  reg?.delete(HIGHLIGHT_CURRENT);
}

/** Brings a match into view. */
export function revealHit(hit: FindHit | undefined): void {
  if (!hit) return;
  const node = hit.range.startContainer;
  const element = node.nodeType === Node.ELEMENT_NODE ? (node as Element) : node.parentElement;
  element?.scrollIntoView?.({ block: "center" });
}
