/** One piece of a search snippet: plain text, or text that matches the search. */
export interface SnippetPart {
  text: string;
  match: boolean;
}

/**
 * Splits `text` around case-insensitive occurrences of `needle`, so the matches can be marked.
 * The backend folds case the same way for ASCII and common letters; the parts always join back
 * to the original text.
 */
export function splitMatches(text: string, needle: string): SnippetPart[] {
  const term = needle.trim();
  if (term === "") return [{ text, match: false }];
  const haystack = text.toLowerCase();
  const target = term.toLowerCase();
  const parts: SnippetPart[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    const at = haystack.indexOf(target, cursor);
    if (at < 0) break;
    if (at > cursor) parts.push({ text: text.slice(cursor, at), match: false });
    parts.push({ text: text.slice(at, at + target.length), match: true });
    cursor = at + target.length;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor), match: false });
  return parts.length > 0 ? parts : [{ text, match: false }];
}
