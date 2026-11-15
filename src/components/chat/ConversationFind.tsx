import { useEffect, useRef, useState, type KeyboardEvent, type RefObject } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { FIND_INPUT_ID } from "../../hooks/useGlobalShortcuts";
import {
  clearHits,
  findHits,
  paintHits,
  revealHit,
  supportsHighlights,
  type FindHit,
} from "../../lib/find";

interface ConversationFindProps {
  /** The message scroll box. Matches are searched in the messages it contains. */
  rootRef: RefObject<HTMLElement | null>;
}

/**
 * Find in the open conversation. Enter and Shift+Enter move between matches, Escape clears the
 * search. The match list is rebuilt whenever the messages change, including while a reply streams.
 */
export function ConversationFind({ rootRef }: ConversationFindProps) {
  const [query, setQuery] = useState("");
  const [count, setCount] = useState(0);
  const [index, setIndex] = useState(0);
  const hitsRef = useRef<FindHit[]>([]);
  const indexRef = useRef(0);
  const highlighting = supportsHighlights();
  const needle = query.trim();

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    indexRef.current = 0;
    const refresh = (reveal: boolean) => {
      const hits = needle === "" ? [] : findHits(root, needle);
      hitsRef.current = hits;
      const next = hits.length === 0 ? 0 : Math.min(indexRef.current, hits.length - 1);
      indexRef.current = next;
      setCount(hits.length);
      setIndex(next);
      paintHits(hits, next);
      if (reveal) revealHit(hits[next]);
    };

    refresh(true);

    // Streaming replies and re-renders change the text under the search, so rebuild when it does.
    let frame = 0;
    const observer = new MutationObserver(() => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => refresh(false));
    });
    observer.observe(root, { childList: true, subtree: true, characterData: true });

    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(frame);
      hitsRef.current = [];
      clearHits();
    };
  }, [needle, rootRef]);

  function step(delta: number) {
    const hits = hitsRef.current;
    if (hits.length === 0) return;
    const next = (indexRef.current + delta + hits.length) % hits.length;
    indexRef.current = next;
    setIndex(next);
    paintHits(hits, next);
    revealHit(hits[next]);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      step(event.shiftKey ? -1 : 1);
    } else if (event.key === "Escape") {
      // Keep Escape here: it clears the search and must not stop a reply.
      event.stopPropagation();
      setQuery("");
    }
  }

  const status =
    needle === "" ? "" : count === 0 ? "No matches" : `${index + 1} of ${count}`;

  return (
    <div className="chat-find" role="search">
      <label htmlFor={FIND_INPUT_ID} className="sr-only">
        Find in conversation
      </label>
      <input
        id={FIND_INPUT_ID}
        type="search"
        placeholder="Find in conversation"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={onKeyDown}
        aria-describedby="chat-find-status"
      />
      <span id="chat-find-status" className="muted small chat-find-status" aria-live="polite">
        {status}
      </span>
      <button
        type="button"
        className="icon-btn"
        aria-label="Previous match"
        disabled={count === 0}
        onClick={() => step(-1)}
      >
        <ChevronUp size={15} aria-hidden="true" />
      </button>
      <button
        type="button"
        className="icon-btn"
        aria-label="Next match"
        disabled={count === 0}
        onClick={() => step(1)}
      >
        <ChevronDown size={15} aria-hidden="true" />
      </button>
      {needle !== "" && !highlighting && (
        <span className="muted small">Highlighting is not supported in this browser. Matches are still found.</span>
      )}
    </div>
  );
}
