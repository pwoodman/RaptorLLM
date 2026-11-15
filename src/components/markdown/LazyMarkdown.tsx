import { lazy, Suspense } from "react";

// The Markdown renderer, syntax highlighter and their grammars load on first use, which keeps the
// main bundle small. Until they arrive, the reply shows as plain text.
const Markdown = lazy(() => import("./Markdown").then((module) => ({ default: module.Markdown })));

export function LazyMarkdown({ text }: { text: string }) {
  return (
    <Suspense fallback={<div className="pre-wrap">{text}</div>}>
      <Markdown text={text} />
    </Suspense>
  );
}
