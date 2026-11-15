import { useRef, isValidElement, type ReactNode } from "react";
import { CopyButton } from "../CopyButton";

/** The language is carried on the `<code>` child as `language-<name>` by react-markdown. */
function languageOf(children: ReactNode): string | null {
  if (!isValidElement<{ className?: string }>(children)) return null;
  const match = /language-(\S+)/.exec(children.props.className ?? "");
  return match?.[1] ?? null;
}

export function CodeBlock({ children }: { children: ReactNode }) {
  const preRef = useRef<HTMLPreElement>(null);
  const language = languageOf(children);
  return (
    <div className="code-block">
      <div className="code-head">
        <span className="code-lang">{language ?? "text"}</span>
        <CopyButton label="Copy code" getText={() => preRef.current?.textContent ?? ""} />
      </div>
      <pre ref={preRef} tabIndex={0} aria-label={`${language ?? "Plain text"} code`}>
        {children}
      </pre>
    </div>
  );
}
