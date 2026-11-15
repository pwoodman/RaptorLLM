import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";

interface CopyButtonProps {
  getText: () => string;
  label?: string;
}

export function CopyButton({ getText, label = "Copy" }: CopyButtonProps) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const timer = window.setTimeout(() => setState("idle"), 1500);
    return () => window.clearTimeout(timer);
  }, [state]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(getText());
      setState("copied");
    } catch {
      setState("failed");
    }
  }

  const text = state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : label;
  return (
    <button type="button" className="btn btn-ghost btn-sm" onClick={() => void copy()}>
      {state === "copied" ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
      <span>{text}</span>
    </button>
  );
}
