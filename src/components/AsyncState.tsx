import type { ReactNode } from "react";
import { Loader2, RotateCw, type LucideIcon } from "lucide-react";
import { describeError } from "../lib/errors";
import { Notice } from "./Notice";

export function LoadingState({ label }: { label: string }) {
  return (
    <div className="state" role="status">
      <Loader2 className="spin" size={16} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const ui = describeError(error);
  return (
    <Notice
      tone="danger"
      title={ui.title}
      actions={
        onRetry && (
          <button type="button" className="btn btn-sm" onClick={onRetry}>
            <RotateCw size={14} aria-hidden="true" /> Try again
          </button>
        )
      }
    >
      {ui.message}
    </Notice>
  );
}

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}

export function EmptyState({ icon: Icon, title, children, action }: EmptyStateProps) {
  return (
    <div className="empty">
      <Icon size={26} aria-hidden="true" />
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}
