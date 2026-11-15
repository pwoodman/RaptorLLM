import type { ReactNode } from "react";
import { AlertOctagon, AlertTriangle, CheckCircle2, Info } from "lucide-react";

export type NoticeTone = "info" | "success" | "warn" | "danger";

const ICONS = {
  info: Info,
  success: CheckCircle2,
  warn: AlertTriangle,
  danger: AlertOctagon,
} as const;

interface NoticeProps {
  tone?: NoticeTone;
  title?: string;
  children?: ReactNode;
  actions?: ReactNode;
}

/** Inline status message. Warnings and errors use role="alert" so screen readers announce them. */
export function Notice({ tone = "info", title, children, actions }: NoticeProps) {
  const Icon = ICONS[tone];
  const role = tone === "warn" || tone === "danger" ? "alert" : "status";
  return (
    <div className={`notice notice-${tone}`} role={role}>
      <Icon size={16} aria-hidden="true" className="notice-icon" />
      <div className="notice-body">
        {title && <strong className="notice-title">{title}</strong>}
        {children && <div className="notice-text">{children}</div>}
        {actions && <div className="notice-actions">{actions}</div>}
      </div>
    </div>
  );
}
