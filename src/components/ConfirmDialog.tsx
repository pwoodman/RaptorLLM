import { useEffect, useId, useRef, type KeyboardEvent, type ReactNode } from "react";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  tone?: "default" | "danger";
  secondaryAction?: { label: string; onClick: () => void };
  busy?: boolean;
  /** Blocks the confirm button, for example while the facts it confirms have not loaded. */
  confirmDisabled?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Modal confirmation. Focus starts on Cancel (the safe choice), stays inside the dialog,
 * closes on Escape, and returns to the element that opened it.
 */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  cancelLabel = "Cancel",
  tone = "default",
  secondaryAction,
  busy = false,
  confirmDisabled = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    if (!open) return;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    cancelRef.current?.focus();
    return () => opener?.focus();
  }, [open]);

  if (!open) return null;

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape" && !busy) {
      event.stopPropagation();
      onCancel();
      return;
    }
    if (event.key !== "Tab") return;
    const items = Array.from(dialogRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? []);
    const first = items[0];
    const last = items[items.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="modal-backdrop">
      <div
        ref={dialogRef}
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        onKeyDown={handleKeyDown}
      >
        <h2 id={titleId} className="modal-title">
          {title}
        </h2>
        <div id={descriptionId} className="modal-body">
          {children}
        </div>
        <div className="modal-actions">
          <button ref={cancelRef} type="button" className="btn" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </button>
          {secondaryAction && (
            <button type="button" className="btn" disabled={busy} onClick={secondaryAction.onClick}>
              {secondaryAction.label}
            </button>
          )}
          <button
            type="button"
            className={tone === "danger" ? "btn btn-danger" : "btn btn-primary"}
            onClick={onConfirm}
            disabled={busy || confirmDisabled}
          >
            {busy ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
