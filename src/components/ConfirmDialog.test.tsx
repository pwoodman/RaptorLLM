import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfirmDialog } from "./ConfirmDialog";

function setup(open = true) {
  const onConfirm = vi.fn();
  const onCancel = vi.fn();
  const view = render(
    <ConfirmDialog open={open} title="Delete model?" confirmLabel="Delete" tone="danger" onConfirm={onConfirm} onCancel={onCancel}>
      This cannot be undone.
    </ConfirmDialog>,
  );
  return { ...view, onConfirm, onCancel };
}

describe("ConfirmDialog", () => {
  it("renders nothing while closed", () => {
    setup(false);
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("is a labelled alert dialog that starts focus on Cancel", () => {
    setup();
    const dialog = screen.getByRole("alertdialog", { name: "Delete model?" });
    expect(dialog).toHaveAccessibleDescription("This cannot be undone.");
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });

  it("closes on Escape without confirming", async () => {
    const user = userEvent.setup();
    const { onCancel, onConfirm } = setup();
    await user.keyboard("{Escape}");
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("calls onConfirm only from the confirm button", async () => {
    const user = userEvent.setup();
    const { onConfirm, onCancel } = setup();
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("keeps Tab focus inside the dialog", async () => {
    const user = userEvent.setup();
    setup();
    await user.tab();
    expect(screen.getByRole("button", { name: "Delete" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });
});
