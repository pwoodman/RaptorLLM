import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SHORTCUTS } from "../hooks/useGlobalShortcuts";
import { ShortcutsDialog } from "./ShortcutsDialog";

describe("ShortcutsDialog", () => {
  it("lists every shortcut the app handles", () => {
    render(<ShortcutsDialog open onClose={vi.fn()} />);
    const dialog = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    for (const shortcut of SHORTCUTS) {
      expect(dialog).toHaveTextContent(shortcut.keys);
    }
    expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  });

  it("closes on Escape and on the Close button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ShortcutsDialog open onClose={onClose} />);
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("renders nothing while closed", () => {
    render(<ShortcutsDialog open={false} onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
