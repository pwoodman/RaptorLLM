import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThinkingBlock } from "./ThinkingBlock";

describe("ThinkingBlock", () => {
  const writeText = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    writeText.mockClear();
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
  });

  it("is open while the model is still thinking, and says it is not the answer", () => {
    render(<ThinkingBlock text="Weighing options" streaming answerStarted={false} />);
    expect(screen.getByRole("button", { name: /Thinking\.\.\./ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Weighing options")).toBeInTheDocument();
  });

  it("collapses when the answer begins, and the text leaves the page", () => {
    const { rerender } = render(<ThinkingBlock text="Weighing options" streaming answerStarted={false} />);
    rerender(<ThinkingBlock text="Weighing options" streaming answerStarted />);

    const toggle = screen.getByRole("button", { name: /Thinking/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Weighing options")).toBeNull();
  });

  it("expands after collapsing, and copies only the reasoning when the user asks", async () => {
    const user = userEvent.setup();
    // user-event installs its own clipboard stub on setup, so the mock goes in after it.
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<ThinkingBlock text="Step one" streaming={false} answerStarted />);
    expect(screen.queryByRole("button", { name: "Copy thinking" })).toBeNull();

    await user.click(screen.getByRole("button", { name: /Thinking/ }));
    expect(screen.getByText("Step one")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Copy thinking" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("Step one"));
  });

  it("stays open when the user reopens it after the answer has started", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<ThinkingBlock text="Notes" streaming answerStarted={false} />);
    rerender(<ThinkingBlock text="Notes" streaming answerStarted />);
    await user.click(screen.getByRole("button", { name: /Thinking/ }));
    rerender(<ThinkingBlock text="Notes more" streaming answerStarted />);
    expect(screen.getByText("Notes more")).toBeInTheDocument();
  });
});
