import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Markdown } from "./Markdown";

describe("Markdown", () => {
  it("copies the code of a fenced block with the copy button", async () => {
    const user = userEvent.setup();
    // userEvent.setup() installs its own clipboard stub, so the mock is applied after it.
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });

    render(<Markdown text={"Example:\n\n```ts\nconst answer = 42;\n```"} />);

    expect(screen.getByLabelText("ts code")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /copy code/i }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0]?.[0]).toContain("const answer = 42;");
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("does not render raw HTML from model output", () => {
    const { container } = render(<Markdown text={'<img src="x" onerror="alert(1)"> safe text'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("safe text");
  });
});
