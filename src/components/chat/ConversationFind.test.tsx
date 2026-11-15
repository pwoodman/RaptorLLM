import { createRef } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConversationFind } from "./ConversationFind";

function Conversation() {
  const ref = createRef<HTMLDivElement>();
  return (
    <>
      <ConversationFind rootRef={ref} />
      <div ref={ref}>
        <div data-find-block="m1">The parser fails on empty input.</div>
        <div data-find-block="m2">Add a guard before the parser runs.</div>
      </div>
    </>
  );
}

describe("ConversationFind", () => {
  it("counts matches and steps forward and backward with wrap-around", async () => {
    const user = userEvent.setup();
    render(<Conversation />);

    const input = screen.getByRole("searchbox", { name: "Find in conversation" });
    await user.type(input, "parser");
    expect(screen.getByText("1 of 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next match" }));
    expect(screen.getByText("2 of 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next match" }));
    expect(screen.getByText("1 of 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Previous match" }));
    expect(screen.getByText("2 of 2")).toBeInTheDocument();
  });

  it("uses Enter for next and Shift+Enter for previous", async () => {
    const user = userEvent.setup();
    render(<Conversation />);
    const input = screen.getByRole("searchbox", { name: "Find in conversation" });
    await user.type(input, "parser");

    await user.keyboard("{Enter}");
    expect(screen.getByText("2 of 2")).toBeInTheDocument();

    await user.keyboard("{Shift>}{Enter}{/Shift}");
    expect(screen.getByText("1 of 2")).toBeInTheDocument();
  });

  it("reports no matches and disables the step buttons", async () => {
    const user = userEvent.setup();
    render(<Conversation />);
    await user.type(screen.getByRole("searchbox", { name: "Find in conversation" }), "zebra");

    expect(screen.getByText("No matches")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next match" })).toBeDisabled();
  });

  it("clears the search on Escape", async () => {
    const user = userEvent.setup();
    render(<Conversation />);
    const input = screen.getByRole("searchbox", { name: "Find in conversation" });
    await user.type(input, "parser");
    await user.keyboard("{Escape}");
    expect(input).toHaveValue("");
  });
});
