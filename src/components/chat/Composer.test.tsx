import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { chatStore, sendMessage } from "../../state/chat";
import { Composer } from "./Composer";

vi.mock("../../state/chat", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../state/chat")>();
  return {
    ...actual,
    sendMessage: vi.fn(async () => ({ ok: true as const })),
  };
});

function textFile(name: string, body: string): File {
  return new File([body], name, { type: "text/plain" });
}

describe("Composer attachments", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    chatStore.set({ drafts: new Map() });
  });

  it("attaches a dropped text file as a chip and sends it inside a delimited block", async () => {
    const user = userEvent.setup();
    render(<Composer conversationId="c1" blockedReason={null} />);

    const form = screen.getByRole("form", { name: "Message composer" });
    fireEvent.drop(form, { dataTransfer: { files: [textFile("lib.ts", "export const x = 1;")], types: ["Files"] } });

    expect(await screen.findByText("lib.ts")).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Message" }), "What is wrong here?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    const [text, attachments] = vi.mocked(sendMessage).mock.calls[0] ?? [];
    expect(text).toBe("What is wrong here?");
    expect(attachments).toEqual([expect.objectContaining({ name: "lib.ts", content: "export const x = 1;" })]);
    await waitFor(() => expect(screen.queryByText("lib.ts")).toBeNull());
  });

  it("refuses an image with a clear message and attaches nothing", async () => {
    // applyAccept off: the file picker filter would hide the image before the app ever sees it.
    const user = userEvent.setup({ applyAccept: false });
    render(<Composer conversationId="c1" blockedReason={null} />);

    const input = screen.getByLabelText("Attach text file");
    await user.upload(input, new File([new Uint8Array([137, 80, 78, 71])], "shot.png", { type: "image/png" }));

    expect(await screen.findByText("File not attached")).toBeInTheDocument();
    expect(screen.getByText(/Images and binary files cannot be attached/)).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Attached files" })).toBeNull();
  });

  it("can remove an attached file before sending", async () => {
    const user = userEvent.setup();
    render(<Composer conversationId="c1" blockedReason={null} />);
    await user.upload(screen.getByLabelText("Attach text file"), textFile("notes.md", "# notes"));

    await user.click(await screen.findByRole("button", { name: "Remove notes.md" }));

    expect(screen.queryByText("notes.md")).toBeNull();
  });

  it("keeps Send disabled with no text and no files", () => {
    render(<Composer conversationId="c1" blockedReason={null} />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("keeps text and attachments with their own chat after navigating away", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<Composer key="c1" conversationId="c1" blockedReason={null} />);
    await user.type(screen.getByRole("textbox", { name: "Message" }), "Review these notes");
    await user.upload(screen.getByLabelText("Attach text file"), textFile("notes.md", "# Notes"));
    expect(await screen.findByText("notes.md")).toBeInTheDocument();

    rerender(<Composer key="c2" conversationId="c2" blockedReason={null} />);
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("");
    expect(screen.queryByText("notes.md")).not.toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Message" }), "A different prompt");

    rerender(<Composer key="c1" conversationId="c1" blockedReason={null} />);
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("Review these notes");
    expect(screen.getByText("notes.md")).toBeInTheDocument();
  });
});
