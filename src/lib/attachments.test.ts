import { describe, expect, it } from "vitest";
import {
  checkAttachmentFile,
  composeMessage,
  MAX_ATTACHMENT_BYTES,
  MAX_ATTACHMENTS,
  readAttachment,
} from "./attachments";

describe("checkAttachmentFile", () => {
  it("accepts a small text or code file", () => {
    expect(checkAttachmentFile({ name: "main.rs", size: 120, type: "" }, 0)).toEqual({ ok: true });
  });

  it("refuses images and other media by MIME type", () => {
    const result = checkAttachmentFile({ name: "shot.bin", size: 10, type: "image/png" }, 0);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("Images and binary files cannot be attached");
  });

  it("refuses binary extensions even when the type is empty", () => {
    const result = checkAttachmentFile({ name: "model.gguf", size: 10, type: "" }, 0);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("binary file");
  });

  it("refuses files over the size cap, naming the limit", () => {
    const result = checkAttachmentFile({ name: "big.log", size: MAX_ATTACHMENT_BYTES + 1, type: "text/plain" }, 0);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("256 KB");
  });

  it("refuses empty files and files beyond the per-message count", () => {
    expect(checkAttachmentFile({ name: "a.txt", size: 0, type: "" }, 0).ok).toBe(false);
    const full = checkAttachmentFile({ name: "a.txt", size: 5, type: "" }, MAX_ATTACHMENTS);
    expect(full.ok).toBe(false);
  });
});

describe("readAttachment", () => {
  it("reads UTF-8 text", async () => {
    const file = new File(["fn main() {}\n"], "main.rs", { type: "text/plain" });
    const result = await readAttachment(file);
    expect(result).toEqual({
      ok: true,
      attachment: { name: "main.rs", size: file.size, content: "fn main() {}\n" },
    });
  });

  it("refuses content with NUL bytes as binary, even with a text name", async () => {
    const file = new File([new Uint8Array([0x4d, 0x5a, 0x00, 0x01])], "tool.txt", { type: "text/plain" });
    const result = await readAttachment(file);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("looks like a binary file");
  });
});

describe("composeMessage", () => {
  it("places each attachment in a delimited block after the typed text", () => {
    const text = composeMessage("Why does this fail?", [{ name: "lib.ts", size: 9, content: "export {}" }]);
    expect(text).toBe(
      [
        "Why does this fail?",
        "",
        "--- BEGIN ATTACHMENT: lib.ts (9 bytes) ---",
        "export {}",
        "--- END ATTACHMENT: lib.ts ---",
      ].join("\n"),
    );
  });

  it("returns the text alone when there are no attachments", () => {
    expect(composeMessage("hello", [])).toBe("hello");
  });
});
