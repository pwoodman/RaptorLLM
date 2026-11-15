import { describe, expect, it } from "vitest";
import { RaptorApiError } from "../api";
import { describeError, isErrorKind, messageOf } from "./errors";

describe("describeError", () => {
  it("maps a backend error kind to a readable title", () => {
    const error = new RaptorApiError({ kind: "Network", message: "Hugging Face is unreachable" });
    expect(describeError(error)).toEqual({ title: "Network error", message: "Hugging Face is unreachable" });
  });

  it("maps integrity failures to their own title", () => {
    const error = new RaptorApiError({ kind: "Integrity", message: "hash mismatch" });
    expect(describeError(error).title).toBe("Integrity check failed");
  });

  it("falls back to the Error message for unknown values", () => {
    expect(describeError(new Error("boom"))).toEqual({ title: "Unexpected error", message: "boom" });
    expect(describeError("plain string").message).toBe("plain string");
  });
});

describe("messageOf and isErrorKind", () => {
  it("returns only the message", () => {
    expect(messageOf(new RaptorApiError({ kind: "Conflict", message: "already installed" }))).toBe("already installed");
  });

  it("detects a specific kind", () => {
    const error = new RaptorApiError({ kind: "Unauthorized", message: "no key" });
    expect(isErrorKind(error, "Unauthorized")).toBe(true);
    expect(isErrorKind(error, "Network")).toBe(false);
    expect(isErrorKind(new Error("x"), "Unauthorized")).toBe(false);
  });
});
