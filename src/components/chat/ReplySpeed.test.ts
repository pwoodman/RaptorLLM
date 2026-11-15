import { describe, expect, it } from "vitest";
import { replySpeedText } from "./MessageList";

describe("replySpeedText", () => {
  it("shows tokens per second and time to first token from the reported stats", () => {
    expect(replySpeedText({ tokens_per_second: 21.349, time_to_first_token_ms: 283 })).toBe(
      "21.3 tok/s · 283 ms to first token",
    );
    expect(replySpeedText({ tokens_per_second: 5, time_to_first_token_ms: 2400 })).toBe(
      "5.0 tok/s · 2.40 s to first token",
    );
  });

  it("shows nothing when the server reported no timings", () => {
    expect(replySpeedText(null)).toBeNull();
    expect(replySpeedText({})).toBeNull();
    expect(replySpeedText({ tokens_per_second: null })).toBeNull();
  });
});
