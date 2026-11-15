import { describe, expect, it } from "vitest";
import { logLineText } from "./DiagnosticsPanel";

describe("logLineText", () => {
  it("unwraps the backend's JSON envelope to the server's own text", () => {
    expect(logLineText('{"stream":"stderr","text":"0.00.9 I print_info: model params = 27.32 B"}')).toBe(
      "0.00.9 I print_info: model params = 27.32 B",
    );
  });

  it("leaves plain lines and JSON without a text field alone", () => {
    expect(logLineText("plain line")).toBe("plain line");
    expect(logLineText('{"level":"info"}')).toBe('{"level":"info"}');
    expect(logLineText("42")).toBe("42");
  });
});
