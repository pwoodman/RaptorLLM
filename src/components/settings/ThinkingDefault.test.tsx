import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { settingsGet, settingsSetThinkingEffort } from "../../api";
import { settingsView } from "../../test/fixtures";
import { ThinkingDefault } from "./ThinkingDefault";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, settingsGet: vi.fn(), settingsSetThinkingEffort: vi.fn() };
});

describe("ThinkingDefault", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is Medium when the shell reports nothing", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView());
    render(<ThinkingDefault />);
    const select = await screen.findByRole("combobox", { name: "Default thinking effort" });
    await waitFor(() => expect(select).not.toBeDisabled());
    expect(select).toHaveValue("medium");
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["Low", "Medium", "High", "Max"]);
  });

  it("shows the saved level", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView({ thinking_effort: "max" }));
    render(<ThinkingDefault />);
    await waitFor(() => expect(screen.getByRole("combobox")).toHaveValue("max"));
  });

  it("saves the level the user picks", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView());
    vi.mocked(settingsSetThinkingEffort).mockResolvedValue(settingsView({ thinking_effort: "high" }));
    const user = userEvent.setup();
    render(<ThinkingDefault />);
    const select = await screen.findByRole("combobox");
    await waitFor(() => expect(select).not.toBeDisabled());

    await user.selectOptions(select, "high");

    await waitFor(() => expect(settingsSetThinkingEffort).toHaveBeenCalledWith("high"));
    await waitFor(() => expect(screen.getByRole("combobox")).toHaveValue("high"));
  });

  it("shows the refusal when the setting cannot be saved", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView());
    vi.mocked(settingsSetThinkingEffort).mockRejectedValue(new Error("settings file is read-only"));
    const user = userEvent.setup();
    render(<ThinkingDefault />);
    const select = await screen.findByRole("combobox");
    await waitFor(() => expect(select).not.toBeDisabled());

    await user.selectOptions(select, "low");

    expect(await screen.findByText("settings file is read-only")).toBeInTheDocument();
  });
});
