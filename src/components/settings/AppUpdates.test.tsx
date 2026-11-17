import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const mocks = vi.hoisted(() => ({ check: vi.fn() }));
vi.mock("@tauri-apps/plugin-updater", () => ({ check: mocks.check }));
import { AppUpdates } from "./AppUpdates";

beforeEach(() => { mocks.check.mockReset(); });
describe("update failures", () => {
  it("explains a missing release JSON without blaming the user's connection", async () => {
    mocks.check.mockRejectedValue("Could not fetch a valid release JSON from the remote");
    render(<AppUpdates />);
    await userEvent.click(screen.getByRole("button", { name: "Check for updates" }));
    expect(await screen.findByText(/signed release feed may be missing or invalid/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open GitHub Releases" })).toHaveAttribute("href", "https://github.com/pwoodman/RaptorLLM/releases");
    expect(screen.queryByText(/Check your internet connection/)).not.toBeInTheDocument();
  });

  it("offers a manual installer link for network errors too", async () => {
    mocks.check.mockRejectedValue(new Error("connection timed out"));
    render(<AppUpdates />);
    await userEvent.click(screen.getByRole("button", { name: "Check for updates" }));
    expect(await screen.findByText(/Couldn't reach the update service/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open GitHub Releases" })).toBeInTheDocument();
  });
});
