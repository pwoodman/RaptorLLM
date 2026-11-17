import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { settingsView } from "../test/fixtures";

vi.mock("../api", async (original) => ({ ...await original<typeof import("../api")>(), settingsGet: async () => settingsView() }));
vi.mock("../components/settings/AppearancePanel", () => ({ AppearancePanel: () => <section aria-label="Appearance panel" /> }));
vi.mock("../components/settings/HardwarePanel", () => ({ HardwarePanel: () => <section aria-label="Hardware panel" /> }));
vi.mock("../components/settings/RuntimePanel", () => ({ RuntimePanel: () => <section aria-label="Runtime panel" /> }));
vi.mock("../components/settings/DependenciesPanel", () => ({ DependenciesPanel: () => <section aria-label="Dependencies panel" /> }));
vi.mock("../components/settings/ServerPanel", () => ({ ServerPanel: () => <section aria-label="Server panel" /> }));
vi.mock("../components/settings/StoragePanel", () => ({ StoragePanel: () => <section aria-label="Storage panel" /> }));
vi.mock("../components/settings/ContextPanel", () => ({ ContextPanel: () => <section aria-label="Context panel" /> }));
vi.mock("../components/settings/DiagnosticsPanel", () => ({ DiagnosticsPanel: () => <section aria-label="Diagnostics panel" /> }));
vi.mock("../components/settings/AppUpdates", () => ({ AppUpdates: () => <p>Update controls</p> }));
import SettingsPage from "./Settings";
import { openSettings, uiStore } from "../state/ui";

beforeEach(() => uiStore.set({ settingsSection: "appearance", anchor: null }));
describe("separate settings screens", () => {
  it("renders only the selected screen and remembers the selection on return", async () => {
    const view = render(<SettingsPage />);
    expect(screen.getByRole("region", { name: "Appearance panel" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Hardware panel" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Hardware" }));
    expect(screen.getByRole("region", { name: "Hardware panel" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Appearance panel" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hardware" })).toHaveAttribute("aria-current", "page");
    view.unmount();
    render(<SettingsPage />);
    expect(screen.getByRole("region", { name: "Hardware panel" })).toBeInTheDocument();
  });

  it("opens a requested section from a setup shortcut", async () => {
    openSettings("storage");
    render(<SettingsPage />);
    await waitFor(() => expect(screen.getByRole("region", { name: "Storage panel" })).toBeInTheDocument());
    expect(screen.queryByRole("region", { name: "Appearance panel" })).not.toBeInTheDocument();
  });
});
