import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, type SettingsView } from "../../api";
import { StoragePanel } from "./StoragePanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      dialog: { ...actual.api.dialog, pickFolder: vi.fn() },
      settings: { ...actual.api.settings, setStorageDir: vi.fn() },
    },
  };
});

function settings(overrides: Partial<SettingsView> = {}): SettingsView {
  return {
    storage_dir: null,
    default_storage_dir: "C:/Raptor/data",
    active_storage_dir: "C:/Raptor/data",
    restart_required: false,
    gateway_enabled: true,
    gateway_port: 8790,
    lan_enabled: false,
    api_key: null,
    context_target: 65_536,
    workspaces: [],
    min_context_target: 65_536,
    max_context_target: 200_000,
    notes: [],
    ...overrides,
  };
}

describe("StoragePanel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("asks before switching live services to a folder selected in Browse", async () => {
    const selected = "E:/Models/Raptor";
    vi.mocked(api.dialog.pickFolder).mockResolvedValue(selected);
    vi.mocked(api.settings.setStorageDir).mockResolvedValue(
      settings({ storage_dir: selected, restart_required: true }),
    );
    const onChange = vi.fn();

    render(<StoragePanel settings={settings()} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "Browse..." }));

    expect(await screen.findByRole("alertdialog", { name: "Switch storage folder now?" })).toBeInTheDocument();
    expect(api.settings.setStorageDir).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Switch folder now" }));

    expect(api.settings.setStorageDir).toHaveBeenCalledWith(selected);
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ storage_dir: selected, restart_required: true }),
    );
    expect(screen.getByRole("textbox", { name: "Raptor data folder" })).toHaveValue(selected);
  });

  it("shows folder picker errors instead of silently ignoring them", async () => {
    vi.mocked(api.dialog.pickFolder).mockRejectedValue(new Error("Folder dialog failed"));
    render(<StoragePanel settings={settings()} onChange={vi.fn()} />);

    await userEvent.click(screen.getByRole("button", { name: "Browse..." }));

    expect(await screen.findByText("Folder dialog failed")).toBeInTheDocument();
    expect(api.settings.setStorageDir).not.toHaveBeenCalled();
  });
});
