import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, type SettingsView } from "../../api";
import { ApiKeyPanel } from "./ApiKeyPanel";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: { settings: { rotateApiKey: vi.fn(), get: vi.fn(), revokeApiKey: vi.fn() } },
  };
});

function view(overrides: Partial<SettingsView> = {}): SettingsView {
  return {
    storage_dir: null,
    default_storage_dir: "C:/data",
    active_storage_dir: "C:/data",
    restart_required: false,
    gateway_enabled: true,
    gateway_port: 8790,
    lan_enabled: false,
    api_key: null,
    context_target: 65536,
    workspaces: [],
    min_context_target: 65536,
    max_context_target: 200000,
    notes: [],
    ...overrides,
  };
}

describe("ApiKeyPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.settings.get).mockResolvedValue(view({ api_key: { id: "key-1", created_at: 1 } }));
  });

  it("shows a new key once, and forgets it when dismissed", async () => {
    vi.mocked(api.settings.rotateApiKey).mockResolvedValue({ id: "key-1", secret: "sk-raptor-test-123" });
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<ApiKeyPanel settings={view()} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Generate key" }));

    const input = await screen.findByRole("textbox", { name: "New API key" });
    expect(input).toHaveValue("sk-raptor-test-123");
    await waitFor(() => expect(onChange).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "I have saved the key" }));
    expect(screen.queryByRole("textbox", { name: "New API key" })).toBeNull();
    expect(screen.queryByText("sk-raptor-test-123")).toBeNull();
  });

  it("asks before replacing an existing key", async () => {
    const user = userEvent.setup();
    render(<ApiKeyPanel settings={view({ api_key: { id: "key-1", created_at: 1 } })} onChange={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Rotate key" }));

    expect(screen.getByRole("alertdialog", { name: "Replace the API key?" })).toBeInTheDocument();
    expect(api.settings.rotateApiKey).not.toHaveBeenCalled();
  });

  it("keeps revoke disabled while LAN is on", () => {
    render(<ApiKeyPanel settings={view({ api_key: { id: "key-1", created_at: 1 }, lan_enabled: true })} onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Revoke key" })).toBeDisabled();
  });
});
