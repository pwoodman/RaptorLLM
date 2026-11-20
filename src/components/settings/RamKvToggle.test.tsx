import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { settingsGet, settingsSetAllowRamKv } from "../../api";
import { settingsView } from "../../test/fixtures";
import { RamKvToggle } from "./RamKvToggle";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, settingsGet: vi.fn(), settingsSetAllowRamKv: vi.fn() };
});

describe("RamKvToggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is off by default and shows no warning", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView({ allow_ram_kv: false }));
    render(<RamKvToggle />);
    expect(await screen.findByRole("switch", { name: /KV cache in system RAM/ })).not.toBeChecked();
    expect(screen.queryByText("KV cache in RAM is on")).toBeNull();
  });

  it("treats a missing field in an older shell as off", async () => {
    const view = settingsView();
    delete (view as { allow_ram_kv?: boolean }).allow_ram_kv;
    vi.mocked(settingsGet).mockResolvedValue(view);
    render(<RamKvToggle />);
    expect(await screen.findByRole("switch", { name: /KV cache in system RAM/ })).not.toBeChecked();
  });

  it("turning it on warns loudly and saves the setting", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView({ allow_ram_kv: false }));
    vi.mocked(settingsSetAllowRamKv).mockResolvedValue(settingsView({ allow_ram_kv: true }));
    const user = userEvent.setup();
    render(<RamKvToggle />);

    await user.click(await screen.findByRole("switch", { name: /KV cache in system RAM/ }));

    await waitFor(() => expect(settingsSetAllowRamKv).toHaveBeenCalledWith(true));
    expect(await screen.findByText("KV cache in RAM is on")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: /KV cache in system RAM/ })).toBeChecked();
  });

  it("shows the refusal when the setting cannot be saved", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView());
    vi.mocked(settingsSetAllowRamKv).mockRejectedValue(new Error("settings file is read-only"));
    const user = userEvent.setup();
    render(<RamKvToggle />);

    await user.click(await screen.findByRole("switch", { name: /KV cache in system RAM/ }));
    expect(await screen.findByText("Could not change the setting")).toBeInTheDocument();
    expect(screen.getByText("settings file is read-only")).toBeInTheDocument();
  });

  it("is not used as a silent default: a failed read is shown, not assumed off", async () => {
    vi.mocked(settingsGet).mockRejectedValue(new Error("bridge offline"));
    render(<RamKvToggle />);
    expect(await screen.findByText("bridge offline")).toBeInTheDocument();
    expect(screen.queryByRole("switch")).toBeNull();
  });
});
