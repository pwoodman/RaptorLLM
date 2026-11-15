import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { settingsGet, settingsSetAllowCpuFallback } from "../../api";
import { settingsView } from "../../test/fixtures";
import { CpuFallbackToggle } from "./CpuFallbackToggle";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, settingsGet: vi.fn(), settingsSetAllowCpuFallback: vi.fn() };
});

describe("CpuFallbackToggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("is off by default and shows no warning", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView({ allow_cpu_fallback: false }));
    render(<CpuFallbackToggle />);
    expect(await screen.findByRole("switch", { name: /Allow CPU fallback/ })).not.toBeChecked();
    expect(screen.queryByText("CPU fallback is on")).toBeNull();
  });

  it("turning it on warns loudly and saves the setting", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView({ allow_cpu_fallback: false }));
    vi.mocked(settingsSetAllowCpuFallback).mockResolvedValue(settingsView({ allow_cpu_fallback: true }));
    const user = userEvent.setup();
    render(<CpuFallbackToggle />);

    await user.click(await screen.findByRole("switch", { name: /Allow CPU fallback/ }));

    await waitFor(() => expect(settingsSetAllowCpuFallback).toHaveBeenCalledWith(true));
    expect(await screen.findByText("CPU fallback is on")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: /Allow CPU fallback/ })).toBeChecked();
  });

  it("shows the refusal when the setting cannot be saved", async () => {
    vi.mocked(settingsGet).mockResolvedValue(settingsView());
    vi.mocked(settingsSetAllowCpuFallback).mockRejectedValue(new Error("settings file is read-only"));
    const user = userEvent.setup();
    render(<CpuFallbackToggle />);

    await user.click(await screen.findByRole("switch", { name: /Allow CPU fallback/ }));
    expect(await screen.findByText("Could not change the setting")).toBeInTheDocument();
    expect(screen.getByText("settings file is read-only")).toBeInTheDocument();
  });

  it("is not used as a silent default: a failed read is shown, not assumed off", async () => {
    vi.mocked(settingsGet).mockRejectedValue(new Error("bridge offline"));
    render(<CpuFallbackToggle />);
    expect(await screen.findByText("bridge offline")).toBeInTheDocument();
    expect(screen.queryByRole("switch")).toBeNull();
  });
});
