import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const mocks = vi.hoisted(() => ({ check: vi.fn(), install: vi.fn(), status: vi.fn(), refresh: vi.fn() }));
vi.mock("../../api", async (original) => {
  const actual = await original<typeof import("../../api")>();
  return { ...actual, api: { ...actual.api,
    dependencies: { ...actual.api.dependencies, check: mocks.check, installUpdate: mocks.install },
    runtime: { ...actual.api.runtime, installStatus: mocks.status },
  } };
});
vi.mock("../../state/engine", async (original) => ({ ...await original<typeof import("../../state/engine")>(), refreshEngine: mocks.refresh }));
import { DependenciesPanel } from "./DependenciesPanel";
import { runtimeSetupStore } from "../../state/runtimeSetup";
import { engineStore } from "../../state/engine";

const update = { id: "llama-hip-win", current_version: "b1", latest_version: "b2", available: true, url: "https://example.com/runtime.zip", size_bytes: 1000, sha256: "a".repeat(64), note: "New build" };
const report = { updates: [update], hip_sdk: null, vulkan: null, check_error: null, latest_tag: "b2" };
beforeEach(() => {
  vi.resetAllMocks();
  engineStore.set({ loaded: null, busy: null });
  runtimeSetupStore.set({ running: false, stage: "idle", label: "", error: null });
  mocks.check.mockResolvedValue(report);
  mocks.status.mockResolvedValue({ stage: "downloading", downloaded: 400 });
  mocks.refresh.mockResolvedValue(undefined);
});
describe("easy dependency setup", () => {
  it("shows real progress after confirmation and Ready after the installation finishes", async () => {
    let finish!: (value: unknown) => void;
    mocks.install.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    render(<DependenciesPanel />);
    await userEvent.click(await screen.findByRole("button", { name: "Update to b2" }));
    await userEvent.click(screen.getByRole("button", { name: "Download and install" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    expect(await screen.findByText("Downloading…")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Dependency download progress" })).toHaveAttribute("value", "400");
    finish({ id: update.id, version: "b2" });
    await waitFor(() => expect(screen.getByText(/Ready to use/)).toBeInTheDocument());
    expect(mocks.install).toHaveBeenCalledWith(update.id);
  });

  it("offers recommended setup directly when no runtime is installed", async () => {
    mocks.check.mockResolvedValue({ ...report, updates: [] });
    render(<DependenciesPanel />);
    expect(await screen.findByRole("button", { name: "Set up recommended engine" })).toBeInTheDocument();
  });
});
