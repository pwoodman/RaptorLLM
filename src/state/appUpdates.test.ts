import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ check: vi.fn(), idle: vi.fn(), download: vi.fn(), install: vi.fn(), close: vi.fn() }));
vi.mock("@tauri-apps/plugin-updater", () => ({ check: mocks.check }));
vi.mock("../api", async (original) => {
  const actual = await original<typeof import("../api")>();
  return { ...actual, api: { ...actual.api, runtime: { ...actual.api.runtime, idleStatus: mocks.idle } } };
});
beforeEach(() => {
  vi.resetModules();
  vi.resetAllMocks();
  vi.useFakeTimers();
  vi.setSystemTime(1_000_000);
  localStorage.clear();
  document.body.innerHTML = "";
  mocks.check.mockResolvedValue({ version: "0.1.4", body: "Update", download: mocks.download, install: mocks.install, close: mocks.close });
  mocks.idle.mockResolvedValue({ idle_seconds: 300, work_active: false });
  mocks.download.mockResolvedValue(undefined);
  mocks.install.mockResolvedValue(undefined);
  mocks.close.mockResolvedValue(undefined);
});
afterEach(() => vi.useRealTimers());

describe("idle automatic updates", () => {
  it("requires five full minutes of OS idle time, then a countdown before installation", async () => {
    const updates = await import("./appUpdates");
    mocks.idle.mockResolvedValue({ idle_seconds: 299, work_active: false });
    await updates.runIdleUpdateCycle();
    expect(mocks.check).toHaveBeenCalledTimes(1);
    expect(mocks.download).not.toHaveBeenCalled();
    mocks.idle.mockResolvedValue({ idle_seconds: 300, work_active: false });
    await updates.runIdleUpdateCycle();
    expect(mocks.download).toHaveBeenCalledTimes(1);
    expect(mocks.install).not.toHaveBeenCalled();
    expect(updates.appUpdatesStore.get().countdownAt).toBe(Date.now() + updates.UPDATE_COUNTDOWN_MS);
    await vi.advanceTimersByTimeAsync(updates.UPDATE_COUNTDOWN_MS);
    await updates.runIdleUpdateCycle();
    expect(mocks.install).toHaveBeenCalledWith({ restartAfterInstall: true });
  });

  it("cancels the countdown when the user returns and waits for a fresh idle period", async () => {
    const updates = await import("./appUpdates");
    await updates.runIdleUpdateCycle();
    mocks.idle.mockResolvedValue({ idle_seconds: 0, work_active: false });
    await vi.advanceTimersByTimeAsync(updates.UPDATE_COUNTDOWN_MS);
    await updates.runIdleUpdateCycle();
    expect(updates.appUpdatesStore.get().countdownAt).toBeNull();
    expect(mocks.install).not.toHaveBeenCalled();
    mocks.idle.mockResolvedValue({ idle_seconds: 300, work_active: false });
    await updates.runIdleUpdateCycle();
    expect(mocks.download).toHaveBeenCalledTimes(1);
    expect(mocks.install).not.toHaveBeenCalled();
    expect(updates.appUpdatesStore.get().countdownAt).toBe(Date.now() + updates.UPDATE_COUNTDOWN_MS);
  });

  it("rechecks activity after downloading so a returning user is not restarted", async () => {
    const updates = await import("./appUpdates");
    mocks.download.mockImplementation(async () => { mocks.idle.mockResolvedValue({ idle_seconds: 1, work_active: false }); });
    await updates.runIdleUpdateCycle();
    expect(updates.appUpdatesStore.get().phase).toBe("ready");
    expect(updates.appUpdatesStore.get().countdownAt).toBeNull();
    expect(mocks.install).not.toHaveBeenCalled();
  });

  it("postpones for API work, tuning and unsent chat text", async () => {
    const updates = await import("./appUpdates");
    const { engineStore } = await import("./engine");
    mocks.idle.mockResolvedValue({ idle_seconds: 900, work_active: true });
    await updates.runIdleUpdateCycle();
    expect(mocks.download).not.toHaveBeenCalled();
    mocks.idle.mockResolvedValue({ idle_seconds: 900, work_active: false });
    engineStore.set({ busy: "optimizing" });
    await updates.runIdleUpdateCycle();
    expect(mocks.download).not.toHaveBeenCalled();
    engineStore.set({ busy: null });
    document.body.innerHTML = '<form data-unsaved-work="true"></form>';
    await updates.runIdleUpdateCycle();
    expect(mocks.download).not.toHaveBeenCalled();
    document.body.innerHTML = "";
    await updates.runIdleUpdateCycle();
    expect(mocks.download).toHaveBeenCalledTimes(1);
  });

  it("honors a persisted opt-out and lets the user postpone an update for one hour", async () => {
    const updates = await import("./appUpdates");
    await updates.runIdleUpdateCycle();
    updates.postponeUpdate();
    await vi.advanceTimersByTimeAsync(updates.UPDATE_COUNTDOWN_MS);
    await updates.runIdleUpdateCycle();
    expect(mocks.install).not.toHaveBeenCalled();
    updates.setAutomaticUpdates(false);
    expect(localStorage.getItem("raptor.autoUpdates")).toBe("false");
    vi.resetModules();
    const reopened = await import("./appUpdates");
    expect(reopened.appUpdatesStore.get().enabled).toBe(false);
    await reopened.runIdleUpdateCycle();
    expect(mocks.check).toHaveBeenCalledTimes(1);
  });

  it("never automatically installs without supported OS idle detection or valid signed download", async () => {
    const updates = await import("./appUpdates");
    mocks.idle.mockResolvedValue({ idle_seconds: null, work_active: false });
    await updates.runIdleUpdateCycle();
    expect(mocks.download).not.toHaveBeenCalled();
    mocks.idle.mockResolvedValue({ idle_seconds: 300, work_active: false });
    mocks.download.mockRejectedValue(new Error("invalid signature"));
    await updates.runIdleUpdateCycle();
    expect(mocks.install).not.toHaveBeenCalled();
    expect(updates.appUpdatesStore.get().error).toContain("invalid signature");
    await updates.runIdleUpdateCycle();
    expect(mocks.download).toHaveBeenCalledTimes(1);
  });
});
