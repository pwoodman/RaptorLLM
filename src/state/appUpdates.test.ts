import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ check: vi.fn(), idle: vi.fn(), download: vi.fn(), install: vi.fn(), close: vi.fn(), discover: vi.fn(), importDiscovered: vi.fn() }));
vi.mock("@tauri-apps/plugin-updater", () => ({ check: mocks.check }));
vi.mock("../api", async (original) => {
  const actual = await original<typeof import("../api")>();
  return { ...actual, api: {
    ...actual.api,
    runtime: { ...actual.api.runtime, idleStatus: mocks.idle },
    models: { ...actual.api.models, discover: mocks.discover, importDiscovered: mocks.importDiscovered },
  } };
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

describe("manual updates", () => {
  it("installs during discovery and stops the scan from starting an import during the download", async () => {
    const { checkAppUpdates, installAppUpdate } = await import("./appUpdates");
    const { syncLocalModels, localModelsStore } = await import("./localModels");
    const folders = [{ path: "C:/models", source: "Local", items: [{ path: "C:/models/a.gguf", size_bytes: 100 }] }];
    let finishScan!: (value: typeof folders) => void;
    mocks.discover.mockReturnValue(new Promise((resolve) => { finishScan = resolve; }));
    const scan = syncLocalModels();
    expect(localModelsStore.get().phase).toBe("scanning");
    mocks.download.mockImplementationOnce(async () => {
      finishScan(folders);
      await scan;
    });
    mocks.idle.mockResolvedValue({ idle_seconds: 0, work_active: false });
    await checkAppUpdates();
    await installAppUpdate();
    expect(mocks.importDiscovered).not.toHaveBeenCalled();
    expect(mocks.install).toHaveBeenCalledWith({ restartAfterInstall: true });
  });

  it("identifies active indexing and allows retry once it finishes", async () => {
    const { appUpdatesStore, checkAppUpdates, installAppUpdate } = await import("./appUpdates");
    const { localModelsStore } = await import("./localModels");
    await checkAppUpdates();
    localModelsStore.set({ phase: "indexing" });
    await installAppUpdate();
    expect(appUpdatesStore.get().message).toContain("registering local models");
    expect(mocks.download).not.toHaveBeenCalled();
    localModelsStore.set({ phase: "idle" });
    await installAppUpdate();
    expect(mocks.install).toHaveBeenCalledTimes(1);
  });

  it("protects a draft created during download and an active API request on retry", async () => {
    const { appUpdatesStore, checkAppUpdates, installAppUpdate } = await import("./appUpdates");
    mocks.download.mockImplementationOnce(async () => { document.body.innerHTML = '<form data-unsaved-work="true"></form>'; });
    await checkAppUpdates();
    await installAppUpdate();
    expect(appUpdatesStore.get().message).toContain("unsent chat draft");
    document.body.innerHTML = "";
    mocks.idle.mockResolvedValue({ idle_seconds: 0, work_active: true });
    await installAppUpdate();
    expect(appUpdatesStore.get().message).toContain("API request");
    expect(mocks.install).not.toHaveBeenCalled();
    mocks.idle.mockResolvedValue({ idle_seconds: 0, work_active: false });
    await installAppUpdate();
    expect(mocks.download).toHaveBeenCalledTimes(1);
    expect(mocks.install).toHaveBeenCalledTimes(1);
  });
});

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
