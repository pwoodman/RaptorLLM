import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ status: vi.fn(), refresh: vi.fn() }));
vi.mock("../api", async (original) => {
  const actual = await original<typeof import("../api")>();
  return { ...actual, api: { ...actual.api, runtime: { ...actual.api.runtime, installStatus: mocks.status } } };
});
vi.mock("./engine", async (original) => ({ ...await original<typeof import("./engine")>(), refreshEngine: mocks.refresh }));
import { engineStore } from "./engine";
import { runtimeSetupStore, runRuntimeSetup, retryRuntimeSetup } from "./runtimeSetup";

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
  engineStore.set({ busy: null });
  runtimeSetupStore.set({ running: false, stage: "idle", label: "", error: null });
  mocks.status.mockResolvedValue({ stage: "preparing", downloaded: 0 });
  mocks.refresh.mockResolvedValue(undefined);
});
afterEach(() => vi.useRealTimers());
describe("persistent runtime setup", () => {
  it("uses actual download, verification and installation phases, and marks Ready only after success", async () => {
    let finish!: (value: string) => void;
    const action = new Promise<string>((resolve) => { finish = resolve; });
    const task = runRuntimeSetup("GPU engine", 1000, () => action);
    await vi.advanceTimersByTimeAsync(0);
    expect(engineStore.get().busy).toBe("installing");
    for (const stage of ["downloading", "verifying", "installing"] as const) {
      mocks.status.mockResolvedValue({ stage, downloaded: 800 });
      await vi.advanceTimersByTimeAsync(500);
      expect(runtimeSetupStore.get()).toMatchObject({ running: true, stage, downloaded: 800 });
    }
    mocks.status.mockResolvedValue({ stage: "ready", downloaded: 1000 });
    await vi.advanceTimersByTimeAsync(500);
    expect(runtimeSetupStore.get().stage).not.toBe("ready");
    finish("installed");
    expect(await task).toBe("installed");
    expect(runtimeSetupStore.get()).toMatchObject({ running: false, stage: "ready" });
    expect(engineStore.get().busy).toBeNull();
  });

  it("keeps a failed operation available for a one-click retry", async () => {
    const action = vi.fn().mockRejectedValueOnce(new Error("download interrupted")).mockResolvedValue("ready");
    await expect(runRuntimeSetup("GPU engine", null, action)).rejects.toThrow("download interrupted");
    expect(runtimeSetupStore.get()).toMatchObject({ running: false, stage: "failed", error: "download interrupted" });
    await retryRuntimeSetup();
    expect(action).toHaveBeenCalledTimes(2);
    expect(runtimeSetupStore.get()).toMatchObject({ stage: "ready", error: null });
  });
});
