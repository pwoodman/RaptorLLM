import { beforeEach, describe, expect, it, vi } from "vitest";
import { engineStatus, loadOutcome, modelEntry } from "../test/fixtures";

const mocks = vi.hoisted(() => ({ offer: vi.fn(), remember: vi.fn(), load: vi.fn(), unload: vi.fn(), tune: vi.fn(), report: vi.fn() }));
vi.mock("./firstLoad", () => ({ offerFirstLoadOptimization: mocks.offer, rememberOptimizationOffer: mocks.remember }));
vi.mock("../api", async (original) => {
  const actual = await original<typeof import("../api")>();
  return { ...actual, modelsLoadWithPreset: mocks.load, api: {
    ...actual.api, models: { ...actual.api.models, unload: mocks.unload, tune: mocks.tune, loadReport: mocks.report, current: async () => null },
    runtime: { ...actual.api.runtime, status: async () => engineStatus({ state: "stopped" }) },
    gateway: { ...actual.api.gateway, status: async () => null },
  } };
});
import { loadModel, engineStore } from "./engine";
import { libraryStore } from "./library";
import { tuningStore } from "./tuning";

beforeEach(() => {
  vi.clearAllMocks();
  engineStore.set({ loaded: null, loadReport: null, busy: null });
  tuningStore.set({ activeId: null, reports: {} });
  libraryStore.set({ models: [modelEntry()] });
  mocks.offer.mockResolvedValue(false);
  mocks.load.mockResolvedValue(loadOutcome());
  mocks.unload.mockResolvedValue(engineStatus({ state: "stopped" }));
  mocks.tune.mockResolvedValue({ model_id: modelEntry().id });
  mocks.report.mockResolvedValue({ ctx_size: 8192, gpu_layers: 32 });
});

describe("optimize before loading", () => {
  it("loads directly after Skip and preserves the chosen preset and context window", async () => {
    const entry = modelEntry();
    await loadModel(entry.id, "coding", { ctx: 8192, cacheType: "q8_0", slots: 1, speculation: false });
    expect(mocks.load).toHaveBeenCalledWith(entry.id, 8192, "coding", "q8_0", 1, false);
    expect(mocks.tune).not.toHaveBeenCalled();
  });

  it("stops the old model, tunes once, and loads for chat automatically", async () => {
    mocks.offer.mockResolvedValue(true);
    await loadModel(modelEntry().id);
    expect(mocks.unload.mock.invocationCallOrder[0]).toBeLessThan(mocks.tune.mock.invocationCallOrder[0]);
    expect(mocks.tune.mock.invocationCallOrder[0]).toBeLessThan(mocks.load.mock.invocationCallOrder[0]);
    expect(mocks.tune).toHaveBeenCalledWith(modelEntry().id, false, null, null);
    expect(mocks.remember).toHaveBeenCalledWith(modelEntry());
    expect(engineStore.get().busy).toBeNull();
    expect(tuningStore.get().activeId).toBeNull();
  });

  it("does not claim completion or load when tuning fails", async () => {
    mocks.offer.mockResolvedValue(true);
    mocks.tune.mockRejectedValueOnce(new Error("benchmark failed"));
    await expect(loadModel(modelEntry().id)).rejects.toThrow("benchmark failed");
    expect(mocks.remember).not.toHaveBeenCalled();
    expect(mocks.load).not.toHaveBeenCalled();
    expect(engineStore.get().busy).toBeNull();
    expect(tuningStore.get().activeId).toBeNull();
  });
});
