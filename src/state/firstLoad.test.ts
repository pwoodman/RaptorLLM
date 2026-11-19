import { beforeEach, describe, expect, it, vi } from "vitest";
import { modelEntry } from "../test/fixtures";

const mocks = vi.hoisted(() => ({ saved: vi.fn() }));
vi.mock("../api", async (original) => ({
  ...await original<typeof import("../api")>(), api: { models: { savedTuning: mocks.saved } },
}));
beforeEach(() => { vi.resetModules(); mocks.saved.mockReset(); mocks.saved.mockResolvedValue([]); localStorage.clear(); });

describe("first-load optimization offer", () => {
  it("offers an unoptimized model once and remembers Skip across app sessions", async () => {
    const first = await import("./firstLoad");
    const entry = modelEntry();
    const decision = first.offerFirstLoadOptimization(entry);
    await vi.waitFor(() => expect(first.firstLoadStore.get().entry?.id).toBe(entry.id));
    first.chooseFirstLoadOptimization(false);
    expect(await decision).toBe(false);
    vi.resetModules();
    const reopened = await import("./firstLoad");
    expect(await reopened.offerFirstLoadOptimization(entry)).toBe(false);
    expect(reopened.firstLoadStore.get().entry).toBeNull();
    expect(mocks.saved).toHaveBeenCalledTimes(1);
  });

  it("does not prompt when the model already has a saved optimization profile", async () => {
    mocks.saved.mockResolvedValue([{ tuning: { qualification: {context_size: 65536}, profile: { kv_cache: "f16", flash_attn: true } } }]);
    const first = await import("./firstLoad");
    expect(await first.offerFirstLoadOptimization(modelEntry())).toBe(false);
    expect(first.firstLoadStore.get().entry).toBeNull();
  });

  it("remembers successful optimization by content even if a model is reimported under another id", async () => {
    const first = await import("./firstLoad");
    const entry = modelEntry();
    first.rememberOptimizationOffer(entry);
    expect(await first.offerFirstLoadOptimization({ ...entry, id: "local/reimport.gguf" })).toBe(false);
    expect(mocks.saved).not.toHaveBeenCalled();
  });

  it("keeps optimization optional if saved profiles cannot be read", async () => {
    mocks.saved.mockRejectedValue(new Error("unavailable"));
    const first = await import("./firstLoad");
    expect(await first.offerFirstLoadOptimization(modelEntry())).toBe(false);
    expect(first.firstLoadStore.get().entry).toBeNull();
  });
});
