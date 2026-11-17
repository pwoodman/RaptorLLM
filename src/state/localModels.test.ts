import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ discover: vi.fn(), register: vi.fn(), refresh: vi.fn() }));
vi.mock("../api", async (original) => ({
  ...await original<typeof import("../api")>(),
  api: { models: { discover: mocks.discover, importDiscovered: mocks.register } },
}));
vi.mock("./library", () => ({ refreshLibrary: mocks.refresh }));

const folder = (...paths: string[]) => [{ path: "C:/models", source: "Local", items: paths.map((path) => ({ path, size_bytes: 100 })) }];
beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  mocks.discover.mockResolvedValue([]);
  mocks.register.mockResolvedValue({});
  mocks.refresh.mockResolvedValue(undefined);
});
afterEach(() => vi.useRealTimers());

describe("automatic local models", () => {
  it("registers every discovered model without user action and refreshes the picker incrementally", async () => {
    mocks.discover.mockResolvedValue(folder("C:/models/a.gguf", "C:/models/b.gguf"));
    const { syncLocalModels, localModelsStore } = await import("./localModels");
    await syncLocalModels();
    expect(mocks.register.mock.calls).toEqual([["C:/models/a.gguf"], ["C:/models/b.gguf"]]);
    expect(mocks.refresh).toHaveBeenCalledTimes(2);
    expect(localModelsStore.get()).toMatchObject({ phase: "idle", added: 2, errors: [] });
  });

  it("shares one queue and deduplicates overlapping Windows folder scans", async () => {
    mocks.discover.mockResolvedValue(folder("C:/models/a.gguf", "c:\\models\\a.gguf"));
    let finish!: () => void;
    mocks.register.mockReturnValue(new Promise<void>((resolve) => { finish = resolve; }));
    const { syncLocalModels } = await import("./localModels");
    const first = syncLocalModels();
    expect(syncLocalModels()).toBe(first);
    await vi.waitFor(() => expect(mocks.register).toHaveBeenCalledTimes(1));
    finish();
    await first;
    await syncLocalModels();
    expect(mocks.register).toHaveBeenCalledTimes(1);
  });

  it("continues after a broken file and retries failures only when requested", async () => {
    mocks.discover.mockResolvedValue(folder("C:/models/bad.gguf", "C:/models/good.gguf"));
    mocks.register.mockRejectedValueOnce(new Error("file unavailable")).mockResolvedValue({});
    const { syncLocalModels, localModelsStore } = await import("./localModels");
    await syncLocalModels();
    expect(localModelsStore.get()).toMatchObject({ added: 1, done: 2 });
    expect(localModelsStore.get().errors[0]).toContain("bad.gguf: file unavailable");
    await syncLocalModels();
    expect(mocks.register).toHaveBeenCalledTimes(2);
    await syncLocalModels(true);
    expect(mocks.register).toHaveBeenCalledTimes(3);
    expect(localModelsStore.get().errors).toEqual([]);
  });

  it("checks for newly arriving files in the background and cleans up its timer", async () => {
    vi.useFakeTimers();
    const { startLocalModelSync, syncLocalModels } = await import("./localModels");
    const stop = startLocalModelSync();
    await syncLocalModels();
    mocks.discover.mockResolvedValue(folder("C:/models/new.gguf"));
    await vi.advanceTimersByTimeAsync(60_000);
    await syncLocalModels();
    expect(mocks.register).toHaveBeenCalledWith("C:/models/new.gguf");
    stop();
    const calls = mocks.discover.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(mocks.discover).toHaveBeenCalledTimes(calls);
  });

  it("rescans a newly chosen folder after the current indexing pass finishes", async () => {
    mocks.discover.mockResolvedValueOnce(folder("C:/models/a.gguf")).mockResolvedValue(folder("D:/new/b.gguf"));
    let finish!: () => void;
    mocks.register.mockReturnValueOnce(new Promise<void>((resolve) => { finish = resolve; })).mockResolvedValue({});
    const { syncLocalModels } = await import("./localModels");
    const first = syncLocalModels();
    await vi.waitFor(() => expect(mocks.register).toHaveBeenCalledTimes(1));
    expect(syncLocalModels(true)).toBe(first);
    finish();
    await first;
    expect(mocks.register).toHaveBeenCalledWith("D:/new/b.gguf");
  });
});
