import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EngineStatus } from "../api";

const mocks = vi.hoisted(() => ({
  status: vi.fn(),
  gateway: vi.fn(),
  current: vi.fn(),
  loadReport: vi.fn(),
}));

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      runtime: { ...actual.api.runtime, status: mocks.status },
      gateway: { ...actual.api.gateway, status: mocks.gateway },
      models: { ...actual.api.models, current: mocks.current, loadReport: mocks.loadReport },
    },
  };
});

import { engineStore, inferenceBlockReason, refreshEngine } from "./engine";

function status(state: EngineStatus["server"]["state"]): EngineStatus {
  return {
    hardware_detected: true,
    runtime: null,
    server: { state, endpoint: null, pid: null, restarts: 0 },
    gpu_verdict: null,
  };
}

describe("inferenceBlockReason", () => {
  it("blocks chat until the server is ready, with a reason for each state", () => {
    expect(inferenceBlockReason(null, null)).toContain("Checking");
    expect(inferenceBlockReason(status({ state: "stopped" }), null)).toContain("not running");
    expect(inferenceBlockReason(status({ state: "starting" }), null)).toContain("starting");
    expect(inferenceBlockReason(status({ state: "crashed", reason: "OOM" }), null)).toContain("OOM");
    expect(inferenceBlockReason(status({ state: "ready" }), null)).toBeNull();
  });

  it("does not block a degraded server, which the chat banner reports separately", () => {
    expect(inferenceBlockReason(status({ state: "degraded", reason: "slow" }), null)).toBeNull();
  });

  it("reports a status read failure before anything else", () => {
    expect(inferenceBlockReason(status({ state: "ready" }), "timeout")).toContain("timeout");
  });
});

describe("refreshEngine after a window reload", () => {
  beforeEach(() => {
    engineStore.set({ status: null, loaded: null, loadReport: null, busy: null, statusError: null });
    mocks.status.mockResolvedValue(status({ state: "ready" }));
    mocks.gateway.mockResolvedValue(null);
    mocks.current.mockResolvedValue({ id: "local/m.gguf" });
  });

  it("restores the planned context and layers the backend reports for the serving load", async () => {
    mocks.loadReport.mockResolvedValue({
      result: { model_id: "local/m.gguf", rationale: ["fits"], estimate: null },
      ctx_size: 8192,
      gpu_layers: 66,
    });
    await refreshEngine();
    const { loaded, loadReport } = engineStore.get();
    expect(loaded).toMatchObject({ modelId: "local/m.gguf", ctxSize: 8192, plannedGpuLayers: 66, rationale: ["fits"] });
    expect(loadReport).not.toBeNull();
  });

  it("falls back to an unknown context when the backend has no report for that model", async () => {
    mocks.loadReport.mockResolvedValue(null);
    await refreshEngine();
    expect(engineStore.get().loaded).toMatchObject({ modelId: "local/m.gguf", ctxSize: null });
  });

  it("does not replace this window's own load state while a load is running", async () => {
    engineStore.set({ busy: "loading" });
    await refreshEngine();
    expect(engineStore.get().loaded).toBeNull();
    expect(engineStore.get().status).not.toBeNull();
  });
});
