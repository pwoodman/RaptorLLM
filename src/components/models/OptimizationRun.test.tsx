import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { modelEntry } from "../../test/fixtures";

const mocks = vi.hoisted(() => ({ progress: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      models: { ...actual.api.models, tuneProgress: mocks.progress },
    },
  };
});

import { OptimizationRun, formatElapsed } from "./OptimizationRun";
import { tuningStore } from "../../state/tuning";
import { libraryStore } from "../../state/library";

describe("formatElapsed", () => {
  it("formats mm:ss under an hour and h:mm:ss beyond it", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(9)).toBe("0:09");
    expect(formatElapsed(75)).toBe("1:15");
    expect(formatElapsed(3599)).toBe("59:59");
    expect(formatElapsed(3600)).toBe("1:00:00");
    expect(formatElapsed(7384)).toBe("2:03:04");
  });
});

describe("OptimizationRun", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.progress.mockResolvedValue({ done: 2, total: 6 });
    tuningStore.set({ activeId: null, startedAt: null, reports: {} });
    libraryStore.set({ models: [modelEntry({ id: "org/repo/model.gguf", file_name: "granite-30b.gguf" })], storage: null, error: null });
  });

  afterEach(() => {
    tuningStore.set({ activeId: null, startedAt: null, reports: {} });
    libraryStore.set({ models: null, storage: null, error: null });
  });

  it("renders nothing when no optimization is running", () => {
    const { container } = render(<OptimizationRun />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the model, elapsed clock, real progress and an honest tip while tuning", async () => {
    tuningStore.set({ activeId: "org/repo/model.gguf", startedAt: Date.now() - 65_000, reports: {} });
    render(<OptimizationRun />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Optimizing")).toBeInTheDocument();
    expect(screen.getByText("granite-30b.gguf")).toBeInTheDocument();
    expect(screen.getByText("1:05", { exact: false })).toBeInTheDocument();
    expect(await screen.findByText("Setup 3 of 6 · 33%")).toBeInTheDocument();
    expect(screen.getByText(/Raptor is benchmarking full-window generations/)).toBeInTheDocument();
    expect(screen.getByText("Good to know", { exact: false })).toBeInTheDocument();
  });

  it("announces the confirmation phase when every candidate has been measured", async () => {
    mocks.progress.mockResolvedValue({ done: 6, total: 6 });
    tuningStore.set({ activeId: "org/repo/model.gguf", startedAt: Date.now(), reports: {} });
    render(<OptimizationRun />);

    expect(await screen.findByText("Confirming the winner across matched runs")).toBeInTheDocument();
  });

  it("falls back to an indeterminate bar before the first progress report arrives", () => {
    mocks.progress.mockReturnValue(new Promise(() => undefined));
    tuningStore.set({ activeId: "org/repo/model.gguf", startedAt: Date.now(), reports: {} });
    render(<OptimizationRun />);

    expect(screen.getByText("Preparing the default setup")).toBeInTheDocument();
  });

  it("rotates the fact of the moment on an interval", async () => {
    vi.useFakeTimers();
    try {
      tuningStore.set({ activeId: "org/repo/model.gguf", startedAt: Date.now(), reports: {} });
      render(<OptimizationRun />);
      expect(screen.getByText(/Every candidate runs the same full-window request/i)).toBeInTheDocument();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(7_000);
      });
      expect(screen.getByText(/only wins after beating the default/i)).toBeInTheDocument();
      expect(screen.queryByText(/Every candidate runs the same full-window request/i)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
