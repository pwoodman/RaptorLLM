import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, RaptorApiError, type RaptorError } from "../../api";
import { engineStore } from "../../state/engine";
import { engineStatus, loadOutcome } from "../../test/fixtures";
import { CpuFallbackBanner } from "./CpuFallbackBanner";
import { LoadFailureCard } from "./LoadFailureCard";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, api: { models: { unload: vi.fn() } } };
});

function refused(): RaptorApiError {
  const error = {
    kind: "Runtime",
    message: "hipblas.dll is needed, but the HIP SDK ships libhipblas.dll",
    backend_failure: {
      backend: "hip",
      cause: "hipblas_name_mismatch",
      message: "The runtime asks for hipblas.dll. The SDK provides libhipblas.dll.",
      suggestions: ["Point PATH at the HIP SDK 7.2 bin folder.", "Use the runtime that matches the SDK."],
    },
  } as RaptorError;
  return new RaptorApiError(error);
}

describe("LoadFailureCard", () => {
  it("shows the GPU diagnosis and its next steps, and says nothing was loaded", () => {
    render(<LoadFailureCard error={refused()} />);

    const card = screen.getByRole("alert");
    expect(card).toHaveTextContent("Model not loaded: The GPU runtime asks for a DLL name the HIP SDK does not ship");
    expect(card).toHaveTextContent("Nothing is loaded and no CPU run was started.");
    expect(screen.getByText("Point PATH at the HIP SDK 7.2 bin folder.")).toBeInTheDocument();
    expect(screen.getByRole("list")).toBeInTheDocument();
  });

  it("falls back to the plain error when there is no GPU diagnosis", () => {
    render(<LoadFailureCard error={new RaptorApiError({ kind: "Conflict", message: "a model is busy" })} />);
    expect(screen.getByText("Action not possible right now")).toBeInTheDocument();
    expect(screen.getByText("a model is busy")).toBeInTheDocument();
    expect(screen.queryByText(/Nothing is loaded/)).toBeNull();
  });
});

describe("CpuFallbackBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    engineStore.set({ loadReport: null, busy: null });
  });

  it("is absent unless the loaded model reported a CPU fallback", () => {
    engineStore.set({ loadReport: loadOutcome({ cpu_fallback: false }) });
    const { container } = render(<CpuFallbackBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it("stays on screen while the model runs on the CPU, and unloads on request", async () => {
    engineStore.set({ loadReport: loadOutcome({ cpu_fallback: true }) });
    vi.mocked(api.models.unload).mockResolvedValue(engineStatus({ state: "stopped" }));
    const user = userEvent.setup();
    render(<CpuFallbackBanner />);

    expect(screen.getByRole("alert", { name: "CPU fallback active" })).toHaveTextContent("NOT running on the GPU");

    await user.click(screen.getByRole("button", { name: "Unload model" }));
    await waitFor(() => expect(api.models.unload).toHaveBeenCalledTimes(1));
  });
});
