import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RaptorApiError, type BackendView } from "../../api";

const mocks = vi.hoisted(() => ({ backend: vi.fn(), setBackend: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    api: { ...actual.api, models: { ...actual.api.models, backend: mocks.backend, setBackend: mocks.setBackend } },
  };
});

import { BackendChoice, backendSummary } from "./BackendChoice";

function view(over: Partial<BackendView> = {}): BackendView {
  return {
    model_id: "local/a",
    preference: "auto",
    effective: "default",
    rocm_installed: true,
    vulkan_installed: true,
    spills: false,
    note: null,
    measured: "Vulkan generated faster on sparse models; ROCm reads prompts 1.2 to 3 times faster.",
    ...over,
  };
}

describe("backendSummary", () => {
  it("says what a load would use, and what is missing", () => {
    expect(backendSummary(view({ effective: "vulkan" }))).toBe("Loads on the Vulkan runtime.");
    expect(backendSummary(view({ effective: "rocm" }))).toBe("Loads on the ROCm runtime.");
    expect(backendSummary(view({ effective: "default" }))).toBe("Loads on the normal runtime.");
    expect(backendSummary(view({ spills: true, vulkan_installed: false }))).toContain("Vulkan runtime is not installed");
    expect(backendSummary(view({ effective: "unavailable", note: "Vulkan is not installed." }))).toBe(
      "Vulkan is not installed.",
    );
  });
});

describe("BackendChoice", () => {
  beforeEach(() => {
    mocks.backend.mockReset();
    mocks.setBackend.mockReset();
  });

  it("shows the saved choice, quotes the measurement, and saves a new choice", async () => {
    mocks.backend.mockResolvedValue(view());
    mocks.setBackend.mockResolvedValue(view({ preference: "vulkan", effective: "vulkan" }));
    render(<BackendChoice modelId="local/a" disabled={false} />);

    const select = await screen.findByRole("combobox");
    expect(select).toHaveValue("auto");
    expect(screen.getByText(/ROCm reads prompts 1.2 to 3 times faster/)).toBeInTheDocument();

    await userEvent.selectOptions(select, "vulkan");
    await waitFor(() => expect(mocks.setBackend).toHaveBeenCalledWith("local/a", "vulkan"));
    expect(await screen.findByText("Loads on the Vulkan runtime.")).toBeInTheDocument();
  });

  it("marks a runtime that is not installed and explains a choice that cannot be honoured", async () => {
    mocks.backend.mockResolvedValue(
      view({
        preference: "vulkan",
        effective: "unavailable",
        vulkan_installed: false,
        note: "You chose Vulkan for this model, but the Vulkan runtime is not installed.",
      }),
    );
    render(<BackendChoice modelId="local/a" disabled={false} />);
    expect(await screen.findByRole("option", { name: "Vulkan (not installed)" })).toBeInTheDocument();
    expect(screen.getByText(/Vulkan runtime is not installed/)).toBeInTheDocument();
  });

  it("is locked while another action runs and reports a failed save", async () => {
    mocks.backend.mockResolvedValue(view());
    mocks.setBackend.mockRejectedValue(new RaptorApiError({ kind: "NotFound", message: "no such model" }));
    const { rerender } = render(<BackendChoice modelId="local/a" disabled={true} />);
    expect(await screen.findByRole("combobox")).toBeDisabled();
    rerender(<BackendChoice modelId="local/a" disabled={false} />);
    await userEvent.selectOptions(screen.getByRole("combobox"), "rocm");
    expect(await screen.findByText(/no such model/)).toBeInTheDocument();
  });

  it("shows the error when the choice cannot be read", async () => {
    mocks.backend.mockRejectedValue(new RaptorApiError({ kind: "Internal", message: "settings unreadable" }));
    render(<BackendChoice modelId="local/a" disabled={false} />);
    expect(await screen.findByText(/settings unreadable/)).toBeInTheDocument();
  });
});
