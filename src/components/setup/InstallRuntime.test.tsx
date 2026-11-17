import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, runtimeInstallPreview, RaptorApiError, type RuntimeInstallPreview } from "../../api";
import { InstallRuntime } from "./InstallRuntime";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, runtimeInstallPreview: vi.fn(), api: { runtime: { ensure: vi.fn() } } };
});

function preview(overrides: Partial<RuntimeInstallPreview> = {}): RuntimeInstallPreview {
  return {
    id: "llama-hip-win",
    version: "b12340",
    backend: "hip",
    degraded: false,
    reason: "",
    gpu_arch: "gfx1200",
    url: "https://github.com/ggml-org/llama.cpp/releases/download/b12340/llama.zip",
    sha256: "d".repeat(64),
    download_bytes: null,
    size_source: "unavailable",
    already_installed: false,
    rejected: [],
    notes: [],
    ...overrides,
  };
}

describe("InstallRuntime confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows the source and SHA-256, and says plainly when the size is unknown", async () => {
    vi.mocked(runtimeInstallPreview).mockResolvedValue(preview());
    const user = userEvent.setup();
    render(<InstallRuntime onInstalled={() => {}} />);

    await user.click(screen.getByRole("button", { name: "Set up recommended engine" }));
    const dialog = screen.getByRole("alertdialog");
    expect(await within(dialog).findByText(/Unknown\. The download server did not report a size/)).toBeInTheDocument();
    expect(within(dialog).getByText(preview().url)).toBeInTheDocument();
    expect(within(dialog).getByText("d".repeat(64))).toBeInTheDocument();
  });

  it("shows the reported size when the server sends Content-Length", async () => {
    vi.mocked(runtimeInstallPreview).mockResolvedValue(
      preview({ download_bytes: 280 * 1024 * 1024, size_source: "content_length" }),
    );
    const user = userEvent.setup();
    render(<InstallRuntime onInstalled={() => {}} />);
    await user.click(screen.getByRole("button", { name: "Set up recommended engine" }));
    expect(await screen.findByText(/reported by the download server/)).toBeInTheDocument();
  });

  it("keeps the install disabled when the preview cannot be read", async () => {
    vi.mocked(runtimeInstallPreview).mockRejectedValue(new RaptorApiError({ kind: "Network", message: "offline" }));
    const user = userEvent.setup();
    render(<InstallRuntime onInstalled={() => {}} />);
    await user.click(screen.getByRole("button", { name: "Set up recommended engine" }));

    expect(await screen.findByText("Install details unavailable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download and continue" })).toBeDisabled();
    expect(api.runtime.ensure).not.toHaveBeenCalled();
  });

  it("names the existing copy when the version is already installed", async () => {
    vi.mocked(runtimeInstallPreview).mockResolvedValue(preview({ already_installed: true }));
    const user = userEvent.setup();
    render(<InstallRuntime onInstalled={() => {}} />);
    await user.click(screen.getByRole("button", { name: "Set up recommended engine" }));

    expect(await screen.findByText(/Nothing would be downloaded/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Use this runtime" })).toBeEnabled());
  });
});
