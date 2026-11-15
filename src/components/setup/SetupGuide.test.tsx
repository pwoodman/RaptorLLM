import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  api,
  runtimeInstallPreview,
  setupStatus,
  type EngineStatus,
  type RuntimeInstallPreview,
  type SetupCheck,
  type SetupStatus,
} from "../../api";
import { engineStore } from "../../state/engine";
import { uiStore } from "../../state/ui";
import { SetupGuide, setupNeedsAttention } from "./SetupGuide";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    setupStatus: vi.fn(),
    runtimeInstallPreview: vi.fn(),
    api: {
      runtime: { ensure: vi.fn(), status: vi.fn() },
      models: { current: vi.fn() },
      gateway: { status: vi.fn() },
    },
  };
});

const LABELS: Record<SetupCheck["id"], string> = {
  hardware: "Hardware detected",
  runtime: "Inference runtime installed",
  gpu: "GPU",
  model: "Model installed",
  storage: "Free storage",
  performance_memory: "Performance memory",
};

function check(id: SetupCheck["id"], ok: boolean, detail = `${id} detail`): SetupCheck {
  return { id, label: LABELS[id], ok, detail };
}

function status(overrides: Partial<SetupStatus> = {}): SetupStatus {
  return {
    checks: [
      check("hardware", true),
      check("runtime", true),
      check("gpu", true),
      check("model", true),
      check("storage", true),
      check("performance_memory", true),
    ],
    hardware: null,
    selected_runtime: { id: "llama-hip-win", version: "b12340", backend: "hip", degraded: false, reason: "", binary: "" },
    installed_runtimes: ["llama-hip-win b12340"],
    available_runtime: null,
    runtime_needs_confirmation: false,
    degraded: false,
    models_installed: 2,
    storage_free_bytes: 50 * 1024 * 1024 * 1024,
    notes: [],
    ...overrides,
  };
}

function installPreview(): RuntimeInstallPreview {
  return {
    id: "llama-hip-win",
    version: "b12340",
    backend: "hip",
    degraded: false,
    reason: "GPU gfx1200 matched.",
    gpu_arch: "gfx1200",
    url: "https://github.com/ggml-org/llama.cpp/releases/download/b12340/llama.zip",
    sha256: "c".repeat(64),
    download_bytes: null,
    size_source: "unavailable",
    already_installed: false,
    rejected: [],
    notes: [],
  };
}

const stoppedEngine: EngineStatus = {
  hardware_detected: true,
  runtime: null,
  server: { state: { state: "stopped" }, endpoint: null, pid: null, restarts: 0 },
  gpu_verdict: null,
};

describe("setupNeedsAttention", () => {
  it("is false when every required check passes and the runtime is not degraded", () => {
    expect(setupNeedsAttention(status())).toBe(false);
  });

  it("ignores informational checks that fail", () => {
    expect(setupNeedsAttention(status({ checks: [check("performance_memory", false)] }))).toBe(false);
  });

  it("is true for a missing model, a pending runtime install, or a CPU build", () => {
    expect(setupNeedsAttention(status({ checks: [check("model", false)] }))).toBe(true);
    expect(setupNeedsAttention(status({ runtime_needs_confirmation: true }))).toBe(true);
    expect(setupNeedsAttention(status({ degraded: true }))).toBe(true);
  });
});

describe("SetupGuide", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    engineStore.set({ status: null, statusError: null });
    uiStore.set({ page: "chats", anchor: null, flash: null });
  });

  it("renders nothing when setup is complete", async () => {
    vi.mocked(setupStatus).mockResolvedValue(status());
    const { container } = render(<SetupGuide />);
    await waitFor(() => expect(setupStatus).toHaveBeenCalled());
    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("lists each unmet item with a clear action", async () => {
    vi.mocked(setupStatus).mockResolvedValue(
      status({
        checks: [
          check("hardware", true),
          check("runtime", true),
          check("model", false, "No model is installed."),
          check("storage", false, "Less than 1 GB free."),
        ],
        storage_free_bytes: 1024,
      }),
    );
    const user = userEvent.setup();
    render(<SetupGuide />);

    const list = await screen.findByRole("list");
    expect(within(list).getAllByRole("listitem")).toHaveLength(4);

    await user.click(screen.getByRole("button", { name: "Go to Models" }));
    expect(uiStore.get().page).toBe("models");

    await user.click(screen.getByRole("button", { name: "Open storage settings" }));
    expect(uiStore.get()).toMatchObject({ page: "settings", anchor: "storage" });
  });

  it("warns loudly when the runtime is a CPU build", async () => {
    vi.mocked(setupStatus).mockResolvedValue(
      status({
        degraded: true,
        selected_runtime: {
          id: "llama-cpu",
          version: "b1",
          backend: "cpu",
          degraded: true,
          reason: "No GPU build is compatible.",
          binary: "",
        },
      }),
    );
    render(<SetupGuide />);
    expect(await screen.findByText("CPU build in use: generation will be slow")).toBeInTheDocument();
    expect(screen.getAllByText("No GPU build is compatible.").length).toBeGreaterThan(0);
  });

  it("shows a retry when the setup check itself fails", async () => {
    vi.mocked(setupStatus).mockRejectedValue(new Error("bridge offline"));
    render(<SetupGuide />);
    expect(await screen.findByText("Setup check unavailable")).toBeInTheDocument();
    expect(screen.getByText("bridge offline")).toBeInTheDocument();
  });

  it("installs the runtime only after the user confirms in the dialog", async () => {
    vi.mocked(setupStatus).mockResolvedValue(
      status({
        checks: [check("runtime", false, "Compatible runtime not installed.")],
        selected_runtime: null,
        available_runtime: { id: "llama-hip-win", version: "b12340", backend: "hip", degraded: false, reason: "GPU gfx1200 matched." },
        runtime_needs_confirmation: true,
      }),
    );
    vi.mocked(api.runtime.ensure).mockResolvedValue({
      id: "llama-hip-win",
      version: "b12340",
      backend: "hip",
      degraded: false,
      reason: "",
      gpu_arch: "gfx1200",
      rejected: [],
    });
    vi.mocked(api.runtime.status).mockResolvedValue(stoppedEngine);
    vi.mocked(api.models.current).mockResolvedValue(null);
    vi.mocked(api.gateway.status).mockResolvedValue({
      running: false,
      addr: null,
      metrics: null,
      last_error: null,
      lan: false,
      tls: false,
      tls_fingerprint_sha256: null,
      insecure: false,
    });

    vi.mocked(runtimeInstallPreview).mockResolvedValue(installPreview());

    const user = userEvent.setup();
    render(<SetupGuide />);

    await user.click(await screen.findByRole("button", { name: "Install llama-hip-win b12340" }));
    const dialog = screen.getByRole("alertdialog", { name: "Install the inference runtime?" });
    expect(dialog).toHaveTextContent("config/runtime-manifest.json");
    expect(api.runtime.ensure).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(api.runtime.ensure).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Install llama-hip-win b12340" }));
    const confirm = await screen.findByRole("button", { name: "Download and install" });
    await waitFor(() => expect(confirm).toBeEnabled());
    await user.click(confirm);
    await waitFor(() => expect(api.runtime.ensure).toHaveBeenCalledTimes(1));
  });
});
