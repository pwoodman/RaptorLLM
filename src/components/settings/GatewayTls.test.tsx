import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api, gatewayRotateTls, RaptorApiError, type GatewayStatus } from "../../api";
import { engineStore } from "../../state/engine";
import { GatewayTls } from "./GatewayTls";

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return { ...actual, gatewayRotateTls: vi.fn(), api: { gateway: { status: vi.fn() } } };
});

const FINGERPRINT = "a".repeat(64);

function gateway(overrides: Partial<GatewayStatus> = {}): GatewayStatus {
  return {
    running: true,
    addr: "0.0.0.0:8790",
    metrics: null,
    last_error: null,
    lan: true,
    tls: true,
    tls_fingerprint_sha256: FINGERPRINT,
    insecure: false,
    ...overrides,
  };
}

describe("GatewayTls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    engineStore.set({ gateway: gateway() });
  });

  it("shows the LAN state, the TLS state, and the fingerprint to pin", () => {
    render(<GatewayTls />);
    expect(screen.getByText("On: listening on all interfaces")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Certificate fingerprint (SHA-256)" })).toHaveValue(FINGERPRINT);
    expect(screen.queryByText("Insecure mode is on")).toBeNull();
  });

  it("copies the fingerprint with the copy button", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<GatewayTls />);

    await user.click(screen.getByRole("button", { name: /Copy fingerprint/ }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(FINGERPRINT));
  });

  it("warns loudly when insecure mode is on", () => {
    engineStore.set({ gateway: gateway({ insecure: true, lan: false, tls: false, tls_fingerprint_sha256: null }) });
    render(<GatewayTls />);
    expect(screen.getByText("Insecure mode is on")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Certificate fingerprint (SHA-256)" })).toHaveValue("No certificate: TLS is off");
  });

  it("rotates the certificate only after confirmation, then shows the new fingerprint", async () => {
    const next = "b".repeat(64);
    vi.mocked(gatewayRotateTls).mockResolvedValue(gateway({ tls_fingerprint_sha256: next }));
    const user = userEvent.setup();
    render(<GatewayTls />);

    await user.click(screen.getByRole("button", { name: "Rotate certificate" }));
    const dialog = screen.getByRole("alertdialog", { name: "Rotate the certificate?" });
    expect(gatewayRotateTls).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(gatewayRotateTls).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Rotate certificate" }));
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Rotate certificate" }));

    await waitFor(() => expect(gatewayRotateTls).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("Certificate rotated")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Certificate fingerprint (SHA-256)" })).toHaveValue(next);
  });

  it("says so when the gateway status cannot be read at all", async () => {
    engineStore.set({ gateway: null });
    vi.mocked(api.gateway.status).mockRejectedValue(new RaptorApiError({ kind: "Internal", message: "bridge offline" }));
    render(<GatewayTls />);
    expect(await screen.findByText("bridge offline")).toBeInTheDocument();
  });

  it("disables rotation when the gateway is not running with TLS", () => {
    engineStore.set({ gateway: gateway({ running: false, tls: false, tls_fingerprint_sha256: null }) });
    render(<GatewayTls />);
    expect(screen.getByRole("button", { name: "Rotate certificate" })).toBeDisabled();
  });

  it("shows the refusal when rotation fails", async () => {
    vi.mocked(gatewayRotateTls).mockRejectedValue(new RaptorApiError({ kind: "Conflict", message: "gateway is not running" }));
    const user = userEvent.setup();
    render(<GatewayTls />);
    await user.click(screen.getByRole("button", { name: "Rotate certificate" }));
    const dialog = screen.getByRole("alertdialog");
    await user.click(within(dialog).getByRole("button", { name: "Rotate certificate" }));
    expect(await screen.findByText("gateway is not running")).toBeInTheDocument();
  });
});

