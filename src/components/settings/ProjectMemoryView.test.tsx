import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ProjectMemoryEntry } from "../../api";

const mocks = vi.hoisted(() => ({ list: vi.fn(), forget: vi.fn(), remember: vi.fn() }));

vi.mock("../../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    contextProjectMemory: mocks.list,
    contextForgetProjectMemory: mocks.forget,
    contextRememberProject: mocks.remember,
  };
});

import { orderMemory, ProjectMemoryView } from "./ProjectMemoryView";

function entry(id: string, seq: number, over: Partial<ProjectMemoryEntry> = {}): ProjectMemoryEntry {
  return { id, kind: "decision", text: `text ${id}`, sources: [], stale: false, stale_reason: null, seq, ...over };
}

describe("orderMemory", () => {
  it("puts current entries first, newest first, and stale ones last", () => {
    const list = [entry("a", 1), entry("b", 3, { stale: true }), entry("c", 2)];
    expect(orderMemory(list).map((e) => e.id)).toEqual(["c", "a", "b"]);
  });
});

describe("ProjectMemoryView", () => {
  beforeEach(() => {
    mocks.list.mockReset();
    mocks.forget.mockReset();
    mocks.remember.mockReset();
  });

  it("lists entries with protection and staleness, and forgets one", async () => {
    mocks.list.mockResolvedValue([
      entry("a", 1, { text: "Use rustls, never OpenSSL", sources: [{ path: "Cargo.toml", hash: "h" }] }),
      entry("b", 2, { kind: "failed_attempt", text: "mmap on network drive", stale: true, stale_reason: "src/a.rs changed" }),
    ]);
    mocks.forget.mockResolvedValue(undefined);
    render(<ProjectMemoryView workspacePath="C:/p" />);

    expect(await screen.findByText("Use rustls, never OpenSSL")).toBeInTheDocument();
    expect(screen.getByText("always included")).toBeInTheDocument();
    expect(screen.getByText(/stale: src\/a.rs changed/)).toBeInTheDocument();
    expect(screen.getByText(/Cites: Cargo.toml/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Forget Decision/ }));
    await waitFor(() => expect(mocks.forget).toHaveBeenCalledWith("C:/p", "a"));
  });

  it("adds an entry with the files it cites", async () => {
    mocks.list.mockResolvedValue([]);
    mocks.remember.mockResolvedValue({ id: "n", evicted: [] });
    render(<ProjectMemoryView workspacePath="C:/p" />);
    await screen.findByText(/Nothing remembered yet/);

    await userEvent.type(screen.getByLabelText("Text"), "Keep the gateway on localhost");
    await userEvent.type(screen.getByLabelText(/Files it depends on/), "src/a.rs, src/b.rs");
    await userEvent.click(screen.getByRole("button", { name: "Remember" }));

    await waitFor(() =>
      expect(mocks.remember).toHaveBeenCalledWith("C:/p", "decision", "Keep the gateway on localhost", ["src/a.rs", "src/b.rs"]),
    );
  });
});
